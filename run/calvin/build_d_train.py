"""Build compact Variant A training contexts from CALVIN task_D_D training (LeRobot mirror on HF).

Source: ducido/calvin_task_D_D_scale_100_lerobo_format (5,124 language-annotated segments of
CALVIN D training; per frame: rgb_static 'image', rgb_gripper 'wrist_image', robot_obs 'state',
'rel_actions', 'scene_obs'). Episodes are streamed into memory, never stored raw.

Kept per episode (run/calvin/d_train/contexts_part*.parquet, one row each):
  instruction, task (via CALVIN's annotation yaml), length,
  start/end robot_obs + scene_obs + static/gripper images,
  all frames' robot_obs, rel_actions, scene_obs (small), window_ends.
Kept for A-dense (run/calvin/d_train/frames_part*.parquet): static + gripper images of two
10-frame history windows per episode (GR-MG seq_len 10), ending at t with 5 future actions
(GR-MG act_len 5) available.

Resumable: finished 500-episode blocks are skipped.
Usage: python run/calvin/build_d_train.py [--workers 8]
"""
import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml
from huggingface_hub import hf_hub_download, hf_hub_url

REPO = "ducido/calvin_task_D_D_scale_100_lerobo_format"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "d_train")
ANN = os.path.expanduser("~/calvin/calvin_models/conf/annotations")
COLUMNS = ["image", "wrist_image", "state", "rel_actions", "scene_obs", "frame_index"]
SEQ_LEN, ACT_LEN, N_WINDOWS, BLOCK = 10, 5, 2, 500


def instruction_to_task():
    table = {}
    for name in ("new_playtable.yaml", "new_playtable_validation.yaml"):
        with open(os.path.join(ANN, name)) as f:
            for task, phrases in yaml.safe_load(f).items():
                for ph in phrases:
                    table.setdefault(ph.strip().lower(), task)
    return table


def fetch(idx, retries=6):
    url = hf_hub_url(REPO, f"data/chunk-{idx // 1000:03d}/episode_{idx:06d}.parquet", repo_type="dataset")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            return pq.read_table(io.BytesIO(r.content), columns=COLUMNS)
        except Exception as exc:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise RuntimeError(f"episode {idx}: {exc}") from exc
            time.sleep(5 * (attempt + 1))


def img_bytes(cell):
    return cell["bytes"] if isinstance(cell, dict) else cell


def window_ends(n, rng):
    lo, hi = SEQ_LEN - 1, n - ACT_LEN  # history t-9..t inside the episode, actions t..t+4 available
    if hi < lo:
        return []
    cands = np.arange(lo, hi + 1)
    return sorted(int(t) for t in rng.choice(cands, size=min(N_WINDOWS, len(cands)), replace=False))


def process(idx, table, meta, task_of):
    t = table.sort_by("frame_index").to_pydict()
    n = len(t["frame_index"])
    instr = meta["tasks"][0]
    ends = window_ends(n, np.random.default_rng(idx))
    ctx = {
        "episode_index": idx, "instruction": instr, "task": task_of.get(instr.strip().lower(), ""),
        "length": n, "window_ends": ends,
        "start_robot_obs": t["state"][0], "start_scene_obs": t["scene_obs"][0],
        "end_robot_obs": t["state"][-1], "end_scene_obs": t["scene_obs"][-1],
        "start_image": img_bytes(t["image"][0]), "start_wrist": img_bytes(t["wrist_image"][0]),
        "end_image": img_bytes(t["image"][-1]), "end_wrist": img_bytes(t["wrist_image"][-1]),
        "robot_obs": [x for row in t["state"] for x in row],
        "rel_actions": [x for row in t["rel_actions"] for x in row],
        "scene_obs": [x for row in t["scene_obs"] for x in row],
    }
    keep = sorted({f for e in ends for f in range(e - SEQ_LEN + 1, e + 1)})
    frames = [{"episode_index": idx, "frame": f, "image": img_bytes(t["image"][f]),
               "wrist": img_bytes(t["wrist_image"][f])} for f in keep]
    return ctx, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in ("meta/episodes.jsonl", "meta/tasks.jsonl", "meta/info.json"):
        hf_hub_download(REPO, name, repo_type="dataset", local_dir=OUT)
    episodes = [json.loads(line) for line in open(os.path.join(OUT, "meta/episodes.jsonl"))]
    meta = {e["episode_index"]: e for e in episodes}
    task_of = instruction_to_task()
    unmapped = sorted({e["tasks"][0] for e in episodes if e["tasks"][0].strip().lower() not in task_of})
    print(f"{len(episodes)} episodes; {len(unmapped)} distinct instructions without a task mapping: {unmapped[:10]}")

    ids = sorted(meta)
    with ThreadPoolExecutor(args.workers) as pool:
        for b in range(0, len(ids), BLOCK):
            block = ids[b:b + BLOCK]
            ctx_path = os.path.join(OUT, f"contexts_part{b // BLOCK:02d}.parquet")
            if os.path.isfile(ctx_path):
                print(f"block {b // BLOCK} done, skipping")
                continue
            t0 = time.time()
            ctxs, frames = [], []
            for idx, table in zip(block, pool.map(fetch, block)):
                c, f = process(idx, table, meta[idx], task_of)
                ctxs.append(c)
                frames.extend(f)
            pq.write_table(pa.Table.from_pylist(frames), os.path.join(OUT, f"frames_part{b // BLOCK:02d}.parquet"))
            pq.write_table(pa.Table.from_pylist(ctxs), ctx_path + ".tmp")
            os.replace(ctx_path + ".tmp", ctx_path)  # contexts file marks the block as complete
            print(f"block {b // BLOCK}: episodes {block[0]}-{block[-1]} in {time.time() - t0:.0f}s", flush=True)
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
