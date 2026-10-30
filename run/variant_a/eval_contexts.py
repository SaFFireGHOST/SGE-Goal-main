"""Variant A rollouts on CALVIN D-training contexts: goal variants x start states, frozen GR-MG.

Like run/gate0/eval_gate0.py, but the start state comes from run/calvin/d_train (the demo's first
frame) instead of the test mapping. Goal folders are CALVIN-EVALS style, case <episode_index + 1>.
  perturb 0 = the demo's start state,
  perturb k = arm joints + N(0, perturb_std) noise, seeded by (episode, k) only,
so every variant of a context sees the same start states (paired comparison).
--variants takes name[:n_perturb] items, e.g. "m0a0,m0a1,gt:1".

Run from run/gr_mg_ws with run/eval_env.sh sourced (see run/variant_a/run_pilot.sh).
Writes per_sample_<tag>.jsonl (one record per context x variant x perturb, appended, so a rerun
skips what is already there) and shard_done_<tag>.json.
"""
import argparse
import glob
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pytorch_lightning import seed_everything

import eval_method as em  # downstream/calvin/gr_mg/evaluate/eval_method.py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gate0"))
from eval_gate0 import OBSERVATION_SPACE, perturb_robot_obs  # noqa: E402

D_TRAIN = Path(__file__).resolve().parents[1] / "calvin" / "d_train"


def main():
    seed_everything(0, workers=True)
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True, help="task_D_D (env / scene config only).")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epoch", type=int, default=47)
    p.add_argument("--config_path", required=True)
    p.add_argument("--method_dir", type=Path, required=True)
    p.add_argument("--variants", required=True, help="Comma-separated name[:n_perturb] sub-folders of --method_dir.")
    p.add_argument("--episodes_file", required=True, help="Text file with comma-separated episode indices.")
    p.add_argument("--tag", required=True)
    p.add_argument("--num_perturb", type=int, default=2, help="Default start states per variant.")
    p.add_argument("--perturb_std", type=float, default=0.02, help="Arm-joint noise std (rad).")
    p.add_argument("--ep_len", type=int, default=360)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device_id", type=int, default=0)
    args = p.parse_args()

    eps = [int(e) for e in Path(args.episodes_file).read_text().split(",") if e.strip()]
    variants = []
    for item in filter(None, args.variants.split(",")):
        name, _, n = item.partition(":")
        variants.append((name, int(n) if n else args.num_perturb))
    roots = {v: em._find_method_subdir(args.method_dir / v) for v, _ in variants}
    t = pq.read_table(sorted(glob.glob(str(D_TRAIN / "contexts_part*.parquet"))),
                      columns=["episode_index", "task", "start_robot_obs", "start_scene_obs"])
    ctx = {r["episode_index"]: {**r, "start_robot_obs": np.asarray(r["start_robot_obs"]),
                                "start_scene_obs": np.asarray(r["start_scene_obs"])}
           for r in t.to_pylist() if r["episode_index"] in set(eps)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"per_sample_{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["episode"], r["variant"], r["perturb"]))

    model = em._load_policy(args.ckpt_dir, args.epoch, args.config_path, args.device_id)
    env = em._make_env(args.dataset_path, OBSERVATION_SPACE, args.device_id)
    oracle = em._load_task_oracle()

    n_done = n_success = 0
    for ep in eps:
        c = ctx[ep]
        for variant, n_perturb in variants:
            goal = em._load_method_row(roots[variant], ep)
            for k in range(n_perturb):
                if (ep, variant, k) in done:
                    continue
                env.reset(robot_obs=perturb_robot_obs(c["start_robot_obs"], ep, k, args.perturb_std),
                          scene_obs=c["start_scene_obs"])
                model.reset()
                start_info = env.get_info()
                obs = env.get_obs()
                success, steps = False, args.ep_len
                for i in range(args.ep_len):
                    action, _ = model.step(obs, [deepcopy(goal["edited"])], [goal["instruction"]])
                    obs, _, _, info = env.step(action)
                    if oracle.get_task_info_for_set(start_info, info, {c["task"]}):
                        success, steps = True, i + 1
                        break
                n_done += 1
                n_success += int(success)
                with open(out_path, "a") as f:
                    f.write(json.dumps({"episode": ep, "task": c["task"], "variant": variant, "perturb": k,
                                        "success": success, "steps_used": steps}) + "\n")
                print(f"[va] ep {ep:5d} {variant} p{k} success={int(success)}  "
                      f"running SR={n_success / n_done:.3f} ({n_done})", flush=True)

    with open(args.output_dir / f"shard_done_{args.tag}.json", "w") as f:
        json.dump({"episodes": len(eps), "rollouts_this_run": n_done, "successes_this_run": n_success}, f, indent=2)


if __name__ == "__main__":
    main()
