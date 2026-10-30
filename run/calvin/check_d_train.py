"""Sanity checks for the CALVIN D-training contexts built by build_d_train.py.

1. Overlap: distance of every training start state ([robot_obs, scene_obs]) to the 214 CALVIN
   start frames behind the 1087 test rows; near-duplicates are listed for exclusion.
2. Replay: reset the simulator to stored start states and compare the rendered rgb_static with
   the stored start image (a faithful mirror renders the same scene; a mismatched episode is the
   reference for "different scene").

Run from run/gr_mg_ws with run/eval_env.sh sourced (needs the eval env's CALVIN wrapper):
    python ../calvin/check_d_train.py [--replay 30]
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import eval_method as em  # downstream/calvin/gr_mg/evaluate/eval_method.py

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
OBS_SPACE = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": [], "state_obs": ["robot_obs"],
             "actions": ["rel_actions"], "language": ["language"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=int, default=30)
    p.add_argument("--overlap_tol", type=float, default=1e-2)
    args = p.parse_args()

    cols = ["episode_index", "task", "start_robot_obs", "start_scene_obs", "start_image"]
    t = pq.read_table(sorted(glob.glob(f"{HERE}/d_train/contexts_part*.parquet")), columns=cols).to_pydict()
    starts = np.concatenate([np.array(t["start_robot_obs"]), np.array(t["start_scene_obs"])], 1)

    test = []
    for f in sorted(glob.glob(f"{RUN}/eval_inputs/task_D_D/validation/episode_*.npz")):
        z = np.load(f, allow_pickle=True)
        test.append(np.concatenate([z["robot_obs"], z["scene_obs"]]))
    test = np.array(test)
    dist = np.sqrt(((starts[:, None, :] - test[None, :, :]) ** 2).sum(-1)).min(1)
    overlap = [int(e) for e, d in zip(t["episode_index"], dist) if d < args.overlap_tol]
    report = {"episodes": len(starts), "test_start_frames": len(test),
              "min_dist_to_test": float(dist.min()), "median_dist_to_test": float(np.median(dist)),
              "overlap_episodes": overlap}
    print(f"overlap: {len(overlap)} of {len(starts)} start states within {args.overlap_tol} of a test start "
          f"(min distance {dist.min():.4f}, median {np.median(dist):.3f})")

    env = em._make_env(f"{RUN}/eval_inputs/task_D_D", OBS_SPACE, 0)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(starts), size=min(args.replay, len(starts)), replace=False)
    same, other = [], []
    for j, i in enumerate(pick):
        env.reset(robot_obs=np.array(t["start_robot_obs"][i]), scene_obs=np.array(t["start_scene_obs"][i]))
        obs = env.get_obs()
        rgb = obs["rgb_obs"]["rgb_static"] if "rgb_obs" in obs else obs["rgb_static"]
        rgb = np.asarray(rgb, dtype=float)
        stored = np.asarray(Image.open(io.BytesIO(t["start_image"][i])).convert("RGB"), dtype=float)
        ref = np.asarray(Image.open(io.BytesIO(t["start_image"][pick[(j + 1) % len(pick)]])).convert("RGB"), dtype=float)
        same.append(np.abs(rgb - stored).mean())
        other.append(np.abs(rgb - ref).mean())
    report |= {"replay_n": len(pick), "replay_mae_same": float(np.mean(same)), "replay_mae_same_max": float(np.max(same)),
               "replay_mae_other_episode": float(np.mean(other))}
    print(f"replay: rendered vs stored start image MAE {np.mean(same):.2f} (max {np.max(same):.2f}); "
          f"vs a different episode {np.mean(other):.2f}  (pixel scale 0-255, n={len(pick)})")
    with open(f"{HERE}/d_train/check_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
