"""Gate 0 rollouts: several goal variants per context x several start states, frozen GR-MG.

Reuses the model / env / oracle / goal loading of downstream/calvin/gr_mg/evaluate/eval_method.py
(imported, not modified). For each listed HF row, each goal variant (a sub-folder of --method_dir)
is rolled out from --num-perturb start states:
  perturb 0 = the original CALVIN reset (identical to eval_method.py),
  perturb k = arm joints + N(0, perturb_std) noise, seeded by (row, k) only,
so every variant of a row sees the same start states (paired comparison).

Run from run/gr_mg_ws with run/eval_env.sh sourced (see run/run_gate0.sh).
Writes per_sample_<tag>.jsonl (one record per row x variant x perturb) and shard_done_<tag>.json.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from pytorch_lightning import seed_everything

import eval_method as em  # downstream/calvin/gr_mg/evaluate/eval_method.py

OBSERVATION_SPACE = {
    "rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": [], "state_obs": ["robot_obs"],
    "actions": ["rel_actions"], "language": ["language"],
}


def perturb_robot_obs(robot_obs, hf_row, k, std):
    """CALVIN robot_obs = [tcp_pos 3, tcp_orn 3, gripper_width 1, arm_joints 7, gripper_action 1];
    env.reset uses the arm joints (7:14)."""
    if k == 0:
        return robot_obs
    rng = np.random.default_rng(100_000 * hf_row + k)
    out = np.array(robot_obs, dtype=np.float64, copy=True)
    out[7:14] += rng.normal(0.0, std, 7)
    return out


def main():
    seed_everything(0, workers=True)
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--epoch", type=int, default=47)
    p.add_argument("--config_path", required=True)
    p.add_argument("--mapping_json", required=True)
    p.add_argument("--method_dir", type=Path, required=True)
    p.add_argument("--variants", required=True, help="Comma-separated sub-folders of --method_dir.")
    p.add_argument("--rows_file", required=True, help="Text file with comma-separated 0-indexed HF rows.")
    p.add_argument("--tag", required=True)
    p.add_argument("--num_perturb", type=int, default=2)
    p.add_argument("--perturb_std", type=float, default=0.02, help="Arm-joint noise std (rad).")
    p.add_argument("--ep_len", type=int, default=360)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device_id", type=int, default=0)
    args = p.parse_args()

    rows = [int(r) for r in Path(args.rows_file).read_text().split(",") if r.strip()]
    variants = [v for v in args.variants.split(",") if v]
    roots = {v: em._find_method_subdir(args.method_dir / v) for v in variants}
    with open(args.mapping_json) as f:
        matches = {m["hf_row"]: m for m in json.load(f)["matches"]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"per_sample_{args.tag}.jsonl"
    open(out_path, "w").close()

    model = em._load_policy(args.ckpt_dir, args.epoch, args.config_path, args.device_id)
    env = em._make_env(args.dataset_path, OBSERVATION_SPACE, args.device_id)
    val_dir = Path(args.dataset_path) / "validation"
    oracle = em._load_task_oracle()

    n_done = n_success = 0
    for hf_row in rows:
        m = matches[hf_row]
        task = m["task"]
        robot_obs, scene_obs = em._read_npz_init(val_dir, int(m["start_frame"]))
        for variant in variants:
            goal = em._load_method_row(roots[variant], hf_row)
            for k in range(args.num_perturb):
                env.reset(robot_obs=perturb_robot_obs(robot_obs, hf_row, k, args.perturb_std), scene_obs=scene_obs)
                model.reset()
                start_info = env.get_info()
                obs = env.get_obs()
                success, steps = False, args.ep_len
                for i in range(args.ep_len):
                    action, _ = model.step(obs, [deepcopy(goal["edited"])], [goal["instruction"]])
                    obs, _, _, info = env.step(action)
                    if oracle.get_task_info_for_set(start_info, info, {task}):
                        success, steps = True, i + 1
                        break
                n_done += 1
                n_success += int(success)
                with open(out_path, "a") as f:
                    f.write(json.dumps({"hf_row": hf_row, "task": task, "variant": variant, "perturb": k,
                                        "success": success, "steps_used": steps}) + "\n")
                print(f"[gate0] row {hf_row:4d} {variant} p{k} success={int(success)}  "
                      f"running SR={n_success / n_done:.3f} ({n_done})", flush=True)

    with open(args.output_dir / f"shard_done_{args.tag}.json", "w") as f:
        json.dump({"rows": len(rows), "rollouts": n_done, "successes": n_success}, f, indent=2)


if __name__ == "__main__":
    main()
