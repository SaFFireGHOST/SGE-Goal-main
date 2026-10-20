# `downstream/` — closed-loop policy evaluation

Goal images from every generator are fed, unchanged, to frozen goal-conditioned policies; only the
goal generator varies between runs. Two policies are released checkpoints (GR-MG, GHIL-Glue); the third
(Octo-ft) we fine-tuned ourselves, with the code in `calvin/octo_ft/`. The GR-MG, GHIL-Glue and SIMPLER-Bridge
harnesses run on [Modal](https://modal.com); Octo-ft trains and evaluates on a local GPU.

## `calvin/gr_mg/` — GR-MG on CALVIN

A fork of [bytedance/GR-MG](https://github.com/bytedance/GR-MG) (Apache-2.0) that adds evaluation
with user-supplied goal images. `USAGE.md` documents the changes; the upstream README is kept as
`README.md`.

- `evaluate/eval_method.py` — evaluates one method over the 1,087 CALVIN test rows, with goal
  images from a manifest; also supports the ground-truth-goal oracle and text-only conditioning.
- `evaluate/build_hf_calvin_mapping.py` → `evaluate/build_chains.py` → `evaluate/aggregate_chain_success.py`
  — map the Hugging Face test rows back to CALVIN episodes, find back-to-back manipulation chains,
  and compute chain-length 1–5 success rates from the per-row results.
- `modal_app.py`, `scripts/` — Modal entry point and run orchestration.

Requires CLOVER at `gr_mg/CLOVER-repo` (`bash third_party/fetch_third_party.sh clover`).

## `calvin/ghil_glue/` — GHIL-Glue on CALVIN

The GHIL-Glue / SuSIE low-level goal-conditioned diffusion policy
(checkpoint `kyle-hatch-tri/ghil-glue-checkpoints` on Hugging Face) evaluated on the same rows.
`modal_app.py` is the entry point; `jaxrl_m/` is the policy code it imports. The two
`mini_dataset/*.tfrecord` files used by `inference_minimal.py` are not included in git.

## `calvin/octo_ft/` — Octo fine-tuned on CALVIN (Octo-ft)

The third CALVIN policy of the supplement: Octo-Base fine-tuned by us on real CALVIN frames and evaluated,
conditioned on the goal image only, on the same rows and chains as GR-MG and GHIL-Glue. Contains the
training-data fetch and selection, the RLDS builder, the fine-tuning config and launcher, a two-line patch
to Octo, and the evaluation wrapper. See `calvin/octo_ft/README.md`.

## `simpler_bridge/` — SIMPLER-Bridge (WidowX / BridgeData V2)

Four atomic tasks × 24 episodes with the GHIL-Glue BridgeData V2 goal-conditioned policy.

1. `modal_phaseA_dump_obs.py` — dump the deterministic initial observation of every episode.
2. `orchestrate_goals.py` — generate a goal image per (method, task, episode).
3. `modal_rollout_gc.py` — roll out the goal-conditioned policy and report success rates
   (`modal_rollout.py` is the earlier Octo-based harness).
4. `overnight_run.py` — end-to-end driver (goal generation → rollout → success-rate table).

`modal_sapien_*.py`, `modal_vulkan_diag.py`, `modal_test_ik_backend.py` and
`pinocchio_ik_backend.py` set up and work around the SAPIEN rendering and IK backends on Modal.
