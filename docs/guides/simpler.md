# SIMPLER-Bridge evaluation

SIMPLER-Bridge is the WidowX / BridgeData V2 setup of [SIMPLER](https://github.com/simpler-env/SimplerEnv).
Goal images from each generator drive the frozen GHIL-Glue (SuSIE) BridgeData V2 goal-conditioned
diffusion policy on four atomic tasks, with 24 episodes each:

`widowx_spoon_on_towel` · `widowx_carrot_on_plate` · `widowx_stack_cube` · `widowx_put_eggplant_in_basket`

All code is in `downstream/simpler_bridge/` and runs on Modal; the containers are built by the
scripts themselves.

## 1. Dump the initial observations

```bash
cd downstream/simpler_bridge
modal run modal_phaseA_dump_obs.py::main
```

This resets each task to its deterministic initial layout for all 24 episodes and saves
`original_image.jpg` and `edit_instruction.txt` to the `simpler-eval-data` volume under
`/simpler_obs/<task>/ep<NNN>/`.

## 2. Generate the goal images

`orchestrate_goals.py` downloads those observations, runs one generator on them, and uploads the
goals to `/goals/<method>/<task>/` on the same volume. It expects one folder per generator (code
plus checkpoints); `make_inferencers_layout.sh` builds that layout as symlinks into this
repository and the downloaded checkpoints:

```bash
huggingface-cli download RiddhiCh/SGE-Goal --local-dir /abs/path/sge-goal-checkpoints
bash make_inferencers_layout.sh /abs/path/sge-goal-checkpoints        # → ./inferencers
export SGE_INFERENCERS_DIR=$PWD/inferencers

python orchestrate_goals.py --method flux2-klein-ours-klein           # SGE-Goal (FLUX.2 klein)
python orchestrate_goals.py --method flux-kontext-ours-klein          # SGE-Goal (FLUX.1 Kontext)
python orchestrate_goals.py --method gemini-3-pro-image               # baselines: see --help for all methods
```

The proprietary baselines need `GEMINI_API_KEYS` / `OPENAI_API_KEY`
([installation](installation.md#5-credentials-used-by-individual-scripts)), and SuSIE / CLOVER need
`bash third_party/fetch_third_party.sh susie clover`.

## 3. Roll out the policy

```bash
modal run modal_rollout_gc.py::run_method --method flux2-klein-ours-klein
```

This rolls out the policy for every task and episode with the goals on the volume and reports the
success rate per task. `overnight_run.py` chains steps 2 and 3 for several methods and compiles
the success-rate table; `roll_watcher.sh` launches rollouts as goals appear on the volume.

## Simulator environment

SAPIEN 2.2.2 needs some care in a headless cloud container. The rollout image in
`modal_rollout_gc.py` already includes all of the following; the other scripts are the diagnostics
used to arrive at it:

| Issue | Handled by | Diagnostic |
|---|---|---|
| No GPU display in the container | CPU Vulkan rendering with Mesa lavapipe | `modal_vulkan_diag.py`, `modal_simpler_smoke.py`, `modal_sapien_kisak.py` |
| SAPIEN's bundled pinocchio segfaults in forward kinematics on newer glibc | `pinocchio_ik_backend.py` swaps in standalone pinocchio for the controller | `modal_sapien_2204.py`, `modal_test_ik_backend.py` |
| Other SAPIEN 2.2.2 ABI issues | version pins and patches in the image definition | `modal_env_research.py` |

`modal_rollout.py` is an earlier harness that used Octo as the low-level policy; it is kept for
reference.
