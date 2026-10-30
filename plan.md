# Plan: Learned Initial Mask, then Variant A (Generator Alignment) for SGE-Goal on CALVIN

Follow-up to SGE-Goal, implementing Variant A of
[`SGE-Goal_Follow-up_Research_Proposal_V2.md`](SGE-Goal_Follow-up_Research_Proposal_V2.md)
on top of a learned Step 1. Everything runs from `(I_o, T)` with **no ground-truth masks at test
time**. Policy: frozen public GR-MG (epoch 47) throughout.

---

## 1. Results so far (GR-MG, all 1087 CALVIN test rows, 360-step episodes)

| Goal images (Steps 2–3 = released checkpoints) | CL1 | CL2 | CL3 | CL4 | CL5 |
|---|---|---|---|---|---|
| Ground-truth goals (ours) | 73.51 | 63.25 | 51.15 | 42.03 | 36.84 |
| Ground-truth goals (paper) | 73.6 | — | — | — | — |
| SGE-Goal (paper; ground-truth `M_i` **and** `M_f`) | 73.1 | 59.7 | 47.3 | 36.8 | 30.9 |
| **Learned `M_i` (EVF-SAM `union`) + predicted `M_f`** | **62.74** | **36.40** | **14.38** | **5.16** | **2.55** |
| Ground-truth `M_i` + predicted `M_f` | 57.22 | 24.32 | 7.70 | 2.79 | 0.85 |
| Training-free `M_i` + predicted `M_f` (original pipeline) | 53.17 | 26.09 | 9.34 | 3.47 | 1.84 |

- The harness reproduces the paper's oracle (73.51 vs 73.6). The authors confirmed the paper's
  73.1 uses ground-truth `M_i` and `M_f` (future information).
- **Learned `M_i` vs training-free: +9.6 pts CL1** (McNemar p = 1e-8), closing ~47% of the gap
  to the ground-truth-goal ceiling.
- **Learned `M_i` beats ground-truth `M_i` by 5.5 pts** (p = 5e-5), although Step 2's `M_f` is
  slightly better with ground truth (IoU 0.368 vs 0.341). Unexplained; candidates: cleaner, more
  consistent learned masks; interaction with an imperfect `M_f`; PNG vs JPEG mask files.
- Checkpoints: `calvin/step2_final_mask` (epoch 22), `calvin/step3_goal_image` (epoch 18),
  SHA-256 verified against Hugging Face.

### Mask quality vs ground truth (`run/mask_iou_vs_gt.py`, `run/ier_results/`)

| | `M_i` IoU | `M_f` IoU (Step 2 output) |
|---|---|---|
| Training-free Step 1 | 0.102 | 0.064 |
| **Learned `union`** | **0.715** | 0.341 |
| Ground-truth `M_i` | 1.000 | 0.368 |

**Step 2 is now the bottleneck:** even with perfect `M_i`, only 27% of final masks have IoU > 0.5.

---

## 2. Why the training-free Step 1 failed (paper §3.1)

Ground-truth masks were extracted from videos: EVF-SAM arm masks (BLIP-2 verified), SAM + XMem
object tracks, LoFTR motion + KMeans (K=2) for dynamic objects. **`M_i` = arm ∪ moving objects on
`I_o`; `M_f` = the same on the release frame.** The training-free Step 1 segments only named
objects, never the arm, so it can't match the target that Steps 2–3 were trained on.

---

## 3. Learned initial mask (done)

`sge_goal/grounding/learned_ier/` — EVF-SAM (SAM ViT-H frozen; BEiT-3-L + projection + mask
decoder trained, 678M params), `(I_o, T) → M_i`, BCE + Dice, AdamW (lr 2e-5 BEiT / 1e-4 head),
3 epochs, batch 8, on the 22k HF train tuples.

| Variant | Where | Test IoU |
|---|---|---|
| `union` (instruction prompt only) | local A6000, ~5.5 h | **0.715** |
| `arm_object` (arm prompt + instruction, soft union) | Modal A100, ~2.7 h (~$7) | 0.711 |

- Only the union `M_i` is supervised (the dataset has no separate arm / object masks).
- No memorisation: IoU 0.72 on its own training images vs 0.72 on validation
  (`run/ier_results/ier_train_vs_val_iou.json`), so predicted `M_i` on training tuples is
  representative of test-time inputs.
- Naming (`union` / `arm_object` → `joint` / `decomposed`) deferred.

---

## 4. Gate 0 (proposal §6): passed

`run/run_gate0.sh`, `run/eval_gate0/gate0_result.json`. 300 test contexts (34 tasks) × 4 goal
samples (learned `M_i` fixed; Steps 2–3 re-sampled with seeds 0/1000/2000/3000) × 2 start states
(original reset, and arm joints + N(0, 0.02 rad)) = 2,400 rollouts.

| Measure | Result | Threshold |
|---|---|---|
| Cross-fitted headroom of the best goal | **+9.3 pts** (95% CI 7.2–11.7) | ≥ 5 ✓ |
| Contexts with mixed outcomes between goals | **51%** | ≥ 30% ✓ |
| Single goal / best (cross-fitted) / naive best-of-4 | 60.2% / 69.5% / 79.2% | — |
| Same goal flips between the two start states | 13% | — |
| Re-run of identical goal + start state matches earlier eval | 95% | — |

- Headroom comes from Steps 2–3 sampling alone, which is what Variant A trains. Largest in
  articulated / precise tasks (LED, lightbulb, slider, drawer, rotations).
- Rollouts are not fully deterministic (95% reproducible) and flip 13% under tiny start
  perturbations: **use ≥ 2 rollouts per goal and compare goals within a context and start state.**
- Gate 0 used test rows (measurement only). All Variant A training uses the contexts in §5.

---

## 5. Variant A training contexts (done)

`run/calvin/d_train/` (4.8 GB), from the HF mirror
`ducido/calvin_task_D_D_scale_100_lerobo_format` = CALVIN **D training** (the official Freiburg
server times out). Built by `run/calvin/build_d_train.py`, checked by `check_d_train.py`, split by
`make_va_splits.py`.

- 5,124 language-annotated segments, all mapped to a CALVIN task (34 tasks).
- Per episode: start / end `robot_obs`, `scene_obs`, static + gripper images; all frames' states and
  `rel_actions`; two 10-frame image windows (GR-MG `seq_len` 10, `act_len` 5) for A-dense.
- **No overlap** with the 214 test start frames (closest distance 0.38; duplicates ≈ 0).
- **Faithful replay:** resetting to stored start states renders the stored image within 0.10/255
  (worst 0.72) vs ~25 between episodes (n = 100).
- `splits.json`: **4,824 `va_train` / 300 `va_val`** contexts (task-stratified).
- HF images are 208×208; CALVIN frames are 200×200 and not pixel-exact to the HF images (the test
  mapping is nearest-neighbour). Generator inputs from these contexts: bilinear resize 200 → 208.

---

## 6. Variant A design

All conditions start from the released Step 2 / Step 3 LoRAs with the learned `union` `M_i`
frozen. Base = "Ref" (62.74% CL1 on test).

| Condition | Trains | Signal | Rollouts | Role |
|---|---|---|---|---|
| **A-sup** | Step 2 (then optionally Step 3) | Original losses only, with **predicted** `M_i` as input | No | Control: gain from more supervised training alone |
| **A-sparse pilot** | Steps 2–3 | 1 round of success-based Flow-DPO from Ref | Yes | De-risks the success pipeline early |
| **A-dense** | Step 3 | Original losses + `L_beh` (proposal Eq. 3) | No | Dense policy signal |
| **A-full** | Steps 2–3 | A-dense + 3 success rounds | Yes | Main method (proposal) |

### 6.1 A-sup

- **Data:** 22k HF train tuples. Input `M_i` = learned `union` predictions (precomputed once,
  ~1.5 h). Targets unchanged: ground-truth `M_f` (Step 2), ground-truth `I_e` (Step 3).
- **Step 2 first** (the measured bottleneck). Step 3 later with predicted `M_i` + ground-truth
  `M_f` (predicting `M_f` for 22k tuples would take ~30 h).
- **Loss:** identical to the original trainer: `L_flow + 25·L_chan` for σ < 0.35 (Step 2);
  `L_flow + 7·L_static` for σ < 0.35 (Step 3).
- **Optimisation:** init from the released LoRA (rank 16); lr 5e-5 constant after 100 warm-up
  steps; effective batch 32 (micro-batch and accumulation set by measured memory); 1–2 epochs;
  bf16, gradient checkpointing, 8-bit Adam (as the original).
- **Selection:** `M_f` IoU on the 966 HF validation tuples (predicted `M_i`) every ¼ epoch; final
  CALVIN check on `va_val`, then test once.

**Result (2026-10-27).** `M_f` IoU on 256 HF validation tuples (10-step sampling, fixed seeds):

| Run | Input `M_i` (train and val) | Step 0 | 86 | 172 | 344 | 516 | 687 |
|---|---|---|---|---|---|---|---|
| A-sup (`run/variant_a/training/asup_step2`) | learned | 0.330 | – | 0.289 | 0.264 | 0.283 | 0.284 |
| Diagnostic (`run/variant_a/training/diag_gtmi_step2`) | ground truth | 0.358 | 0.324 | 0.356 | – | – | – |

Validation flow loss falls in A-sup (0.165 → 0.153) while sampled IoU drops: the loss is not a
proxy for mask quality. With ground-truth `M_i` IoU stays flat, so the released LoRA is at a
plateau for supervised training, and supervised training on learned `M_i` (with ground-truth
`M_f` targets) hurts. Consequences: A-sup ≈ Ref as a control; Variant A anchor losses should use
ground-truth `M_i`; select checkpoints on sampled IoU / success, never on flow loss.

### 6.2 A-sparse pilot (one success round from Ref)

- **Contexts:** 300 from `va_train` (task-stratified).
- **Factorial sampling** (proposal §4.4): K_m = 2 final masks × K_a = 2 appearances per mask × K_s = 2
  start states (original + perturbed) = 8 rollouts per context, **2,400 rollouts**.
- **Advantages (Eq. 8):** `U(M_f)` = mean success over its 2 appearances × 2 start states;
  `b(c)` = context mean; `A2 = U(M_f) − b(c)`; `A3 = p̂(x) − U(M_f)`.
- **Updates (Flow-DPO, Eq. 5)** against a frozen copy of the released LoRA as reference:
  - Step 2: mask pairs with `U(M_w) − U(M_l) ≥ 0.5`, conditioned on `(I_o, T, M_i)`.
  - Step 3: appearance pairs under the **same** `M_f` with `p̂(x_w) − p̂(x_l) ≥ 0.5`, conditioned on
    `(I_o, T, M_i, M_f)`.
  - Anchor: the original supervised losses on HF train tuples, mixed in each step.
  - β: Diffusion-DPO used values in the low thousands; tune on the pilot. Fallback: Flow-RWR
    (Eq. 4) on the same rollouts.
- **Expected pairs:** with 51% mixed contexts in Gate 0, on the order of ~100 pairs per stage from
  300 contexts. Enough to test the pipeline and the sign of the effect, not the final effect size.
- **Eval:** `va_val` (300 contexts) × 1 goal (fixed seed) × 2 start states = 600 rollouts, Ref vs
  pilot, paired (McNemar).

### 6.3 A-dense (later, on Modal)

- `L_beh` (Eq. 3): decode the one-step estimate `Î_e` (already computed for `L_static`), feed it
  with differentiable augmentations to the frozen GR-MG on demo windows from `d_train`, and penalise
  the action discrepancy against the same policy conditioned on the ground-truth end frame.
- Needs FLUX + VAE decode + GR-MG in one graph: beyond the ~40 GB free on the shared A6000 →
  Modal A100-80GB / H100.

### 6.4 Code to build (new files only)

| File | Purpose |
|---|---|
| `run/variant_a/make_context_folders.py` | `d_train` contexts → CALVIN-EVALS folders (`I_o` resized to 208, instruction) |
| `run/variant_a/eval_contexts.py` | Rollouts from stored start states (like `eval_gate0.py`), variants × start states |
| `run/variant_a/predict_hf_ier.py` | Learned `M_i` for the HF train / val tuples |
| `sge_goal/variant_a/flux_lora_trainer.py` | Lean FLUX-Kontext LoRA trainer: modes `sup` / `dpo` / `rwr`, Step 2 or 3, init from released LoRA, frozen reference adapter |

---

## 7. Compute and cost (estimates, to be measured before each full run)

| Job | Where | Estimate |
|---|---|---|
| A-sup, Step 2, 1 epoch | local A6000 | measured 1.03 s/sample, 28 GB peak → ~6.3 h train + ~0.5–1 h val — free |
| Pilot generation (600 masks + 1,200 goals) | local | ~3 h |
| Pilot rollouts (2,400) | local, 8 workers | ~10.5 h |
| Pilot DPO update | local (if it fits) | a few hours |
| `va_val` eval per condition (600 rollouts + 300 goals) | local | ~3.5 h |
| A-dense, 1–2 epochs over ~10k windows | Modal | $6–12 (H100) / $10–20 (A100-80GB) |

Modal credit: $30/month; ~$7–8 used this month (check the dashboard). Measure speed and memory
with short runs first; give the projected cost before any full Modal run.

---

## 8. Order of work

| # | Step | Status |
|---|---|---|
| 1 | Learned `M_i` (EVF-SAM), downstream eval | done (62.74%) |
| 2 | Stage isolation: ground-truth `M_i` + predicted `M_f` | done (57.22%) |
| 3 | Gate 0 headroom | done, passed |
| 4 | Variant A training contexts from CALVIN D training | done |
| 5 | Shared code (§6.4) + timing / memory tests | done (trainer `sge_goal/variant_a/flux_lora_trainer.py`) |
| 6 | A-sup (Step 2) locally ‖ A-sparse pilot locally | A-sup Step 2 done: no gain, IoU 0.33 → 0.28 (see §6.1). Pilot running since 2026-10-27 13:01 (`run/variant_a/run_pilot.sh`, log `run/logs/va_pilot.log`): generation ~3.5 h, then 2,700 rollouts (incl. 300 ground-truth-goal references) ~12 h; then `build_pilot_pairs.py` → trainer `--mode dpo` |
| 7 | A-dense on Modal (after cost estimate) | |
| 8 | A-full: 3 success rounds on A-dense | |
| 9 | Final test eval (1087 rows, CL1–5) of Ref, A-sup, A-dense, A-full | |

---

## 9. Conventions

- All outputs, data, logs and checkpoints live under `run/` in this repo; nothing outside it.
- Don't modify tracked files; add new files instead.
- Long jobs run detached (`setsid nohup … &`) so they survive the SSH / VS Code session closing.
- Eval: `bash run/run_eval_full.sh 8 <out_dir> 0 1087 <goal folder under run/CALVIN-EVALS>`
  (generated goals) or `bash run/run_eval_gt.sh 8` (ground-truth goals); aggregate with
  `downstream/calvin/gr_mg/evaluate/aggregate_chain_success.py`.
- Generation: the three `local_*` step scripts (`--skip-existing --low-vram --seed`), or
  `run/run_ier_pipeline.sh` for the learned-`M_i` pipeline.
