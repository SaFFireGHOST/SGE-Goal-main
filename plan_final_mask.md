# Plan, part 2: the final mask (Step 2) for SGE-Goal on CALVIN

Continues [`plan.md`](plan.md), whose §1–5 (learned `M_i`, Gate 0, Variant A contexts) still hold;
its §6–8 are superseded here. Last updated 2026-11-06.

**Summary.**
- **Pipeline:** Step 1 learned `M_i` (EVF-SAM, test IoU 0.715) → Step 2 `M_f` (released FLUX-Kontext
  LoRA) → Step 3 goal image → frozen GR-MG.
- **The bottleneck is `M_f`:**
  - Step 2's `M_f` gives **62.74** CL1; ground-truth `M_f` gives **72.86**, close to the ground-truth-goal
    ceiling (73.51).
  - Step 2's masks are barely better than copying `M_i`.
- **What didn't work:**
  - Variant A (training the generators on policy success): no gain.
  - Fine-tuning Step 2: supervised and success-DPO gave no gain.
  - IoU-DPO: held-out IoU 0.340 → 0.369, but `va_val` success 64.0 vs Ref 64.2 (§3.2).
  - One-step masks: IoU 0.417, but `va_val` success 63.8 (§3.4).
  - `M_i` dropout alone: held-out IoU 0.340 → 0.354, but masks still ignore the instruction (§3.7).
- **Retrieval:** copying `M_f` from similar training tuples (tuned: **71.21** CL1) beats Step 2 and is not
  significantly below ground-truth `M_f` at CL1, though long chains stay well below it. It is CALVIN-only
  memorisation, a diagnostic, not a method.
- **Diagnosis:** Step 2 barely uses the instruction (§3.5). Opposite instructions give masks with IoU 0.975
  with the original, while real masks for opposite tasks differ a lot. The image and `M_i` drive the output,
  and the seed changes it far more than the text does. Sampling variance (§3.1) is a separate weakness.
- **Next:** counterfactual Flow-DPO on same-scene pairs with different instructions (§5). A perfect `M_f`
  beats retrieval by only 1.66 CL1, so a learned model must justify itself beyond this CALVIN CL1 number.

---

## 1. Test results (GR-MG, all 1087 CALVIN test rows)

| Goal images | `M_i` | `M_f` (test IoU) | CL1 | CL2 | CL3 | CL4 | CL5 |
|---|---|---|---|---|---|---|---|
| Ground-truth goals (`ground_truth`) | – | – | 73.51 | 63.25 | 51.15 | 42.03 | 36.84 |
| `sge_goal_union_gtmf` | learned | ground truth (1.0) | 72.86 | 58.63 | 44.71 | 34.05 | 29.08 |
| `sge_goal_union_nnmf2` | learned | tuned retrieval (0.494) | 71.21 | 52.03 | 32.45 | 19.30 | 13.64 |
| `sge_goal_union_nnmf` | learned | retrieval, nn5 vote (0.422) | 66.33 | 43.18 | 26.88 | 16.93 | 12.44 |
| `sge_goal_union_ier` (Ref) | learned | Step 2 (0.341) | 62.74 | 36.40 | 14.38 | 5.16 | 2.55 |
| `sge_goal_gt_ier` | ground truth | Step 2 (0.368) | 57.22 | 24.32 | 7.70 | 2.79 | 0.85 |
| `sge_goal` (original pipeline) | training-free | Step 2 | 53.17 | 26.09 | 9.34 | 3.47 | 1.84 |

- Only `M_f` differs between the `sge_goal_union_*` rows (same learned `M_i`, Step 3 seed 0).
- **With a perfect `M_f`, Steps 1 + 3 are nearly at the ceiling.** Step 3 is not the problem.
- CL1 rises with `M_f` IoU, then flattens: 0.341 → 62.74, 0.422 → 66.33, 0.494 → 71.21, 1.0 → 72.86.
  Long chains don't flatten (CL5: 2.55 → 12.44 → 13.64 → 29.08).
- IoU may not be the only cause: retrieval masks are voted consensus masks, not diffusion samples.
- Paired tests at CL1 (exact McNemar, all 1087 rows), `sge_goal_union_nnmf2` vs:

  | Compared with | Difference | Wins / losses | p |
  |---|---|---|---|
  | `sge_goal_union_ier` (Step 2) | +8.46 | 161 / 69 | 1e-9 |
  | `sge_goal_union_nnmf` (old retrieval) | +4.88 | 112 / 59 | 6e-5 |
  | `sge_goal_union_gtmf` (ground-truth `M_f`) | −1.66 | 70 / 88 | 0.18 |
  | `ground_truth` (ground-truth goals) | −2.30 | 66 / 91 | 0.055 |

- Results: `run/eval_<method>/table.csv`, per-row `per_sample_*.jsonl` (for McNemar).
- Still unexplained: learned `M_i` beats ground-truth `M_i` (62.74 vs 57.22; see plan.md §1). Likely but
  unproven: labels are noisy, with learned `M_i` agreeing 0.94 with itself on near-duplicates vs 0.79 for
  ground truth (ground-truth `M_f` agreement ~0.8).
- **Use cluster-aware statistics.** The 1087 rows come from 214 start frames, so McNemar above is
  pseudoreplicated. Bootstrap over start frames, CL1:

  | Comparison | Difference (95% CI) |
  |---|---|
  | `nnmf2` − `ier` | +8.46 (4.6, 12.5) |
  | `nnmf2` − `gtmf` | −1.66 (−5.0, 1.6) |
  | `ier` − `gt_ier` | +5.52 (2.5, 8.7) |
  | `gtmf` − `nnmf2`, CL5 | CI (−19.9, +18.2) |

- **CL-n is not a closed-loop rollout:** `aggregate_chain_success.py` multiplies per-row CL1 successes over
  chain skeletons. CL5 rests on 13 subchains from 24 clusters, and `stack_block` is in 89% of CL5 instances.
- **What in the mask predicts success:** better IoU predicts success (logit +0.58, CI 0.24–1.01). Precision
  ≥ 0.7 or recall ≥ 0.85 loses almost nothing. Six articulated tasks account for 52 of the 110-row gap
  between Step 2 and ground-truth `M_f`.

---

## 2. Variant A: done, negative

Held-out comparison on the 300 `va_val` contexts × 2 start states (600 paired rollouts). Only Step 3
differs. Source: `run/variant_a/val/summary.json`.

| Step 3 | Success | vs Ref (95% CI) |
|---|---|---|
| Ref (released) | 64.2 | – |
| A-sparse pilot (Flow-DPO, 102 success pairs) | 65.2 | +1.0 (−1.7, +3.7) |
| A-dense (original loss + `L_beh`, 2 epochs, Modal) | 58.5 | −5.7 (−10.2, −1.0) |
| Ground-truth end-frame goal | 85.7 | +21.5 |

- **A-sup** (supervised Step 2 on learned `M_i`): validation IoU fell 0.330 → 0.28. With ground-truth
  `M_i` it stayed flat (0.358). The released LoRA sits at a plateau for supervised training, and flow loss
  is not a proxy for mask quality (plan.md §6.1).
- **Pilot:** 2,400 rollouts on 300 contexts, 47% with mixed outcomes, but that yielded only 76 Step 2 /
  102 Step 3 pairs (`run/variant_a/pilot/pilot_stats.json`).
  - Success labels are noisy: rollouts are 95% reproducible, and 13% flip under tiny start perturbations.
- **Why the sparse pilot failed:** success labels are too few and too noisy.
- **Why `L_beh` failed** (the loss itself worked: validation `L_beh` fell 0.0036 → 0.00018; 55 wins / 89
  losses):
  - it trained Step 3, which has little room: with a perfect `M_f`, Step 3 is 0.65 CL1 from ground-truth
    goals;
  - it saw one-step estimates below noise 0.35 built from the true end frame, not images sampled from pure
    noise (likely, not tested);
  - there is no control: Step 3 was never trained with the same settings without `L_beh`, and plain
    retraining also hurt Step 2, so −5.7 can't be attributed to `L_beh` itself.

---

## 3. Final-mask work

### 3.1 Diagnostics: what is wrong with Step 2

Sources: `run/final_mask/run_diag.sh` → `fer_diag_summary.json`, `nn_baseline.json`, plus a re-score of
`run/final_mask/step2_samples/`.

- **No better than copying `M_i`:** test IoU 0.341 (Step 2) vs 0.343 (learned `M_i` copied as `M_f`).
- **Underfits its own training data:** 256 train tuples with ground-truth `M_i` score 0.345, vs 0.381
  on validation.
- **High sampling variance.** On 256 validation tuples (learned `M_i`), 8 seeds:

  | Single sample | Agreement between 2 samples (IoU) | Vote of 4 | Vote of 8 | Best of 4 | Best of 8 |
  |---|---|---|---|---|---|
  | 0.349 | 0.42 | 0.406 | 0.411 | 0.446 | 0.478 |

  "Best of N" needs the answer to pick the sample, so it is only an upper bound.
- **Same on 2,000 train tuples × 4 seeds:**

  | Copy `M_i` | Single sample | Vote of 4 | Best of 4 | Mean spread between seeds |
  |---|---|---|---|---|
  | 0.325 | 0.338 | 0.392 | 0.433 | 0.19 IoU |

- **Reading:** a diffusion model draws one plausible mask; IoU rewards the consensus. Averaging alone
  is worth +0.05–0.06 IoU. Retrieval shows the same effect: 1 neighbour 0.433, a vote over 25 0.501
  (validation).
- **Change one input at a time** (`run/final_mask/diag_step2_cause.py`, `diag_step2_controls.py` →
  `step2_cause.json`, `step2_controls.json`; validation, IoU with the original mask):

  | Change | Agreement | IoU with truth |
  |---|---|---|
  | Wrong instruction, same seed (187/192 a different task) | 0.95 | 0.355 vs 0.357 |
  | Other image | 0.70 | 0.315 |
  | Different seed | 0.43–0.45 | – |
  | `M_i` of another row | 0.37 | 0.197, output follows the donor |
  | Black `M_i` | empty mask | – |

- **Ruled out:** the VAE (round trip IoU 0.999) and too few steps (20 steps 0.337 vs 10 steps 0.349).
- **Not better than copying:** from ground-truth `M_i`, Step 2 scores 0.371 while copying that mask scores
  0.413 (test).
- **The text has usable information:** retrieval scores 0.282 with the wrong task, 0.415 with no text and
  0.501 with the right one (articulated: 0.195 / 0.349 / 0.515).

### 3.2 IoU-ranked Flow-DPO on Step 2: done, better IoU but no success gain

The idea was to replace the noisy success labels with clean IoU labels.

- **Samples** (`sge_goal/final_mask/sample_step2.py`, `run/final_mask/run_sample_step2.sh`, local):
  - released Step 2 × 4 seeds on 2,000 HF train tuples (fixed permutation, learned `M_i`);
  - each sample scored by IoU vs ground-truth `M_f`.
- **Pairs** (`run/final_mask/build_iou_pairs.py` → `iou_pairs.json`):
  - winner = best-IoU sample, loser = worst, kept if the gap ≥ 0.1;
  - 1,479 train / 83 held-out pairs, split by episode.
- **Training** (`run/final_mask/run_dpo_step2.sh`):
  - the Variant A trainer through `sge_goal/final_mask/train_flux_lora_lowmem.py` (startup peak ~29 GB
    instead of ~39 GB);
  - init and frozen reference = released Step 2 LoRA; anchor = original Step 2 loss with ground-truth
    `M_i`;
  - lr 1e-5, β 1000, 1 epoch = 184 updates (~46 s each).
- **Run 1 result** (`run/final_mask/training/iou_dpo_step2/metrics.jsonl`):
  - validation IoU (256 tuples, learned `M_i`) 0.330 → best 0.338 (step 106) → 0.334 at the end;
  - held-out pair accuracy 0.55–0.63;
  - no rollouts run.
- **Run 2 result** (lr 5e-5, 2 epochs = 368 updates, ~5 h,
  `run/final_mask/training/iou_dpo_step2_lr5e5/metrics.jsonl`):
  - validation IoU by checkpoint: 0.330 (start), 0.354 (74), 0.341 (148), 0.344 (222), 0.355 (296),
    0.364 (368, the last, = `best/`);
  - held-out pair accuracy 0.51–0.67, pair loss well above its starting 0.69 (4.8–8.4);
  - validation flow loss rose 0.183 → 0.201;
  - `best/` was chosen on the same 256 tuples, so 0.364 may be optimistic, and the checkpoint-to-checkpoint
    swings suggest noise of about ±0.01.
  - Still far below retrieval on validation (0.501).
- **Only 2,000 of the 22,000 train tuples were sampled** (× 4 seeds, 7.8 h, 3.63 s per mask), so this is not a
  full-scale test.
- **Run 2 checked** (`run/final_mask/run_iou_dpo_val.sh` → `run/final_mask/ioudpo_val/`):

  | Measure | Released / Ref | IoU-DPO | Difference (95% CI) |
  |---|---|---|---|
  | IoU, 710 held-out rows | 0.340 | 0.369 | +0.029 (0.024, 0.034) |
  | `va_val` success, 600 rollouts | 64.2 | 64.0 | −0.2 (−4.3, +4.2), 57 wins / 58 losses |

  | Family (contexts) | Ref | IoU-DPO | Difference (95% CI) |
  |---|---|---|---|
  | Articulated (81) | 58.6 | 62.3 | +3.7 (−4.9, 12.3) |
  | Lift / place / stack (117) | 67.9 | 67.5 | −0.4 (−7.3, 6.8) |
  | Push (49) | 56.1 | 45.9 | −10.2 (−18.4, −3.1) |
  | Rotate (53) | 71.7 | 75.5 | +3.8 (−4.7, 13.2) |

  The push drop is one of four post-hoc families, and one-step moved push by +10.2, so don't rely on it.
- **Why no gain (not proven):**
  - +0.03 IoU predicts only about 1–2 points (rough), which this test can't detect; it does exclude +5;
  - better masks didn't give better goals: L1 to the true end frame stayed at 15.52 → 15.56, though masks
    (agreement with Ref 0.74) and goals did change;
  - mask area grew 0.114 → 0.138, which raises IoU by hedging without being more correct (one-step shows the
    same pattern); `va_val` has no true masks to check precision;
  - both masks in a pair share one instruction, so nothing pushes the model to use language.
- **Decision:** drop IoU-DPO; don't combine it with counterfactual DPO (§5).

### 3.3 Retrieval (lookup) `M_f`: a CALVIN-only diagnostic

**Method.** No model is learned. For a query (image, instruction, learned `M_i`):
1. **Candidates:** train tuples of the same CALVIN task from other episodes. Instruction → task comes
   from `~/calvin/calvin_models/conf/annotations/new_playtable*.yaml` (34 tasks).
2. **Score:** IoU(query learned `M_i`, candidate ground-truth `M_i`) − α × standardised image L1, both at
   52×52.
3. **Prediction:** a pixel vote over the top-K candidates' ground-truth `M_f`.

| Version | Settings | Validation IoU | Test IoU (≥ 0.5) | CL1 |
|---|---|---|---|---|
| `run/final_mask/nn_baseline.py` | same instruction, α 0.1, K 5, vote ≥ 0.5 | 0.432 | 0.422 (35.9%) | 66.33 |
| `run/final_mask/tune_retrieval.py` | same task, α 0.3, K 25, vote ≥ 0.4 | 0.501 | 0.494 (51.5%) | 71.21 |
| Oracle: best of the 25 neighbours | – | 0.626 | – | – |

- **Tuning protocol:** tuned on the 966 validation tuples with learned `M_i`; test evaluated once.
  Output: `run/final_mask/retrieval_tuning.json`.
- **What mattered:** the task pool (+0.035) and a vote over more neighbours (+0.03).
- **What never helped:** centroid shift and the medoid. Candidates' ground-truth `M_i` rank better than
  their learned `M_i`.
- **Rollouts by task** (successes for tuned retrieval / Step 2 / ground-truth `M_f`, of n rows):

  | Task | n | Retrieval | Step 2 | Ground-truth `M_f` |
  |---|---|---|---|---|
  | `stack_block` | 40 | 20 | 10 | 22 |
  | `turn_on_led` | 27 | 27 | 17 | 27 |
  | `push_red_block_right` | 46 | 27 | 17 | 23 |
  | `turn_off_lightbulb` | 11 | 11 | 2 | 10 |
  | `push_into_drawer` | 41 | 11 | 22 | 23 |
  | `push_blue_block_left` | 43 | 29 | 34 | 37 |
  | `push_blue_block_right` | 43 | 17 | 18 | 29 |

  The largest shortfalls are block-pushing tasks. That fits the weakness below (fixed pixel positions),
  but it hasn't been tested directly.
- **Not a method:**
  - every test instruction appears verbatim in training (1087/1087 rows, 299 distinct);
  - CALVIN is one fixed scene and camera;
  - it uses CALVIN's hand-written task table;
  - it would fail on new tasks, scenes or object positions.
- **Use it only as:** (a) proof that Step 2 underuses its training data (a lookup beats it in-domain);
  (b) a point on the IoU → success curve. Label it "in-domain retrieval baseline" if reported.

### 3.4 One-step masks: not a new baseline

Threshold the model's one-step estimate at 64 (`run/final_mask/predict_onestep.py`, `run_onestep_val.sh`,
`analyze_onestep_val.py` → `run/final_mask/onestep_val/`).

- **Mask quality:** IoU 0.417 on the 966 validation rows, seed agreement 0.88, 8× faster than 10 steps.
- **Success:** 63.8 vs Ref 64.2, −0.3 (−5.5, +4.8), 79 wins / 81 losses.
- **By family (Ref / one-step):** articulated 58.6 / 55.6, lift-place-stack 67.9 / 68.8, push 56.1 / 66.3,
  rotate 71.7 / 63.2. Mask area 0.149, agreement with Ref 0.54.
- **Reading:** IoU alone doesn't select a model; judge by rollouts. Keep 10 steps.

### 3.5 Language test: Step 2 barely uses the instruction

`run/final_mask/diag_language.py` → `run/final_mask/language_test.json`. 96 validation rows from task groups
with opposites (push, rotate, drawer, slider, LED, lightbulb), same image and seed. 95 of 96 opposite swaps
change one word. Agreement with the original mask:

| Instruction changed to | Step 2 text only | `M_i` only | Both |
|---|---|---|---|
| Opposite task | 0.975 | 0.942 | 0.932 |
| Other wording, same task | 0.956 | 0.952 | 0.925 |
| Task from another group | 0.938 | 0.568 | 0.552 |

- Seed-to-seed agreement is 0.42, so noise changes the mask far more than the instruction.
- IoU with the truth: baseline 0.324, another group's text 0.321 (−0.003), its `M_i` 0.219 (−0.104, CI −0.136,
  −0.075).
- Other wording moves the mask more than the opposite meaning (+0.019, CI 0.009, 0.031), so sensitivity tracks
  text difference rather than meaning.
- `M_i` barely reacts to opposites either (IoU 0.977 between the masks). Direction has to enter through
  Step 2's text input.
- `M_i` is mostly the arm: by the paper's definition it is arm + moved objects. In six extreme rows inspected,
  "turn off the yellow lamp" → "lift the blue block" changed 18 pixels. So IoU may mostly score arm position
  (share not measured).

**Real masks do depend on the instruction** (train split, nearest scene per row, L1 < 6 on 32×32 images):

| Group | Same task | Opposite task | Model, opposite instruction (both) |
|---|---|---|---|
| Push | 0.665 | 0.314 | 0.949 |
| Rotate | 0.631 | 0.428 | 0.868 |
| Slider | 0.539 | 0.319 | 0.939 |
| LED | 0.521 | 0.422 | 0.945 |
| Lightbulb | 0.508 | 0.396 | 0.941 |

Drawer has no such pairs. Closer scenes (L1 < 3): push 0.708 vs 0.263 (n = 10), rotate 0.662 vs 0.533 (n = 40).

**Likely causes (not proven):**
1. **The data rarely needs the text.** Close-scene matches (L1 < 3): push 2,332 same-task vs 10 opposite, rotate
   2,447 vs 40. The image nearly gives away the task.
2. **Ground-truth `M_i` already marks which objects will move**, leaving mainly direction for the text.
3. **`M_i` lines up with the output by position**; text has no position.
4. **Noisy labels:** same-task masks agree only ~0.7, so learning direction lowers the loss less.
5. **The prompt:** the instruction is 4–8 of ~285 T5 tokens; the rest is a fixed paragraph. CLIP (233–237 tokens)
   is cut at 77, so it keeps the instruction, which comes first. Encoders are frozen.
6. **The loss has no such term:** flow MSE + 25 × channel consistency (σ < 0.35), no text dropout.

### 3.6 Same-scene pair supply

Scratch scripts (not kept in the repo). Counted on train data only.

Authors' masked rows (HF train, 21,998 with a known task). Only 7 exact images appear under 2+ different tasks.

| Scene L1 (32×32) | Rows with a different-task match | Same-object match | Real `M_f` IoU: same object, different task | Same task |
|---|---|---|---|---|
| < 1.5 | 95 | 51 | 0.482 | 0.687 |
| < 3 | 920 | 325 | 0.411 | 0.652 |
| < 6 | 14,139 | 5,438 | 0.343 | 0.581 |

- 35% of rows have a same-task match below L1 3 (7,615) and 4% a different-task one.
- 537 of the 920 matches are in the same video, so split by video.
- `M_i` between matched rows is only 0.49–0.55 IoU at L1 < 3, so each row's own `M_i` would leak the answer.
- Opposite-only pairs are scarcer: ~61 rows at L1 < 3, ~1,560 at L1 < 6.

CALVIN D demos by simulator state (`run/calvin/d_train`, 5,124 demos, no masks), start scene only:

| Tolerance (articulated / blocks) | Different-task pairs | Same object | Same object, arm < 5 cm |
|---|---|---|---|
| 1 cm / 0.5 cm | 312 | 247 | 59 |
| 2 cm / 2 cm | 1,071 | 570 | 178 |
| 3 cm / 4 cm | 2,243 | 813 | 200 |

Same-object pairs almost never cover articulated tasks (2 cm: drawer 2, slider 0), since an open drawer can only
close.

### 3.7 `M_i` dropout: failed the language test

`sge_goal/final_mask/train_step2_midrop.py`, `run/final_mask/run_midrop_step2.sh` →
`run/final_mask/training/midrop_step2/`, `run/final_mask/midrop_val/`.

- **Setup:** 1 epoch from the released LoRA, ground-truth `M_i`, lr 5e-5, batch 32 (settings of
  `diag_gtmi_step2`). Each row: 25% black `M_i`, 25% damaged (one shift / grow / shrink / cut, mean IoU 0.71 with
  the true `M_i`, like learned `M_i` at 0.715), otherwise unchanged. Validation untouched.
- **IoU, 256 rows:** 0.330 → 0.317 → 0.314 → 0.319 → 0.345 (steps 0 / 172 / 344 / 516 / 687).
- **IoU, 710 held-out rows:** 0.340 → 0.354, +0.014 (0.010, 0.018).

| Agreement with the original mask | Released | Dropout, real `M_i` | Dropout, black `M_i` |
|---|---|---|---|
| Opposite instruction | 0.975 | **0.971** | 0.960 |
| Task from another group | 0.938 | 0.921 | 0.889 |
| IoU with the truth | 0.324 | 0.331 | 0.231 |

- **Failed:** the bar was below ~0.7 for opposite instructions with real `M_i`. No rollouts run.
- **The image is a shortcut too:** with black `M_i` the model now produces real masks but still ignores the
  text (another group's text: 0.231 → 0.227). Push still reaches 0.396 with no `M_i`; drawer 0.107, slider 0.141.
- **Reading:** removing `M_i` doesn't force the text. Only rows where the image and `M_i` are the same and the
  answer flips can do that.

---

## 4. Jobs (as of 2026-11-06)

Nothing is running. Last chain: `run/final_mask/run_midrop_step2.sh`, `MIDROP_DONE` on 2026-11-04 15:46, log
`run/logs/fm_midrop.log`. Its language test crashed four times out of memory because of other users' jobs, costing
about 15 h; the script now retries.

---

## 5. Next steps (recommended order)

Focus is CALVIN only, aiming at a paper-level contribution. A deterministic predictor alone is not considered
that, so it is no longer the main direction.

1. **Counterfactual Flow-DPO + dropout on close-scene pairs (next).** No rollouts or pipeline needed.
   - **Pairs:** same-object, different-instruction rows from the HF train set (about 325 rows at L1 < 3, looser if
     needed). This includes opposite pairs. Split by video. Check an image sheet of the pairs before training.
   - **Conditioning:** keep row A's image and `M_i` for both directions; only the instruction and which mask is good
     change. Mask B is only approximately right for image A, which suits a preference loss better than a
     supervised target.
   - **Loss**, same noise `ε` and `σ` everywhere, reference = frozen starting LoRA:

     ```text
     E_θ(M|T) = || v_θ(z(M), σ, I, M_i, T) − (ε − x(M)) ||²,  z(M) = (1 − σ)·x(M) + σ·ε
     Δ(M|T)   = E_θ(M|T) − E_ref(M|T)
     L_a = −log sigmoid( −β [ Δ(M_a|T_a) − Δ(M_b|T_a) ] )
     L_b = −log sigmoid( −β [ Δ(M_b|T_b) − Δ(M_a|T_b) ] )
     L   = flow MSE + 25 × channel consistency (σ < 0.35) + λ (L_a + L_b) / 2
     ```

     The formula is `dpo_loss` in `sge_goal/variant_a/flux_lora_trainer.py`, except the instruction must switch
     between `L_a` and `L_b`.
   - **Settings:** start from `run/final_mask/training/midrop_step2/last`, which also makes that run the control.
     λ = 1. β ≈ 300 is a starting guess; at 1000, IoU-DPO's held-out pair loss was unstable.
   - **Dropout:** normal rows 25% black + 25% damaged. Pair rows 0% black, 25% damaged with the same damage on both
     sides, because the test always has a real `M_i`.
   - **Estimated time:** a few hours to build, ~3.5 h to train, ~1.75 h to evaluate. Needs ~30 GB free.
   - **Checks:** language test (opposite-text agreement below ~0.7 with real `M_i`), held-out IoU not below 0.340, and
     pair accuracy: is the output closer to the right mask than to the other instruction's mask? Agreement alone could
     fall from noise.

     | Pair accuracy | Meaning | Next |
     |---|---|---|
     | Rises on train and held-out | It works | `va_val` rollouts vs Ref 64.2 |
     | Rises on train only | Too few pairs | Step 2 below |
     | Rises on neither | The approach isn't working | Rethink before spending on rollouts |

2. **GR-MG same-start rollouts (only if short of pairs; the paper author's suggestion).** Run several instructions on
   the same object from one start state, using `--conditioning text_only` (black goal, 59.0 CL1 in the paper).
   - **Why:** exact pairs that exist no other way, and goals the policy can reach.
   - **Cost:** 233 rollouts/h measured (8 shards), so ~17 h for 1,000 scenes × 4 instructions, an upper limit of about
     2,100 pairs at 59% success. Plus pipeline time, not yet measured.
   - **Masks must come from the authors' pipeline** (`data_pipeline/data_processing_v3.py`), not the simulator, to match
     training labels. It isn't set up here (no `lgie_dp` env, XMem or weights).
   - **Changes needed:** save every frame; keep running after success, since the pipeline picks `I_e` at release; one
     shared `M_i` per start state.
   - **Check first with ~60 rollouts** (about 20 scenes × instruction, its opposite, the same instruction again): the
     tuple survival rate, whether paired rollouts get the same `I_o`, and whether opposites differ more than repeats.
   - **Pipeline settings:** don't upload to Hugging Face (its default); put all new rows in train, not its hash split.
   - **Limits:** environment D only; skewed to what the policy does well (push-left 0.38–0.50 even with true goals).
   - **Full videos if needed for real demos:** re-stream `ducido/calvin_task_D_D_scale_100_lerobo_format`, or test
     re-rendering from the per-frame states in `run/calvin/d_train`.
3. **Different-object pairs,** with shared `M_i = M_i^a ∪ M_i^b`, to reach articulated tasks (52 of the 110-row gap). They
   teach which task, not which direction, and each row's own `M_i` would leak the answer.
4. **Retrain Step 2 with a changed recipe** if fine-tuning moves language use but not enough: counterfactual negatives,
   `M_i` dropout, shorter prompt with text dropout (enables true CFG), Dice on the decoded estimate; select on sampled
   IoU + language sensitivity. Same recipe with a new loss is unlikely to help.
5. **Does retrieval hold on harder cases?** Unseen wordings and less familiar scenes, validation only. Paused by choice.
   It matters because a perfect `M_f` beats retrieval by only 1.66 CL1, so a learned model most plausibly earns its place
   on cases retrieval can't match.
6. **Final evaluation:** the best `M_f` predictor on all 1087 test rows, once (CL1–5), cluster bootstrap vs
   `sge_goal_union_ier`, `nnmf`, `nnmf2` and `gtmf`. Select only on HF validation and `va_val`.

**Not recommended:**
- adding IoU-DPO to counterfactual DPO: no success gain, may reward hedging, adds no language signal, and its large
  gradients (norm ~72 against a clip of 1.0) could crowd out the new loss;
- original-image dropout: the model needs the image to locate things;
- counterfactual losers from a completely different scene: the image alone rejects them, so language isn't needed;
- RL as the main method: 2,400 rollouts gave 76 Step 2 pairs, with 13% label flips;
- more plain supervised or success-DPO fine-tuning of FLUX Step 2 (both flat), or one-step as the baseline (§3.4).

**Compute (measured on the A6000):** 1.05 s/sample, 29.9 GB peak, 6.4 h per epoch of 22k. 22 epochs ≈ 6 days. A requested
RTX Pro 6000 is assumed ~2.5× faster (the proposal's assumption, not measured): 22 epochs ≈ 2.5–3 days.

---

## 6. Files from the final-mask work (not on `main` and uncommitted unless noted)

| File | Purpose |
|---|---|
| `run/final_mask/run_diag.sh`, `make_folders.py`, `analyze_fer_diag.py` | GT-`M_f` oracle eval + Step 2 sampling diagnostics (committed) |
| `run/final_mask/nn_baseline.py`, `run_nn_eval.sh` | First retrieval `M_f` + its eval, 66.33 (committed) |
| `sge_goal/final_mask/sample_step2.py` | Step 2 × K samples per train tuple, IoU-scored; resumable |
| `sge_goal/final_mask/modal_sample_step2.py` | Same on Modal (app `sge-final-mask`, volume `sge-variant-a`) |
| `sge_goal/final_mask/train_flux_lora_lowmem.py` | Variant A trainer with a lower startup memory peak |
| `run/final_mask/run_sample_step2.sh` | Local 2,000 × 4 sampling → `run/final_mask/step2_samples/` (gitignored) |
| `run/final_mask/build_iou_pairs.py` → `iou_pairs.json` | IoU-ranked DPO pairs |
| `run/final_mask/run_dpo_step2.sh` | IoU Flow-DPO on Step 2 (env: NAME, LR, EPOCHS, VAL_EVERY) |
| `run/final_mask/tune_retrieval.py` → `retrieval_tuning.json` | Retrieval tuning; `--write-test-folder` |
| `run/final_mask/run_nn2_eval.sh` | Tuned retrieval → Step 3 → 1087-row eval |
| `run/final_mask/training/` | DPO and dropout checkpoints and metrics |
| `run/final_mask/diag_step2_cause.py`, `diag_step2_controls.py` → `step2_cause.json`, `step2_controls.json` | Change one input at a time (§3.1) |
| `run/final_mask/score_step2_ckpts.py` | Paired IoU on the 710 rows not used for selection; `--ref` reuses saved per-row scores |
| `run/final_mask/predict_onestep.py`, `run_onestep_val.sh`, `analyze_onestep_val.py` → `onestep_val/` | One-step masks; `analyze_onestep_val.py --variant --dir` compares any variant with Ref |
| `run/final_mask/run_iou_dpo_val.sh` → `ioudpo_val/` | IoU-DPO held-out IoU + `va_val` rollouts |
| `run/final_mask/diag_language.py` → `language_test.json` | Language test; `--checkpoint-dir`, `--black-mi` |
| `sge_goal/final_mask/train_step2_midrop.py` | Step 2 training with `M_i` dropout (`--mi-drop`, `--mi-damage`) |
| `run/final_mask/run_midrop_step2.sh` → `midrop_val/` | Dropout training + language tests + held-out IoU |

`.gitignore` (tracked) gained `run/final_mask/step2_samples/` (uncommitted).

Pair counts, mask-vs-success and label-noise figures came from session scratch scripts that are not kept; the numbers
are recorded above.

---

## 7. Conventions and environment

- **Storage:** all outputs, data, logs and checkpoints go under `run/` in this repo, never outside it.
  Don't read the sibling folders of `SGE-Goal-Cleaned/`; `~/calvin`, conda envs and caches are fine.
- **Edits:** don't modify tracked files; add new ones.
- **Long jobs:** run detached (`setsid nohup … > log 2>&1 < /dev/null &`), because of VS Code Remote-SSH.
  To stop one, find its PIDs with `ps aux | grep <script>`. Avoid `pgrep -f`/`pkill -f` with a pattern that
  also appears in your own command line.
- **GPU:** one shared 49 GB A6000; other users take 18–38 GB, and a job can still crash if memory is taken after the
  check, so retry on failure. Measured needs:

  | Job | GPU memory |
  |---|---|
  | Step 2/3 inference (`--low-vram`) | ~25 GB |
  | FLUX LoRA DPO | ~29.5 GB |
  | Rollouts | ~2.5 GB per shard |

  Scripts wait for free memory before starting.
- **Python envs:** `~/.conda/envs/sgegen` (generation, training), `~/.conda/envs/gate0` (CALVIN eval).
- **Eval:** `bash run/run_eval_full.sh 8 <out_dir> 0 1087 <folder under run/CALVIN-EVALS>`, then
  `downstream/calvin/gr_mg/evaluate/aggregate_chain_success.py` with `run/eval_inputs/chain_skeletons.json`.
- **Modal:**
  - free tier; every single run must cost < $30;
  - measure with a short run and get approval before any full run;
  - H100 ≈ $3.95/h; volume `sge-variant-a`; secret `huggingface-secret`.
  - Never print the HF token.
- **Git:**
  - branch `followup/learned-mask-variant-a`; commit only when asked;
  - only this file has been pushed to `main`. `plan.md` and the code it depends on are on the branch (2 commits, not
    pushed) or uncommitted (§6), so a fresh clone of `main` doesn't have them;
  - identity `git -c user.name="SaFFireGHOST" -c user.email="Koushik.Mupparapu@iiitb.ac.in"`;
  - messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
