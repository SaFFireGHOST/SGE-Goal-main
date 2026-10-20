# `evaluator/` — 15-dimension diagnostic evaluator

Scores each `(I_o, I_e, T)` triplet along 15 manipulation-specific dimensions using only open
models (VideoChat-Flash `OpenGVLab/VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B`, SAM, XMem,
BLIP-2, LoFTR, …). The VLM prompts are defined verbatim inside each `R*.py` script.

| Script | Dimensions | Group |
|---|---|---|
| `R1.py` | R1 unmentioned stationary objects | object preservation |
| `R2.py` | R2 unmentioned objects moved within the scene | object preservation |
| `R3.py` | R3 unmentioned objects removed from the scene | object preservation |
| `R4.py` | R4 hallucinated objects | object preservation |
| `R5-6-7.py` | R5–R7 unrelated colour / texture / shape change | scene artifacts |
| `R8-9.py` | R8–R9 scene darker / brighter | scene artifacts |
| `R10-11-12.py` | R10–R12 camera zoom / crop / perspective | scene artifacts |
| `R13.py` | R13 photorealism | photorealism |
| `R14-15.py` | R14 source adherence, R15 destination adherence | adherence |

## Running

The scripts resolve their weights and each other relative to this directory, so keep them here.

```bash
conda env create -f evaluator/environment.yml
bash third_party/fetch_third_party.sh xmem sam      # → evaluator/XMem, evaluator/segment-anything
# weights: sam_vit_h_4b8939.pth, XMem/saves/XMem.pth, loftr_weights/*.ckpt — see third_party/README.md

cd evaluator
python lgie_evaluator.py --all --data_dir /path/to/test_data [--skip-existing]
python sge_goal_eval_aggregator.py /path/to/test_data        # → aggregated_evaluation_results.json
```

`lgie_evaluator.py` runs all dimensions in four phases so each model is loaded once.
`modal_lgie_eval.py` / `modal_lgie_eval_partitioned.py` run the same evaluation on Modal (see
`MODAL_README.md`); `setup_shared_volumes.py` and `upload_weights.py` stage weights and data on
Modal volumes, and `verify_integrity.py` checks completed runs.

`results/aggregated-eval-results/` holds the aggregated per-method, per-dataset scores produced by
`sge_goal_eval_aggregator.py`.

## I2I-Bench comparison (`i2i_bench/`)

Our implementation of the 15 single-reference dimensions of I2I-Bench (Wang et al.,
arXiv:2512.04660), used as the general-purpose editing-quality comparison. See
`i2i_bench/README.md`. `modal_i2ibench_eval.py` runs it on Modal; aggregated scores are in
`i2i_bench/results/` and the formatted tables in `i2ibench_evaluation_results*.txt`
(`_calvin_make_table.py` produced the CALVIN-ablation table).
