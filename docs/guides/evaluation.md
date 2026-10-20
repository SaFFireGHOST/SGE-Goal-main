# Evaluation

Two kinds of evaluation are released: **editing-metric diagnostics** that score goal images
directly, and **closed-loop policy evaluation**, where frozen goal-conditioned policies are driven by
the goal images. Downstream success is the measure of utility; the diagnostics explain it.

All of them read goal images from the same folder format the generators write:

```
<test_data>/<case>/
├── original_image.jpg
├── edit_instruction.txt
└── edited_image.jpg        # the generated goal
```

## 1. Diagnostic evaluator (15 dimensions)

Scores each `(I_o, I_e, T)` triplet on R1–R15 (object preservation, scene artifacts,
photorealism, source and destination adherence) with open models only. See
[`evaluator/README.md`](../../evaluator/README.md) for the per-script dimension list.

```bash
conda activate lgie_eval
bash third_party/fetch_third_party.sh xmem sam
cd evaluator          # place sam_vit_h_4b8939.pth, XMem/saves/XMem.pth and loftr_weights/ here
python lgie_evaluator.py --all --data_dir /path/to/test_data --skip-existing
python sge_goal_eval_aggregator.py /path/to/test_data
```

Each case gets an `evaluation_results.json`; the aggregator writes
`aggregated_evaluation_results.json` with the mean of every dimension. To run on Modal instead,
follow [`evaluator/MODAL_README.md`](../../evaluator/MODAL_README.md). Aggregated scores for every
method are in `evaluator/results/aggregated-eval-results/`.

## 2. I2I-Bench comparison

Our implementation of I2I-Bench's 15 single-reference dimensions, used as the general-purpose
editing benchmark:

```bash
cd evaluator/i2i_bench
pip install -r requirements.txt
python runner.py --data-dir /path/to/test_data            # --dimensions ... / --test-cases ...
```

`modal_i2ibench_eval.py` runs it on Modal. Aggregated results and the formatted tables are in
`evaluator/i2i_bench/results/` and `i2ibench_evaluation_results*.txt`.

## 3. CALVIN: GR-MG

`downstream/calvin/gr_mg/` is a fork of GR-MG that evaluates user-supplied goal images on the
1,087 CALVIN test rows. Its [`USAGE.md`](../../downstream/calvin/gr_mg/USAGE.md) walks through the
Modal setup (data and checkpoint upload, the Hugging Face ↔ CALVIN row mapping) and evaluation.

```bash
cd downstream/calvin/gr_mg
bash ../../../third_party/fetch_third_party.sh clover          # CLOVER-repo/ is used by the image
modal run modal_app.py::run_method --method-name <method>      # <method>: folder of goal images on the eval volume
modal run modal_app.py::run_method --method-name ground_truth --use-hf-parquet-goal   # GT-goal oracle
modal run modal_app.py::run_method --method-name <method> --conditioning text_only   # text-only ablation
```

Per-row results are written to the runs volume as `/runs/eval_method/<method>/per_sample_*.jsonl`;
`scripts/post_eval.py` pulls them into `.runlogs/results/<method>/`. Chain-length success rates
(chain lengths 1–5) are then computed from the per-row results:

```bash
python evaluate/build_chains.py --mapping_json <hf_calvin_mapping.json> --out_json chain_skeletons.json
python evaluate/aggregate_chain_success.py \
  --chain_skeletons chain_skeletons.json --results_dir .runlogs/results \
  --methods <method> [<method> ...] --out_csv success.csv --out_json success.json
```

## 4. CALVIN: GHIL-Glue

`downstream/calvin/ghil_glue/modal_app.py` evaluates the GHIL-Glue / SuSIE goal-conditioned
diffusion policy (checkpoint `kyle-hatch-tri/ghil-glue-checkpoints`) on the same rows:

```bash
cd downstream/calvin/ghil_glue
modal run modal_app.py::bootstrap                                 # CALVIN data + policy checkpoint
modal run modal_app.py::smoke_test --method-name ground_truth --num-rows 10
modal run modal_app.py::run_method --method-name <method>
```

Its per-row results feed the same chain aggregation as GR-MG.

## 5. CALVIN: Octo-ft (Octo fine-tuned by us)

`downstream/calvin/octo_ft/` fine-tunes Octo-Base on real CALVIN frames and evaluates it, conditioned on
the goal image only, on the same 1,087 rows (supplement, Sec. A). Unlike the other harnesses it runs on a
local GPU. Its [`README.md`](../../downstream/calvin/octo_ft/README.md) covers data, training and
evaluation; in short:

```bash
cd downstream/calvin/octo_ft
bash setup_octo.sh && export CALVIN_ROOT=$PWD/calvin_data
python data/calvin_train_fetch.py list && python data/calvin_train_fetch.py fetch 0
python data/select_frames.py && python data/calvin_train_fetch.py fetch_selected
CALVIN_RLDS_DIR=$PWD/rlds python calvin_rlds/build_rlds.py
CALVIN_RLDS_DIR=$PWD/rlds bash finetune.sh                     # paper settings are the defaults
bash evaluate/run_eval_shard.sh <method> <row_start> <row_end> <ckpt_dir> results
```

Its per-row results feed the same chain aggregation as GR-MG.

## 6. SIMPLER-Bridge

See the [SIMPLER-Bridge guide](simpler.md).

## Producing the goal images

Every generator writes `edited_image.jpg` into the test folders: SGE-Goal through
`sge_goal/pipelines/` ([quickstart](quickstart.md)), the baselines through the scripts in
[`baselines/`](../../baselines/README.md). The test folders are built from the test splits of the
released datasets.
