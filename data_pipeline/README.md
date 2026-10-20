# `data_pipeline/` — automated tuple extraction from robot videos

`data_processing_v3.py` turns raw manipulation episodes into `(I_o, I_e, M_i, M_f, T)` tuples
and pushes them to the Hugging Face Hub in batches. Stages as numbered in the code:

| Code stage | Paper stage | What it does | Models |
|---|---|---|---|
| 0 | 1 | Robotic-arm masking and first-frame selection | EVF-SAM, BLIP-2 (`Salesforce/blip2-flan-t5-xl`) |
| 1 | 2a | Class-agnostic object proposals, filtering and merging | SAM ViT-H |
| 2 | 2b | Temporal object tracking | XMem |
| 3 | 3 + 4 | Dynamic-object identification (LoFTR + 2-means), edit masks, edited-frame detection | LoFTR |
| 4 | — | Deterministic MD5 split (seed 42, 5% val / 5% test) and upload to the Hub | — |
| `generate_instructions.py` | 5 | Edit-instruction generation for datasets without language annotations (run before the stages above) | VideoChat-Flash |

## Setup

```bash
conda env create -f data_pipeline/environment.yml        # env name: lgie_dp
bash third_party/fetch_third_party.sh xmem sam            # → data_pipeline/XMem, data_pipeline/segment-anything
# weights: sam_vit_h_4b8939.pth, XMem/saves/XMem.pth, loftr_weights/*.ckpt — see third_party/README.md
```

## Input layout

```
data_pipeline/<dataset_name>/
├── JPEGImages/<episode>/<frame>.jpg
└── EditInstructions/<episode>/edit_instruction.txt
```

## Run

```bash
cd data_pipeline
# only for datasets without instructions: writes EditInstructions/<episode>/edit_instruction.txt
python3 generate_instructions.py --dataset_name <dataset_name>

PYTHONPATH=./XMem python3 data_processing_v3.py \
  --dataset_name <dataset_name> --batch_size 100 \
  --hf_repo_id <your-user>/<dataset> --hf_token $HF_TOKEN
```

`--resume N` restarts from batch `N`. `combine_libero_datasets.py` merges the LIBERO-10 and
LIBERO-90 Hub datasets into the combined LIBERO-10/90 release. `scripts/` holds small helpers
(copying a test subset, corrupt-image detection and de-duplication). `benchmark_pipeline.py` times
each stage (results in `scripts/runtime/results/`).

The released, processed datasets are listed in the top-level README.
