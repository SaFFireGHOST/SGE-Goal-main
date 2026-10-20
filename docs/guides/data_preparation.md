# Data preparation

The processed training data is released, so you only need this guide to rebuild it or to extract
tuples from a new robot dataset.

## Use the released datasets

| Dataset | Hugging Face | Train / Val / Test |
|---|---|---|
| BridgeDataV2 (teleoperated) | [`RiddhiCh/LGRM_BridgeDataV2_Teleoperated`](https://huggingface.co/datasets/RiddhiCh/LGRM_BridgeDataV2_Teleoperated) | 40,851 / 1,000 / 2,000 |
| CALVIN-ABCD | [`RiddhiCh/Calvin-Dataset-Final`](https://huggingface.co/datasets/RiddhiCh/Calvin-Dataset-Final) | 22,000 / 966 / 1,087 |
| LIBERO-10/90 (combined) | [`RiddhiCh/LGRM_Libero_10_90_Combined`](https://huggingface.co/datasets/RiddhiCh/LGRM_Libero_10_90_Combined) | 6,464 / 358 / 376 |

```python
from datasets import load_dataset
ds = load_dataset("RiddhiCh/LGRM_BridgeDataV2_Teleoperated")
row = ds["train"][0]
# row["original_image"], row["edited_image"], row["initial_edit_region_mask"],
# row["final_edit_region_mask"] (PIL images), row["edit_instruction"] (str), row["id"]
```

The training scripts read these datasets directly from the Hub (`dataset_name` in each launcher's
`TRAINING_CONFIG`).

## Build tuples from robot videos

Everything below runs from `data_pipeline/` in the `lgie_dp` environment
([installation](installation.md#3-local-environments-optional)), after
`bash third_party/fetch_third_party.sh xmem sam` and placing the SAM, XMem and LoFTR weights.

### 1. Lay out the episodes

```
data_pipeline/<dataset_name>/
├── JPEGImages/<episode>/<frame>.jpg
└── EditInstructions/<episode>/edit_instruction.txt     # one atomic instruction per episode
```

### 2. Generate instructions (only if the dataset has none)

```bash
python3 generate_instructions.py --dataset_name <dataset_name>
```

VideoChat-Flash watches each episode and writes `EditInstructions/<episode>/edit_instruction.txt`.

### 3. Extract the tuples

```bash
export HF_TOKEN=<your-hf-token>
PYTHONPATH=./XMem python3 data_processing_v3.py \
  --dataset_name <dataset_name> --batch_size 100 \
  --hf_repo_id <your-user>/<dataset-repo>
```

For each batch of episodes the script

1. masks the robot arm and selects the reference frame (EVF-SAM, verified with BLIP-2),
2. proposes, filters and merges object masks on that frame (SAM),
3. tracks the objects through the episode (XMem),
4. separates moved from stationary objects with LoFTR matching and 2-means clustering, writes the
   initial and final edit-region masks and picks the edited frame,
5. assigns each sample to train / validation / test by MD5 hash (seed 42; 5% validation, 5% test)
   and uploads the batch to the Hub repository.

`--resume N` restarts from batch `N`. `benchmark_pipeline.py` runs the same stages with per-stage
timing.

### 4. LIBERO

LIBERO-10 and LIBERO-90 were processed separately and merged with `combine_libero_datasets.py`
(reads `HF_TOKEN`; the source and target repository names are set in the script).
