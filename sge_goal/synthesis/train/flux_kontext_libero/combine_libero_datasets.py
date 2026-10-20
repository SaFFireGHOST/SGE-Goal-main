"""
Combine LGRM_Libero_10 and LGRM_Libero_90 into a single HuggingFace dataset.

- All Libero_10 samples come first, then Libero_90 samples (per split).
- The `id` field is prefixed with "Libero10_" or "Libero90_" respectively.
- Uploaded as SuyashAjitChavan2002/LGRM_Libero_10_90_Combined.
"""

import os

from datasets import load_dataset, concatenate_datasets, DatasetDict
from huggingface_hub import login

HF_TOKEN = os.environ["HF_TOKEN"]

login(token=HF_TOKEN)

print("Loading LGRM_Libero_10...")
ds_10 = load_dataset("SuyashAjitChavan2002/LGRM_Libero_10")
print(f"  Splits: { {s: len(ds_10[s]) for s in ds_10} }")

print("Loading LGRM_Libero_90...")
ds_90 = load_dataset("SuyashAjitChavan2002/LGRM_Libero_90")
print(f"  Splits: { {s: len(ds_90[s]) for s in ds_90} }")

# Prepend identifiers to the id field
def prepend_id(example, prefix):
    example["id"] = prefix + example["id"]
    return example

combined = DatasetDict()
for split in ds_10.keys():
    print(f"\nProcessing split: {split}")
    split_10 = ds_10[split].map(lambda ex: prepend_id(ex, "Libero10_"), desc=f"Prefixing Libero10 {split}")
    split_90 = ds_90[split].map(lambda ex: prepend_id(ex, "Libero90_"), desc=f"Prefixing Libero90 {split}")
    combined[split] = concatenate_datasets([split_10, split_90])
    print(f"  Libero_10: {len(split_10)}, Libero_90: {len(split_90)}, Combined: {len(combined[split])}")

# Verify a few samples
print("\n--- Verification ---")
for split in combined:
    first_id = combined[split][0]["id"]
    last_id = combined[split][-1]["id"]
    print(f"  {split}: {len(combined[split])} samples, first_id={first_id}, last_id={last_id}")

# Upload
repo_id = "SuyashAjitChavan2002/LGRM_Libero_10_90_Combined"
print(f"\nUploading to {repo_id}...")
combined.push_to_hub(repo_id, token=HF_TOKEN)
print("Done!")
