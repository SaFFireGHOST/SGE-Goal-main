"""Build the CALVIN RLDS dataset by calling TFDS directly.

Invoking the builder in-process avoids `tfds build`'s module-discovery rules
(which want a `<name>_dataset_builder.py` inside a `<name>/` directory).

Output: $CALVIN_RLDS_DIR (default ./rlds), ~3.4 GB for the 1,214 windows used in the paper.
"""
import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

OUT = os.path.abspath(os.environ.get("CALVIN_RLDS_DIR", "rlds"))

from calvin_rlds.calvin_dataset_builder import CalvinDD  # noqa: E402

os.makedirs(OUT, exist_ok=True)
builder = CalvinDD(data_dir=OUT)
print(f"[build] dataset={builder.name} -> {OUT}", flush=True)
builder.download_and_prepare()
info = builder.info
print(info, flush=True)
for split, s in info.splits.items():
    print(f"[build] split={split} episodes={s.num_examples}", flush=True)
print("RLDS_BUILD_DONE", flush=True)
