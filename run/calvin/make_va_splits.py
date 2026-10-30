"""Split the CALVIN D-training contexts into Variant A train / val (by episode, stratified by task).

Excludes episodes flagged by check_d_train.py (start state near a test start) and episodes without
a task name. Val = 300 contexts (task-proportional, >= 1 per task), never used for training.
Writes run/calvin/d_train/splits.json.
"""
import glob
import json
import os
import random
from collections import defaultdict

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "d_train")
N_VAL, SEED = 300, 0

t = pq.read_table(sorted(glob.glob(f"{D}/contexts_part*.parquet")), columns=["episode_index", "task"]).to_pydict()
with open(f"{D}/check_report.json") as f:
    excluded = set(json.load(f)["overlap_episodes"])
by_task = defaultdict(list)
for e, task in zip(t["episode_index"], t["task"]):
    if task and e not in excluded:
        by_task[task].append(e)
total = sum(len(v) for v in by_task.values())

quota = {k: max(1, int(N_VAL * len(v) / total)) for k, v in by_task.items()}
order = sorted(by_task, key=lambda k: N_VAL * len(by_task[k]) / total - quota[k], reverse=True)
for k in order[: N_VAL - sum(quota.values())]:
    quota[k] += 1
rng = random.Random(SEED)
val = sorted(e for k, v in by_task.items() for e in rng.sample(v, quota[k]))
train = sorted(set(e for v in by_task.values() for e in v) - set(val))

splits = {"source": "ducido/calvin_task_D_D_scale_100_lerobo_format (CALVIN D training)",
          "excluded_overlap": sorted(excluded), "n_train": len(train), "n_val": len(val),
          "n_tasks": len(by_task), "train": train, "val": val}
with open(f"{D}/splits.json", "w") as f:
    json.dump(splits, f)
print(f"train {len(train)}, val {len(val)}, tasks {len(by_task)}, excluded {len(excluded)}")
