#!/usr/bin/env python3
"""Pick a task-stratified subset of CALVIN training windows within a frame budget.

Whole annotated windows are kept (not individual frames) so each RLDS episode is
a coherent trajectory and Octo's `uniform` goal relabeling can sample a real
future frame from within it.
"""
import json, collections, os, numpy as np, random

ROOT = os.path.abspath(os.environ.get("CALVIN_ROOT", "."))   # same root as calvin_train_fetch.py
T = f"{ROOT}/data/task_D_D/training"
BUDGET = int(os.environ.get("FRAME_BUDGET", 64000))          # the paper's run used 64,000
random.seed(0)

ann = np.load(f"{T}/lang_annotations/auto_lang_ann.npy", allow_pickle=True).item()
idx = ann["info"]["indx"]
tasks = ann["language"]["task"]
texts = ann["language"]["ann"]

by_task = collections.defaultdict(list)
for w, (t, txt) in enumerate(zip(tasks, texts)):
    by_task[t].append(w)
for t in by_task:
    random.shuffle(by_task[t])

# round-robin across tasks until the frame budget is hit -> balanced coverage
chosen, frames, order, ptr = [], set(), sorted(by_task), {t: 0 for t in by_task}
while len(frames) < BUDGET:
    progressed = False
    for t in order:
        if ptr[t] >= len(by_task[t]):
            continue
        w = by_task[t][ptr[t]]; ptr[t] += 1; progressed = True
        a, b = int(idx[w][0]), int(idx[w][1])
        new = set(range(a, b + 1))
        if len(frames | new) > BUDGET and frames:
            ptr[t] = len(by_task[t])      # this task is done
            continue
        frames |= new
        chosen.append({"window": w, "task": t, "start": a, "end": b,
                       "instruction": str(texts[w])})
        if len(frames) >= BUDGET:
            break
    if not progressed:
        break

cnt = collections.Counter(c["task"] for c in chosen)
print(f"[select] windows={len(chosen)} frames={len(frames)} tasks={len(cnt)}")
print(f"[select] per-task windows: min={min(cnt.values())} max={max(cnt.values())}")
json.dump({"windows": chosen, "frames": sorted(frames)},
          open(f"{ROOT}/calvin_train_selection.json", "w"))
print("[select] wrote calvin_train_selection.json")
