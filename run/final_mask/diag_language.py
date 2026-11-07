"""
Does language reach the final mask, and by which route? HF validation tuples from tasks with an opposite task (push /
rotate left-right, drawer open-close, slider left-right, LED and lightbulb on-off), released Step 1 and Step 2, same
image and seed throughout. Each row gets three replacement instructions:

  close   the opposite task, by swapping one word where possible ("... to the right" -> "... to the left")
  para    another wording of the same task
  rand    a task from another family

For each, three routes through Steps 1-2, each compared with the baseline (T, M_i(T)):
  text    (T', M_i(T))    only Step 2's text input changes
  mi      (T, M_i(T'))    only Step 1 sees T'
  both    (T', M_i(T'))   the instruction changes in the whole mask pipeline

Also: IoU(M_i(T), M_i(T')), the seed-to-seed floor, and IoU of each output with the ground truth of the original T.
Per-row values are saved.

    python run/final_mask/diag_language.py --per-pair 16 --out run/final_mask/language_test.json
"""

import argparse
import glob
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_onestep import RUN, INFERENCE_CONFIG, iou, load_pipe, prompt  # noqa: E402
import tune_retrieval  # noqa: E402

COLUMNS = ["id", "original_image", "final_edit_region_mask", "edit_instruction"]
SUB = "calvin_abcd_test_data"
OPPOSITE_TASKS = {}
GROUPS = {}
for c in ("red", "blue", "pink"):
    for verb in ("push", "rotate"):
        a, b = f"{verb}_{c}_block_left", f"{verb}_{c}_block_right"
        OPPOSITE_TASKS[a], OPPOSITE_TASKS[b] = b, a
        GROUPS[a] = GROUPS[b] = verb
for name, a, b in (("drawer", "open_drawer", "close_drawer"), ("slider", "move_slider_left", "move_slider_right"),
                   ("led", "turn_on_led", "turn_off_led"), ("lightbulb", "turn_on_lightbulb", "turn_off_lightbulb")):
    OPPOSITE_TASKS[a], OPPOSITE_TASKS[b] = b, a
    GROUPS[a] = GROUPS[b] = name
WORDS = {"left": "right", "right": "left", "open": "close", "close": "open", "on": "off", "off": "on",
         "up": "down", "down": "up", "opening": "closing", "closing": "opening"}


def swap_word(s):
    """Every candidate with exactly one opposite word swapped."""
    tokens = re.findall(r"\w+|\W+", s)
    return ["".join(tokens[:i] + [WORDS[t]] + tokens[i + 1:]) for i, t in enumerate(tokens) if t in WORDS]


def choose_rows(rows, tasks, per_pair, rng):
    by_group = {}
    for i, r in enumerate(rows):
        t = tasks.get(r["edit_instruction"].strip().lower())
        if t in GROUPS:
            by_group.setdefault(GROUPS[t], []).append(i)
    out = []
    for g in sorted(by_group):
        idx = by_group[g][:]
        rng.shuffle(idx)
        out += idx[:per_pair]
    return out


def replacements(instr, tasks, by_task, rng):
    low = instr.strip().lower()
    t = tasks[low]
    opp = OPPOSITE_TASKS[t]
    minimal = [c for c in swap_word(low) if tasks.get(c) == opp]
    close = rng.choice(minimal) if minimal else rng.choice(by_task[opp])
    para = rng.choice([s for s in by_task[t] if s != low])
    others = [k for k in by_task if GROUPS.get(k) != GROUPS[t]]
    rand = rng.choice(by_task[rng.choice(others)])
    return {"close": close, "para": para, "rand": rand}, bool(minimal)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-pair", type=int, default=16, help="Rows per task group (push, rotate, drawer, ...).")
    p.add_argument("--out", default=f"{RUN}/final_mask/language_test.json")
    p.add_argument("--work", default=f"{RUN}/CALVIN-EVALS/fm_lang")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--checkpoint-dir", default=f"{RUN}/ckpts/sge-goal-checkpoints/flux1-kontext-dev/calvin/step2_final_mask")
    p.add_argument("--ier-checkpoint", default=f"{RUN}/ier_training/evf_union/best.pt")
    p.add_argument("--black-mi", action="store_true", help="Black M_i in Step 2, text route only.")
    p.add_argument("--dry-run", action="store_true", help="Print the chosen instructions and stop.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    rng = random.Random(0)
    tasks = tune_retrieval.task_map()
    by_task = {}
    for s, t in tasks.items():
        by_task.setdefault(t, []).append(s)
    for t in by_task:
        by_task[t].sort()
    table = pq.read_table(sorted(glob.glob(f"{RUN}/calvin/hf_dataset/data/validation-*.parquet")), columns=COLUMNS).to_pylist()
    rows = [table[i] for i in choose_rows(table, tasks, args.per_pair, rng)]
    for r in rows:
        r["instr"] = r["edit_instruction"].strip()
        r["task"] = tasks[r["instr"].lower()]
        r["group"] = GROUPS[r["task"]]
        r["alt"], r["one_word"] = replacements(r["instr"], tasks, by_task, rng)
    print(f"{len(rows)} rows; close pairs with a one-word swap: {sum(r['one_word'] for r in rows)}", flush=True)
    for r in rows[::max(1, len(rows) // 12)]:
        print(f"  [{r['group']}] {r['instr']!r} | close {r['alt']['close']!r} | para {r['alt']['para']!r} | rand {r['alt']['rand']!r}", flush=True)
    if args.dry_run:
        return

    # Step 1 under each instruction.
    conds = ["orig", "close", "para", "rand"]
    for c in conds:
        src = f"{args.work}/src_{c}/{SUB}"
        for k, r in enumerate(rows, start=1):
            d = f"{src}/{k}"
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/original_image.png", "wb") as f:
                f.write(r["original_image"]["bytes"])
            with open(f"{d}/edit_instruction.txt", "w") as f:
                f.write((r["instr"] if c == "orig" else r["alt"][c]) + "\n")
        subprocess.run(["bash", f"{RUN}/predict_ier.sh", "--checkpoint", args.ier_checkpoint, "--data-dir", src,
                        "--out-dir", f"{args.work}/mi_{c}/{SUB}", "--skip-existing"], check=True)
    torch.cuda.empty_cache()
    mi = {c: [Image.open(f"{args.work}/mi_{c}/{SUB}/{k}/initial_edit_region_mask.jpg").convert("RGB")
              for k in range(1, len(rows) + 1)] for c in conds}
    if args.black_mi:
        mi = {c: [Image.new("RGB", m.size) for m in ms] for c, ms in mi.items()}
    origs = [Image.open(io.BytesIO(r["original_image"]["bytes"])).convert("RGB") for r in rows]
    gts = [np.asarray(Image.open(io.BytesIO(r["final_edit_region_mask"]["bytes"])).convert("L")) > 127 for r in rows]

    def text_of(i, c):
        return rows[i]["instr"] if c == "orig" else rows[i]["alt"][c]

    all_prompts = [prompt(text_of(i, c)) for i in range(len(rows)) for c in conds]
    pipe, cache = load_pipe(args.checkpoint_dir, all_prompts, args.device)
    w, h = origs[0].size
    tw, th = w // 32 * 32, h // 32 * 32

    def step2(name, text_cond, mi_cond, seed=0):
        t0, out = time.perf_counter(), []
        for b in range(0, len(rows), args.batch):
            idx = list(range(b, min(b + args.batch, len(rows))))
            with torch.no_grad(), torch.autocast("cuda"):
                ims = pipe(original_image=[origs[i] for i in idx], initial_edit_region_mask=[mi[mi_cond][i] for i in idx],
                           prompt_embeds=torch.cat([cache[prompt(text_of(i, text_cond))][0] for i in idx]).to(args.device),
                           pooled_prompt_embeds=torch.cat([cache[prompt(text_of(i, text_cond))][1] for i in idx]).to(args.device),
                           generator=[torch.Generator(device=args.device).manual_seed(seed + i) for i in idx],
                           height=th, width=tw, guidance_scale=1.0, num_inference_steps=10,
                           max_sequence_length=INFERENCE_CONFIG["max_sequence_length"],
                           _auto_resize=False, max_area=tw * th).images
            out += [np.asarray(im.resize((w, h), Image.BILINEAR).convert("L")) > 127 for im in ims]
        print(f"{name}: {time.perf_counter() - t0:.0f}s", flush=True)
        return out

    base = step2("base", "orig", "orig")
    seed2 = step2("seed2", "orig", "orig", seed=10_000)
    bin_mi = {c: [np.asarray(m.convert("L")) > 127 for m in mi[c]] for c in conds}
    per_row = [{"id": r["id"], "group": r["group"], "task": r["task"], "instr": r["instr"], "alt": r["alt"],
                "one_word": r["one_word"], "base_iou_gt": iou(base[i], gts[i]), "seed_floor": iou(base[i], seed2[i])}
               for i, r in enumerate(rows)]
    routes = {"text": lambda c: (c, "orig"), "mi": lambda c: ("orig", c), "both": lambda c: (c, c)}
    if args.black_mi:
        del routes["mi"], routes["both"]
    for c in ("close", "para", "rand"):
        for i in range(len(rows)):
            per_row[i][f"{c}_mi_vs_orig_mi"] = iou(bin_mi[c][i], bin_mi["orig"][i])
        for route, f in routes.items():
            masks = step2(f"{c}_{route}", *f(c))
            for i in range(len(rows)):
                per_row[i][f"{c}_{route}_vs_base"] = iou(masks[i], base[i])
                per_row[i][f"{c}_{route}_iou_gt"] = iou(masks[i], gts[i])

    keys = [k for k in per_row[0] if k.endswith(("_vs_base", "_iou_gt", "_vs_orig_mi", "floor"))]
    summary = {"n": len(rows), "all": {k: float(np.mean([r[k] for r in per_row])) for k in keys}, "by_group": {}}
    for g in sorted({r["group"] for r in per_row}):
        sel = [r for r in per_row if r["group"] == g]
        summary["by_group"][g] = {"n": len(sel), **{k: float(np.mean([r[k] for r in sel])) for k in keys}}
    print(json.dumps(summary, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_row": per_row}, f, indent=1)
    shutil.rmtree(f"{args.work}", ignore_errors=True)
    print("LANGUAGE_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
