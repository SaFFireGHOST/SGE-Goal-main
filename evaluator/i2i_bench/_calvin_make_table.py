#!/usr/bin/env python3
"""Generate an I2I-Bench results table for the 4 new CALVIN ablation methods,
matching the exact format of i2ibench_evaluation_results.txt."""
import json, os, sys
from decimal import Decimal, ROUND_HALF_UP

def hud(x, n):
    """Round half-up to n decimals (matches the existing table's convention)."""
    q = Decimal(1).scaleb(-n)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))

# Folder holding one <method>/ sub-directory of I2I-Bench outputs per ablation method.
ROOT = os.environ.get("I2IBENCH_ROOT", ".")
SUB = "calvin_abcd_ier_fer_gt_test_data"

# (column abbrev, dimension key, weight) in the exact display order
COLS = [
    ("ObjManip",  "object_manipulation_accuracy", 0.100),
    ("ActState",  "action_state_change_accuracy", 0.100),
    ("InstrFol",  "instruction_following_macro",  0.100),
    ("Spatial",   "spatial_accuracy",             0.080),
    ("LocAttr",   "local_attribute_accuracy",     0.080),
    ("NonEdit",   "non_edited_region_fidelity",   0.120),
    ("SubjIdent", "subject_identity_fidelity",    0.120),
    ("PhysPlaus", "physical_plausibility",        0.050),
    ("BlendNat",  "blending_naturalness",         0.050),
    ("EditArt",   "editing_artifacts",            0.050),
    ("CompIntr",  "composition_interaction",      0.050),
    ("WorldKnow", "world_knowledge_reasoning",    0.025),
    ("TextStyle", "text_content_style_accuracy",  0.025),
    ("ImgQual",   "overall_image_quality",        0.025),
    ("Aesthet",   "aesthetic_quality",            0.025),
]

def aggregate(folder):
    """Mean of each dimension over all TCs (from raw per-TC scores), + weighted avg."""
    base = os.path.join(ROOT, folder, SUB) if not folder.startswith("/") else folder
    tcs = sorted([d for d in os.listdir(base) if d.isdigit()], key=int)
    sums = {k: 0.0 for _, k, _ in COLS}
    n = {k: 0 for _, k, _ in COLS}
    for tc in tcs:
        d = json.load(open(os.path.join(base, tc, "evaluation_results.json")))
        for _, k, _ in COLS:
            v = d[k]["score"]
            sums[k] += float(v); n[k] += 1
    # 6-dp means (as stored in the aggregated JSONs); wavg = weighted sum of those, 6-dp
    means = {k: hud(sums[k] / n[k] if n[k] else 0.0, 6) for _, k, _ in COLS}
    wavg = hud(sum(means[k] * w for _, k, w in COLS), 6)
    return means, wavg, len(tcs)

def fmt_row(name, means, wavg):
    # table cells: 6-dp value rounded half-up to 4 dp (matches existing table convention)
    parts = [f"{name:<49}"]
    parts += [f"{hud(means[k], 4):>13.4f} " for _, k, _ in COLS]
    parts += [f"{hud(wavg, 4):>15.4f}"]
    return "|".join(parts)

# ---- header block: reuse verbatim from the existing file ----
existing = open(os.path.join(ROOT, "i2ibench_evaluation_results.txt")).read().split("\n")
HDR  = [l for l in existing if l.startswith("Method")][0]
UNIT = [l for l in existing if "(x0.1)" in l and "(x0.08)" in l][0]
SEP  = [l for l in existing if l.startswith("---")][0]
BAR  = "=" * 120

# ---- self-check: reproduce a published row (flux-kontext-dev-ft on CALVIN) ----
if "--verify" in sys.argv:
    ref = os.path.join(ROOT, "FINAL-EVAL-RUNS-DICEnI2IBench/flux-kontext-dev-ft", SUB)
    m, w, ntc = aggregate(ref)
    print(f"[verify] flux-kontext-dev-ft CALVIN  n={ntc}  WeightedAvg={w:.4f}  (published 0.6157)")
    print("[verify] NonEdit=%.4f (pub 0.9572)  WorldKnow=%.4f (pub 0.7360)  PhysPlaus=%.4f (pub 0.5269)"
          % (m["non_edited_region_fidelity"], m["world_knowledge_reasoning"], m["physical_plausibility"]))
    print(fmt_row("FLUX.1 Kontext [Dev] SGE-Goal (Ours)", m, w))
    sys.exit(0)

# ---- the 4 new methods (folder -> display name) ----
METHODS = [
    ("flux2_klein_calvin_bft_i1",   "FLUX.2 [klein] BFT-i1"),
    ("flux2_klein_calvin_bft_i2",   "FLUX.2 [klein] BFT-i2"),
    ("flux_kontext_calvin_bft_i1",  "FLUX.1 Kontext [Dev] BFT-i1"),
    ("flux_kontext_calvin_bft_i2",  "FLUX.1 Kontext [Dev] BFT-i2"),
]

rows = []
for folder, name in METHODS:
    fpath = os.path.join("CALVIN-ABLATION-I2IBench", folder)
    means, wavg, ntc = aggregate(fpath)
    rows.append((name, folder, means, wavg, ntc))
    # also write the per-method aggregated JSON (matches repo convention)
    out = {
        "model_group": folder,
        "dataset": SUB,
        "total_test_cases": ntc,
        "dimensions": {k: {"average_score": means[k], "num_test_cases": ntc}
                        for _, k, _ in COLS},
        "weighted_average_of_dimensions": wavg,
    }
    os.makedirs(os.path.join(ROOT, "aggregated_i2ibench_eval_results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "aggregated_i2ibench_eval_results",
              f"{folder}__{SUB}.json"), "w"), indent=2, sort_keys=True)

rows.sort(key=lambda r: r[3], reverse=True)  # by weighted avg, descending

# ---- build the output file ----
L = []
L.append(BAR)
L.append("I2I-BENCH EVALUATION RESULTS  —  NEW CALVIN ABLATION METHODS")
L.append(BAR)
L.append("")
L.append("")
L.append(BAR)
L.append("TABLE: CALVIN ABLATION — BFT ITERATIONS (4 methods)")
L.append("Dataset: calvin_abcd_ier_fer_gt_test_data   (1087 test cases per method)")
L.append("Rows sorted by Weighted Average (descending)")
L.append(BAR)
L.append("")
L.append(HDR)
L.append(UNIT)
L.append(SEP)
for name, folder, means, wavg, ntc in rows:
    L.append(fmt_row(name, means, wavg))
L.append(SEP)
L.append("")
L.append("")
L.append("Method  ->  source folder (CALVIN-ABLATION-I2IBench/<folder>/calvin_abcd_ier_fer_gt_test_data):")
for name, folder, means, wavg, ntc in rows:
    L.append(f"    {name:<32s}  ->  {folder}   (n={ntc})")
L.append("")
L.append(BAR)
L.append("DIMENSION WEIGHTS:")
for ab, k, w in COLS:
    L.append(f"    {ab:>9s} ({k}): {w:.3f}")
L.append(BAR)
L.append("")

out_path = os.path.join(ROOT, "i2ibench_evaluation_results_NEW_calvin_ablation.txt")
open(out_path, "w").write("\n".join(L))
print("Wrote", out_path)
print()
print("\n".join(L))
