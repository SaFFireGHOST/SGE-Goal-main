#!/usr/bin/env python3
"""
Local orchestrator for per-method goal-image generation on SIMPLER-Bridge.

For each (method, task) it:
  1. downloads the deterministic SIMPLER initial obs from the rollout volume
     (`simpler-eval-data:/simpler_obs/<task>/ep<NNN>/{original_image.jpg,
     edit_instruction.txt}`) into a local stage dir,
  2. runs the method's editor pipeline (the user's existing Modal scripts under
     Inferencers/) on the stage dir -> each case dir gets `edited_image.jpg`
     (the goal image; masked methods also get IER/FER masks first),
  3. uploads the stage dir to `simpler-eval-data:/goals/<method>/<task>/`.

Afterwards `modal run modal_rollout.py::run_method --method <method>` runs the
shared Octo goal-conditioned rollouts and computes the success rate.

This script only does I/O + launches Modal jobs locally (no heavy local compute);
all model inference happens on Modal, per the project constraint.

Usage:
    python orchestrate_goals.py --method flux-kontext-dev --tasks all
    python orchestrate_goals.py --method flux-kontext-dev-bft-i2 --tasks widowx_spoon_on_towel
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Folder holding one sub-directory per goal generator (code + checkpoints), e.g.
# flux2-klein-ft/, flux-kontext-dev-ft/bridge_and_calvin/, susie/ — see docs/guides/simpler.md.
INF = os.environ.get("SGE_INFERENCERS_DIR", os.path.join(HERE, "inferencers"))
ROLLOUT = os.path.join(HERE, "modal_rollout.py")
VOL = "simpler-eval-data"
# Local staging root. Overridable so parallel per-account drivers don't collide
# on the same /tmp dir (each account-driver sets a unique SGE_STAGE_ROOT).
STAGE_ROOT = os.environ.get("SGE_STAGE_ROOT", "/tmp/sge_stage")
TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]
# IER mask prediction code (Grounded-SAM) shared by the masked methods
IER_DIR = os.path.join(INF, "flux-kontext-dev-ft/bridge_and_calvin/ier-mask-prediction-code")


def sh(cmd, cwd=None):
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd or os.getcwd()})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def stage_task(task, parent=STAGE_ROOT):
    """Export SIMPLER obs for `task` into <parent>/<task>/ (via the rollout app's
    tar-based `stage` entrypoint; `modal volume get -r` is buggy). `parent` is
    method-specific so concurrent method runs don't race on the same dir."""
    dest = os.path.join(parent, task)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    sh(["modal", "run", ROLLOUT + "::stage", "--task", task, "--dest", dest])
    cases = sorted(d for d in os.listdir(dest)
                   if os.path.isdir(os.path.join(dest, d)) and d.startswith("ep"))
    assert cases, f"no episodes staged for {task} at {dest}"
    return dest, cases


def push_goals(method, task, stage_task_dir):
    """Upload edited_image.jpg per case to /goals/<method>/<task>/ on the volume
    (via the rollout app's `push` entrypoint)."""
    sh(["modal", "run", ROLLOUT + "::push", "--method", method,
        "--task", task, "--src", stage_task_dir])


def verify_goals(stage_task_dir, cases):
    missing = [c for c in cases
               if not os.path.isfile(os.path.join(stage_task_dir, c, "edited_image.jpg"))]
    if missing:
        print(f"  WARNING: {len(missing)}/{len(cases)} cases missing edited_image.jpg: {missing[:5]}")
    else:
        print(f"  OK: all {len(cases)} cases have edited_image.jpg")
    return not missing


# ---------------------------------------------------------------------------
# Per-method editor recipes. Each takes the staged task dir (case subdirs with
# original_image.jpg + edit_instruction.txt) and must leave edited_image.jpg in
# each case dir. They invoke the user's Modal scripts under Inferencers/.
# ---------------------------------------------------------------------------
def recipe_ier(stage_task_dir):
    """Predict initial_edit_region_mask.jpg (Grounded-SAM) for each case."""
    sh(["modal", "run", "modal_ier_mask_prediction.py", "--data-dir", stage_task_dir],
       cwd=IER_DIR)


def recipe_flux_kontext_dev(stage_task_dir):
    sh(["modal", "run", "modal_flux_kontext_pretrained_inference.py",
        "--data-dir", stage_task_dir],
       cwd=os.path.join(INF, "flux-kontext-dev"))


def recipe_flux2_klein(stage_task_dir):
    sh(["modal", "run", "modal_flux2_klein_pretrained_inference.py",
        "--data-dir", stage_task_dir],
       cwd=os.path.join(INF, "flux2-klein"))


def recipe_flux_kontext_dev_bft_i1(stage_task_dir):
    sh(["modal", "run", "modal_i1_inference.py", "--data-dir", stage_task_dir,
        "--checkpoint-dir", "models/kontext-i2i/checkpoint-epoch-13"],
       cwd=os.path.join(INF, "flux-kontext-dev-bft-i1"))


def recipe_flux_kontext_dev_bft_i2(stage_task_dir):
    recipe_ier(stage_task_dir)  # i2 needs the initial mask as editor input
    sh(["modal", "run", "modal_i2_inference.py", "--data-dir", stage_task_dir,
        "--checkpoint-dir", "models/kontext-i2i/checkpoint-epoch-9"],
       cwd=os.path.join(INF, "flux-kontext-dev-bft-i2"))


def recipe_flux2_klein_bft_i1(stage_dir):
    sh(["modal", "run", "modal_flux2_klein_ie_inference.py", "--data-dir", stage_dir,
        "--checkpoint-dir",
        "models/flux-2-klein-lora-bridge-v2-ablation-no-masks/checkpoint-epoch-10"],
       cwd=os.path.join(INF, "flux2-klein-bft-i1"))


def recipe_flux2_klein_bft_i2(stage_dir):
    recipe_ier(stage_dir)
    sh(["modal", "run", "modal_flux2_klein_ie_inference.py", "--data-dir", stage_dir,
        "--checkpoint-dir", "models/flux-2-klein-lora-bridge-v2/checkpoint-epoch-9"],
       cwd=os.path.join(INF, "flux2-klein-bft-i2"))


def recipe_qwen_image_edit(stage_dir):
    sh(["modal", "run", "modal_qwen_ie_inference.py", "--data-dir", stage_dir],
       cwd=os.path.join(INF, "qwen-image-edit"))


def recipe_susie(stage_dir):
    sh(["modal", "run", "modal_susie_inference.py", "--data-dir", stage_dir],
       cwd=os.path.join(INF, "SuSIE"))


def recipe_clover(stage_dir):
    sh(["modal", "run", "modal_clover_inference.py", "--data-dir", stage_dir],
       cwd=os.path.join(INF, "Clover"))


def recipe_gemini(stage_dir):
    # local script calling the paid Gemini API (resume-safe; embedded API keys)
    sh([sys.executable, "inference.py", stage_dir],
       cwd=os.path.join(INF, "gemini-3-pro-image"))


def recipe_gpt_image(stage_dir):
    # local script calling the paid GPT-Image API (resume-safe; embedded API key)
    sh([sys.executable, "inference.py", "--data-dir", stage_dir],
       cwd=os.path.join(INF, "gpt-image-1.5"))


# --- composite "Ours" recipes: IER -> FER (klein) -> IE (klein or kontext).
# Each stage runs `modal run` on the SAME stage dir so masks accumulate in place.
KLEIN_FT = os.path.join(INF, "flux2-klein-ft")
KONTEXT_FT = os.path.join(INF, "flux-kontext-dev-ft/bridge_and_calvin")
KLEIN_FER_CKPT = os.path.join(KLEIN_FT, "bridge-fer-mask-prediction-checkpoint/klein-fer-checkpoint-epoch-17")
KLEIN_IE_CKPT = os.path.join(KLEIN_FT, "bridge-image-editing-checkpoint/klein-bridge-image-editing-checkpoint-epoch-9")
KONTEXT_IE_CKPT = os.path.join(KONTEXT_FT, "bridge-image-editing-checkpoint/kontext-ie-checkpoint-epoch-18")


def _ier(stage_dir, cwd):
    sh(["modal", "run", "modal_ier_mask_prediction.py", "--data-dir", stage_dir],
       cwd=os.path.join(cwd, "ier-mask-prediction-code"))


def recipe_flux2_klein_ours_klein(stage_dir):  # klein-FER + klein-IE
    _ier(stage_dir, KLEIN_FT)
    sh(["modal", "run", "modal_flux2_klein_erm_inference.py", "--data-dir", stage_dir,
        "--checkpoint-dir", KLEIN_FER_CKPT], cwd=os.path.join(KLEIN_FT, "fer-mask-prediction-code"))
    sh(["modal", "run", "modal_flux2_klein_ie_inference.py", "--data-dir", stage_dir,
        "--checkpoint-dir", KLEIN_IE_CKPT], cwd=os.path.join(KLEIN_FT, "image-editing-code"))


def recipe_flux_kontext_ours_klein(stage_dir):  # klein-FER + kontext-IE (cross-repo)
    _ier(stage_dir, KONTEXT_FT)
    sh(["modal", "run", "modal_flux2_klein_erm_inference.py", "--data-dir", stage_dir,
        "--checkpoint-dir", KLEIN_FER_CKPT], cwd=os.path.join(KLEIN_FT, "fer-mask-prediction-code"))
    sh(["modal", "run", "modal_flux_kontext_inference.py", "--data-dir", stage_dir,
        "--checkpoint-dir", KONTEXT_IE_CKPT], cwd=os.path.join(KONTEXT_FT, "image-editing-code"))


RECIPES = {
    "flux-kontext-dev": recipe_flux_kontext_dev,
    "flux2-klein": recipe_flux2_klein,
    "flux-kontext-dev-bft-i1": recipe_flux_kontext_dev_bft_i1,
    "flux-kontext-dev-bft-i2": recipe_flux_kontext_dev_bft_i2,
    "flux2-klein-bft-i1": recipe_flux2_klein_bft_i1,
    "flux2-klein-bft-i2": recipe_flux2_klein_bft_i2,
    "qwen-image-edit": recipe_qwen_image_edit,
    "susie": recipe_susie,
    "clover": recipe_clover,
    "gemini-3-pro-image": recipe_gemini,
    "gpt-image-1.5": recipe_gpt_image,
    "flux2-klein-ours-klein": recipe_flux2_klein_ours_klein,
    "flux-kontext-ours-klein": recipe_flux_kontext_ours_klein,
}


def stage_combined(method, tasks):
    """Stage all tasks into ONE dir with case names `<task>__ep<NNN>` so the
    editor downloads its (huge) model only ONCE per method instead of per task."""
    cdir = os.path.join(STAGE_ROOT, f"{method}__combined")
    tmp_parent = os.path.join(STAGE_ROOT, f"{method}__tmp")
    for d in (cdir, tmp_parent):
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(cdir)
    for task in tasks:
        tdir, cases = stage_task(task, parent=tmp_parent)
        for c in cases:
            shutil.copytree(os.path.join(tdir, c), os.path.join(cdir, f"{task}__{c}"))
    return cdir


def push_combined(method, cdir, tasks):
    """Split the combined dir's edited_image.jpg back per task and push."""
    for task in tasks:
        tmp = os.path.join(STAGE_ROOT, f"{method}__push__{task}")
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        n_ok = 0
        for cc in sorted(os.listdir(cdir)):
            if not cc.startswith(f"{task}__ep"):
                continue
            ep = cc.split("__", 1)[1]
            src = os.path.join(cdir, cc, "edited_image.jpg")
            os.makedirs(os.path.join(tmp, ep), exist_ok=True)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp, ep, "edited_image.jpg"))
                n_ok += 1
        print(f"  [{task}] {n_ok} goal images -> push")
        push_goals(method, task, tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(RECIPES))
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--per-task", action="store_true",
                    help="stage/run each task separately (default: combined run)")
    args = ap.parse_args()
    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")
    recipe = RECIPES[args.method]
    if args.per_task:
        for task in tasks:
            print(f"\n{'='*70}\n[{args.method}] {task}\n{'='*70}")
            stage_task_dir, cases = stage_task(task)
            recipe(stage_task_dir)
            verify_goals(stage_task_dir, cases)
            push_goals(args.method, task, stage_task_dir)
    else:
        print(f"\n{'='*70}\n[{args.method}] combined staging over {tasks}\n{'='*70}")
        cdir = stage_combined(args.method, tasks)
        recipe(cdir)
        cases = sorted(d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d)))
        verify_goals(cdir, cases)
        push_combined(args.method, cdir, tasks)
    print("\nDone. Now run: modal run simpler-eval/modal_rollout.py::run_method "
          f"--method {args.method}")


if __name__ == "__main__":
    main()
