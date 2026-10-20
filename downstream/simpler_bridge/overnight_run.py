#!/usr/bin/env python3
"""
Autonomous overnight driver: generate goals + run SIMPLER-Bridge GC-policy
rollouts for ALL 13 methods, pick the best 'Ours' candidate per category, and
compile the final SR table. Robust to transient Modal/network failures (retries)
and RESUMABLE (skips work already on the volume / in the results file).

Pipeline per method:  goal-gen (orchestrate_goals.py) -> rollout
(modal_rollout_gc.py run_method, early-stop reached-success SR, 24 eps/task).

Producer/consumer: a goal-gen pool feeds a rollout pool so the two overlap.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = "/tmp/gc_overnight_results.json"
LOG = "/tmp/gc_overnight.log"
TABLE = "/tmp/gc_final_table.md"
PY = sys.executable
CONDA_PY = os.environ.get("SGE_PYTHON", sys.executable)

TASKS = ["widowx_spoon_on_towel", "widowx_carrot_on_plate",
         "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

# 5 methods already evaluated this session (goals on volume, SRs known)
DONE = {
    "flux-kontext-dev": {"widowx_spoon_on_towel": 4.2, "widowx_carrot_on_plate": 4.2,
                         "widowx_stack_cube": 0.0, "widowx_put_eggplant_in_basket": 0.0},
    "flux2-klein": {"widowx_spoon_on_towel": 4.2, "widowx_carrot_on_plate": 4.2,
                    "widowx_stack_cube": 0.0, "widowx_put_eggplant_in_basket": 0.0},
    "qwen-image-edit": {"widowx_spoon_on_towel": 8.3, "widowx_carrot_on_plate": 4.2,
                        "widowx_stack_cube": 0.0, "widowx_put_eggplant_in_basket": 0.0},
    "flux-kontext-dev-bft-i1": {"widowx_spoon_on_towel": 12.5, "widowx_carrot_on_plate": 8.3,
                                "widowx_stack_cube": 0.0, "widowx_put_eggplant_in_basket": 0.0},
    "flux-kontext-dev-bft-i2": {"widowx_spoon_on_towel": 16.7, "widowx_carrot_on_plate": 4.2,
                                "widowx_stack_cube": 0.0, "widowx_put_eggplant_in_basket": 0.0},
}

# "Ours" = best (by mean SR) of the candidate recipe keys
OURS = {
    "flux-kontext-ours": ["flux-kontext-ours-klein"],
    "flux2-klein-ours": ["flux2-klein-ours-klein"],
}
# single-recipe methods still needing goals (fast/non-klein first)
SINGLE = ["gemini-3-pro-image", "gpt-image-1.5", "susie", "clover",
          "flux2-klein-bft-i1", "flux2-klein-bft-i2"]

# generation order: front-load high-value (Ours candidates + paid baselines) and
# fast/non-klein recipes; defer klein-heavy gens (FLUX.2-klein download contention)
# and the lower-priority klein ablations to the end.
GEN_ORDER = ["gemini-3-pro-image", "gpt-image-1.5",
             "susie", "clover",
             "flux-kontext-ours-klein", "flux2-klein-ours-klein",
             "flux2-klein-bft-i1", "flux2-klein-bft-i2"]

GEN_CONCURRENCY = 4
ROLL_CONCURRENCY = 2
_lock = threading.Lock()


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with _lock:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def load_results():
    if os.path.exists(RESULTS):
        try:
            return json.load(open(RESULTS))
        except Exception:
            return {}
    return {}


def save_result(method, sr_dict):
    with _lock:
        r = load_results()
        r[method] = sr_dict
        json.dump(r, open(RESULTS, "w"), indent=2)


def run(cmd, cwd, retries=3, timeout=10800, tag=""):
    """Run a command with retries on failure (transient Modal/network errors)."""
    for attempt in range(1, retries + 1):
        log(f"  [{tag}] attempt {attempt}/{retries}: {' '.join(cmd[:3])}... (cwd={os.path.basename(cwd)})")
        try:
            p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            if p.returncode == 0:
                return p.stdout + p.stderr
            log(f"  [{tag}] rc={p.returncode}; tail: {(p.stdout + p.stderr).strip()[-300:]}")
        except subprocess.TimeoutExpired:
            log(f"  [{tag}] TIMEOUT after {timeout}s")
        except Exception as e:
            log(f"  [{tag}] exception: {e}")
        time.sleep(30 * attempt)
    return None


def goals_on_volume(method):
    out = run(["modal", "volume", "ls", "simpler-eval-data", f"goals/{method}"],
              cwd=HERE, retries=2, timeout=120, tag=f"chk:{method}")
    if out is None:
        return False
    # count ep dirs across tasks: look for 'ep0' entries
    return out.count("ep0") + out.count("widowx") > 0 and "Error" not in out[:200]


def gen_goals(method):
    if goals_on_volume(method):
        log(f"GEN skip {method} (goals already on volume)")
        return True
    log(f"GEN start {method}")
    out = run([CONDA_PY, "orchestrate_goals.py", "--method", method],
              cwd=HERE, retries=2, timeout=14400, tag=f"gen:{method}")
    ok = out is not None and goals_on_volume(method)
    log(f"GEN {'DONE' if ok else 'FAIL'} {method}")
    return ok


def rollout(method):
    existing = load_results()
    if method in existing and existing[method].get("_status") == "ok":
        log(f"ROLL skip {method} (already have SR)")
        return existing[method]
    log(f"ROLL start {method}")
    out = run([CONDA_PY, "-m", "modal", "run",
               "modal_rollout_gc.py::run_method", "--method", method,
               "--n-eps", "24", "--chunk", "6"],
              cwd=HERE, retries=2, timeout=10800, tag=f"roll:{method}")
    sr = {"_status": "fail"}
    if out:
        m = re.search(r"RESULT_JSON (\{.*\})", out)
        if m:
            try:
                data = json.loads(m.group(1))
                sr = {t: round(100 * data.get(t, {}).get("SR", 0.0), 1) for t in TASKS}
                sr["_status"] = "ok"
            except Exception as e:
                log(f"  [roll:{method}] parse error: {e}")
    save_result(method, sr)
    log(f"ROLL {'DONE' if sr.get('_status') == 'ok' else 'FAIL'} {method}: "
        f"{ {t: sr.get(t) for t in TASKS} }")
    try:
        compile_table()  # keep /tmp/gc_final_table.md always current
    except Exception as e:
        log(f"  compile_table error: {e}")
    return sr


def mean_sr(sr):
    vals = [sr.get(t, 0.0) for t in TASKS if isinstance(sr.get(t), (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


# ---- producer/consumer: gen pool feeds roll queue; roll pool consumes ----
_roll_q = []
_roll_q_cv = threading.Condition()
_gen_done = threading.Event()


def gen_worker(queue):
    while True:
        with _lock:
            if not queue:
                return
            method = queue.pop(0)
        try:
            gen_goals(method)
        except Exception as e:
            log(f"GEN worker error {method}: {e}")
        with _roll_q_cv:
            _roll_q.append(method)
            _roll_q_cv.notify()


def roll_worker():
    while True:
        with _roll_q_cv:
            while not _roll_q and not _gen_done.is_set():
                _roll_q_cv.wait(timeout=60)
            if not _roll_q:
                if _gen_done.is_set():
                    return
                continue
            method = _roll_q.pop(0)
        if goals_on_volume(method):
            try:
                rollout(method)
            except Exception as e:
                log(f"ROLL worker error {method}: {e}")
        else:
            log(f"ROLL skip {method} (no goals produced -> gen failed)")
            save_result(method, {"_status": "no_goals"})


def compile_table():
    r = load_results()
    all_sr = dict(DONE)
    for m, sr in r.items():
        if sr.get("_status") == "ok":
            all_sr[m] = {t: sr.get(t, 0.0) for t in TASKS}
    # resolve Ours = best candidate
    final_rows = {}
    for ours_name, cands in OURS.items():
        best, best_mean = None, -1
        for c in cands:
            if c in all_sr:
                mm = mean_sr(all_sr[c])
                if mm > best_mean:
                    best, best_mean = c, mm
        if best:
            final_rows[f"{ours_name} (best={best})"] = all_sr[best]
    # everything else
    for m, sr in all_sr.items():
        if m not in sum(OURS.values(), []):
            final_rows[m] = sr
    lines = ["# SIMPLER-Bridge GC-policy SR table (early-stop reached-success, 24 eps/task, %)",
             "# policy: SuSIE/GHIL-Glue bridge gcdiffusion auggoaldiff",
             "", "| method | spoon | carrot | stack | eggplant | mean |",
             "|---|---|---|---|---|---|"]
    for m in sorted(final_rows, key=lambda k: mean_sr(final_rows[k])):
        sr = final_rows[m]
        lines.append(f"| {m} | {sr.get(TASKS[0])} | {sr.get(TASKS[1])} | "
                     f"{sr.get(TASKS[2])} | {sr.get(TASKS[3])} | {mean_sr(sr)} |")
    # note any failures
    fails = [m for m, sr in r.items() if sr.get("_status") != "ok"]
    if fails:
        lines += ["", f"# FAILED/incomplete: {', '.join(fails)}"]
    open(TABLE, "w").write("\n".join(lines))
    log("TABLE written:\n" + "\n".join(lines))


def main():
    log("=" * 60)
    log("OVERNIGHT RUN start")
    # seed results with the 5 done methods
    for m, sr in DONE.items():
        d = dict(sr); d["_status"] = "ok"
        save_result(m, d)
    queue = list(GEN_ORDER)
    gens = [threading.Thread(target=gen_worker, args=(queue,)) for _ in range(GEN_CONCURRENCY)]
    rolls = [threading.Thread(target=roll_worker) for _ in range(ROLL_CONCURRENCY)]
    for t in rolls:
        t.start()
    for t in gens:
        t.start()
    for t in gens:
        t.join()
    _gen_done.set()
    with _roll_q_cv:
        _roll_q_cv.notify_all()
    for t in rolls:
        t.join()
    compile_table()
    log("OVERNIGHT RUN complete")


if __name__ == "__main__":
    main()
