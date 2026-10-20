#!/usr/bin/env python3
"""
Nano Banana Pro (gemini-3-pro-image-preview) — Robotic Manipulation Image Editing
==================================================================================

Language-guided goal-state image editing for robotic manipulation test cases.
Uses three API keys from separate Google Cloud projects for symmetric load
balancing, effectively tripling rate limits.

Features:
  - Triple API key support with round-robin symmetric load balancing
  - Multiple concurrent workers per key (pipelining to maximise RPM usage)
  - Thread-safe sliding-window rate limiter (RPM) + daily counter (RPD)
  - Infinite retry with exponential back-off + jitter for rate limits / errors
  - Auto-sleep until midnight PT when daily quota is exhausted
  - Resume support: skips test cases that already have edited_image.jpg
  - Optimised prompt template for robotic manipulation goal-state prediction

Usage:
  python inference.py dummy_test_data
  python inference.py bridge_data_v2_tel_test_data --rpm 10 --rpd 1500

Cost estimate (3 463 cases @ 1K):
  ~$0.16–0.23 per request  →  ~$555–$800 total  (depends on thinking tokens)
"""

from __future__ import annotations
import argparse
import datetime
import glob
import io
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — three API keys from DIFFERENT Google Cloud projects/accounts
#  so that rate limits are independent.  Symmetric round-robin balancing.
# ═════════════════════════════════════════════════════════════════════════════

# Comma-separated in GEMINI_API_KEYS (or a single key in GEMINI_API_KEY); keys from separate
# projects give independent rate limits and are used round-robin.
API_KEYS = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",") if k.strip()]

API_KEY_LABELS = [f"key{i}" for i in range(len(API_KEYS))]

MODEL_ID = "gemini-3-pro-image-preview"

# ── Prompt template ─────────────────────────────────────────────────────────
# Designed for robotic manipulation goal-state prediction.  The model receives
# the *current* scene image and must produce the *post-manipulation* image.

EDIT_PROMPT_TEMPLATE = """\
You are an expert robotic manipulation simulator.

INPUT:
  • An image showing the CURRENT state of a robotic workspace (table-top with
    objects, possibly a robot arm / gripper visible).
  • A manipulation instruction describing what action a robot should perform.

YOUR TASK:
  Edit the provided image to produce a photorealistic photograph of the scene
  AFTER the manipulation has been successfully completed.

MANIPULATION INSTRUCTION:
  "{instruction}"

STRICT REQUIREMENTS:
  1. ONLY modify the objects and spatial relationships directly affected by the
     instruction.  Every other object, the table/surface, background, camera
     angle, perspective, and lighting must remain PIXEL-PERFECT unchanged.
  2. The result must be physically plausible — objects obey gravity, maintain
     realistic proportions, and respect spatial constraints.
  3. If a robot arm or gripper is visible in the input, either (a) show it in a
     natural resting / retracted pose consistent with having just completed the
     action, or (b) leave it in its final grasping pose if the instruction
     implies it is still holding an object.
  4. Preserve complete image quality, resolution, colour grading, and style
     consistency with the input.
  5. Do NOT add any text, watermarks, or annotations to the output image.
  6. Output ONLY the edited image — no extra commentary is needed."""


# ═════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER  (thread-safe, sliding-window RPM + daily RPD counter)
# ═════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Thread-safe rate limiter that enforces:
      - requests-per-minute  (sliding window of timestamps)
      - requests-per-day     (counter, resets at midnight Pacific Time)
    """

    def __init__(self, rpm: int, rpd: int, label: str = ""):
        self.rpm = rpm
        self.rpd = rpd
        self.label = label
        self._lock = threading.Lock()
        self._timestamps: list[float] = []      # last 60 s of request times
        self._day_count = 0
        self._day_reset = self._next_midnight_pt()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _next_midnight_pt() -> float:
        """Unix timestamp of the next midnight Pacific Time."""
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("America/Los_Angeles")
        except ImportError:
            tz = datetime.timezone(datetime.timedelta(hours=-8))
        now = datetime.datetime.now(tz)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        midnight += datetime.timedelta(days=1)
        return midnight.timestamp()

    # ── public API ───────────────────────────────────────────────────────────

    def acquire(self):
        """Block the calling thread until a request slot is available."""
        while True:
            with self._lock:
                now = time.time()

                # ── daily reset ──────────────────────────────────────────────
                if now >= self._day_reset:
                    self._day_count = 0
                    self._day_reset = self._next_midnight_pt()

                # ── daily quota exhausted → sleep until reset ────────────────
                if self._day_count >= self.rpd:
                    wait = self._day_reset - now + 5
                    hrs = wait / 3600
                    print(
                        f"  [{self.label}] ⏳ Daily limit ({self.rpd} RPD) "
                        f"reached — sleeping {hrs:.1f} h until midnight PT …"
                    )
                    # release lock while sleeping
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    self._day_count = 0
                    self._day_reset = self._next_midnight_pt()
                    now = time.time()

                # ── sliding-window RPM check ─────────────────────────────────
                cutoff = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > cutoff]

                if len(self._timestamps) >= self.rpm:
                    # wait until the oldest request "expires" from the window
                    wait = self._timestamps[0] + 60.0 - now + 0.05
                    if wait > 0:
                        self._lock.release()
                        time.sleep(wait)
                        self._lock.acquire()
                        continue          # re-check after waking

                # ── slot available ───────────────────────────────────────────
                self._timestamps.append(time.time())
                self._day_count += 1
                return


# ═════════════════════════════════════════════════════════════════════════════
#  PROGRESS TRACKER  (thread-safe)
# ═════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.skipped = 0
        self.retries = 0
        self.start = time.time()
        self._lock = threading.Lock()

    def done(self, case_id: int, label: str):
        with self._lock:
            self.completed += 1
            elapsed = time.time() - self.start
            rate = self.completed / (elapsed / 60) if elapsed > 60 else self.completed
            remaining = self.total - self.completed - self.skipped
            eta = remaining / rate if rate > 0 else float("inf")
            pct = self.completed * 100 // self.total
            print(
                f"  ✓ [{label}] Case {case_id:>5d} done │ "
                f"{self.completed}/{self.total} ({pct}%) │ "
                f"{rate:.1f}/min │ ETA {eta:.0f} min"
            )

    def skip(self):
        with self._lock:
            self.skipped += 1

    def retry(self):
        with self._lock:
            self.retries += 1


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def discover_test_cases(data_dir: str) -> list[tuple[str, int]]:
    """Return [(path, numeric_id), …] sorted by id."""
    cases = []
    for i, name in enumerate(sorted(os.listdir(data_dir))):
        full = os.path.join(data_dir, name)
        if not os.path.isdir(full):
            continue
        if name.isdigit():
            cases.append((full, int(name)))
        elif os.path.exists(os.path.join(full, "edit_instruction.txt")):
            # combined-staging case dirs named <task>__ep<NNN>
            cases.append((full, i))
    cases.sort(key=lambda x: x[1])
    return cases


def find_original_image(case_dir: str) -> str | None:
    for ext in ("jpg", "jpeg", "png"):
        p = os.path.join(case_dir, f"original_image.{ext}")
        if os.path.exists(p):
            return p
    return None


def read_instruction(case_dir: str) -> str:
    p = os.path.join(case_dir, "edit_instruction.txt")
    with open(p) as f:
        lines = f.read().strip().splitlines()
    # First line = instruction; subsequent lines may be metadata (confidence…)
    return lines[0].strip() if lines else ""


# ═════════════════════════════════════════════════════════════════════════════
#  CORE:  process one test case (with infinite retry)
# ═════════════════════════════════════════════════════════════════════════════

def process_test_case(
    case_dir: str,
    case_id: int,
    client: genai.Client,
    limiter: RateLimiter,
    label: str,
    tracker: ProgressTracker,
    resolution: str,
):
    """Process a single test case.  Retries indefinitely on any failure."""

    output_path = os.path.join(case_dir, "edited_image.jpg")

    # ── skip if already done ─────────────────────────────────────────────────
    if os.path.exists(output_path):
        tracker.skip()
        return

    # ── load inputs ──────────────────────────────────────────────────────────
    img_path = find_original_image(case_dir)
    if img_path is None:
        print(f"  ⚠ [{label}] Case {case_id}: no original_image found — skipping")
        tracker.skip()
        return

    instruction = read_instruction(case_dir)
    if not instruction:
        print(f"  ⚠ [{label}] Case {case_id}: empty instruction — skipping")
        tracker.skip()
        return

    prompt = EDIT_PROMPT_TEMPLATE.format(instruction=instruction)
    img = Image.open(img_path)

    # ── retry loop ───────────────────────────────────────────────────────────
    backoff = 5.0            # initial back-off seconds
    max_backoff = 600.0      # 10 min cap
    attempt = 0

    while True:
        attempt += 1
        try:
            limiter.acquire()

            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    # google-genai 1.47 ImageConfig dropped image_size; use API
                    # default output resolution (goals are downscaled to 200x200
                    # for the policy anyway). `resolution` retained for logging.
                ),
            )

            # ── extract final (non-thought) image ───────────────────────────
            final_image = None   # will be a PIL Image
            text_parts: list[str] = []

            def _to_pil(part):
                """Convert a genai Part with inline_data to a PIL Image."""
                if part.inline_data is not None and hasattr(part.inline_data, 'data'):
                    return Image.open(io.BytesIO(part.inline_data.data))
                return None

            if response.candidates:
                for cand in response.candidates:
                    if not (cand.content and cand.content.parts):
                        continue
                    for part in cand.content.parts:
                        if getattr(part, "thought", False):
                            continue                       # skip thought images
                        if part.text:
                            text_parts.append(part.text)
                        if part.inline_data is not None:
                            try:
                                im = _to_pil(part)
                                if im is not None:
                                    final_image = im
                            except Exception:
                                pass

            # fallback: accept the last thought image if no non-thought image
            if final_image is None and response.candidates:
                for cand in response.candidates:
                    if not (cand.content and cand.content.parts):
                        continue
                    for part in cand.content.parts:
                        if part.inline_data is not None:
                            try:
                                im = _to_pil(part)
                                if im is not None:
                                    final_image = im
                            except Exception:
                                pass

            if final_image is not None:
                # Resize to match original image dimensions exactly
                orig_w, orig_h = img.size
                if final_image.size != (orig_w, orig_h):
                    final_image = final_image.resize(
                        (orig_w, orig_h), Image.LANCZOS
                    )
                final_image.save(output_path, "JPEG", quality=95)
                # optionally save model text
                if text_parts:
                    tp = os.path.join(case_dir, "edit_response_text.txt")
                    with open(tp, "w") as f:
                        f.write("\n".join(text_parts))
                tracker.done(case_id, label)
                return   # ✓ success

            # ── no image returned ────────────────────────────────────────────
            snippet = (" | ".join(text_parts))[:150] if text_parts else "empty"
            print(
                f"  ⚠ [{label}] Case {case_id}: no image in response "
                f"(attempt {attempt}): {snippet}"
            )
            tracker.retry()

        except Exception as e:
            err = str(e)
            tracker.retry()

            if any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "quota",
                                       "rate_limit", "rateLimitExceeded")):
                jitter = random.uniform(0, backoff * 0.3)
                wait = backoff + jitter
                print(
                    f"  🔄 [{label}] Case {case_id}: rate-limited "
                    f"(attempt {attempt}) — waiting {wait:.0f}s"
                )
                time.sleep(wait)
                backoff = min(backoff * 2, max_backoff)
                continue

            if any(k in err for k in ("500", "503", "INTERNAL", "UNAVAILABLE",
                                       "DeadlineExceeded")):
                wait = min(backoff, 30) + random.uniform(0, 5)
                print(
                    f"  🔄 [{label}] Case {case_id}: server error "
                    f"(attempt {attempt}) — waiting {wait:.0f}s"
                )
                time.sleep(wait)
                continue

            if any(k in err.upper() for k in ("SAFETY", "BLOCKED", "RECITATION",
                                                "PROHIBITED")):
                wait = backoff + random.uniform(0, 5)
                print(
                    f"  ⚠ [{label}] Case {case_id}: content filter "
                    f"(attempt {attempt}) — retrying in {wait:.0f}s: "
                    f"{err[:120]}"
                )
                time.sleep(wait)
                backoff = min(backoff * 1.5, 300)
                continue

            # unknown error
            wait = backoff + random.uniform(0, 5)
            print(
                f"  ❌ [{label}] Case {case_id}: error (attempt {attempt}) — "
                f"retrying in {wait:.0f}s: {err[:200]}"
            )
            time.sleep(wait)
            backoff = min(backoff * 2, max_backoff)
            continue

        # fell through (no image, no exception) — back off and retry
        time.sleep(backoff + random.uniform(0, 3))
        backoff = min(backoff * 1.5, 300)


# ═════════════════════════════════════════════════════════════════════════════
#  WORKER  (one per thread — pulls from a per-key queue)
# ═════════════════════════════════════════════════════════════════════════════

def key_worker(
    q: queue.Queue,
    client: genai.Client,
    limiter: RateLimiter,
    label: str,
    tracker: ProgressTracker,
    resolution: str,
):
    while True:
        try:
            case_dir, case_id = q.get_nowait()
        except queue.Empty:
            return
        try:
            process_test_case(
                case_dir, case_id, client, limiter, label, tracker, resolution
            )
        except Exception as e:
            print(f"  ❌ [{label}] Case {case_id}: fatal — {e}")
        finally:
            q.task_done()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Nano Banana Pro — robotic manipulation image editing inference"
    )
    ap.add_argument(
        "test_data_dir",
        help="Path to test data folder containing numbered sub-directories",
    )
    ap.add_argument("--rpm", type=int, default=10,
                    help="Requests per minute per API key (default: 10)")
    ap.add_argument("--rpd", type=int, default=1500,
                    help="Requests per day per API key (default: 1500)")
    ap.add_argument("--workers-per-key", type=int, default=5,
                    help="Concurrent worker threads per key (default: 5)")
    ap.add_argument("--resolution", choices=["1K", "2K", "4K"], default="1K",
                    help="Output image resolution (default: 1K) — 1K & 2K same price")
    ap.add_argument("--reprocess", action="store_true",
                    help="Re-process cases that already have edited_image.jpg")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.test_data_dir)
    if not os.path.isdir(data_dir):
        sys.exit(f"Error: '{data_dir}' is not a directory")

    # ── discover test cases ──────────────────────────────────────────────────
    all_cases = discover_test_cases(data_dir)
    if not all_cases:
        sys.exit("No numbered sub-directories found.")

    if args.reprocess:
        cases = all_cases
    else:
        cases = [
            (d, i) for d, i in all_cases
            if not os.path.exists(os.path.join(d, "edited_image.jpg"))
        ]
        skipped = len(all_cases) - len(cases)
        if skipped:
            print(f"ℹ  Skipping {skipped} already-processed test cases "
                  f"(use --reprocess to redo)")

    total = len(cases)
    num_keys = len([k for k in API_KEYS if k and not k.startswith("API_KEY")])
    if num_keys == 0:
        sys.exit(
            "ERROR: No valid API keys found.\n"
            "       Set GEMINI_API_KEYS to one or more comma-separated Gemini\n"
            "       API keys (ideally from separate Google Cloud projects)."
        )

    eff_rpm = args.rpm * num_keys
    eff_rpd = args.rpd * num_keys

    # ── cost / time estimates ────────────────────────────────────────────────
    cost_img = {"1K": 0.134, "2K": 0.134, "4K": 0.24}[args.resolution]
    cost_input = 0.002       # ~860 input tokens at $2/1M
    cost_think = 0.04        # ~3 300 thinking+text tokens at $12/1M (mid est.)
    cost_req = cost_input + cost_think + cost_img
    total_cost = cost_req * total
    min_by_rpm = total / eff_rpm               # minutes
    days_by_rpd = total / eff_rpd if eff_rpd else float("inf")

    print(f"\n{'═'*72}")
    print(f"  Nano Banana Pro — Robotic Manipulation Image Editing Inference")
    print(f"{'═'*72}")
    print(f"  Model:           {MODEL_ID}")
    print(f"  Resolution:      {args.resolution}")
    print(f"  Test cases:      {total} remaining / {len(all_cases)} total")
    print(f"  API keys:        {num_keys}")
    print(f"  Workers / key:   {args.workers_per_key}")
    print(f"  Per-key limits:  {args.rpm} RPM, {args.rpd} RPD")
    print(f"  Combined limits: {eff_rpm} RPM, {eff_rpd} RPD")
    print(f"{'─'*72}")
    print(f"  Est. cost/req:   ~${cost_req:.3f}  "
          f"(input ${cost_input:.4f} + think ${cost_think:.3f} + img ${cost_img:.3f})")
    print(f"  Est. total cost: ~${total_cost:.0f}  "
          f"(range ${total*0.16:.0f}–${total*0.23:.0f})")
    print(f"  Min time (RPM):  {min_by_rpm:.0f} min  ({min_by_rpm/60:.1f} h)")
    print(f"  Days (RPD):      {days_by_rpd:.2f}")
    print(f"{'═'*72}\n")

    if total == 0:
        print("Nothing to process.")
        return

    # ── create clients, rate limiters, queues ────────────────────────────────
    active_keys = [k for k in API_KEYS if k and not k.startswith("API_KEY")]
    clients = []
    limiters = []
    queues: list[queue.Queue] = []

    for i, key in enumerate(active_keys):
        label = API_KEY_LABELS[i] if i < len(API_KEY_LABELS) else f"Key{i}"
        clients.append(genai.Client(api_key=key))
        limiters.append(RateLimiter(args.rpm, args.rpd, label=label))
        queues.append(queue.Queue())

    # round-robin assignment for symmetric credit usage
    for idx, (case_dir, case_id) in enumerate(cases):
        queues[idx % num_keys].put((case_dir, case_id))

    for i in range(num_keys):
        lbl = API_KEY_LABELS[i] if i < len(API_KEY_LABELS) else f"Key{i}"
        print(f"  {lbl}: {queues[i].qsize()} test cases queued")
    print()

    tracker = ProgressTracker(total)

    # ── launch workers ───────────────────────────────────────────────────────
    threads: list[threading.Thread] = []
    for ki in range(num_keys):
        lbl = API_KEY_LABELS[ki] if ki < len(API_KEY_LABELS) else f"Key{ki}"
        for w in range(args.workers_per_key):
            t = threading.Thread(
                target=key_worker,
                args=(queues[ki], clients[ki], limiters[ki],
                      lbl, tracker, args.resolution),
                daemon=True,
            )
            t.start()
            threads.append(t)

    # wait for completion
    for t in threads:
        t.join()

    # ── summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - tracker.start
    print(f"\n{'═'*72}")
    print(f"  FINISHED")
    print(f"  Completed: {tracker.completed}")
    print(f"  Skipped:   {tracker.skipped}  (no image / no instruction / already done)")
    print(f"  Retries:   {tracker.retries}")
    print(f"  Wall time: {elapsed/60:.1f} min  ({elapsed/3600:.2f} h)")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
