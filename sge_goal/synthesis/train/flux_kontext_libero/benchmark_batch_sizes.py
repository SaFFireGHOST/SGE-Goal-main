"""
Quick benchmark: test different batch sizes at resolution 128 to find GPU VRAM limits.
Runs a few training steps and reports peak GPU memory.
"""
import subprocess
import sys
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
SCRIPT = os.path.join(HERE, "flux_kontext_lora_finetuning.py")
OUTPUT_DIR = os.path.join(HERE, "kontext-i2i-libero")
DISK_CACHE_DIR = os.path.join(OUTPUT_DIR, "disk_cache")
DATASET = "RiddhiCh/LGRM_Libero_10_90_Combined"

# Test configs: (train_batch_size, gradient_accumulation_steps, val_batch_size)
CONFIGS = [
    (32, 1, 32),
    (48, 1, 48),
    (64, 1, 64),
    (96, 1, 64),
    (128, 1, 64),
]

def run_test(train_bs, grad_accum, val_bs, max_steps=5):
    """Run training for a few steps and check if it OOMs."""
    cmd = [
        PYTHON, "-m", "accelerate.commands.launch", SCRIPT,
        "--pretrained_model_name_or_path", "black-forest-labs/FLUX.1-Kontext-dev",
        "--dataset_name", DATASET,
        "--edited_image_column", "edited_image",
        "--original_image_column", "original_image",
        "--edit_instruction_column", "edit_instruction",
        "--initial_edit_region_mask_column", "initial_edit_region_mask",
        "--final_edit_region_mask_column", "final_edit_region_mask",
        "--output_dir", OUTPUT_DIR,
        "--logging_dir", "logs",
        "--cache_dir", "cache",
        "--resolution", "128",
        "--train_batch_size", str(train_bs),
        "--gradient_accumulation_steps", str(grad_accum),
        "--num_train_epochs", "75",
        "--max_train_steps", str(max_steps),
        "--learning_rate", "0.0001",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--optimizer", "adamw",
        "--rank", "16",
        "--lora_alpha", "16",
        "--lora_dropout", "0.0",
        "--checkpointing_steps", "999999",
        "--seed", "0",
        "--max_sequence_length", "512",
        "--guidance_scale", "1.0",
        "--weighting_scheme", "none",
        "--dataloader_num_workers", "2",
        "--max_val_samples", "25",
        "--val_batch_size", str(val_bs),
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--allow_tf32",
        "--mixed_precision", "bf16",
        "--cache_text_embeddings_to_disk",
        "--cache_latents_to_disk",
        "--cache_batch_size", "64",
        "--disk_cache_dir", DISK_CACHE_DIR,
    ]

    print(f"\n{'='*60}")
    print(f"Testing: train_bs={train_bs}, grad_accum={grad_accum}, val_bs={val_bs}")
    print(f"Effective batch size = {train_bs * grad_accum}")
    print(f"{'='*60}")

    log_path = f"/tmp/bench_bs{train_bs}.log"
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
            cwd=HERE,
        )
        elapsed = time.time() - start

        # Save output
        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)

        # Check for OOM
        combined_output = result.stdout + result.stderr
        if "CUDA out of memory" in combined_output or "OutOfMemoryError" in combined_output:
            print(f"  RESULT: OOM at bs={train_bs}")
            return False, elapsed
        elif result.returncode != 0:
            # Check for other errors
            if "Killed" in combined_output or result.returncode == -9:
                print(f"  RESULT: KILLED (likely OOM) at bs={train_bs}")
                return False, elapsed
            else:
                print(f"  RESULT: FAILED (rc={result.returncode}) at bs={train_bs}")
                # Print last few lines of stderr
                stderr_lines = result.stderr.strip().split('\n')
                for line in stderr_lines[-5:]:
                    print(f"    {line}")
                return False, elapsed
        else:
            print(f"  RESULT: SUCCESS at bs={train_bs} in {elapsed:.1f}s")
            return True, elapsed

    except subprocess.TimeoutExpired:
        print(f"  RESULT: TIMEOUT at bs={train_bs}")
        # Kill any remaining processes
        subprocess.run(["pkill", "-f", "flux_kontext_lora_finetuning"], capture_output=True)
        time.sleep(3)
        return False, 600
    except Exception as e:
        print(f"  RESULT: ERROR at bs={train_bs}: {e}")
        return False, 0


def get_gpu_memory():
    """Get current GPU memory usage in MiB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        used, total = result.stdout.strip().split(", ")
        return int(used), int(total)
    except:
        return 0, 0


if __name__ == "__main__":
    print("GPU Memory Benchmark for resolution=128")
    print(f"Dataset: {DATASET}")

    used, total = get_gpu_memory()
    print(f"GPU before: {used} / {total} MiB")

    results = {}
    for train_bs, grad_accum, val_bs in CONFIGS:
        success, elapsed = run_test(train_bs, grad_accum, val_bs)
        results[(train_bs, grad_accum, val_bs)] = (success, elapsed)

        # Clean up GPU
        subprocess.run(["pkill", "-f", "flux_kontext_lora_finetuning"], capture_output=True)
        time.sleep(5)

        used, total = get_gpu_memory()
        print(f"  GPU after cleanup: {used} / {total} MiB")

        if not success:
            # If this batch size failed, no need to try larger ones
            # unless it was a non-OOM failure
            pass

    print("\n\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    for (train_bs, grad_accum, val_bs), (success, elapsed) in results.items():
        status = "OK" if success else "FAIL"
        print(f"  bs={train_bs}, accum={grad_accum}, val_bs={val_bs}: {status} ({elapsed:.1f}s)")
