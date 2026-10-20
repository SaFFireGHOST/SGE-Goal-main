"""
Resize all edited images to match their corresponding original image dimensions.

Safety measures:
1. Creates a backup of each edited image before overwriting (edited_image_backup_256x256.jpg)
2. Verifies output dimensions match original after resize
3. Logs all operations for audit
"""

import os
import shutil
import sys
from PIL import Image

FOLDERS = [
    "bridge_data_v2_tel_test_data",
    "calvin_abcd_test_data",
    "libero_10_90_combi_test_data",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def process_folder(folder_name):
    folder_path = os.path.join(BASE_DIR, folder_name)
    subdirs = sorted(
        [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))],
        key=lambda x: int(x),
    )

    total = len(subdirs)
    resized = 0
    skipped = 0
    errors = []

    print(f"\n{'='*60}")
    print(f"Processing: {folder_name} ({total} test cases)")
    print(f"{'='*60}")

    for i, subdir in enumerate(subdirs):
        case_path = os.path.join(folder_path, subdir)
        files = os.listdir(case_path)

        # Find original and edited image files
        orig_files = [f for f in files if f.startswith("original_image")]
        edit_files = [f for f in files if f == "edited_image.jpg"]

        if not orig_files or not edit_files:
            msg = f"  [{subdir}] SKIP: missing files (orig={orig_files}, edit={edit_files})"
            print(msg)
            errors.append(msg)
            skipped += 1
            continue

        orig_path = os.path.join(case_path, orig_files[0])
        edit_path = os.path.join(case_path, edit_files[0])

        try:
            # Get original dimensions
            with Image.open(orig_path) as orig_img:
                target_size = orig_img.size  # (width, height)

            # Get edited dimensions
            with Image.open(edit_path) as edit_img:
                edit_size = edit_img.size

            # Skip if already matching
            if edit_size == target_size:
                skipped += 1
                continue

            # Create backup (only if backup doesn't already exist)
            backup_name = f"edited_image_backup_{edit_size[0]}x{edit_size[1]}.jpg"
            backup_path = os.path.join(case_path, backup_name)
            if not os.path.exists(backup_path):
                shutil.copy2(edit_path, backup_path)

            # Resize edited image to match original dimensions
            with Image.open(edit_path) as edit_img:
                # Convert to RGB if necessary (JPEG doesn't support alpha)
                if edit_img.mode != "RGB":
                    edit_img = edit_img.convert("RGB")
                resized_img = edit_img.resize(target_size, Image.LANCZOS)
                # Save with high quality to minimize compression artifacts
                resized_img.save(edit_path, "JPEG", quality=95)

            # Verify the saved image has correct dimensions
            with Image.open(edit_path) as verify_img:
                if verify_img.size != target_size:
                    msg = f"  [{subdir}] ERROR: verification failed! Expected {target_size}, got {verify_img.size}"
                    print(msg)
                    errors.append(msg)
                    # Restore from backup
                    shutil.copy2(backup_path, edit_path)
                    print(f"  [{subdir}] Restored from backup")
                    continue

            resized += 1

        except Exception as e:
            msg = f"  [{subdir}] ERROR: {e}"
            print(msg)
            errors.append(msg)
            # If backup exists and something went wrong, restore
            backup_candidates = [f for f in os.listdir(case_path) if f.startswith("edited_image_backup_")]
            if backup_candidates:
                shutil.copy2(os.path.join(case_path, backup_candidates[0]), edit_path)
                print(f"  [{subdir}] Restored from backup after error")

        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{total}")

    print(f"\n  Results: {resized} resized, {skipped} skipped, {len(errors)} errors")
    if errors:
        print(f"  Errors:")
        for e in errors:
            print(f"    {e}")

    return resized, skipped, errors


def main():
    total_resized = 0
    total_skipped = 0
    total_errors = []

    for folder in FOLDERS:
        resized, skipped, errors = process_folder(folder)
        total_resized += resized
        total_skipped += skipped
        total_errors.extend(errors)

    print(f"\n{'='*60}")
    print(f"OVERALL: {total_resized} resized, {total_skipped} skipped, {len(total_errors)} errors")
    print(f"{'='*60}")

    if total_errors:
        print("\nAll errors:")
        for e in total_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()
