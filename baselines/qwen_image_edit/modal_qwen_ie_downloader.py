"""
Real-time sync of edited_image.jpg from Modal volumes to local folders.

Runs continuously, checking both the Calvin and Libero inference volumes
for newly committed edited images and downloading them to the corresponding
local test data directories.

Usage:
    MODAL_PROFILE=modal-account-5-test \
    modal run modal_qwen_ie_downloader.py --interval 120
"""

import modal
import os

# ============================================================================
# Volume & Local Path Configuration
# ============================================================================
CALVIN_VOLUME_NAME = "qwen-ie-calvin-parallel-volume"
LIBERO_VOLUME_NAME = "qwen-ie-libero-parallel-volume"

CALVIN_LOCAL_DIR = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "calvin_abcd_test_data")
)
LIBERO_LOCAL_DIR = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libero_10_90_combi_test_data")
)

# Max images to return per remote call (keeps payload manageable)
MAX_BATCH_SIZE = 100

# ============================================================================
# Modal App
# ============================================================================
app = modal.App("qwen-ie-downloader")

calvin_volume = modal.Volume.from_name(CALVIN_VOLUME_NAME, create_if_missing=True)
libero_volume = modal.Volume.from_name(LIBERO_VOLUME_NAME, create_if_missing=True)

downloader_image = modal.Image.debian_slim(python_version="3.12")


# ============================================================================
# Remote Sync Function (CPU – no GPU needed)
# ============================================================================
@app.function(
    image=downloader_image,
    volumes={"/calvin": calvin_volume, "/libero": libero_volume},
    timeout=600,
)
def fetch_new_edits(
    calvin_known: list[str],
    libero_known: list[str],
) -> dict:
    """
    Scan both inference volumes for edited_image.jpg files that are
    NOT in the known lists.  Return their bytes so the local entrypoint
    can write them to disk.

    Returns:
        {
            "calvin": {case_id: bytes, ...},
            "libero": {case_id: bytes, ...},
        }
    """
    calvin_known_set = set(calvin_known)
    libero_known_set = set(libero_known)
    result: dict = {"calvin": {}, "libero": {}}
    count = 0

    # Reload to see latest commits from inference workers
    calvin_volume.reload()
    libero_volume.reload()

    # ---- Calvin --------------------------------------------------------
    calvin_data = "/calvin/test_data"
    if os.path.isdir(calvin_data):
        for name in sorted(os.listdir(calvin_data)):
            if count >= MAX_BATCH_SIZE:
                break
            if name in calvin_known_set:
                continue
            edited = os.path.join(calvin_data, name, "edited_image.jpg")
            if os.path.isfile(edited):
                with open(edited, "rb") as f:
                    result["calvin"][name] = f.read()
                count += 1

    # ---- Libero --------------------------------------------------------
    libero_data = "/libero/test_data"
    if os.path.isdir(libero_data):
        for name in sorted(os.listdir(libero_data)):
            if count >= MAX_BATCH_SIZE:
                break
            if name in libero_known_set:
                continue
            edited = os.path.join(libero_data, name, "edited_image.jpg")
            if os.path.isfile(edited):
                with open(edited, "rb") as f:
                    result["libero"][name] = f.read()
                count += 1

    nc = len(result["calvin"])
    nl = len(result["libero"])
    print(f"Returning {nc} Calvin + {nl} Libero new edits ({nc + nl} total)")
    return result


# ============================================================================
# Local Entrypoint
# ============================================================================
@app.local_entrypoint()
def main(interval: int = 120):
    """
    Continuously sync edited images from Modal volumes to local folders.

    Checks every ``interval`` seconds.  Exits automatically when every
    case in both datasets has an edited_image.jpg locally.

    Args:
        interval: Seconds between sync cycles (default 120).
    """
    import time

    # Count total cases locally
    calvin_total = len(
        [
            d
            for d in os.listdir(CALVIN_LOCAL_DIR)
            if os.path.isdir(os.path.join(CALVIN_LOCAL_DIR, d))
        ]
    )
    libero_total = len(
        [
            d
            for d in os.listdir(LIBERO_LOCAL_DIR)
            if os.path.isdir(os.path.join(LIBERO_LOCAL_DIR, d))
        ]
    )

    print("=" * 70)
    print("Qwen IE Downloader – Real-time Volume Sync")
    print("=" * 70)
    print(f"  Calvin:   {CALVIN_LOCAL_DIR}  ({calvin_total} cases)")
    print(f"  Libero:   {LIBERO_LOCAL_DIR}  ({libero_total} cases)")
    print(f"  Interval: {interval}s")
    print(f"  Batch:    {MAX_BATCH_SIZE} images/cycle")
    print("=" * 70)

    cycle = 0
    while True:
        cycle += 1

        # Gather locally-known case IDs
        calvin_known = [
            d
            for d in os.listdir(CALVIN_LOCAL_DIR)
            if os.path.isdir(os.path.join(CALVIN_LOCAL_DIR, d))
            and os.path.isfile(
                os.path.join(CALVIN_LOCAL_DIR, d, "edited_image.jpg")
            )
        ]
        libero_known = [
            d
            for d in os.listdir(LIBERO_LOCAL_DIR)
            if os.path.isdir(os.path.join(LIBERO_LOCAL_DIR, d))
            and os.path.isfile(
                os.path.join(LIBERO_LOCAL_DIR, d, "edited_image.jpg")
            )
        ]

        ts = time.strftime("%H:%M:%S")
        print(
            f"\n[{ts}] Cycle {cycle}: "
            f"Calvin {len(calvin_known)}/{calvin_total}, "
            f"Libero {len(libero_known)}/{libero_total}"
        )

        # Done?
        if len(calvin_known) >= calvin_total and len(libero_known) >= libero_total:
            print("All cases synced locally! Exiting.")
            break

        try:
            result = fetch_new_edits.remote(calvin_known, libero_known)

            new_calvin = result.get("calvin", {})
            new_libero = result.get("libero", {})

            for case_id, img_bytes in new_calvin.items():
                out = os.path.join(CALVIN_LOCAL_DIR, case_id, "edited_image.jpg")
                with open(out, "wb") as f:
                    f.write(img_bytes)

            for case_id, img_bytes in new_libero.items():
                out = os.path.join(LIBERO_LOCAL_DIR, case_id, "edited_image.jpg")
                with open(out, "wb") as f:
                    f.write(img_bytes)

            if new_calvin or new_libero:
                print(
                    f"  Synced {len(new_calvin)} Calvin + "
                    f"{len(new_libero)} Libero edits"
                )
            else:
                print("  No new edits found")

        except Exception as e:
            print(f"  Sync error: {e}")

        time.sleep(interval)

    print("\nDownloader finished – all edited images synced to local folders.")
