"""Modal app for GCBC (Goal-Conditioned Diffusion BC) policy evaluation on CALVIN.

Uses the GHIL-Glue / SuSIE low-level GCBC diffusion policy, which is specifically
trained for goal-image conditioning on CALVIN.  Checkpoint from HuggingFace:
  kyle-hatch-tri/ghil-glue-checkpoints/susie_low_level/calvin/gcdiffusion/auggoaldiff/...

The policy takes (current_obs_image, goal_image) → 7-dim action.
"""

from __future__ import annotations
import os, sys, json, time, shutil, struct, io, zipfile
from pathlib import Path
import modal

# ---------------------------------------------------------------------------
APP_NAME = "rm-policy"
DATA_MOUNT = "/data"
CKPT_MOUNT = "/ckpts"
RUNS_MOUNT = "/runs"
EVAL_DATA_MOUNT = "/eval_data"
CALVIN_REPO = "/opt/calvin"
GCBC_CODE = "/opt/gcbc"
HF_CKPT_DIR = "/ckpts/gcbc"

data_vol = modal.Volume.from_name("rm-policy-calvin-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("rm-policy-ckpts", create_if_missing=True)
runs_vol = modal.Volume.from_name("rm-policy-runs", create_if_missing=True)
eval_data_vol = modal.Volume.from_name("rm-policy-eval-data", create_if_missing=True)

VOLUMES = {
    DATA_MOUNT: data_vol,
    CKPT_MOUNT: ckpt_vol,
    RUNS_MOUNT: runs_vol,
    EVAL_DATA_MOUNT: eval_data_vol,
}

# ---------------------------------------------------------------------------
_apt_packages = [
    "git", "wget", "build-essential", "cmake",
    "libegl1-mesa", "libgl1-mesa-glx", "libgles2-mesa",
    "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev",
    "libosmesa6", "libosmesa6-dev", "libglvnd-dev", "pkg-config",
    "ffmpeg", "clang", "llvm",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "MESA_GL_VERSION_OVERRIDE": "3.3",
        "PYOPENGL_PLATFORM": "egl",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".4",
    })
    .run_commands(
        "sed -i 's/^# deb http/deb http/g' /etc/apt/sources.list; "
        "grep -q universe /etc/apt/sources.list || "
        "echo 'deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse' >> /etc/apt/sources.list; "
        "echo 'deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list; "
        "echo 'deb http://archive.ubuntu.com/ubuntu jammy-security main restricted universe multiverse' >> /etc/apt/sources.list",
    )
    .run_commands(
        "rm -rf /var/lib/apt/lists/* && "
        "ok=0; for i in 1 2 3 4 5; do "
        "apt-get update && apt-get install -y --fix-missing "
        + " ".join(_apt_packages)
        + " && ok=1 && break; "
        "echo \"apt retry $i/5\"; rm -rf /var/lib/apt/lists/*; sleep 15; done; "
        "[ $ok -eq 1 ] || { echo 'FATAL: apt failed'; exit 1; }",
    )
    .pip_install("numpy==1.24.4")
    # JAX with CUDA 12 - install all together with numpy pinned
    .run_commands(
        "pip install 'jax[cuda12_pip]==0.4.20' 'numpy==1.24.4' 'scipy<1.12' 'flax==0.7.4' 'optax==0.1.7' 'orbax-checkpoint==0.4.4' 'distrax==0.1.4' 'ml_collections' 'einops' 'requests' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        # Verify versions
        "python -c \"import jax; print('JAX:', jax.__version__); import flax; print('flax:', flax.__version__); import numpy; print('numpy:', numpy.__version__)\"",
    )
    # CALVIN env (PyBullet + hydra)
    .pip_install(
        "scipy==1.11.4", "pillow", "tqdm", "omegaconf", "hydra-core==1.1.1",
        "opencv-python-headless", "pybullet", "pyarrow", "pandas",
        "huggingface_hub",
    )
    .run_commands(
        "pip install 'setuptools==65.5.0' 'wheel==0.38.4'",
        "pip install --no-build-isolation 'gym==0.21.0'",
    )
    # Install CALVIN
    .run_commands(
        "pip install 'setuptools<58' wheel",
        f"git clone --recurse-submodules https://github.com/mees/calvin.git {CALVIN_REPO}",
        f"cd {CALVIN_REPO}/calvin_env && pip install -e . || true",
        f"cd {CALVIN_REPO}/calvin_env/tacto && pip install -e . || true",
        f"cd {CALVIN_REPO}/calvin_models && pip install --no-deps -e .",
        "mkdir -p /tools/llvm/bin && ln -sf $(which clang) /tools/llvm/bin/clang && ln -sf $(which llvm-ar) /tools/llvm/bin/llvm-ar",
        "pip install pyhash || true",
        # Fix Python 3.10 collections.Mapping removal in tacto/pyrender
        "find /opt/calvin /usr/local/lib/python3.10 -name '*.py' -exec grep -l 'from collections import Mapping\\|collections.Mapping' {} \\; 2>/dev/null | head -20 | xargs -I{} sed -i 's/from collections import Mapping/from collections.abc import Mapping/g; s/collections.Mapping/collections.abc.Mapping/g' {} || true",
    )
    # Copy GCBC code (jaxrl_m + configs)
    .add_local_dir(
        str(Path(__file__).parent / "jaxrl_m"),
        f"{GCBC_CODE}/jaxrl_m",
    )
    .add_local_dir(
        str(Path(__file__).parent / "mini_dataset"),
        f"{GCBC_CODE}/mini_dataset",
    )
    .add_local_file(
        str(Path(__file__).parent / "gcbc_train_config.py"),
        f"{GCBC_CODE}/gcbc_train_config.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "diffusion_gc_policy.py"),
        f"{GCBC_CODE}/diffusion_gc_policy.py",
    )
)

app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# Data helpers (reuse GR-MG pattern for CALVIN-D download)
# ---------------------------------------------------------------------------

CALVIN_ZIP_URL = "https://huggingface.co/datasets/oakink/OakInk/resolve/main/task_D_D.zip"
CALVIN_ZIP_URL2 = "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"

def _fetch_calvin_d():
    """Download CALVIN-D validation split via partial ZIP extraction."""
    import numpy as np
    val_dir = Path(DATA_MOUNT) / "task_D_D" / "validation"
    if val_dir.exists() and len(list(val_dir.glob("episode_*.npz"))) > 1000:
        print(f"[fetch] CALVIN-D already present ({len(list(val_dir.glob('episode_*.npz')))} npz files)")
        return

    import requests, struct, io, zipfile as zf

    url = "https://huggingface.co/datasets/oakink/OakInk/resolve/main/task_D_D.zip"

    # Get ZIP size
    r = requests.head(url, allow_redirects=True, timeout=30)
    total = int(r.headers.get("Content-Length", 0))
    if total == 0:
        raise RuntimeError("Cannot determine ZIP size")
    print(f"[fetch] ZIP size: {total/1e9:.2f} GB")

    # Read EOCD
    eocd_fetch = 65536
    r = requests.get(url, headers={"Range": f"bytes={total - eocd_fetch}-{total - 1}"}, timeout=30)
    eocd_data = r.content
    eocd_sig = b'\x50\x4b\x05\x06'
    idx = eocd_data.rfind(eocd_sig)
    if idx < 0:
        raise RuntimeError("EOCD not found")
    cd_size = struct.unpack_from('<I', eocd_data, idx + 12)[0]
    cd_offset = struct.unpack_from('<I', eocd_data, idx + 16)[0]
    print(f"[fetch] CD offset={cd_offset}, size={cd_size}")

    # Read central directory
    r = requests.get(url, headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"}, timeout=60)
    cd_data = r.content

    # Parse entries
    entries = []
    pos = 0
    while pos < len(cd_data):
        if cd_data[pos:pos+4] != b'\x50\x4b\x01\x02':
            break
        fname_len = struct.unpack_from('<H', cd_data, pos + 28)[0]
        extra_len = struct.unpack_from('<H', cd_data, pos + 30)[0]
        comment_len = struct.unpack_from('<H', cd_data, pos + 32)[0]
        comp_size = struct.unpack_from('<I', cd_data, pos + 20)[0]
        local_offset = struct.unpack_from('<I', cd_data, pos + 42)[0]
        fname = cd_data[pos + 46: pos + 46 + fname_len].decode('utf-8', errors='replace')
        entries.append((fname, local_offset, comp_size))
        pos += 46 + fname_len + extra_len + comment_len

    # Filter validation entries
    val_entries = [e for e in entries if "/validation/" in e[0] and not e[0].endswith("/")]
    print(f"[fetch] {len(val_entries)} validation files to extract")

    # Extract
    extracted = 0
    for fname, offset, comp_size in val_entries:
        # strip task_D_D/ prefix
        rel = fname.split("task_D_D/", 1)[-1] if "task_D_D/" in fname else fname
        out_path = Path(DATA_MOUNT) / "task_D_D" / rel
        if out_path.exists():
            extracted += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read local file header + data
        fetch_size = 30 + 300 + comp_size + 100
        r = requests.get(url, headers={"Range": f"bytes={offset}-{offset + fetch_size - 1}"}, timeout=30)
        local_data = r.content

        if local_data[:4] != b'\x50\x4b\x03\x04':
            continue
        lfname_len = struct.unpack_from('<H', local_data, 26)[0]
        lextra_len = struct.unpack_from('<H', local_data, 28)[0]
        data_start = 30 + lfname_len + lextra_len
        comp_method = struct.unpack_from('<H', local_data, 8)[0]
        raw = local_data[data_start:data_start + comp_size]

        if comp_method == 0:
            out_path.write_bytes(raw)
        elif comp_method == 8:
            import zlib
            out_path.write_bytes(zlib.decompress(raw, -15))
        extracted += 1
        if extracted % 2000 == 0:
            print(f"[fetch] extracted {extracted}/{len(val_entries)}")

    # Also get training scene_info
    train_dir = Path(DATA_MOUNT) / "task_D_D" / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_entries = [e for e in entries if "/training/" in e[0] and "scene_info" in e[0]]
    for fname, offset, comp_size in train_entries:
        rel = fname.split("task_D_D/", 1)[-1] if "task_D_D/" in fname else fname
        out_path = Path(DATA_MOUNT) / "task_D_D" / rel
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fetch_size = 30 + 300 + comp_size + 100
        r = requests.get(url, headers={"Range": f"bytes={offset}-{offset + fetch_size - 1}"}, timeout=30)
        local_data = r.content
        if local_data[:4] != b'\x50\x4b\x03\x04':
            continue
        lfname_len = struct.unpack_from('<H', local_data, 26)[0]
        lextra_len = struct.unpack_from('<H', local_data, 28)[0]
        data_start = 30 + lfname_len + lextra_len
        comp_method = struct.unpack_from('<H', local_data, 8)[0]
        raw = local_data[data_start:data_start + comp_size]
        if comp_method == 0:
            out_path.write_bytes(raw)
        elif comp_method == 8:
            import zlib
            out_path.write_bytes(zlib.decompress(raw, -15))

    # Create training symlink if needed
    train_npz = list(train_dir.glob("episode_*.npz"))
    if len(train_npz) == 0:
        # Symlink training -> validation so CALVIN data module doesn't crash
        for item in val_dir.iterdir():
            dst = train_dir / item.name
            if not dst.exists():
                try:
                    dst.symlink_to(item)
                except Exception:
                    pass

    print(f"[fetch] done. {extracted} files extracted")
    data_vol.commit()


def _download_gcbc_ckpt():
    """Download GCBC checkpoint from HuggingFace."""
    ckpt_dir = Path(HF_CKPT_DIR)
    marker = ckpt_dir / ".downloaded"
    if marker.exists():
        print("[ckpt] GCBC checkpoint already downloaded")
        return

    from huggingface_hub import hf_hub_download, snapshot_download
    print("[ckpt] Downloading GCBC checkpoint from HuggingFace...")

    # Download just the calvin GCBC checkpoint
    local_dir = str(ckpt_dir / "checkpoint_150000")
    snapshot_download(
        repo_id="kyle-hatch-tri/ghil-glue-checkpoints",
        allow_patterns="susie_low_level/calvin/gcdiffusion/auggoaldiff/seed_0/20240227_194024/checkpoint_150000/*",
        local_dir=str(ckpt_dir / "hf_download"),
        local_dir_use_symlinks=False,
    )
    # Move to expected path
    src = ckpt_dir / "hf_download" / "susie_low_level" / "calvin" / "gcdiffusion" / "auggoaldiff" / "seed_0" / "20240227_194024" / "checkpoint_150000"
    dst = ckpt_dir / "checkpoint_150000"
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
    # Cleanup
    hf_dl = ckpt_dir / "hf_download"
    if hf_dl.exists():
        shutil.rmtree(str(hf_dl), ignore_errors=True)

    marker.touch()
    print(f"[ckpt] GCBC checkpoint at {dst}")
    ckpt_vol.commit()


def _upload_mapping():
    """Ensure hf_calvin_mapping.json is on the runs volume."""
    mapping_path = Path(RUNS_MOUNT) / "hf_calvin_mapping.json"
    if mapping_path.exists():
        print("[mapping] already present")
        return
    # Download from the same HF dataset we use for eval data
    import requests
    url = "https://huggingface.co/datasets/riddhi-more/CALVIN-EVALS/resolve/main/hf_calvin_mapping.json"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_bytes(r.content)
    print(f"[mapping] downloaded ({len(r.content)} bytes)")
    runs_vol.commit()


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

def _init_gcbc_policy(ckpt_path: str):
    """Initialize the GCBC diffusion policy from checkpoint."""
    sys.path.insert(0, GCBC_CODE)
    sys.path.insert(0, f"{GCBC_CODE}/jaxrl_m")

    import jax
    import numpy as np
    from ml_collections import ConfigDict
    from jaxrl_m.jaxrl_m.vision import encoders
    from jaxrl_m.jaxrl_m.agents import agents
    import orbax.checkpoint

    # Config for calvin_gcdiffusion_noactnorm-sagemaker-auggoaldiff
    agent_type = "gc_ddpm_bc"
    encoder_name = "resnetv1-34-bridge"
    encoder_kwargs = dict(
        pooling_method="avg",
        add_spatial_coordinates=True,
        act="swish",
    )
    agent_kwargs = dict(
        score_network_kwargs=dict(
            time_dim=32,
            num_blocks=3,
            dropout_rate=0.1,
            hidden_dim=256,
            use_layer_norm=True,
        ),
        early_goal_concat=True,
        shared_goal_encoder=True,
        use_proprio=False,
        beta_schedule="cosine",
        diffusion_steps=20,
        action_samples=1,
        repeat_last_step=0,
        learning_rate=3e-4,
        warmup_steps=2000,
        actor_decay_steps=int(2e6),
    )

    # Create dummy batch for initialization
    example_actions = np.zeros((1, 4, 7), dtype=np.float32)  # act_pred_horizon=4, action_dim=7
    example_obs = {"image": np.zeros((1, 1, 200, 200, 3), dtype=np.uint8)}  # obs_horizon=1
    example_goal = {"image": np.zeros((1, 200, 200, 3), dtype=np.uint8)}
    example_batch = {
        "observations": example_obs,
        "actions": example_actions,
        "goals": example_goal,
    }
    example_batch["goals"]["language"] = np.zeros((1, 512), dtype=np.float32)

    # Initialize encoder
    encoder_def = encoders[encoder_name](**encoder_kwargs)

    # Create agent
    rng = jax.random.PRNGKey(42)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[agent_type].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **agent_kwargs,
    )

    # Load checkpoint
    print(f"[policy] Loading checkpoint from {ckpt_path}...")
    restored = orbax.checkpoint.PyTreeCheckpointer().restore(ckpt_path, item=agent)
    if agent is restored:
        raise FileNotFoundError(f"Cannot load checkpoint from {ckpt_path}")
    print("[policy] Checkpoint loaded successfully")
    return restored


class GCBCPolicyWrapper:
    """Wraps the GCBC diffusion policy with temporal ensembling."""

    def __init__(self, agent):
        import jax
        import numpy as np
        self.agent = agent
        self.jax = jax
        self.np = np
        # Action normalization stats (from diffusion_gc_policy.py)
        self.act_mean = np.array([2.9842544e-04, -2.6099570e-04, -1.5863389e-04,
                                   5.8916201e-05, -4.4560504e-05, 8.2349771e-04, 9.4075650e-02])
        self.act_std = np.array([0.27278143, 0.23548537, 0.2196189,
                                  0.15881406, 0.17537235, 0.27875036, 1.0049515])
        self.reset()

    def reset(self):
        self.action_buffer = self.np.zeros((4, 4, 7))
        self.action_buffer_mask = self.np.zeros((4, 4), dtype=bool)

    def predict_action(self, image_obs, goal_image):
        """
        Args:
            image_obs: uint8 array (200, 200, 3)
            goal_image: uint8 array (200, 200, 3)
        Returns:
            action: float32 array (7,) - 3 pos + 3 rot + 1 gripper
        """
        np = self.np
        jax = self.jax

        # Add observation history dimension: (200,200,3) -> (1,200,200,3)
        obs_input = {"image": image_obs[np.newaxis, ...]}
        goal_input = {"image": goal_image}

        action = self.agent.sample_actions(
            obs_input,
            goal_input,
            seed=jax.random.PRNGKey(42),
            temperature=0.0,
        )
        action = np.array(action.tolist())  # (4, 7)

        # Temporal ensembling
        # Shift buffers
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * np.array(
            [[True, True, True, True],
             [True, True, True, False],
             [True, True, False, False],
             [True, False, False, False]], dtype=bool)

        self.action_buffer[0] = action
        self.action_buffer_mask[0] = np.array([True, True, True, True], dtype=bool)

        action_prediction = np.sum(
            self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0
        ) / np.sum(self.action_buffer_mask[:, 0], axis=0)

        # Binarize gripper
        if action_prediction[-1] < 0:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        return action_prediction


# ---------------------------------------------------------------------------
# Read CALVIN npz helpers
# ---------------------------------------------------------------------------

def _read_npz_full(val_dir, frame_idx):
    import numpy as np
    p = val_dir / f"episode_{frame_idx:07d}.npz"
    d = np.load(str(p))
    return {
        "robot_obs": d["robot_obs"],
        "scene_obs": d["scene_obs"],
        "rgb_static": d["rgb_static"],
        "rgb_gripper": d["rgb_gripper"],
        "depth_static": d["depth_static"],
        "depth_gripper": d["depth_gripper"],
    }


def _read_npz_init(val_dir, frame_idx):
    import numpy as np
    p = val_dir / f"episode_{frame_idx:07d}.npz"
    d = np.load(str(p))
    return {"robot_obs": d["robot_obs"], "scene_obs": d["scene_obs"]}


# ---------------------------------------------------------------------------
# Load goal images from eval-data or CALVIN validation
# ---------------------------------------------------------------------------

def _load_eval_data_rows(method_name, mapping, row_start, row_end):
    """Load goal images from eval-data volume (pre-uploaded method images)."""
    import numpy as np
    from PIL import Image

    matches = mapping["matches"]
    # Some fine-tuned methods use a different subdirectory name
    method_dir = Path(EVAL_DATA_MOUNT) / method_name / "calvin_abcd_test_data"
    if not method_dir.exists():
        alt_dir = Path(EVAL_DATA_MOUNT) / method_name / "calvin_abcd_ier_fer_gt_test_data"
        if alt_dir.exists():
            method_dir = alt_dir

    rows = []
    for i in range(row_start, min(row_end, len(matches))):
        m = matches[i]
        # Directories are 1-indexed: hf_row 0 → dir "1"
        dir_num = i + 1
        img_path = method_dir / str(dir_num) / "edited_image.jpg"
        if img_path.exists():
            img = Image.open(str(img_path)).convert("RGB")
            img_np = np.array(img.resize((200, 200), Image.BILINEAR))
        else:
            # Try png
            img_path_png = method_dir / str(dir_num) / "edited_image.png"
            if img_path_png.exists():
                img = Image.open(str(img_path_png)).convert("RGB")
                img_np = np.array(img.resize((200, 200), Image.BILINEAR))
            else:
                print(f"[warn] no goal image for row {i} dir {dir_num}")
                img_np = np.zeros((200, 200, 3), dtype=np.uint8)
        rows.append({
            "hf_row": m["hf_row"],
            "hf_id": m.get("hf_id", f"row_{m['hf_row']}"),
            "task": m["task"],
            "calvin_ann": m.get("ann", m.get("calvin_ann", "")),
            "start_frame": m["start_frame"],
            "end_frame": m["end_frame"],
            "goal_image": img_np,
        })
    return rows


def _load_ground_truth_rows(mapping, val_dir, row_start, row_end):
    """Load ground truth goal images from CALVIN validation end frames."""
    import numpy as np
    from pathlib import Path

    matches = mapping["matches"]
    rows = []
    for i in range(row_start, min(row_end, len(matches))):
        m = matches[i]
        end_frame = m["end_frame"]
        start_frame = m["start_frame"]
        # Load the end-frame image as the ground truth goal
        npz_path = val_dir / f"episode_{end_frame:07d}.npz"
        if npz_path.exists():
            d = np.load(str(npz_path))
            goal_img = d["rgb_static"]  # (200, 200, 3) uint8
        else:
            goal_img = np.zeros((200, 200, 3), dtype=np.uint8)
        rows.append({
            "hf_row": m["hf_row"],
            "hf_id": m.get("hf_id", f"row_{m['hf_row']}"),
            "task": m["task"],
            "calvin_ann": m.get("ann", ""),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "goal_image": goal_img,
        })
    return rows


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=21600,
    volumes=VOLUMES,
)
def eval_method_shard(
    method_name: str,
    row_start: int,
    row_end: int,
    ep_len: int = 360,
    run_tag: str = "",
    force_rerun: bool = False,
):
    """Evaluate one shard of rows for a given method."""
    # Fix Python 3.10 collections.Mapping removal
    import collections
    import collections.abc
    for attr in ('Mapping', 'MutableMapping', 'Sequence', 'MutableSequence',
                 'Set', 'MutableSet', 'Callable', 'Iterable', 'Iterator'):
        if not hasattr(collections, attr):
            setattr(collections, attr, getattr(collections.abc, attr))

    import numpy as np
    import hydra
    from omegaconf import OmegaConf
    from pathlib import Path

    tag = run_tag or method_name
    out_dir = Path(RUNS_MOUNT) / "eval_method_gcbc" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    done_marker = out_dir / f"shard_done_{row_start}_{row_end}.json"
    if done_marker.exists() and not force_rerun:
        print(f"[gcbc-eval] shard [{row_start},{row_end}) already done, skipping")
        return
    elif done_marker.exists() and force_rerun:
        print(f"[gcbc-eval] shard [{row_start},{row_end}) force re-run, removing done marker")
        done_marker.unlink()

    jsonl_path = out_dir / f"per_sample_{row_start}_{row_end}.jsonl"

    # Resume: read already-done hf_rows from existing jsonl
    done_hf_rows = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_hf_rows.add(rec["hf_row"])
        if done_hf_rows:
            print(f"[gcbc-eval] shard [{row_start},{row_end}) resuming: {len(done_hf_rows)} rows already done")

    # Load mapping
    mapping_path = Path(RUNS_MOUNT) / "hf_calvin_mapping.json"
    with open(mapping_path) as f:
        mapping = json.load(f)

    val_dir = Path(DATA_MOUNT) / "task_D_D" / "validation"

    # Load GCBC policy
    ckpt_path = f"{HF_CKPT_DIR}/checkpoint_150000"
    agent = _init_gcbc_policy(ckpt_path)
    policy = GCBCPolicyWrapper(agent)

    # Set up CALVIN environment
    from calvin_env.envs.play_table_env import get_env
    obs_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language'],
    }
    env = get_env(val_dir, show_gui=False, obs_space=obs_space)

    # Load task oracle
    conf_dir = Path(CALVIN_REPO) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    # Load goal images
    if method_name == "ground_truth":
        goal_rows = _load_ground_truth_rows(mapping, val_dir, row_start, row_end)
    else:
        goal_rows = _load_eval_data_rows(method_name, mapping, row_start, row_end)

    n_success = 0
    n_skipped = 0
    n_total = 0

    with open(jsonl_path, "a") as fout:
        for ri, grow in enumerate(goal_rows):
            hf_row = grow["hf_row"]

            # Skip if already processed (resume)
            if hf_row in done_hf_rows:
                continue

            task = grow["task"]
            start_frame = grow["start_frame"]

            # Reset env to start state
            init_data = _read_npz_init(val_dir, start_frame)
            robot_obs = init_data["robot_obs"]
            scene_obs = init_data["scene_obs"]
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

            # Get goal image
            goal_img = grow["goal_image"]  # (200, 200, 3) uint8

            # Reset policy
            policy.reset()

            # Run episode
            obs = env.get_obs()
            start_info = env.get_info()
            success = False
            steps_used = ep_len

            for step in range(ep_len):
                rgb_static = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3)

                action = policy.predict_action(rgb_static, goal_img)
                obs, _, _, current_info = env.step(action)

                # Check success
                current_task_info = task_oracle.get_task_info_for_set(
                    start_info, current_info, {task}
                )
                if len(current_task_info) > 0:
                    success = True
                    steps_used = step + 1
                    break

            n_total += 1
            if success:
                n_success += 1

            sr = n_success / n_total if n_total > 0 else 0
            rec = {
                "hf_row": hf_row,
                "hf_id": grow["hf_id"],
                "task": task,
                "calvin_ann": grow["calvin_ann"],
                "start_frame": start_frame,
                "end_frame": grow["end_frame"],
                "success": success,
                "skipped": False,
                "steps_used": steps_used,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

            # Periodic volume commit every 25 rows
            if n_total % 25 == 0:
                runs_vol.commit()

            if n_total % 5 == 0:
                print(f"[gcbc-eval] row {hf_row:4d}  SR={sr:.3f}  task={task:35s}  success={int(success)}")

    # Write done marker
    done_data = {
        "method": method_name,
        "row_start": row_start,
        "row_end": row_end,
        "n_total": n_total,
        "n_success": n_success,
        "n_skipped": n_skipped,
        "sr": n_success / max(n_total, 1),
    }
    with open(done_marker, "w") as f:
        json.dump(done_data, f)

    runs_vol.commit()
    sr = n_success / max(n_total, 1)
    print(f"[gcbc-eval] DONE shard [{row_start},{row_end}) SR={sr:.4f} ({n_success}/{n_total}) skipped={n_skipped}")


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def bootstrap():
    """Download CALVIN data + GCBC checkpoint + mapping."""
    _bootstrap.remote()

@app.function(gpu="T4", timeout=3600, volumes=VOLUMES)
def _bootstrap():
    _fetch_calvin_d()
    _download_gcbc_ckpt()
    _upload_mapping()
    print("[bootstrap] Done. Data + checkpoint + mapping ready.")


@app.local_entrypoint()
def run_setup_check():
    _setup_check.remote()

@app.function(gpu="T4", timeout=600, volumes=VOLUMES)
def _setup_check():
    import subprocess
    print("== CUDA ==")
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    print(f"== JAX ==")
    import jax
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    val_dir = Path(DATA_MOUNT) / "task_D_D" / "validation"
    n_ep = len(list(val_dir.glob("episode_*.npz"))) if val_dir.exists() else 0
    print(f"Validation episodes: {n_ep}")

    ckpt = Path(HF_CKPT_DIR) / "checkpoint_150000"
    print(f"GCBC ckpt exists: {ckpt.exists()}")

    mapping = Path(RUNS_MOUNT) / "hf_calvin_mapping.json"
    print(f"Mapping exists: {mapping.exists()}")

    # Check eval-data
    ed = Path(EVAL_DATA_MOUNT)
    if ed.exists():
        methods = [d.name for d in ed.iterdir() if d.is_dir()]
        print(f"Eval-data methods: {sorted(methods)}")


@app.local_entrypoint()
def smoke_test(method_name: str = "ground_truth", num_rows: int = 10):
    """Quick test with a few rows."""
    if num_rows == 0:
        _setup_check.remote()
        return
    handle = eval_method_shard.spawn(
        method_name=method_name,
        row_start=0,
        row_end=num_rows,
        run_tag=f"{method_name}-smoke",
    )
    handle.get()
    print(f"[smoke] done")


@app.local_entrypoint()
def run_method(method_name: str, num_shards: int = 5, run_tag: str = ""):
    """Run full evaluation for a method across shards."""
    TOTAL_ROWS = 1087
    shard_size = (TOTAL_ROWS + num_shards - 1) // num_shards
    tag = run_tag or method_name

    handles = []
    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, TOTAL_ROWS)
        h = eval_method_shard.spawn(
            method_name=method_name,
            row_start=start,
            row_end=end,
            run_tag=tag,
        )
        handles.append(h)
        print(f"[run_method] spawned shard {i}: [{start}, {end})")

    for h in handles:
        h.get()
    print(f"[run_method] All {num_shards} shards done for {method_name}")


@app.local_entrypoint()
def run_missing(shards_json: str, force: bool = False):
    """Re-run specific incomplete shards. Input is JSON list of [method, start, end].

    Example:
        modal run GCBC/modal_app.py::run_missing --shards-json '["susie",0,218]]' --force
    """
    import json as _json
    shards = _json.loads(shards_json)
    handles = []
    for method, start, end in shards:
        h = eval_method_shard.spawn(
            method_name=method,
            row_start=start,
            row_end=end,
            run_tag=method,
            force_rerun=force,
        )
        handles.append((method, start, end, h))
        print(f"[run_missing] spawned {method} [{start}, {end})")

    for method, start, end, h in handles:
        h.get()
        print(f"[run_missing] DONE {method} [{start}, {end})")
    print(f"[run_missing] All {len(handles)} shards complete.")


# ---------------------------------------------------------------------------
# Robust Freiburg-based CALVIN-D validation fetch (ZIP64-aware partial unzip).
# Ported from GR-MG/modal_app.py because the GCBC default oakink HF mirror is
# now gated (HTTP 401). Writes into the shared rm-policy-calvin-data volume at
# task_D_D/validation, which is exactly what the GCBC eval reads.
# ---------------------------------------------------------------------------

_FREIBURG_CALVIN_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
_FREIBURG_SCENE_FIX = "http://calvin.cs.uni-freiburg.de/scene_info_fix/task_D_D_scene_info.zip"


def _partial_zip_extract_freiburg(url, prefix, dest, workers=16):
    import struct, zlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests

    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (CALVIN-fetch)"
    head = sess.head(url, allow_redirects=True, timeout=30)
    total_size = int(head.headers.get("Content-Length", 0))
    accept_ranges = head.headers.get("Accept-Ranges", "")
    print(f"[pz] total={total_size/1e9:.2f}GB ranges={accept_ranges!r}")
    if total_size == 0 or "bytes" not in accept_ranges.lower():
        return None

    eocd_read = min(4 * 1024 * 1024, total_size)
    r = sess.get(url, headers={"Range": f"bytes={total_size-eocd_read}-{total_size-1}"}, timeout=60)
    r.raise_for_status()
    tail = r.content
    eocd_pos = tail.rfind(b"\x50\x4b\x05\x06")
    if eocd_pos == -1:
        return None
    eocd = tail[eocd_pos:eocd_pos + 22]
    (_, _, _, _, _, cd_size, cd_offset, _) = struct.unpack("<IHHHHIIH", eocd)
    zip64_loc_pos = tail.rfind(b"\x50\x4b\x06\x07")
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or zip64_loc_pos != -1:
        if zip64_loc_pos != -1:
            loc = tail[zip64_loc_pos:zip64_loc_pos + 20]
            _, _, zip64_eocd_offset, _ = struct.unpack("<IIQI", loc)
            r64 = sess.get(url, headers={"Range": f"bytes={zip64_eocd_offset}-{zip64_eocd_offset+55}"}, timeout=60)
            r64.raise_for_status()
            z64 = r64.content
            if z64[:4] == b"\x50\x4b\x06\x06":
                cd_size = struct.unpack("<Q", z64[40:48])[0]
                cd_offset = struct.unpack("<Q", z64[48:56])[0]
    r = sess.get(url, headers={"Range": f"bytes={cd_offset}-{cd_offset+cd_size-1}"}, timeout=120)
    r.raise_for_status()
    cd_data = r.content

    entries = []
    pos = 0
    while pos < len(cd_data) - 46:
        if cd_data[pos:pos + 4] != b"\x50\x4b\x01\x02":
            break
        (sig, _vm, _vn, _fl, cm, _mt, _md, _crc, comp, uncomp,
         nlen, elen, clen, _ds, _ia, _ea, lho) = struct.unpack(
            "<4sHHHHHHIIIHHHHHII", cd_data[pos:pos + 46])
        name = cd_data[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd_data[pos + 46 + nlen:pos + 46 + nlen + elen]
        if uncomp == 0xFFFFFFFF or comp == 0xFFFFFFFF or lho == 0xFFFFFFFF:
            ep = 0
            while ep < len(extra) - 4:
                tag, sz = struct.unpack("<HH", extra[ep:ep + 4])
                if tag == 0x0001:
                    vals = extra[ep + 4:ep + 4 + sz]; idx = 0
                    if uncomp == 0xFFFFFFFF and idx + 8 <= len(vals):
                        uncomp = struct.unpack("<Q", vals[idx:idx + 8])[0]; idx += 8
                    if comp == 0xFFFFFFFF and idx + 8 <= len(vals):
                        comp = struct.unpack("<Q", vals[idx:idx + 8])[0]; idx += 8
                    if lho == 0xFFFFFFFF and idx + 8 <= len(vals):
                        lho = struct.unpack("<Q", vals[idx:idx + 8])[0]; idx += 8
                    break
                ep += 4 + sz
        entries.append({"name": name, "compress_method": cm,
                        "compressed_size": comp, "local_header_offset": lho})
        pos += 46 + nlen + elen + clen
    print(f"[pz] entries={len(entries)}")

    wanted = [e for e in entries if f"/{prefix}" in e["name"] or e["name"].startswith(prefix)]
    print(f"[pz] matching '{prefix}': {len(wanted)}")
    if not wanted:
        return None
    dest_path = Path(dest); dest_path.mkdir(parents=True, exist_ok=True)
    done = [0]; errors = []

    def fetch_one(entry):
        name = entry["name"]
        if name.endswith("/"):
            (dest_path / name).mkdir(parents=True, exist_ok=True); return
        offset = entry["local_header_offset"]
        hr = sess.get(url, headers={"Range": f"bytes={offset}-{offset+29}"}, timeout=30)
        hr.raise_for_status(); hdr = hr.content
        ln = struct.unpack("<H", hdr[26:28])[0]; le = struct.unpack("<H", hdr[28:30])[0]
        data_off = offset + 30 + ln + le; cs = entry["compressed_size"]
        out = dest_path / name; out.parent.mkdir(parents=True, exist_ok=True)
        if cs == 0:
            out.write_bytes(b""); return
        dr = sess.get(url, headers={"Range": f"bytes={data_off}-{data_off+cs-1}"}, timeout=120)
        dr.raise_for_status(); raw = dr.content
        m = entry["compress_method"]
        out.write_bytes(raw if m == 0 else zlib.decompress(raw, -15))
        done[0] += 1
        if done[0] % 2000 == 0:
            print(f"[pz] extracted {done[0]}/{len(wanted)}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_one, e): e for e in wanted}
        for f in as_completed(futs):
            exc = f.exception()
            if exc:
                errors.append((futs[f]["name"], str(exc)))
                if len(errors) > 20:
                    break
    print(f"[pz] extracted {done[0]}/{len(wanted)} errors={len(errors)}")
    return dest_path


@app.function(volumes={DATA_MOUNT: data_vol}, timeout=60 * 60 * 4, cpu=4, memory=16384)
def fetch_calvin_freiburg(force: bool = False):
    import shutil, zipfile
    target = Path(DATA_MOUNT) / "task_D_D"
    val_dir = target / "validation"
    if val_dir.exists() and not force and len(list(val_dir.glob("episode_*.npz"))) > 1000:
        print(f"[fetch] validation already present ({len(list(val_dir.glob('episode_*.npz')))} npz)")
        return str(val_dir)
    target.mkdir(parents=True, exist_ok=True)
    scratch = Path("/tmp/calvin_dl"); scratch.mkdir(parents=True, exist_ok=True)
    res = _partial_zip_extract_freiburg(_FREIBURG_CALVIN_URL, "validation/", str(scratch), workers=24)
    if res is None:
        raise SystemExit("partial-zip failed")
    src_val = None
    for c in [scratch / "task_D_D" / "validation", scratch / "validation"]:
        if c.exists():
            src_val = c; break
    if src_val is None:
        for p in scratch.rglob("validation"):
            if p.is_dir():
                src_val = p; break
    if src_val is None:
        raise SystemExit("validation/ not found after extract")
    if val_dir.exists():
        shutil.rmtree(val_dir)
    shutil.move(str(src_val), str(val_dir))
    # scene_info fix
    try:
        import requests
        fix = scratch / "fix.zip"
        rr = requests.get(_FREIBURG_SCENE_FIX, timeout=120)
        if rr.status_code == 200:
            fix.write_bytes(rr.content)
            with zipfile.ZipFile(fix) as zf:
                zf.extractall(target)
            print("[fetch] scene_info fix applied")
    except Exception as e:
        print(f"[fetch] scene_info fix skipped: {e}")
    # training scene_info symlink fallback so data module doesn't crash
    train_dir = target / "training"; train_dir.mkdir(parents=True, exist_ok=True)
    if len(list(train_dir.glob("episode_*.npz"))) == 0:
        for item in val_dir.iterdir():
            dst = train_dir / item.name
            if not dst.exists():
                try:
                    dst.symlink_to(item)
                except Exception:
                    pass
    shutil.rmtree(scratch, ignore_errors=True)
    n = len(list(val_dir.glob("episode_*.npz")))
    print(f"[fetch] DONE episode_npz={n}")
    data_vol.commit()
    return str(val_dir)


@app.local_entrypoint()
def run_fetch_calvin(force: bool = False):
    fetch_calvin_freiburg.remote(force=force)
    print("[run_fetch_calvin] done")
