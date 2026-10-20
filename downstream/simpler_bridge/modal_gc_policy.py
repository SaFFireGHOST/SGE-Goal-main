"""
SuSIE / GHIL-Glue BridgeData-V2 goal-conditioned diffusion policy on Modal.

This REPLACES Octo as the shared low-level policy for the SR experiment. It is a
language-agnostic, goal-IMAGE-conditioned diffusion policy (jaxrl_m gc_ddpm_bc)
pretrained on BridgeData V2, and is robust to GENERATED goal images (the SuSIE
paper shows it beats a ground-truth-goal oracle) — exactly what we need since the
editor goals are generated/edited images.

  policy = GCPolicy(get_config("bridge_gcdiffusion_noactnorm-sagemaker-auggoaldiff"))
  action_7 = policy.predict_action(obs_200x200x3, goal_200x200x3[None])  # xyz,rpy,grip

Checkpoint: hf://kyle-hatch-tri/ghil-glue-checkpoints
  susie_low_level/bridge/gcdiffusion/auggoaldiff/seed_0/20240602_011058/checkpoint_150000

Smoke (no env): policy loads + predicts on dummy images
  modal run simpler-eval/modal_gc_policy.py::smoke
"""
import modal

app = modal.App("simpler-gc-policy")

JAXRL_FORK = "https://github.com/kyle-hatch-tri/bridge_data_v2_ghil-glue.git"
CKPT_REPO = "kyle-hatch-tri/ghil-glue-checkpoints"
CKPT_SUBDIR = "susie_low_level/bridge/gcdiffusion/auggoaldiff/seed_0/20240602_011058/checkpoint_150000"
CONFIG_STRING = "bridge_gcdiffusion_noactnorm-sagemaker-auggoaldiff"

# GC-policy image: jaxrl_m fork + jax 0.4.11 (cuda11) + flax 0.7 / distrax 0.1.2.
# tf 2.13 is needed by jaxrl_m.data imports but forced to CPU at runtime.
gc_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.10")
    .apt_install("git", "wget", "build-essential", "ffmpeg", "libglib2.0-0")
    .run_commands("python -m pip install --upgrade pip 'setuptools<70' wheel")
    .run_commands("git clone " + JAXRL_FORK + " /root/bridge_data_v2")
    # install jaxrl_m (the fork's package) without its (heavy/pinned) deps, then
    # add the exact pinned deps it needs.
    .run_commands("cd /root/bridge_data_v2 && python -m pip install -e . --no-deps")
    .pip_install(
        "numpy==1.24.3", "ml_collections>=0.1.0", "einops==0.6.1",
        "distrax==0.1.2", "flax==0.7.0", "optax==0.1.5", "chex==0.1.7",
        "orbax-checkpoint==0.2.3",
        "tensorflow==2.13.0", "tensorflow_probability==0.21.0",
        "tensorflow_hub==0.15.0",
        "transformers==4.34.1", "pillow", "opencv-python-headless",
        "imageio", "tqdm", "wandb", "func_timeout", "plotly",
        "huggingface_hub", "hf_transfer",
    )
    .run_commands(
        "python -m pip install 'jax[cuda11_pip]==0.4.11' "
        "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        # jax 0.4.11 needs ml_dtypes with float8_e4m3b11 (removed in >=0.4); deps
        # otherwise pull ml_dtypes 0.5.x -> AttributeError at jax import.
        # scipy>=1.13 removed scipy.linalg.tril that jax/tfp import -> pin <1.13.
        "python -m pip install 'ml_dtypes==0.2.0' 'numpy==1.24.3' 'scipy==1.10.1'",
        # jaxlib 0.4.11 is built against cudnn 8.6 ('cuda11.cudnn86'); a mismatched
        # nvidia-cudnn-cu11 -> 'DNN library initialization failed'. Pin 8.6.
        "python -m pip install 'nvidia-cudnn-cu11==8.6.0.163'",
        # new tensorstore needs ml_dtypes>=0.5 (float8_e3m4) which conflicts with
        # jax 0.4.11's ml_dtypes 0.2.0 -> pin old tensorstore for orbax import.
        "python -m pip install 'tensorstore==0.1.45'",
    )
    .env({
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "HF_HOME": "/cache/hf",
        # jaxrl_m is a top-level pkg in the fork; the editable install didn't
        # register on sys.path, so add the repo root directly.
        "PYTHONPATH": "/root/bridge_data_v2",
    })
    .add_local_python_source("gcbc_train_config", "diffusion_gc_policy")
)

cache = modal.Volume.from_name("simpler-gc-cache", create_if_missing=True)


def _download_ckpt():
    """Snapshot the bridge GC-diffusion checkpoint dir to /cache; return local path."""
    import os
    from huggingface_hub import snapshot_download
    # `checkpoint` and `serialized_policy_ckpt` are LFS files; the default
    # symlink mode left dangling symlinks (blobs not materialized) ->
    # local_dir_use_symlinks=False downloads real file content.
    snapshot_download(
        repo_id=CKPT_REPO,
        allow_patterns=[CKPT_SUBDIR + "/*"],
        local_dir="/cache/ghil_ckpts3",
        local_dir_use_symlinks=False,
    )
    path = os.path.join("/cache/ghil_ckpts3", CKPT_SUBDIR)
    assert os.path.isdir(path), f"checkpoint not found at {path}"
    return path


def load_gc_policy():
    """Build the GCPolicy for bridge gcdiffusion auggoaldiff."""
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # leave GPU to jax
    import os
    from gcbc_train_config import get_config
    from diffusion_gc_policy import GCPolicy
    ckpt = _download_ckpt()
    # ckpt dir holds `serialized_policy_ckpt` (flax) + `checkpoint` (orbax) files;
    # GCPolicy.__init__ prefers the flax serialized_policy_ckpt.
    agent_config, env_name, agent_type, _ = get_config(CONFIG_STRING)
    policy = GCPolicy(agent_config, env_name, agent_type, ckpt,
                      normalize_actions=False, use_temporal_ensembling=True,
                      text_processor=None)
    return policy


@app.function(image=gc_image, gpu="L4", volumes={"/cache": cache}, timeout=1800)
def smoke():
    import numpy as np
    import jax
    out = {"jax_devices": [str(d) for d in jax.devices()]}
    policy = load_gc_policy()
    out["loaded"] = True
    # both obs and goal are (H,W,C) [3D]; predict_action adds the obs newaxis and
    # sample_actions adds the batch dim (obs->5D B,T,H,W,C ; goal->4D B,H,W,C).
    obs = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    goal = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    a = policy.predict_action(obs, goal)
    out["action_shape"] = list(np.asarray(a).shape)
    out["action_sample"] = [round(float(x), 4) for x in np.asarray(a)]
    return out


@app.function(image=gc_image, gpu="L4", volumes={"/cache": cache}, timeout=1800)
def inspect_ckpt():
    """Dump the raw checkpoint pytree top-level structure vs the freshly-created
    agent structure, to fix the orbax restore KeyError('state')."""
    import jax, numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import orbax.checkpoint
    from gcbc_train_config import get_config
    from jaxrl_m.vision import encoders
    from jaxrl_m.agents import agents
    import os
    ckpt = _download_ckpt()
    out = {}
    # list the on-disk orbax layout (all levels, dir names only)
    layout = []
    for root, dirs, files in os.walk(ckpt):
        rel = os.path.relpath(root, ckpt)
        layout.append(rel + " :: " + ",".join(sorted(dirs)[:30]) +
                      " | files: " + ",".join(sorted(files)[:8]))
    out["ckpt_layout"] = layout[:60]
    # also try raw restore from the inner checkpoint dir
    inner = os.path.join(ckpt, "checkpoint")
    try:
        raw = orbax.checkpoint.PyTreeCheckpointer().restore(inner)
        out["raw_inner_keys"] = list(raw.keys())[:20] if isinstance(raw, dict) else str(type(raw))
        if isinstance(raw, dict):
            for k in list(raw.keys())[:3]:
                v = raw[k]
                out[f"raw_inner[{k}]_keys"] = (list(v.keys())[:15] if isinstance(v, dict) else str(type(v)))
    except Exception as e:
        out["raw_inner_err"] = repr(e)[:300]

    def topkeys(x, depth=0, prefix=""):
        r = []
        if isinstance(x, dict):
            for k in list(x.keys())[:12]:
                r.append(prefix + str(k))
                if depth < 1:
                    r += topkeys(x[k], depth + 1, prefix + str(k) + "/")
        return r
    # build the agent the same way diffusion_gc_policy does
    agent_config, env_name, agent_type, _ = get_config(CONFIG_STRING)
    example_obs = {"image": np.zeros((1, 200, 200, 3), np.uint8)}
    example_goal = {"image": np.zeros((1, 200, 200, 3), np.uint8),
                    "language": np.zeros((1, 512), np.float32)}
    example_batch = {"observations": example_obs, "goals": example_goal,
                     "actions": np.zeros((1, 4, 7), np.float32)}
    enc = encoders[agent_config.encoder](**agent_config.encoder_kwargs)
    agent = agents[agent_config.agent].create(
        rng=jax.random.PRNGKey(42), observations=example_batch["observations"],
        goals=example_batch["goals"], actions=example_batch["actions"],
        encoder_def=enc, **agent_config.agent_kwargs)
    out["agent_keys"] = topkeys(agent if isinstance(agent, dict)
                                else getattr(agent, "__dict__", {}))[:40]
    out["agent_type"] = str(type(agent))
    return out


@app.local_entrypoint()
def inspect():
    import json
    print(json.dumps(inspect_ckpt.remote(), indent=2, default=str))


@app.local_entrypoint()
def smoke_main():
    import json
    print(json.dumps(smoke.remote(), indent=2, default=str))
