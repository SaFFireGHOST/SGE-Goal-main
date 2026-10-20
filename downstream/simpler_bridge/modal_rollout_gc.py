"""
SIMPLER-Bridge rollout harness with the SuSIE/GHIL-Glue BridgeData-V2
goal-conditioned DIFFUSION policy (replaces Octo — Octo gave meaningless ~0 SR on
generated goals). Combined image: env stack (sapien 2.2.2 + SimplerEnv + lavapipe
render + v1-table platform fix + numpy-control/pinocchio patches) PLUS the GC
policy stack (jaxrl_m fork + jax-0.4.11-cuda11 + flax-0.7 + the version pins).

The GC policy (jaxrl_m gc_ddpm_bc, bridge gcdiffusion auggoaldiff) is robust to
GENERATED goal images, so editor goals drive it meaningfully.

  validate_gc : clean-success-frame test (policy can reach a real success state)
  run_method  : per-method goal-conditioned rollouts -> SR (goals already on vol)
"""
import modal
import os

app = modal.App("simpler-rollout-gc")

CUDA_TAG = "12.8.1-devel-ubuntu24.04"
JAXRL_FORK = "https://github.com/kyle-hatch-tri/bridge_data_v2_ghil-glue.git"
CKPT_REPO = "kyle-hatch-tri/ghil-glue-checkpoints"
CKPT_SUBDIR = "susie_low_level/bridge/gcdiffusion/auggoaldiff/seed_0/20240602_011058/checkpoint_150000"
CONFIG_STRING = "bridge_gcdiffusion_noactnorm-sagemaker-auggoaldiff"

# --- combined image: GC-policy stack first (it pins old jax/numpy), then the
#     env stack (sapien + SimplerEnv + lavapipe). numpy re-pinned 1.24.3 last. ---
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    # ---- env apt (kisak lavapipe + GL/Vulkan/X libs) ----
    .apt_install("software-properties-common", "gpg-agent", "ca-certificates")
    .run_commands("add-apt-repository -y ppa:kisak/kisak-mesa", "apt-get update")
    .apt_install(
        "git", "wget", "bzip2", "libglib2.0-0", "libgl1", "libglx-mesa0",
        "libvulkan1", "vulkan-tools", "libvulkan-dev",
        "libegl1", "libgles2", "libglvnd-dev", "libosmesa6-dev", "ffmpeg",
        "build-essential", "g++", "cmake",
        "mesa-vulkan-drivers", "libgl1-mesa-dri",
        "libx11-6", "libxext6", "libxrender1", "libxi6", "libxfixes3",
        "libxxf86vm1", "libxrandr2", "libxinerama1", "libxcursor1",
        "libxcb1", "libxau6", "libxdmcp6", "libsm6", "libice6", "libxkbcommon0",
    )
    .env({"CC": "gcc", "CXX": "g++"})
    .run_commands("python -m pip install --upgrade pip 'setuptools<70' wheel")
    # ---- env stack: sapien + SimplerEnv + standalone pinocchio ----
    .pip_install("numpy==1.24.3", "scipy==1.10.1", "pillow",
                 "opencv-python-headless", "mediapy")
    .pip_install("sapien==2.2.2", "ruckig", "pin")
    .run_commands(
        "git clone https://github.com/simpler-env/SimplerEnv /root/SimplerEnv --recurse-submodules",
        "cd /root/SimplerEnv/ManiSkill2_real2sim && python -m pip install -e .",
        "cd /root/SimplerEnv && python -m pip install -e . --no-deps",
    )
    # ---- GC policy stack: jaxrl_m fork + pinned jax-ecosystem ----
    .run_commands("git clone " + JAXRL_FORK + " /root/bridge_data_v2")
    .run_commands("cd /root/bridge_data_v2 && python -m pip install -e . --no-deps")
    .pip_install(
        "ml_collections>=0.1.0", "einops==0.6.1", "distrax==0.1.2",
        "flax==0.7.0", "optax==0.1.5", "chex==0.1.7", "orbax-checkpoint==0.2.3",
        "tensorflow==2.13.0", "tensorflow_probability==0.21.0", "tensorflow_hub==0.15.0",
        "transformers==4.34.1", "huggingface_hub", "hf_transfer",
        "func_timeout", "plotly", "imageio", "tqdm", "wandb",
    )
    .run_commands(
        "python -m pip install 'jax[cuda11_pip]==0.4.11' "
        "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    )
    # cheap final layer of version reconciliations (keeps heavy layers cached)
    .pip_install("ml_dtypes==0.2.0", "scipy==1.10.1", "numpy==1.24.3",
                 "tensorstore==0.1.45", "nvidia-cudnn-cu11==8.6.0.163")
    .env({
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json",
        "XDG_RUNTIME_DIR": "/tmp",
        "DISPLAY": "",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "HF_HOME": "/cache/hf",
        "PYTHONPATH": "/root/bridge_data_v2",
        # opt-in to the finetuned ckpt (for ablation/eval); baked from launch env
        # so the seed-table rollouts (no var) keep using the pretrained policy.
        "SGE_FINETUNED": os.environ.get("SGE_FINETUNED", ""),
    })
    .add_local_python_source("pinocchio_ik_backend", "gcbc_train_config",
                             "diffusion_gc_policy")
)

CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
PREPKG = {
    "widowx_spoon_on_towel": dict(
        env_name="PutSpoonOnTableClothInScene-v0", robot="widowx",
        scene_name="bridge_table_1_v1", overlay="real_inpainting/bridge_real_eval_1.png",
        robot_init_xy=[0.147, 0.028], max_steps=60, init_z=0.95),
    "widowx_carrot_on_plate": dict(
        env_name="PutCarrotOnPlateInScene-v0", robot="widowx",
        scene_name="bridge_table_1_v1", overlay="real_inpainting/bridge_real_eval_1.png",
        robot_init_xy=[0.147, 0.028], max_steps=60, init_z=0.95),
    "widowx_stack_cube": dict(
        env_name="StackGreenCubeOnYellowCubeBakedTexInScene-v0", robot="widowx",
        scene_name="bridge_table_1_v1", overlay="real_inpainting/bridge_real_eval_1.png",
        robot_init_xy=[0.147, 0.028], max_steps=60, init_z=0.95),
    "widowx_put_eggplant_in_basket": dict(
        env_name="PutEggplantInBasketScene-v0", robot="widowx_sink_camera_setup",
        scene_name="bridge_table_1_v2", overlay="real_inpainting/bridge_sink.png",
        robot_init_xy=[0.127, 0.06], max_steps=120, init_z=None),
}
BRIDGE_TASKS = list(PREPKG.keys())
N_EPISODES = 24

io_image = modal.Image.debian_slim(python_version="3.10").pip_install("numpy")
vol = modal.Volume.from_name("simpler-eval-data", create_if_missing=True)
cache = modal.Volume.from_name("simpler-gc-cache", create_if_missing=True)


# ===========================================================================
# Env helpers (identical to the validated Octo harness)
# ===========================================================================
def _force_offscreen():
    import sapien.core as sapien
    _orig = sapien.SapienRenderer

    def _off(*a, **k):
        k.setdefault("offscreen_only", True)
        return _orig(*a, **k)

    sapien.SapienRenderer = _off


def _install_platform(half_xy=0.6, bottom_z=-0.2, center_xy=(-0.15, 0.0)):
    import sapien.core as sapien
    import mani_skill2_real2sim.envs  # noqa: F401
    from mani_skill2_real2sim.utils.registration import REGISTERED_ENVS
    spec = REGISTERED_ENVS["PutSpoonOnTableClothInScene-v0"]
    leaf = getattr(spec, "cls", None) or spec.env_class
    defcls = None
    for c in leaf.__mro__:
        if "_load_actors" in c.__dict__:
            defcls = c
            break
    if defcls is None or getattr(defcls, "_platform_installed", False):
        return
    orig = defcls.__dict__["_load_actors"]

    def patched(self, *a, **k):
        orig(self, *a, **k)
        if "v1" not in (getattr(self, "scene_name", "") or ""):
            return
        top_z = float(getattr(self, "scene_table_height", 0.87))
        half_z = (top_z - bottom_z) / 2.0
        b = self._scene.create_actor_builder()
        b.add_box_collision(half_size=[half_xy, half_xy, half_z])
        plat = b.build_static(name="table_platform")
        plat.set_pose(sapien.Pose([center_xy[0], center_xy[1], bottom_z + half_z]))

    defcls._load_actors = patched
    defcls._platform_installed = True


def apply_env_patches():
    from pinocchio_ik_backend import patch_pd_ee_controllers
    _force_offscreen()
    patch_pd_ee_controllers()
    _install_platform()


def make_env(task):
    from simpler_env.utils.env.env_builder import build_maniskill2_env
    from mani_skill2_real2sim import ASSET_DIR
    cfg = PREPKG[task]
    env = build_maniskill2_env(
        cfg["env_name"], obs_mode="rgbd", robot=cfg["robot"],
        sim_freq=500, control_mode=CONTROL_MODE, control_freq=5,
        max_episode_steps=cfg["max_steps"], scene_name=cfg["scene_name"],
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=str(ASSET_DIR / cfg["overlay"]),
    )
    return env, cfg


def reset_env(env, cfg, episode_id):
    import numpy as np
    obj_opts = {"episode_id": int(episode_id)}
    if cfg["init_z"] is not None:
        obj_opts["init_z"] = float(cfg["init_z"])
    return env.reset(options={
        "robot_init_options": {"init_xy": np.array(cfg["robot_init_xy"]),
                               "init_rot_quat": [0, 0, 0, 1]},
        "obj_init_options": obj_opts,
    })


def _resolve_goal(method, task, ep):
    import os
    base = f"/data/goals/{method}/{task}/ep{ep:03d}"
    for name in ("edited_image.jpg", "goal.jpg", "edited_image.png", "goal.png"):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    return None


# ===========================================================================
# GC policy wrapper: load once; per-episode reset; obs/goal -> SIMPLER action.
# ===========================================================================
def _download_ckpt():
    import os
    # FINAL-GOAL FINETUNED policy is the default now (our trained variant; same
    # architecture + action normalization -> drop-in). The tar was transferred to
    # this account's simpler-gc-cache. Set SGE_PRETRAINED=1 to use the original.
    # Default = the pretrained GHIL-Glue policy (the chosen policy for the primary
    # table; finetuning did not beat it). Set SGE_FINETUNED=1 to use the
    # final-goal finetuned checkpoint instead (the ablation; tar must be present
    # on this account's simpler-gc-cache).
    if os.environ.get("SGE_FINETUNED"):
        import tarfile, glob, shutil
        # Extract to a CONTAINER-LOCAL dir (not the shared /cache volume): the
        # rollout fans out to several containers that would otherwise race to
        # untar into the same volume path -> corrupt/incomplete checkpoint. Also
        # re-extract fresh each time so a warm container can't serve a stale
        # different-step checkpoint from a previous eval.
        local = "/root/finetuned_ckpt"
        if os.path.isdir(local):
            shutil.rmtree(local, ignore_errors=True)
        os.makedirs(local, exist_ok=True)
        with tarfile.open("/cache/finetuned_ft.tar") as t:
            t.extractall(local)
        found = [d for d in glob.glob(f"{local}/checkpoint_*") if os.path.isdir(d)]
        assert found, "finetuned ckpt not found after untar"
        return found[0]
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=CKPT_REPO, allow_patterns=[CKPT_SUBDIR + "/*"],
                      local_dir="/cache/ghil_ckpts3", local_dir_use_symlinks=False)
    path = os.path.join("/cache/ghil_ckpts3", CKPT_SUBDIR)
    assert os.path.isdir(path), f"checkpoint not found at {path}"
    return path


# bridge_data_v2 (GHIL-Glue) action stats — the policy outputs RAW actions in
# this scale (config is "noactnorm"). SIMPLER's widowx controller has
# normalize_action=False -> it wants delta pose directly in the Open-X bridge
# meter/radian scale (what octo emits). So re-normalize per-axis:
#   a_sim[i] = (a_v2[i]-MEAN_v2[i])/STD_v2[i] * OPENX_STD[i] + OPENX_MEAN[i]
V2_MEAN = [2.9842544e-04, -2.6099570e-04, -1.5863389e-04,
           5.8916201e-05, -4.4560504e-05, 8.2349771e-04, 9.4075650e-02]
V2_STD = [0.27278143, 0.23548537, 0.2196189, 0.15881406, 0.17537235, 0.27875036, 1.0049515]
# octo bridge_dataset action statistics (rail-berkeley/octo-*-1.5)
OPENX_MEAN = [0.00022, 0.00013, -0.00017, -0.00016, -0.00025, 0.00025, 0.58795]
OPENX_STD = [0.00963, 0.0135, 0.01251, 0.02815, 0.03028, 0.07586, 0.48772]


class GCWrapper:
    """Wrap jaxrl_m GCPolicy for SIMPLER-Bridge rollouts."""

    def __init__(self, flip_gripper: bool = False):
        self.flip_gripper = flip_gripper
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")  # GPU is jax's
        import cv2
        from gcbc_train_config import get_config
        from diffusion_gc_policy import GCPolicy
        self._cv2 = cv2
        ckpt = _download_ckpt()
        agent_config, env_name, agent_type, _ = get_config(CONFIG_STRING)
        self.policy = GCPolicy(agent_config, env_name, agent_type, ckpt,
                               normalize_actions=False, use_temporal_ensembling=True,
                               text_processor=None)
        self.goal200 = None

    def _resize(self, img):
        import numpy as np
        return self._cv2.resize(np.asarray(img), (200, 200),
                                interpolation=self._cv2.INTER_AREA).astype(np.uint8)

    def reset_goal(self, goal_image):
        self.goal200 = self._resize(goal_image)        # (200,200,3) uint8
        self.policy.reset()

    def _to_sim_action(self, a):
        """a: raw 7-dim GHIL-Glue action -> SIMPLER widowx action."""
        import numpy as np
        from transforms3d.euler import euler2axangle
        a = np.asarray(a, dtype=np.float64)
        # per-axis re-normalize dims 0..5 (xyz, rpy) from v2-scale to openx meters/rad
        conv = np.array([(a[i] - V2_MEAN[i]) / V2_STD[i] * OPENX_STD[i] + OPENX_MEAN[i]
                         for i in range(6)])
        world_vector = conv[:3]
        ax, ang = euler2axangle(float(conv[3]), float(conv[4]), float(conv[5]))
        rot_axangle = ax * ang
        g = float(a[6])                                 # already sign-forced to {-1,1}
        g = -g if self.flip_gripper else g              # env: +1=open, -1=close
        return np.concatenate([world_vector, rot_axangle, [g]]).astype(np.float64)

    def step(self, obs_image):
        """obs_image: HxWx3 uint8 SIMPLER frame. Returns SIMPLER 7-dim action."""
        obs200 = self._resize(obs_image)               # (200,200,3)
        a = self.policy.predict_action(obs200, self.goal200)  # 7-dim
        return self._to_sim_action(a)


@app.function(image=image, gpu="L4", cpu=8.0, volumes={"/data": vol, "/cache": cache},
              timeout=3600)
def debug_rollout(method: str, task: str, ep: int = 0, flip_gripper: bool = False):
    """Run ONE episode, save a frame montage + goal + action trace to /data/debug/."""
    import os, numpy as np
    from PIL import Image
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    apply_env_patches()
    gc = GCWrapper(flip_gripper=flip_gripper)
    env, cfg = make_env(task)
    gp = _resolve_goal(method, task, ep)
    goal = np.asarray(Image.open(gp).convert("RGB"))
    ret = reset_env(env, cfg, ep)
    obs = ret[0] if isinstance(ret, tuple) else ret
    image_frame = get_image_from_maniskill2_obs_dict(env, obs)
    gc.reset_goal(goal)
    frames, raws, sims, ever = [image_frame], [], [], False
    truncated, info, t = False, {}, 0
    while not truncated:
        a = np.asarray(gc.policy.predict_action(gc._resize(image_frame), gc.goal200))
        action = gc._to_sim_action(a)
        raws.append([round(float(x), 4) for x in a])
        sims.append([round(float(x), 4) for x in action])
        obs, _r, done, truncated, info = env.step(action)
        if bool(info.get("success", done)):
            ever = True
        image_frame = get_image_from_maniskill2_obs_dict(env, obs)
        frames.append(image_frame)
        t += 1
    idxs = np.linspace(0, len(frames) - 1, 8).astype(int)
    sel = [frames[i] for i in idxs]
    h, w = sel[0].shape[:2]
    montage = np.concatenate([np.asarray(Image.fromarray(f).resize((w, h))) for f in sel], axis=1)
    goal_r = np.asarray(Image.fromarray(goal).resize((w, h)))
    montage = np.concatenate([montage, goal_r], axis=1)  # last panel = GOAL
    od = "/data/debug"; os.makedirs(od, exist_ok=True)
    tag = f"{method}__{task}__ep{ep:03d}{'__flip' if flip_gripper else ''}"
    Image.fromarray(montage).save(f"{od}/{tag}.png")
    vol.commit()
    swv = np.array([s[:3] for s in sims]); grips = [s[6] for s in sims]
    print(f"DEBUG {tag}: steps={t} ever_success={ever} success={bool(info.get('success', False))}")
    print(f"  SIM |world_vector| mean per-axis (m): {np.abs(swv).mean(0).round(4).tolist()}  total_path={np.abs(swv).sum(0).round(3).tolist()}")
    print(f"  gripper: n_open(+1)={sum(1 for g in grips if g>0)} n_close(-1)={sum(1 for g in grips if g<0)}")
    print(f"  raw action[0]={raws[0]}  sim action[0]={sims[0]}")
    print(f"  raw action[mid]={raws[len(raws)//2]}  sim action[mid]={sims[len(sims)//2]}")
    return tag


@app.function(image=io_image, volumes={"/data": vol}, timeout=300)
def fetch_debug(tag: str):
    with open(f"/data/debug/{tag}.png", "rb") as f:
        return f.read()


@app.local_entrypoint()
def debug(method: str = "flux-kontext-dev-bft-i2", task: str = "widowx_spoon_on_towel",
          ep: int = 0, flip: bool = False):
    tag = debug_rollout.remote(method, task, ep, flip)
    data = fetch_debug.remote(tag)
    out = f"/tmp/gc_debug_{tag}.png"
    with open(out, "wb") as f:
        f.write(data)
    print(f"MONTAGE_SAVED {out}")


def rollout_one_gc(gc, env, cfg, episode_id, goal_image):
    import numpy as np
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    ret = reset_env(env, cfg, episode_id)
    obs = ret[0] if isinstance(ret, tuple) else ret
    image = get_image_from_maniskill2_obs_dict(env, obs)
    gc.reset_goal(goal_image)
    truncated, ever, info, t = False, False, {}, 0
    while not truncated:
        action = gc.step(image)
        obs, _r, done, truncated, info = env.step(action)
        if bool(info.get("success", done)):
            ever = True
            break  # early-stop on first success (goal-reaching protocol)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        t += 1
    stats = info.get("episode_stats", {}) or {}
    clean = {}
    for k, v in (stats.items() if hasattr(stats, "items") else []):
        try:
            clean[str(k)] = v.item() if hasattr(v, "item") else v
        except Exception:
            clean[str(k)] = str(v)
    # with early-stop, success == ever_success (episode terminates at first success)
    return {"success": bool(ever), "ever_success": bool(ever),
            "n_steps": int(t), "episode_stats": clean}


# ===========================================================================
# Validation: clean-success-frame test (does the GC policy reach a REAL success
# state captured from a successful LANGUAGE-Octo... no octo here, so use the
# stock-language path is unavailable; instead condition on the GROUND-TRUTH
# editor-free check: run a few eggplant eps with the goal == a forward-rendered
# success — simplest is to just report SR on a chosen method's goals).
# We validate by running on flux2-klein-ours-klein (mask-guided, best goals).
# ===========================================================================
@app.function(image=image, gpu="L4", cpu=8.0, volumes={"/data": vol, "/cache": cache},
              timeout=7200)
def run_chunk(method: str, task: str, ep_ids: list):
    import numpy as np
    from PIL import Image
    apply_env_patches()
    gc = GCWrapper()
    env, cfg = make_env(task)
    results = []
    for ep in ep_ids:
        gp = _resolve_goal(method, task, ep)
        if gp is None:
            results.append({"ep": ep, "success": None, "error": "no_goal_image"})
            continue
        goal = np.asarray(Image.open(gp).convert("RGB"))
        r = rollout_one_gc(gc, env, cfg, ep, goal)
        r["ep"] = ep
        results.append(r)
    return results


@app.function(image=io_image, volumes={"/data": vol}, timeout=300)
def write_results(method: str, summary: dict):
    import os, json
    od = f"/data/results_gc/{method}"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/summary.json", "w") as f:
        json.dump({"method": method, "policy": "susie_gc_ddpm", "summary": summary},
                  f, indent=2, default=str)
    vol.commit()
    return od


@app.local_entrypoint()
def run_method(method: str, n_eps: int = N_EPISODES, chunk: int = 6, tasks: str = "all"):
    import json
    task_list = BRIDGE_TASKS if tasks == "all" else tasks.split(",")
    args = []
    for t in task_list:
        for s in range(0, n_eps, chunk):
            args.append((method, t, list(range(s, min(s + chunk, n_eps)))))
    chunk_results = list(run_chunk.starmap(args))
    per_task = {t: [] for t in task_list}
    for (m, t, ep_ids), res in zip(args, chunk_results):
        per_task[t].extend(res)
    summary = {}
    print(f"\n=== {method} (SuSIE GC) SR SUMMARY ===")
    srs = []
    for t in task_list:
        results = sorted(per_task[t], key=lambda r: r["ep"])
        valid = [r for r in results if r.get("success") is not None]
        sr = float(sum(r["success"] for r in valid) / len(valid)) if valid else None
        ever = float(sum(r["ever_success"] for r in valid) / len(valid)) if valid else None
        summary[t] = {"SR": sr, "ever_SR": ever, "n_valid": len(valid),
                      "n_eps": len(results), "results": results}
        srs.append(sr if sr is not None else 0.0)
        print(f"  {t:35s} SR={sr}  ever_SR={ever}  ({len(valid)}/{len(results)})")
    overall = sum(srs) / len(srs) if srs else None
    print(f"  {'MEAN':35s} SR={overall}")
    print("RESULT_JSON " + json.dumps({t: {"SR": summary[t]["SR"]} for t in task_list}))
    try:
        write_results.remote(method, summary)
    except Exception as e:
        print(f"[warn] write_results failed: {e}")
