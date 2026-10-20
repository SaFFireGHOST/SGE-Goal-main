"""
Shared SIMPLER-Bridge goal-conditioned rollout harness.

The success-rate experiment compares many goal-image generators (FLUX Kontext /
FLUX2 Klein / Qwen / Gemini / GPT-Image / SuSIE / CLOVER ...). The LOW-LEVEL
policy is held FIXED across all of them: Octo in goal-image conditioning mode.
So a "method" only differs in the goal image it produces for each (task,
episode); this file runs the shared rollout:

    SIMPLER initial obs  --[METHOD]-->  goal image
    goal image --> Octo goal-conditioned policy --> rollout in SIMPLER
    SIMPLER `done` (terminated) == task success.

This module owns the env (sapien + SimplerEnv + the v1-table platform fix + the
standalone-pinocchio / numpy-control patches) AND the Octo policy (jax). A rollout
runs env.step + model.step in the same process, so both live in one image.

Smoke test (no env): octo loads + goal-conditioned sample_actions works
    modal run simpler-eval/modal_rollout.py::smoke
"""
import modal

app = modal.App("simpler-rollout")

CUDA_TAG = "12.8.1-devel-ubuntu24.04"
OCTO_COMMIT = "653c54acde686fde619855f2eac0dd6edad7116b"  # octo-1.0

# --- combined image: the proven env stack (24.04 + kisak lavapipe + sapien +
#     SimplerEnv) PLUS the Octo policy stack (jax-cuda + tf + octo). numpy is
#     re-pinned to 1.24.4 last so sapien stays happy. ---
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install("software-properties-common", "gpg-agent", "ca-certificates")
    .run_commands(
        "add-apt-repository -y ppa:kisak/kisak-mesa",
        "apt-get update",
    )
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
    # env stack
    .pip_install("numpy==1.24.4", "scipy", "pillow", "opencv-python-headless", "mediapy")
    .pip_install("sapien==2.2.2", "ruckig", "pin")
    .run_commands(
        "git clone https://github.com/simpler-env/SimplerEnv /root/SimplerEnv --recurse-submodules",
        "cd /root/SimplerEnv/ManiSkill2_real2sim && python -m pip install -e .",
        "cd /root/SimplerEnv && python -m pip install -e . --no-deps",
    )
    # policy stack: octo-1.0 + jax(cuda12) + tf. octo's setup.py pins flax/distrax/
    # optax/einops/tf-probability; we add jax-cuda after so it has GPU support.
    .run_commands(
        "git clone https://github.com/octo-models/octo /root/octo",
        f"cd /root/octo && git checkout {OCTO_COMMIT}",
        # octo keeps its real deps in requirements.txt (flax/distrax/optax/orbax/
        # ml_collections/...), NOT setup.py — install both.
        "cd /root/octo && python -m pip install -e . && python -m pip install -r requirements.txt",
        "python -m pip install 'tensorflow==2.15.1'",
        "python -m pip install git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972",
        # pin GPU jax LAST (octo's requirements may install a CPU/other jax)
        "python -m pip install 'jax[cuda12_pip]==0.4.20' "
        "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        # sapien needs numpy 1.24.4; re-pin LAST so octo/tf didn't bump it
        "python -m pip install 'numpy==1.24.4'",
    )
    # separate cheap layer (keeps the heavy octo/jax layer cached):
    #  - octo's deps pull scipy>=1.13 which removed scipy.linalg.tril/triu that
    #    jax/tfp import -> pin scipy<1.13 (1.10.1 has tril, ok with numpy 1.24.4).
    #  - a too-new transformers hides FlaxAutoModel from octo (its flax-available
    #    check fails against jax 0.4.20 / flax 0.7.5) -> pin the octo-1.0 era 4.34.1.
    #  - octo deps bumped cuDNN to 9.x (libcudnn.so.9) but jaxlib 0.4.20+cuda12 links
    #    libcudnn.so.8 (needs >=8.9.4) -> without this jax silently falls back to CPU.
    .pip_install("scipy==1.10.1", "numpy==1.24.4", "transformers==4.34.1",
                 "nvidia-cudnn-cu12==8.9.7.29")
    .env({
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json",
        "XDG_RUNTIME_DIR": "/tmp",
        "DISPLAY": "",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "HF_HOME": "/cache/hf",
    })
    .add_local_python_source("pinocchio_ik_backend")
)

CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"

# Per-task SIMPLER-Bridge config (matches the official visual-matching eval).
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

# light image for volume I/O (staging obs out, goals in) — avoids loading the
# heavy octo image just to move files. (`modal volume get` recursive download is
# buggy in this CLI version, so we tar through a function instead.)
io_image = modal.Image.debian_slim(python_version="3.10").pip_install("numpy")

vol = modal.Volume.from_name("simpler-eval-data", create_if_missing=True)
cache = modal.Volume.from_name("simpler-octo-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(image=image, gpu="L4", volumes={"/cache": cache},
              secrets=[hf_secret], timeout=1800)
def smoke(model_type: str = "octo-small"):
    """Load Octo, build a GOAL-IMAGE task, and sample actions on a dummy obs.
    De-risks the jax/octo setup independent of the SIMPLER env."""
    import os
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import numpy as np
    import jax
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # leave the GPU to jax
    from octo.model.octo_model import OctoModel

    out = {"jax_devices": [str(d) for d in jax.devices()]}
    model = OctoModel.load_pretrained(f"hf://rail-berkeley/{model_type}")
    out["loaded"] = model_type
    out["dataset_ids"] = list(model.dataset_statistics.keys())

    # GOAL-IMAGE task (256x256 for octo image_primary)
    goal = (np.random.rand(1, 256, 256, 3) * 255).astype(np.uint8)
    task_goal = model.create_tasks(goals={"image_primary": goal})
    out["goal_task_keys"] = list(task_goal.keys())

    # dummy obs: horizon=2 history of 256x256 images
    obs = {
        "image_primary": (np.random.rand(1, 2, 256, 256, 3) * 255).astype(np.uint8),
        "pad_mask": np.ones((1, 2), dtype=np.float64),
    }
    rng = jax.random.PRNGKey(0)
    actions = model.sample_actions(obs, task_goal, rng=rng)
    out["action_shape"] = list(np.asarray(actions).shape)

    # also confirm TEXT task still works (sanity)
    task_text = model.create_tasks(texts=["put the spoon on the towel"])
    out["text_task_keys"] = list(task_text.keys())
    cache.commit()
    return out


@app.local_entrypoint()
def smoke_main(model_type: str = "octo-small"):
    import json
    print(json.dumps(smoke.remote(model_type=model_type), indent=2, default=str))


# ===========================================================================
# Shared env construction (used by BOTH the Phase-A initial-obs dump and the
# rollout, so the goal image is generated from EXACTLY the state the rollout
# starts from). Applies: offscreen render, control/pinocchio patches, v1 table
# platform fix.
# ===========================================================================
def _force_offscreen():
    import sapien.core as sapien
    _orig = sapien.SapienRenderer

    def _off(*a, **k):
        k.setdefault("offscreen_only", True)
        return _orig(*a, **k)

    sapien.SapienRenderer = _off


def _install_platform(half_xy=0.6, bottom_z=-0.2, center_xy=(-0.15, 0.0)):
    """Invisible THICK static collision box at scene_table_height for v1 scenes
    (the v1 arena nonconvex table-top is mis-cooked in this sapien build so
    objects fall through). See memory simpler-on-modal-setup. Idempotent."""
    import sapien.core as sapien
    import mani_skill2_real2sim.envs  # noqa: F401  complete registration first
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
    """All monkeypatches that must run BEFORE any env is built."""
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


# ===========================================================================
# Phase A: dump deterministic initial obs for each (task, episode) in the
# editor's expected layout: /data/simpler_obs/<task>/ep<NNN>/{original_image.jpg,
# edit_instruction.txt}. Uses the SAME env path as the rollout.
# ===========================================================================
@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def dump_initial_obs(task: str):
    import os, json, hashlib
    import numpy as np
    from PIL import Image
    apply_env_patches()
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    env, cfg = make_env(task)
    hashes = []
    for ep in range(N_EPISODES):
        ret = reset_env(env, cfg, ep)
        obs = ret[0] if isinstance(ret, tuple) else ret
        u = env.unwrapped
        instr = u.get_language_instruction()
        img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
        d = f"/data/simpler_obs/{task}/ep{ep:03d}"
        os.makedirs(d, exist_ok=True)
        Image.fromarray(img).save(f"{d}/original_image.jpg", quality=95)
        with open(f"{d}/edit_instruction.txt", "w") as f:
            f.write(instr)
        with open(f"{d}/meta.json", "w") as f:
            json.dump({"task": task, "episode_id": ep, "instruction": instr,
                       "img_shape": list(img.shape)}, f)
        hashes.append(hashlib.md5(img.tobytes()).hexdigest()[:8])
    vol.commit()
    return {"task": task, "n_episodes": N_EPISODES,
            "n_unique_layouts": len(set(hashes)),
            "instruction": instr}


@app.local_entrypoint()
def dumpobs():
    import json
    res = list(dump_initial_obs.map(BRIDGE_TASKS))
    print(json.dumps(res, indent=2))


# ===========================================================================
# Octo goal-image-conditioned policy (the ONLY change vs stock OctoInference is
# conditioning on a goal IMAGE instead of language text).
# ===========================================================================
def build_goal_policy(model_type="octo-base", init_rng=0):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # GPU is jax's
    from simpler_env.policies.octo.octo_model import OctoInference

    class OctoGoalInference(OctoInference):
        def reset_goal(self, goal_image):
            import numpy as np
            goal = self._resize_image(np.asarray(goal_image).astype(np.uint8))
            self.task = self.model.create_tasks(goals={"image_primary": goal[None]})
            self.task_description = "<goal-image>"
            self.image_history.clear()
            if self.action_ensemble:
                self.action_ensembler.reset()
            self.num_image_history = 0
            self.sticky_action_is_on = False
            self.gripper_action_repeat = 0
            self.sticky_gripper_action = 0.0
            self.previous_gripper_action = None

    return OctoGoalInference(model_type=model_type, policy_setup="widowx_bridge",
                             init_rng=init_rng)


def rollout_one(model, env, cfg, episode_id, goal_image):
    """Run a single goal-conditioned episode. Returns (success, n_steps, info)."""
    import numpy as np
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    ret = reset_env(env, cfg, episode_id)
    obs = ret[0] if isinstance(ret, tuple) else ret
    image = get_image_from_maniskill2_obs_dict(env, obs)
    model.reset_goal(goal_image)
    predicted_terminated, truncated = False, False
    ever_success = False
    info = {}
    t = 0
    while not (predicted_terminated or truncated):
        raw_action, action = model.step(image, task_description=None)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]))
        if bool(info.get("success", done)):
            ever_success = True
        image = get_image_from_maniskill2_obs_dict(env, obs)
        t += 1
    final_success = bool(info.get("success", False))
    # SIMPLER's episode_stats hold numpy scalars; sanitize to pure-python so the
    # results serialize cleanly (the light io_image used by write_results lacks
    # numpy, and json.dump can't handle np.bool_/np.float64).
    stats = info.get("episode_stats", {}) or {}
    clean = {}
    for k, v in (stats.items() if hasattr(stats, "items") else []):
        try:
            clean[str(k)] = v.item() if hasattr(v, "item") else v
        except Exception:
            clean[str(k)] = str(v)
    # SIMPLER's official metric is terminal success; we also track ever_success
    return {"success": final_success, "ever_success": bool(ever_success),
            "n_steps": int(t), "episode_stats": clean}


# ===========================================================================
# VALIDATION: run STOCK language-conditioned Octo on the env. If the resulting
# SRs land in the published SIMPLER ballpark, the env + control patches + v1
# platform fix + success detection are all correct, isolating goal-conditioning
# as the only new variable for the real experiment.
# ===========================================================================
@app.function(image=image, gpu="L4", volumes={"/data": vol, "/cache": cache},
              secrets=[hf_secret], timeout=3600)
def validate_lang(task: str, n_eps: int = 6, model_type: str = "octo-base"):
    import numpy as np
    apply_env_patches()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from simpler_env.policies.octo.octo_model import OctoInference
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    model = OctoInference(model_type=model_type, policy_setup="widowx_bridge", init_rng=0)
    env, cfg = make_env(task)
    results = []
    for ep in range(n_eps):
        ret = reset_env(env, cfg, ep)
        obs = ret[0] if isinstance(ret, tuple) else ret
        instr = env.unwrapped.get_language_instruction()
        image = get_image_from_maniskill2_obs_dict(env, obs)
        model.reset(instr)
        pterm, trunc, ever = False, False, False
        info = {}
        t = 0
        while not (pterm or trunc):
            _ra, action = model.step(image, instr)
            pterm = bool(action["terminate_episode"][0] > 0)
            obs, _r, done, trunc, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]))
            if bool(info.get("success", done)):
                ever = True
            image = get_image_from_maniskill2_obs_dict(env, obs)
            t += 1
        results.append({"ep": ep, "success": bool(info.get("success", False)),
                        "ever_success": ever, "n_steps": t})
    sr = float(np.mean([r["success"] for r in results]))
    ever_sr = float(np.mean([r["ever_success"] for r in results]))
    return {"task": task, "model": model_type, "n_eps": n_eps,
            "SR": sr, "ever_SR": ever_sr, "results": results}


@app.function(image=image, gpu="L4", cpu=8.0, volumes={"/data": vol, "/cache": cache},
              secrets=[hf_secret], timeout=5400)
def verify_gc(task: str = "widowx_put_eggplant_in_basket", n_eps: int = 10,
              model_type: str = "octo-base"):
    """Decisive goal-conditioning test, independent of editor quality. For each
    ep: run LANGUAGE-octo, capture its final frame + success. For eps the policy
    actually SOLVED (lang success), that final frame is a REAL achievable goal —
    re-run the SAME ep with GOAL-octo conditioned on it. If goal-octo then also
    succeeds, octo goal-conditioning genuinely steers behavior, so low method SRs
    reflect goal QUALITY, not a broken harness."""
    import numpy as np
    apply_env_patches()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from simpler_env.policies.octo.octo_model import OctoInference
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    lang = OctoInference(model_type=model_type, policy_setup="widowx_bridge", init_rng=0)
    goalp = build_goal_policy(model_type=model_type)
    env, cfg = make_env(task)
    rows = []
    for ep in range(n_eps):
        ret = reset_env(env, cfg, ep)
        obs = ret[0] if isinstance(ret, tuple) else ret
        instr = env.unwrapped.get_language_instruction()
        img = get_image_from_maniskill2_obs_dict(env, obs)
        lang.reset(instr)
        pterm, trunc, info = False, False, {}
        frames = [img]
        succ_img, succ_step = None, None
        step = 0
        while not (pterm or trunc):
            _ra, a = lang.step(img, instr)
            pterm = bool(a["terminate_episode"][0] > 0)
            obs, _r, done, trunc, info = env.step(
                np.concatenate([a["world_vector"], a["rot_axangle"], a["gripper"]]))
            img = get_image_from_maniskill2_obs_dict(env, obs)
            frames.append(img)
            step += 1
            if succ_img is None and bool(info.get("success", False)):
                # CLEAN success frame: a couple steps after success first triggers
                # (so the object is settled), captured before any later disturbance
                succ_step = step
        ever = succ_step is not None
        row = {"ep": ep, "lang_ever_success": ever}
        if ever:
            # clean success goal = frame a few steps after first success
            si = min(succ_step + 3, len(frames) - 1)
            succ_img = frames[si]
            r_succ = rollout_one(goalp, env, cfg, ep, succ_img)
            row["goal_on_cleanSucc_success"] = r_succ["success"]
            row["goal_on_cleanSucc_ever"] = r_succ["ever_success"]
            # near subgoal = frame at ~40% to first success (reachable horizon)
            ni = max(1, int(0.4 * succ_step))
            r_near = rollout_one(goalp, env, cfg, ep, frames[ni])
            row["goal_on_nearSubgoal_ever"] = r_near["ever_success"]
        rows.append(row)
    succ_eps = [r for r in rows if r["lang_ever_success"]]
    cleanOK = [r for r in succ_eps if r.get("goal_on_cleanSucc_success")]
    cleanEver = [r for r in succ_eps if r.get("goal_on_cleanSucc_ever")]
    return {"task": task, "n_eps": n_eps,
            "lang_ever_SR": float(len(succ_eps) / n_eps),
            "n_lang_success": len(succ_eps),
            "goal_cleanSucc_SR_over_langsucc": (len(cleanOK) / len(succ_eps)) if succ_eps else None,
            "goal_cleanSucc_EVER_over_langsucc": (len(cleanEver) / len(succ_eps)) if succ_eps else None,
            "rows": rows}


@app.local_entrypoint()
def vgc(task: str = "widowx_put_eggplant_in_basket", n_eps: int = 10):
    import json
    print(json.dumps(verify_gc.remote(task=task, n_eps=n_eps), indent=2, default=str))


@app.local_entrypoint()
def vlang(n_eps: int = 6, model_type: str = "octo-base"):
    import json
    args = [(t, n_eps, model_type) for t in BRIDGE_TASKS]
    res = list(validate_lang.starmap(args))
    print(json.dumps(res, indent=2, default=str))
    print("=== LANG-CONDITIONED SR SUMMARY ===")
    for r in res:
        print(f"  {r['task']:35s} SR={r['SR']:.3f}  ever_SR={r['ever_SR']:.3f}")


# ===========================================================================
# Batch goal-conditioned rollout for one (method, task): load Octo once, build
# env once, run all episodes against the method's goal images on the volume.
# Goal image path: /data/goals/<method>/<task>/ep<NNN>/{edited_image.jpg|goal.jpg}
# ===========================================================================
def _resolve_goal(method, task, ep):
    import os
    base = f"/data/goals/{method}/{task}/ep{ep:03d}"
    for name in ("edited_image.jpg", "goal.jpg", "edited_image.png", "goal.png"):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    return None


@app.function(image=image, gpu="L4", cpu=8.0, volumes={"/data": vol, "/cache": cache},
              secrets=[hf_secret], timeout=7200)
def run_chunk(method: str, task: str, ep_ids: list, model_type: str = "octo-base"):
    """Run a chunk of episodes for one (method, task). Loads Octo + env ONCE,
    rolls out each ep against its goal image. cpu=8 speeds the per-step lavapipe
    (CPU Vulkan) render, the rollout bottleneck.
    rolls out each ep against its goal image. Returns per-ep results (the
    local aggregator writes the per-task json)."""
    import numpy as np
    from PIL import Image
    apply_env_patches()
    model = build_goal_policy(model_type=model_type)
    env, cfg = make_env(task)
    results = []
    for ep in ep_ids:
        gp = _resolve_goal(method, task, ep)
        if gp is None:
            results.append({"ep": ep, "success": None, "error": "no_goal_image"})
            continue
        goal = np.asarray(Image.open(gp).convert("RGB"))
        r = rollout_one(model, env, cfg, ep, goal)
        r["ep"] = ep
        results.append(r)
    return results


@app.local_entrypoint()
def run_method(method: str, model_type: str = "octo-base",
               n_eps: int = N_EPISODES, chunk: int = 6, tasks: str = "all"):
    """Parallel goal-conditioned rollouts: chunk episodes across containers."""
    import json
    task_list = BRIDGE_TASKS if tasks == "all" else tasks.split(",")
    args = []
    for t in task_list:
        for s in range(0, n_eps, chunk):
            args.append((method, t, list(range(s, min(s + chunk, n_eps))), model_type))
    chunk_results = list(run_chunk.starmap(args))
    # aggregate per task
    per_task = {t: [] for t in task_list}
    for (m, t, ep_ids, mt), res in zip(args, chunk_results):
        per_task[t].extend(res)
    summary = {}
    for t, results in per_task.items():
        results = sorted(results, key=lambda r: r["ep"])
        valid = [r for r in results if r.get("success") is not None]
        sr = float(sum(r["success"] for r in valid) / len(valid)) if valid else None
        ever = float(sum(r["ever_success"] for r in valid) / len(valid)) if valid else None
        summary[t] = {"SR": sr, "ever_SR": ever, "n_valid": len(valid),
                      "n_eps": len(results), "results": results}
    # print FIRST (so results survive any persistence error), then persist
    print(f"\n=== {method} ({model_type}) SR SUMMARY ===")
    srs = []
    for t in task_list:
        s = summary[t]
        srs.append(s["SR"] if s["SR"] is not None else 0.0)
        print(f"  {t:35s} SR={s['SR']}  ever_SR={s['ever_SR']}  "
              f"({s['n_valid']}/{s['n_eps']} valid)")
    overall = sum(srs) / len(srs) if srs else None
    print(f"  {'MEAN':35s} SR={overall:.4f}" if overall is not None else "  MEAN SR=None")
    try:
        write_results.remote(method, model_type, summary)
    except Exception as e:
        print(f"  [warn] write_results failed: {e}")
    print("RESULT_JSON " + json.dumps(
        {t: {"SR": summary[t]["SR"], "ever_SR": summary[t]["ever_SR"]} for t in task_list}))


@app.function(image=io_image, volumes={"/data": vol}, timeout=300)
def write_results(method: str, model_type: str, summary: dict):
    import os, json
    od = f"/data/results/{method}"
    os.makedirs(od, exist_ok=True)
    with open(f"{od}/summary.json", "w") as f:
        json.dump({"method": method, "model": model_type, "summary": summary},
                  f, indent=2, default=str)
    vol.commit()
    return od


# ===========================================================================
# Volume I/O helpers (tar-based; `modal volume get` recursive download is buggy).
# stage:  export SIMPLER obs for a task -> local dir of case subdirs.
# push:   read edited_image.jpg from a local stage dir -> /goals/<method>/<task>.
# ===========================================================================
@app.function(image=io_image, volumes={"/data": vol}, timeout=900)
def export_obs_tar(task: str):
    import io, tarfile, os
    root = f"/data/simpler_obs/{task}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for case in sorted(os.listdir(root)):
            cd = os.path.join(root, case)
            if not os.path.isdir(cd):
                continue
            for fn in ("original_image.jpg", "edit_instruction.txt"):
                p = os.path.join(cd, fn)
                if os.path.isfile(p):
                    tf.add(p, arcname=f"{case}/{fn}")
    return buf.getvalue()


@app.function(image=io_image, volumes={"/data": vol}, timeout=900)
def import_goals(method: str, task: str, files: dict):
    """files: {case_name: edited_image_bytes}."""
    import os
    n = 0
    for case, data in files.items():
        d = f"/data/goals/{method}/{task}/{case}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/edited_image.jpg", "wb") as f:
            f.write(data)
        n += 1
    vol.commit()
    return n


@app.local_entrypoint()
def stage(task: str, dest: str):
    import io, tarfile, os
    data = export_obs_tar.remote(task)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(dest)
    cases = sorted(d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d)))
    print(f"STAGED {task}: {len(cases)} cases -> {dest}")


@app.local_entrypoint()
def push(method: str, task: str, src: str):
    import os
    files = {}
    for case in sorted(os.listdir(src)):
        cd = os.path.join(src, case)
        if not os.path.isdir(cd):
            continue
        for name in ("edited_image.jpg", "goal.jpg", "edited_image.png", "goal.png"):
            p = os.path.join(cd, name)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    files[case] = f.read()
                break
    n = import_goals.remote(method, task, files)
    print(f"PUSHED {n} goal images for {method}/{task}")


@app.function(image=image, gpu="L4", volumes={"/cache": cache}, secrets=[hf_secret], timeout=900)
def dump_create_tasks():
    import inspect, numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from octo.model.octo_model import OctoModel
    out = {}
    out["create_tasks_src"] = inspect.getsource(OctoModel.create_tasks)
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
    goal = (np.random.rand(1, 256, 256, 3) * 255).astype(np.uint8)
    task = model.create_tasks(goals={"image_primary": goal})
    # inspect pad masks
    def describe(d, prefix=""):
        r = {}
        for k, v in d.items():
            if hasattr(v, "shape"):
                r[prefix+k] = {"shape": list(v.shape), "dtype": str(v.dtype),
                               "sample": np.asarray(v).flatten()[:4].tolist() if np.asarray(v).size <= 100 or "mask" in k else "..."}
            elif isinstance(v, dict):
                r.update(describe(v, prefix+k+"/"))
            else:
                r[prefix+k] = str(v)
        return r
    out["goal_task"] = describe(task)
    txt = model.create_tasks(texts=["put eggplant in basket"])
    out["text_task"] = describe(txt)
    return out


@app.local_entrypoint()
def dctasks():
    import json
    print(json.dumps(dump_create_tasks.remote(), indent=2, default=str))


@app.function(image=image, timeout=600)
def dump_octo_examples():
    import os, glob, subprocess
    out = {}
    # find goal-conditioning usage examples in the octo repo
    hits = subprocess.run(["grep", "-rn", "create_tasks", "/root/octo"],
                          capture_output=True, text=True).stdout
    out["create_tasks_usages"] = hits[:4000]
    # obs key naming: pad_mask vs timestep_pad_mask
    pm = subprocess.run(["grep", "-rn", "timestep_pad_mask\|'pad_mask'\|\"pad_mask\"", "/root/octo/octo"],
                        capture_output=True, text=True).stdout
    out["pad_mask_refs"] = pm[:3000]
    # example batch task keys
    return out


@app.local_entrypoint()
def doctoex():
    import json
    print(json.dumps(dump_octo_examples.remote(), indent=2, default=str))
