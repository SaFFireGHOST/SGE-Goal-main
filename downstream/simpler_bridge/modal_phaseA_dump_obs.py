"""
Phase A: dump deterministic SIMPLER-Bridge initial observations.

For each of the 4 WidowX/Bridge tasks and each of the 24 canonical episodes
(obj_variation_mode="episode"), reset the env to its deterministic initial
layout and save:
    /data/simpler_obs/<task>/ep<NNN>/original_image.jpg   (RGB initial obs)
    /data/simpler_obs/<task>/ep<NNN>/edit_instruction.txt  (manipulation instr)
    /data/simpler_obs/<task>/ep<NNN>/meta.json

This folder layout matches what the klein IER/FER + Kontext IE inference scripts
consume, so Phase B (composite goal generation) runs on these folders unchanged.
Because the episodes are deterministic, the obs dumped here are byte-identical to
the obs the policy will see at the start of the Phase C rollout (same episode_id)
-> the generated goal image is consistent with the rollout's true initial state.

Rendering is headless lavapipe (CPU Vulkan); no GPU needed for this phase.

Run:  modal run simpler-eval/modal_phaseA_dump_obs.py
"""
import modal

app = modal.App("simpler-phaseA-dump")

CUDA_TAG = "12.8.1-devel-ubuntu24.04"

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
    .pip_install("numpy==1.24.4", "scipy", "pillow", "opencv-python-headless", "mediapy")
    .pip_install("sapien==2.2.2")
    .pip_install("ruckig")
    .run_commands(
        "git clone https://github.com/simpler-env/SimplerEnv /root/SimplerEnv --recurse-submodules",
        "cd /root/SimplerEnv/ManiSkill2_real2sim && python -m pip install -e .",
        "cd /root/SimplerEnv && python -m pip install -e . --no-deps",
    )
    .env({
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json",
        "XDG_RUNTIME_DIR": "/tmp",
        "DISPLAY": "",
    })
)

vol = modal.Volume.from_name("simpler-eval-data", create_if_missing=True)

BRIDGE_TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]
N_EPISODES = 24


def _force_offscreen():
    """lavapipe is headless -> SAPIEN renderer must be offscreen_only."""
    import sapien.core as sapien
    _orig = sapien.SapienRenderer

    def _off(*a, **k):
        k.setdefault("offscreen_only", True)
        return _orig(*a, **k)

    sapien.SapienRenderer = _off


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def dump_task(task: str):
    import os, json, hashlib
    import numpy as np
    from PIL import Image

    _force_offscreen()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    env = simpler_env.make(task)
    results = []
    hashes = []
    for ep in range(N_EPISODES):
        reset_ret = env.reset(options={"obj_init_options": {"episode_id": ep}})
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        instr = env.unwrapped.get_language_instruction()
        img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")

        d = f"/data/simpler_obs/{task}/ep{ep:03d}"
        os.makedirs(d, exist_ok=True)
        Image.fromarray(img).save(f"{d}/original_image.jpg", quality=95)
        with open(f"{d}/edit_instruction.txt", "w") as f:
            f.write(instr)
        with open(f"{d}/meta.json", "w") as f:
            json.dump({"task": task, "episode_id": ep, "instruction": instr,
                       "img_shape": list(img.shape)}, f)
        h = hashlib.md5(img.tobytes()).hexdigest()[:8]
        hashes.append(h)
        results.append({"ep": ep, "instr": instr, "shape": list(img.shape), "hash": h})

    vol.commit()
    # distinct-layout sanity: episodes should NOT all be identical
    n_unique = len(set(hashes))
    return {"task": task, "n_episodes": len(results), "n_unique_layouts": n_unique,
            "instruction": results[0]["instr"] if results else None,
            "shape": results[0]["shape"] if results else None}


@app.function(image=image, timeout=900)
def probe_variation():
    """Why do spoon/carrot/stack give identical layouts across episode_id but
    eggplant varies? Inspect how the bridge envs consume episode variation."""
    import inspect
    _force_offscreen()
    import simpler_env
    out = {}
    # what does simpler_env.make map each task to?
    try:
        out["ENVIRONMENT_MAP"] = {k: str(v) for k, v in
                                  simpler_env.ENVIRONMENT_MAP.items()
                                  if k in BRIDGE_TASKS}
    except Exception as e:
        out["ENVIRONMENT_MAP"] = repr(e)
    for task in ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]:
        env = simpler_env.make(task)
        u = env.unwrapped
        info = {}
        for attr in ["obj_variation_mode", "obj_episode_range", "_obj_init_options",
                     "episode_id", "obj_init_options"]:
            info[attr] = repr(getattr(u, attr, "<none>"))
        # source of the model-placement step that reads episode info
        for meth in ["reset", "_initialize_episode", "_set_model", "_load_model",
                     "_setup_model_init", "reconfigure"]:
            fn = getattr(type(u), meth, None)
            if fn is not None:
                try:
                    info[f"src_{meth}"] = inspect.getsource(fn)[:1500]
                except Exception as e:
                    info[f"src_{meth}"] = repr(e)
        out[task] = info
    return out


@app.function(image=image, timeout=900)
def probe_grids():
    """How many (xy, quat) configs does each bridge task actually have, and how
    are they constructed? Determines how many distinct episodes are possible."""
    import inspect
    _force_offscreen()
    import simpler_env
    out = {}
    for task in BRIDGE_TASKS:
        env = simpler_env.make(task)
        u = env.unwrapped
        xy = getattr(u, "_xy_configs", None)
        quat = getattr(u, "_quat_configs", None)
        info = {
            "n_xy_configs": (len(xy) if xy is not None else None),
            "n_quat_configs": (len(quat) if quat is not None else None),
            "n_distinct_layouts": ((len(xy) * len(quat)) if (xy is not None and quat is not None) else None),
        }
        # where are the grids built? find the method that assigns _xy_configs
        for meth in ["__init__", "_setup_prepackaged_env_init_config",
                     "_additional_prepackaged_config_reset", "set_episode_rng"]:
            fn = getattr(type(u), meth, None) or getattr(
                type(u).__bases__[0] if type(u).__bases__ else object, meth, None)
            if fn is not None:
                try:
                    src = inspect.getsource(fn)
                    if "_xy_configs" in src or "prepackaged" in src.lower():
                        info[f"src_{meth}"] = src[:2500]
                except Exception:
                    pass
        out[task] = info
    return out


PREPKG = {
    "widowx_spoon_on_towel": dict(
        env_name="PutSpoonOnTableClothInScene-v0", robot="widowx",
        scene_name="bridge_table_1_v1", overlay="real_inpainting/bridge_real_eval_1.png",
        robot_init_xy=[0.147, 0.028], max_steps=60),
    "widowx_carrot_on_plate": dict(
        env_name="PutCarrotOnPlateInScene-v0", robot="widowx",
        scene_name="bridge_table_1_v1", overlay="real_inpainting/bridge_real_eval_1.png",
        robot_init_xy=[0.147, 0.028], max_steps=60),
    "widowx_stack_cube": dict(
        env_name="StackGreenCubeOnYellowCubeBakedTexInScene-v0", robot="widowx",
        scene_name="bridge_table_1_v1", overlay="real_inpainting/bridge_real_eval_1.png",
        robot_init_xy=[0.147, 0.028], max_steps=60),
    "widowx_put_eggplant_in_basket": dict(
        env_name="PutEggplantInBasketScene-v0", robot="widowx_sink_camera_setup",
        scene_name="bridge_table_1_v2", overlay="real_inpainting/bridge_sink.png",
        robot_init_xy=[0.127, 0.06], max_steps=120),
}
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


def _build_official(task: str):
    """Reproduce the official SimplerEnv build_maniskill2_env path for one task."""
    import numpy as np
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


def _reset_official(env, cfg, episode_id, init_z=None):
    import numpy as np
    obj_opts = {"episode_id": int(episode_id)}
    if init_z is not None:
        # Spawn just above the table so thin objects can't tunnel through it.
        # (This SAPIEN build's PhysX lacks working CCD; the default +0.5 drop
        # makes thin objects like the spoon tunnel to the arena floor.)
        obj_opts["init_z"] = float(init_z)
    return env.reset(options={
        "robot_init_options": {"init_xy": np.array(cfg["robot_init_xy"]),
                               "init_rot_quat": [0, 0, 0, 1]},
        "obj_init_options": obj_opts,
    })


@app.function(image=image, timeout=900)
def probe_table_height():
    """scene_table_height for spoon (broken) vs eggplant (ok), arena pose, and
    where the value comes from. Disambiguates 'wrong height' vs 'no collision'."""
    import inspect
    _force_offscreen()
    import simpler_env
    out = {}
    for task in ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]:
        env, cfg = _build_official(task)
        _reset_official(env, cfg, 0)
        u = env.unwrapped
        info = {
            "scene_table_height": repr(getattr(u, "scene_table_height", "<none>")),
            "robot_init_height_attr": repr(getattr(u, "robot_init_height", "<none>")),
        }
        try:
            arena = [a for a in u._scene.get_all_actors() if a.name == "arena"]
            info["arena_pose_p"] = ([round(float(x), 4) for x in arena[0].pose.p]
                                    if arena else "no-arena")
            # collision shapes on the arena actor
            if arena:
                cs = arena[0].get_collision_shapes()
                info["arena_n_collision_shapes"] = len(cs)
        except Exception as e:
            info["arena_err"] = repr(e)
        # bbox of the source object (to know how far it should rest above table)
        objs = getattr(u, "episode_objs", None)
        if objs:
            info["src_obj_p"] = [round(float(x), 4) for x in objs[0].pose.p]
            info["src_obj_n_collision"] = len(objs[0].get_collision_shapes())
        out[task] = info
    # where is scene_table_height assigned?
    import mani_skill2_real2sim, os, glob
    pkgdir = os.path.dirname(mani_skill2_real2sim.__file__)
    grep = {}
    for f in glob.glob(os.path.join(pkgdir, "**", "*.py"), recursive=True):
        try:
            with open(f) as fh:
                for ln, line in enumerate(fh, 1):
                    if "scene_table_height" in line and ("=" in line and "==" not in line):
                        grep.setdefault(os.path.relpath(f, pkgdir), []).append(
                            f"{ln}: {line.strip()[:120]}")
        except Exception:
            pass
    out["_scene_table_height_assignments"] = grep
    return out


@app.local_entrypoint()
def tableh():
    import json
    print(json.dumps(probe_table_height.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_placement_src():
    """Dump the object placement + settle code to see why spoon objs fall through
    the bridge_table_1_v1 table (z=-1.05) while eggplant rests fine."""
    import inspect
    _force_offscreen()
    import simpler_env
    env, cfg = _build_official("widowx_spoon_on_towel")
    _reset_official(env, cfg, 0)
    u = env.unwrapped
    out = {"mro": [c.__name__ for c in type(u).__mro__]}
    methods = ["_initialize_actors", "_initialize_actors_pre", "_load_actors",
               "initialize_episode", "_initialize_episode", "_settle",
               "_set_model", "_get_init_z", "_initialize_agent"]
    seen = set()
    for c in type(u).__mro__:
        for meth in methods:
            if meth in seen:
                continue
            fn = c.__dict__.get(meth)
            if fn is not None:
                try:
                    out[f"{c.__name__}.{meth}"] = inspect.getsource(fn)[:2200]
                    seen.add(meth)
                except Exception as e:
                    out[f"{c.__name__}.{meth}"] = repr(e)
    # also: table actor & its collision, plus object z right after reset vs after settle
    objs = getattr(u, "episode_objs", None)
    out["episode_obj_p_after_reset"] = (
        [round(float(x), 4) for x in objs[0].pose.p] if objs else None)
    # scene actors (look for table/ground)
    try:
        out["scene_actors"] = [a.name for a in u._scene.get_all_actors()][:40]
    except Exception as e:
        out["scene_actors"] = repr(e)
    return out


@app.local_entrypoint()
def placesrc():
    import json
    d = probe_placement_src.remote()
    for k, v in d.items():
        print(f"\n===== {k} =====")
        print(v if not isinstance(v, str) else v)


def _patch_table_platform(half_xy=0.55, top_z=0.870, thickness=0.04):
    """The bridge_table_1_v1 collision mesh does NOT cover the spoon/carrot/stack
    object positions (a clean test box there rests on the arena floor at z~0, not
    the table at z=0.87), while the visual/overlay table IS there. Add an invisible
    static collision platform at the true table height so objects rest on the
    visually-correct surface and the robot interacts with them there.
    Must be paired with a low object spawn (init_z) so objects don't tunnel through
    the thin platform during the drop (this SAPIEN build's PhysX lacks CCD)."""
    import mani_skill2_real2sim.envs.custom_scenes.move_near_in_scene as mn
    import sapien.core as sapien
    cls = mn.MoveNearInSceneEnv
    if getattr(cls, "_platform_patched", False):
        return
    _orig = cls._load_actors

    def _patched(self):
        _orig(self)
        b = self._scene.create_actor_builder()
        b.add_box_collision(half_size=[half_xy, half_xy, thickness / 2.0])
        plat = b.build_static(name="table_platform")
        plat.set_pose(sapien.Pose([0.0, 0.0, top_z - thickness / 2.0]))

    cls._load_actors = _patched
    cls._platform_patched = True


# spawn just above the platform top (0.87) so objects settle without tunneling:
# obj_init_z = init_z + 0.5  -> init_z = 0.40 gives spawn 0.90 (0.03m above table)
PLATFORM_INIT_Z = 0.40


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def verify_platform():
    """With the table platform: do spoon/carrot/stack objects now rest at table
    height, vary per episode, and render visibly?"""
    import os, numpy as np, hashlib
    from PIL import Image
    _force_offscreen()
    _patch_table_platform()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    os.makedirs("/data/debug3", exist_ok=True)
    out = {}
    for task in ["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube"]:
        env, cfg = _build_official(task)
        rows = []
        for ep in [0, 2, 4, 8, 16]:
            ret = _reset_official(env, cfg, ep, init_z=PLATFORM_INIT_Z)
            obs = ret[0] if isinstance(ret, tuple) else ret
            u = env.unwrapped
            sz = round(float(u.episode_objs[0].pose.p[2]), 4)
            tz = round(float(u.episode_objs[1].pose.p[2]), 4)
            img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
            if ep in (0, 8):
                Image.fromarray(img).save(f"/data/debug3/{task}_ep{ep}.png")
            rows.append({"ep": ep, "src_z": sz, "tgt_z": tz,
                         "img_hash": hashlib.md5(img.tobytes()).hexdigest()[:8]})
        out[task] = {"rows": rows, "n_unique_img": len({r["img_hash"] for r in rows})}
    vol.commit()
    return out


@app.local_entrypoint()
def vplat():
    import json
    print(json.dumps(verify_platform.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_shapeprops():
    """Final check: per-shape contact_offset / rest_offset / is_trigger / physical
    material for spoon (falls through) vs eggplant (rests). A zero/negative contact
    offset or trigger flag would explain zero collision response."""
    _force_offscreen()
    import simpler_env
    out = {}
    for task in ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]:
        env, cfg = _build_official(task)
        _reset_official(env, cfg, 0)
        u = env.unwrapped
        obj = u.episode_objs[0]
        shapes = []
        for s in obj.get_collision_shapes():
            d = {}
            for attr in ["contact_offset", "rest_offset", "min_patch_radius",
                         "patch_radius", "is_trigger"]:
                try:
                    v = getattr(s, attr)
                    d[attr] = (v() if callable(v) else v)
                except Exception as e:
                    d[attr] = f"err:{e!r}"
            try:
                m = s.get_physical_material()
                d["material"] = {"sf": round(m.get_static_friction(), 3),
                                 "df": round(m.get_dynamic_friction(), 3),
                                 "rest": round(m.get_restitution(), 3)}
            except Exception as e:
                d["material"] = f"err:{e!r}"
            try:
                d["collision_groups"] = list(s.get_collision_groups())
            except Exception as e:
                d["collision_groups"] = repr(e)
            shapes.append(d)
        out[task] = {"n_shapes": len(shapes), "shapes": shapes}
    return out


@app.local_entrypoint()
def shapeprops():
    import json
    print(json.dumps(probe_shapeprops.remote(), indent=2, default=str))


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def probe_nolock():
    """Hypothesis: Actor.lock_motion() (broken-ABI family) leaves the object with
    collision disabled in this build -> objects fall through everything. A fresh
    box (never lock_motion'd) collides fine. Test: no-op lock_motion, then reset
    spoon/cube and see if objects rest on the table (with the platform)."""
    import os, numpy as np, hashlib
    import sapien.core as sapien
    from PIL import Image
    _force_offscreen()
    # neutralize lock_motion BEFORE building any env
    if not getattr(sapien.Actor, "_lockmotion_noop", False):
        sapien.Actor.lock_motion = lambda self, *a, **k: None
        sapien.Actor._lockmotion_noop = True
    _patch_table_platform()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    os.makedirs("/data/debug6", exist_ok=True)
    out = {}
    for task in ["widowx_spoon_on_towel", "widowx_stack_cube"]:
        env, cfg = _build_official(task)
        rows = []
        for ep in [0, 4, 8]:
            ret = _reset_official(env, cfg, ep, init_z=PLATFORM_INIT_Z)
            obs = ret[0] if isinstance(ret, tuple) else ret
            u = env.unwrapped
            sz = round(float(u.episode_objs[0].pose.p[2]), 4)
            tz = round(float(u.episode_objs[1].pose.p[2]), 4)
            img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
            if ep == 0:
                Image.fromarray(img).save(f"/data/debug6/{task}_ep{ep}.png")
            rows.append({"ep": ep, "src_z": sz, "tgt_z": tz,
                         "hash": hashlib.md5(img.tobytes()).hexdigest()[:8]})
        out[task] = {"rows": rows, "n_unique_img": len({r["hash"] for r in rows})}
    vol.commit()
    return out


@app.local_entrypoint()
def nolock():
    import json
    print(json.dumps(probe_nolock.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_drop():
    """Decisive: with platform, drop spoon from just above it w/ zero velocity and
    log z. Also: platform present? collision groups spoon vs eggplant? kinematic?"""
    import numpy as np
    import sapien.core as sapien
    _force_offscreen()
    _patch_table_platform()
    import simpler_env
    out = {}
    env, cfg = _build_official("widowx_spoon_on_towel")
    _reset_official(env, cfg, 0, init_z=PLATFORM_INIT_Z)
    u = env.unwrapped
    scene = u._scene
    out["actors"] = [a.name for a in scene.get_all_actors()]
    out["platform_present"] = any(a.name == "table_platform" for a in scene.get_all_actors())
    obj = u.episode_objs[0]
    out["obj_type"] = type(obj).__name__
    out["obj_kinematic"] = bool(getattr(obj, "is_kinematic", lambda: "?")() if callable(getattr(obj, "is_kinematic", None)) else getattr(obj, "is_kinematic", "?"))
    # collision groups
    def groups(a):
        try:
            return [list(s.get_collision_groups()) for s in a.get_collision_shapes()[:2]]
        except Exception as e:
            return repr(e)
    out["spoon_cgroups"] = groups(obj)
    plat = [a for a in scene.get_all_actors() if a.name == "table_platform"]
    if plat:
        out["platform_cgroups"] = groups(plat[0])
    arena = [a for a in scene.get_all_actors() if a.name == "arena"]
    if arena:
        out["arena_cgroups"] = groups(arena[0])
    # eggplant comparison
    env2, cfg2 = _build_official("widowx_put_eggplant_in_basket")
    _reset_official(env2, cfg2, 0)
    out["eggplant_cgroups"] = groups(env2.unwrapped.episode_objs[0])

    # drop test: force spoon to canonical xy at 0.95 with zero velocity, step
    xys = np.array(u.obj_init_options["init_xys"])
    obj.set_pose(sapien.Pose([float(xys[0][0]), float(xys[0][1]), 0.95]))
    try:
        obj.set_velocity([0, 0, 0]); obj.set_angular_velocity([0, 0, 0])
    except Exception as e:
        out["setvel_err"] = repr(e)
    zs = []
    for i in range(300):
        scene.step()
        if i % 50 == 0 or i == 299:
            zs.append((i, round(float(obj.pose.p[2]), 4)))
    out["drop_z_trace"] = zs
    return out


@app.local_entrypoint()
def drop():
    import json
    print(json.dumps(probe_drop.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_coacd():
    """Is object collision convex/working? Check coacd availability, the collision
    build path, and the spoon collision shape geometry types + contact offsets."""
    import inspect
    out = {}
    try:
        import coacd
        out["coacd"] = f"available {getattr(coacd, '__version__', '?')}"
    except Exception as e:
        out["coacd"] = f"MISSING: {e!r}"
    _force_offscreen()
    import simpler_env, numpy as np
    import mani_skill2_real2sim.envs.custom_scenes.base_env as be
    # how does _build_actor_helper add collision?
    for cls in [be.CustomSceneEnv]:
        fn = getattr(cls, "_build_actor_helper", None)
        if fn:
            out["_build_actor_helper_src"] = inspect.getsource(fn)[:3000]
    for task in ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]:
        env, cfg = _build_official(task)
        _reset_official(env, cfg, 0)
        u = env.unwrapped
        obj = u.episode_objs[0]
        shapes = obj.get_collision_shapes()
        info = {"n_shapes": len(shapes), "obj_pose_z": round(float(obj.pose.p[2]), 4),
                "obj_mass": round(float(obj.mass), 5)}
        sh = []
        for s in shapes[:5]:
            g = s.geometry
            d = {"geom": type(g).__name__}
            try:
                if hasattr(g, "vertices"):
                    v = np.asarray(g.vertices)
                    d["n_verts"] = int(v.shape[0])
                    d["extent"] = (v.max(0) - v.min(0)).round(4).tolist() if v.size else None
            except Exception as e:
                d["err"] = repr(e)
            sh.append(d)
        info["shapes"] = sh
        out[task] = info
    return out


@app.local_entrypoint()
def coacd():
    import json
    print(json.dumps(probe_coacd.remote(), indent=2, default=str))


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def probe_forceshow():
    """Place spoon+towel at their CANONICAL xy at table height and render. If they
    appear, objects render fine and the only bug is the fall-through; the fix is to
    make them rest on the table."""
    import os, numpy as np, hashlib
    import sapien.core as sapien
    from PIL import Image
    _force_offscreen()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    os.makedirs("/data/debug5", exist_ok=True)
    out = {}
    env, cfg = _build_official("widowx_spoon_on_towel")
    for ep in [0, 2]:
        ret = _reset_official(env, cfg, ep)
        obs = ret[0] if isinstance(ret, tuple) else ret
        u = env.unwrapped
        xys = np.array(u.obj_init_options["init_xys"])  # [src, tgt]
        quats = np.array(u.obj_init_options["init_rot_quats"])
        for i, obj in enumerate(u.episode_objs):
            obj.set_pose(sapien.Pose([float(xys[i][0]), float(xys[i][1]), 0.885],
                                     list(quats[i])))
        u._scene.update_render()
        fresh = u.get_obs()  # FRESH render (reset obs is stale: objects had fallen)
        img = np.asarray(get_image_from_maniskill2_obs_dict(env, fresh)).astype("uint8")
        Image.fromarray(img).save(f"/data/debug5/spoon_forced_canonical_ep{ep}.png")
        out[f"ep{ep}"] = {
            "init_xys": xys.round(3).tolist(),
            "obj_p_set": [[round(float(x), 3) for x in o.pose.p] for o in u.episode_objs],
            "img_hash": hashlib.md5(img.tobytes()).hexdigest()[:8]}
    vol.commit()
    return out


@app.local_entrypoint()
def forceshow():
    import json
    print(json.dumps(probe_forceshow.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_modeldir():
    """List mesh files in spoon vs eggplant model dirs + the loaded visual-body
    bounding boxes. Definitive: 'mesh missing' vs 'mesh fails to load'."""
    import os
    from mani_skill2_real2sim import ASSET_DIR
    out = {}
    base = os.path.join(str(ASSET_DIR), "custom", "models")
    for d in ["bridge_spoon_generated_modified", "eggplant",
              "bridge_carrot_generated_modified", "baked_green_cube_3cm",
              "table_cloth_generated_shorter"]:
        p = os.path.join(base, d)
        if os.path.isdir(p):
            out[d] = {f: os.path.getsize(os.path.join(p, f))
                      for f in sorted(os.listdir(p))}
        else:
            out[d] = "MISSING"
    # loaded visual-body bbox for spoon vs eggplant
    _force_offscreen()
    import simpler_env, numpy as np
    for task in ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]:
        env, cfg = _build_official(task)
        _reset_official(env, cfg, 0)
        u = env.unwrapped
        obj = u.episode_objs[0]
        vbs = obj.get_visual_bodies()
        info = {"n_visual_bodies": len(vbs)}
        try:
            vb = vbs[0]
            shapes = vb.get_render_shapes()
            info["n_render_shapes"] = len(shapes)
            if shapes:
                mesh = shapes[0].mesh
                verts = np.asarray(mesh.vertices)
                info["mesh_n_verts"] = int(verts.shape[0])
                if verts.size:
                    info["mesh_bbox_extent"] = (verts.max(0) - verts.min(0)).round(4).tolist()
            info["vb_scale"] = list(np.asarray(getattr(vb, "scale", [1, 1, 1])).round(4))
        except Exception as e:
            info["vb_err"] = repr(e)
        out[f"loaded::{task}"] = info
    return out


@app.local_entrypoint()
def modeldir():
    import json
    print(json.dumps(probe_modeldir.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_assets():
    """Are the bridge object mesh assets actually present? (image build never
    runs the SIMPLER asset download.) Compare spoon/carrot/cube vs eggplant."""
    import os, glob
    from mani_skill2_real2sim import ASSET_DIR
    out = {"ASSET_DIR": str(ASSET_DIR), "ASSET_DIR_exists": os.path.isdir(str(ASSET_DIR))}
    # list the custom model dirs
    candidates = [
        os.path.join(str(ASSET_DIR), "custom"),
        os.path.join(str(ASSET_DIR), "models"),
        str(ASSET_DIR),
    ]
    for c in candidates:
        if os.path.isdir(c):
            out[f"ls::{os.path.relpath(c, str(ASSET_DIR))}"] = sorted(os.listdir(c))[:60]
    # hunt for specific model assets
    for key in ["bridge_spoon", "spoon", "carrot", "cube", "eggplant",
                "table_cloth", "plate", "bridge_carrot", "green_cube", "yellow_cube"]:
        hits = glob.glob(os.path.join(str(ASSET_DIR), "**", f"*{key}*"), recursive=True)
        out[f"glob::{key}"] = [(os.path.relpath(h, str(ASSET_DIR)),
                                (os.path.getsize(h) if os.path.isfile(h) else "dir"))
                               for h in sorted(hits)[:12]]
    # the model db that lists object -> files
    import mani_skill2_real2sim, importlib, inspect
    try:
        from mani_skill2_real2sim.utils.io_utils import load_json
        # find model_db jsons
        dbs = glob.glob(os.path.join(str(ASSET_DIR), "**", "*model*.json"), recursive=True)
        out["model_db_jsons"] = [os.path.relpath(d, str(ASSET_DIR)) for d in dbs[:20]]
    except Exception as e:
        out["model_db_err"] = repr(e)
    return out


@app.local_entrypoint()
def assets():
    import json
    print(json.dumps(probe_assets.remote(), indent=2, default=str))


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def probe_render():
    """Do spoon objects have visual meshes, and do they render if force-placed in
    a known-visible spot? Disambiguates 'visual mesh missing' vs 'object out of
    frame / overlay masks it'."""
    import os, numpy as np, hashlib
    import sapien.core as sapien
    from PIL import Image
    _force_offscreen()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from mani_skill2_real2sim import ASSET_DIR
    from simpler_env.utils.env.env_builder import build_maniskill2_env
    os.makedirs("/data/debug4", exist_ok=True)
    out = {}

    # (1) with overlay: force source obj to a clearly-visible on-table spot
    env, cfg = _build_official("widowx_spoon_on_towel")
    ret = _reset_official(env, cfg, 0)
    obs = ret[0] if isinstance(ret, tuple) else ret
    u = env.unwrapped
    obj = u.episode_objs[0]
    out["src_name"] = obj.name
    out["src_n_visual_bodies"] = len(obj.get_visual_bodies())
    out["src_n_collision"] = len(obj.get_collision_shapes())
    # force to center of table, render
    for tag, p in [("forced_center", [0.0, 0.0, 0.95]),
                   ("forced_front", [-0.1, 0.0, 0.92])]:
        obj.set_pose(sapien.Pose(p))
        u._scene.step(); u._scene.update_render()
        img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
        Image.fromarray(img).save(f"/data/debug4/spoon_overlay_{tag}.png")
        out[f"hash_overlay_{tag}"] = hashlib.md5(img.tobytes()).hexdigest()[:8]

    # (2) NO overlay: same env config but rgb_overlay_path=None
    env2 = build_maniskill2_env(
        cfg["env_name"], obs_mode="rgbd", robot=cfg["robot"], sim_freq=500,
        control_mode=CONTROL_MODE, control_freq=5, max_episode_steps=cfg["max_steps"],
        scene_name=cfg["scene_name"], camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=None)
    ret2 = _reset_official(env2, cfg, 0)
    obs2 = ret2[0] if isinstance(ret2, tuple) else ret2
    u2 = env2.unwrapped
    obj2 = u2.episode_objs[0]
    for tag, p in [("rest", None), ("forced_center", [0.0, 0.0, 0.95])]:
        if p is not None:
            obj2.set_pose(sapien.Pose(p))
        u2._scene.step(); u2._scene.update_render()
        img = np.asarray(get_image_from_maniskill2_obs_dict(env2, obs2)).astype("uint8")
        Image.fromarray(img).save(f"/data/debug4/spoon_nooverlay_{tag}.png")
        out[f"hash_nooverlay_{tag}"] = hashlib.md5(img.tobytes()).hexdigest()[:8]
    out["src_obj_p_after_reset_nooverlay"] = [round(float(x), 3) for x in obj2.pose.p]
    vol.commit()
    return out


@app.local_entrypoint()
def prender():
    import json
    print(json.dumps(probe_render.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_collision():
    """Isolate: is the v1 TABLE collision broken, or the OBJECT collision?
    Drop a plain test box at the object xy and see if it rests on the table."""
    import numpy as np
    import sapien.core as sapien
    _force_offscreen()
    import simpler_env
    out = {}
    for task in ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]:
        env, cfg = _build_official(task)
        _reset_official(env, cfg, 0)
        u = env.unwrapped
        scene = u._scene
        # arena collision geometry
        arena = [a for a in scene.get_all_actors() if a.name == "arena"][0]
        cs = arena.get_collision_shapes()[0]
        geom = cs.geometry
        ginfo = {"type": type(geom).__name__}
        for attr in ["scale", "vertices", "half_size"]:
            v = getattr(geom, attr, None)
            if v is not None:
                try:
                    arr = np.asarray(v)
                    ginfo[attr] = (arr.shape if arr.size > 12 else arr.round(3).tolist())
                    if attr == "vertices":
                        ginfo["vert_world_z_range"] = [
                            round(float((arena.pose.to_transformation_matrix()[:3, :3] @ arr.T).T[:, 2].min() + arena.pose.p[2]), 3),
                            round(float((arena.pose.to_transformation_matrix()[:3, :3] @ arr.T).T[:, 2].max() + arena.pose.p[2]), 3)]
                except Exception as e:
                    ginfo[attr] = repr(e)
        out[f"{task}_arena_collision"] = ginfo
        # drop a plain 3cm box at object xy from z=1.2
        objx, objy = float(u.episode_objs[0].pose.p[0]), float(u.episode_objs[0].pose.p[1])
        b = scene.create_actor_builder()
        b.add_box_collision(half_size=[0.015, 0.015, 0.015], density=1000)
        b.add_box_visual(half_size=[0.015, 0.015, 0.015])
        box = b.build(name="testbox")
        box.set_pose(sapien.Pose([objx, objy, 1.2]))
        for _ in range(250):
            scene.step()
        out[f"{task}_testbox_rest_z"] = round(float(box.pose.p[2]), 4)
        out[f"{task}_testbox_xy"] = [round(objx, 3), round(objy, 3)]
    return out


@app.local_entrypoint()
def coll():
    import json
    print(json.dumps(probe_collision.remote(), indent=2, default=str))


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def verify_initz(init_z: float = 0.42):
    """Spawn objects just above the table (init_z) so thin objects don't tunnel.
    Check spoon objects now rest at ~table height, images vary, save to inspect."""
    import os, numpy as np, hashlib
    from PIL import Image
    _force_offscreen()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    os.makedirs("/data/debug2", exist_ok=True)
    out = {}
    for task in ["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube"]:
        env, cfg = _build_official(task)
        rows = []
        for ep in [0, 2, 4, 8, 16]:
            ret = _reset_official(env, cfg, ep, init_z=init_z)
            obs = ret[0] if isinstance(ret, tuple) else ret
            u = env.unwrapped
            objs = u.episode_objs
            src_z = round(float(objs[0].pose.p[2]), 4)
            tgt_z = round(float(objs[1].pose.p[2]), 4)
            img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
            if ep in (0, 8):
                Image.fromarray(img).save(f"/data/debug2/{task}_ep{ep}.png")
            rows.append({"ep": ep, "src_z": src_z, "tgt_z": tgt_z,
                         "img_hash": hashlib.md5(img.tobytes()).hexdigest()[:8]})
        out[task] = {"rows": rows, "n_unique_img": len({r["img_hash"] for r in rows})}
    vol.commit()
    return out


@app.local_entrypoint()
def vinitz(init_z: float = 0.42):
    import json
    print(json.dumps(verify_initz.remote(init_z=init_z), indent=2, default=str))


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def save_debug_images():
    """Save spoon vs eggplant reset images (official build) to inspect what is
    actually rendered (is the object visible? does the overlay dominate?)."""
    import os, numpy as np
    from PIL import Image
    _force_offscreen()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    os.makedirs("/data/debug", exist_ok=True)
    saved = []
    for task in ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]:
        env, cfg = _build_official(task)
        for ep in [0, 4]:
            ret = _reset_official(env, cfg, ep)
            obs = ret[0] if isinstance(ret, tuple) else ret
            img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
            p = f"/data/debug/{task}_ep{ep}.png"
            Image.fromarray(img).save(p)
            saved.append(p)
    vol.commit()
    return {"saved": saved}


@app.local_entrypoint()
def savedbg():
    import json
    print(json.dumps(save_debug_images.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_official_build():
    """Does the OFFICIAL build_maniskill2_env path vary spoon images + place
    objects on the table (vs the broken prepackaged make path)?"""
    import numpy as np, hashlib
    _force_offscreen()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    out = {}
    for task in ["widowx_spoon_on_towel", "widowx_stack_cube"]:
        env, cfg = _build_official(task)
        rows = []
        for ep in [0, 1, 2, 3, 4]:
            ret = _reset_official(env, cfg, ep)
            obs = ret[0] if isinstance(ret, tuple) else ret
            u = env.unwrapped
            objs = getattr(u, "episode_objs", None)
            src_p = ([round(float(x), 4) for x in objs[0].pose.p] if objs else None)
            img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
            rows.append({"ep": ep, "src_obj_p": src_p,
                         "img_hash": hashlib.md5(img.tobytes()).hexdigest()[:8]})
        out[task] = {"rows": rows,
                     "n_unique_img": len({r["img_hash"] for r in rows}),
                     "n_unique_obj_p": len({str(r["src_obj_p"]) for r in rows})}
    return out


@app.local_entrypoint()
def obuild():
    import json
    print(json.dumps(probe_official_build.remote(), indent=2, default=str))


@app.function(image=image, timeout=900)
def probe_official():
    """Read the authoritative SimplerEnv eval to see EXACTLY how it makes the
    bridge env and iterates episodes (the visual-matching multi-episode loop)."""
    import os, glob
    out = {}
    roots = ["/root/SimplerEnv/simpler_env"]
    wanted = ["main_inference.py", "maniskill2_evaluator.py"]
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "*.py"), recursive=True):
            base = os.path.basename(f)
            if base in wanted:
                with open(f) as fh:
                    out[f] = fh.read()
    # also grep for episode-loop / make-env patterns across the package
    hits = {}
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "*.py"), recursive=True):
            try:
                with open(f) as fh:
                    txt = fh.read()
            except Exception:
                continue
            for kw in ["obj_episode_range", "episode_id", "additional_env_build_kwargs",
                       "obj_variation_mode"]:
                if kw in txt:
                    hits.setdefault(kw, []).append(os.path.relpath(f, r))
    out["_keyword_files"] = {k: sorted(set(v)) for k, v in hits.items()}
    return out


@app.local_entrypoint()
def official():
    import json
    d = probe_official.remote()
    kw = d.pop("_keyword_files", {})
    print("=== KEYWORD FILES ===")
    print(json.dumps(kw, indent=2))
    for path, txt in d.items():
        print(f"\n===== {path} =====\n{txt}")


@app.function(image=image, timeout=900)
def probe_place():
    """For spoon (broken) test if source-object POSE actually changes per
    episode, and whether reconfigure=True is required to apply it."""
    import numpy as np, hashlib
    _force_offscreen()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    out = {}

    def src_obj_xy(u):
        objs = getattr(u, "episode_objs", None)
        if objs:
            return [round(float(x), 4) for x in objs[0].pose.p]
        src = getattr(u, "episode_source_obj", None)
        if src is not None:
            return [round(float(x), 4) for x in src.pose.p]
        return None

    for mode in ["plain", "reconfigure", "fresh_env"]:
        env = simpler_env.make("widowx_spoon_on_towel")
        rows = []
        for ep in [0, 2, 4, 6]:
            if mode == "fresh_env":
                env = simpler_env.make("widowx_spoon_on_towel")
            opts = {"obj_init_options": {"episode_id": ep}}
            if mode == "reconfigure":
                opts["reconfigure"] = True
            ret = env.reset(options=opts)
            obs = ret[0] if isinstance(ret, tuple) else ret
            u = env.unwrapped
            img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
            rows.append({"ep": ep, "src_obj_p": src_obj_xy(u),
                         "img_hash": hashlib.md5(img.tobytes()).hexdigest()[:8]})
        out[mode] = {"rows": rows,
                     "n_unique_obj_p": len(set(str(r["src_obj_p"]) for r in rows)),
                     "n_unique_img": len(set(r["img_hash"] for r in rows))}
    return out


@app.local_entrypoint()
def place():
    import json
    print(json.dumps(probe_place.remote(), indent=2, default=str))


@app.local_entrypoint()
def grids():
    import json
    print(json.dumps(probe_grids.remote(), indent=2, default=str))


@app.local_entrypoint()
def probe():
    import json
    print(json.dumps(probe_variation.remote(), indent=2, default=str))


@app.local_entrypoint()
def main():
    import json
    # 4 tasks dump concurrently (separate containers; render is CPU lavapipe)
    summary = list(dump_task.map(BRIDGE_TASKS))
    for res in summary:
        print(json.dumps(res))
    print("=== PHASE A SUMMARY ===")
    print(json.dumps(summary, indent=2))


@app.function(image=image, timeout=900)
def probe_table_actor():
    """Nail the v1-vs-v2 table-collision difference. For spoon (v1, broken) and
    eggplant (v2, works): env-class MRO, and EVERY actor with its collision-shape
    geometry types + world-space z bbox. Find the actor that provides the table
    top at z~0.87 in eggplant and is missing/broken in spoon."""
    import numpy as np
    import sapien.core as sapien
    _force_offscreen()
    import simpler_env
    out = {}
    for task in ["widowx_spoon_on_towel", "widowx_put_eggplant_in_basket"]:
        env, cfg = _build_official(task)
        _reset_official(env, cfg, 0)
        u = env.unwrapped
        info = {"env_class": type(u).__name__,
                "mro": [c.__name__ for c in type(u).__mro__],
                "scene_name": cfg["scene_name"]}
        actors = []
        for a in u._scene.get_all_actors():
            entry = {"name": a.name, "type": a.type,
                     "p": [round(float(x), 3) for x in a.pose.p]}
            shapes = []
            for cs in a.get_collision_shapes():
                g = cs.geometry
                sd = {"geom": type(g).__name__}
                # world z bbox of this collision shape
                try:
                    verts = getattr(g, "vertices", None)
                    R = a.pose.to_transformation_matrix()[:3, :3]
                    if verts is not None:
                        w = (R @ np.asarray(verts).T).T + np.asarray(a.pose.p)
                        sd["world_z"] = [round(float(w[:, 2].min()), 3),
                                         round(float(w[:, 2].max()), 3)]
                    elif hasattr(g, "half_size"):
                        hs = np.asarray(g.half_size)
                        sd["half_size"] = [round(float(x), 3) for x in hs]
                        sd["world_z"] = [round(float(a.pose.p[2] - hs[2]), 3),
                                         round(float(a.pose.p[2] + hs[2]), 3)]
                except Exception as e:
                    sd["err"] = repr(e)
                shapes.append(sd)
            entry["collision"] = shapes
            actors.append(entry)
        info["n_actors"] = len(actors)
        info["actors"] = actors
        out[task] = info
    return out


@app.local_entrypoint()
def tableactor():
    import json
    print(json.dumps(probe_table_actor.remote(), indent=2, default=str))


# ---------------------------------------------------------------------------
# THE v1-TABLE FIX. The bridge_table_1_v1 arena's nonconvex collision mesh is
# mis-cooked by this sapien build -> no table-top collision at z~0.87, so
# spoon/carrot/stack objects fall through to z~-1.05. (v2/eggplant works.) Add
# an invisible static box at scene_table_height covering the workspace so
# objects rest on the visually-correct surface. Collision-only => invisible in
# the greenscreen-overlay render. Hook: the real defining class of _load_actors
# (runs before the per-episode object settle), guarded to v1 scenes only.
# ---------------------------------------------------------------------------
def _install_platform(half_xy=0.6, bottom_z=-0.2, center_xy=(-0.15, 0.0)):
    """Patch the class that ACTUALLY defines _load_actors for the put-on bridge
    envs. PutOnInSceneEnv overrides MoveNearInSceneEnv._load_actors and is NOT in
    MoveNearInSceneEnv.__mro__, so we must resolve the defining class from a real
    leaf env's MRO (a base-up search finds the shadowed base method). Idempotent.
    The platform is THICK (top at scene_table_height, bottom well below) so that
    contact depenetration can only push objects UP to the surface (a thin slab
    lets them tunnel out the bottom face in this CCD-less build)."""
    import sapien.core as sapien
    # import envs first so registration completes (avoids circular import), then
    # read the registry to get the leaf class WITHOUT building/resetting an env.
    import mani_skill2_real2sim.envs  # noqa: F401  (triggers full env registration)
    from mani_skill2_real2sim.utils.registration import REGISTERED_ENVS
    spec = REGISTERED_ENVS["PutSpoonOnTableClothInScene-v0"]
    leaf = getattr(spec, "cls", None) or getattr(spec, "env_class", None) or spec.cls
    defcls = None
    for c in leaf.__mro__:
        if "_load_actors" in c.__dict__:
            defcls = c
            break
    if defcls is None or getattr(defcls, "_platform_installed", False):
        return defcls.__name__ if defcls else "NONE"
    orig = defcls.__dict__["_load_actors"]

    def patched(self, *a, **k):
        orig(self, *a, **k)
        scene_name = getattr(self, "scene_name", "") or ""
        if "v1" not in scene_name:
            return  # v2 (eggplant) arena collides fine; leave untouched
        top_z = float(getattr(self, "scene_table_height", 0.87))
        half_z = (top_z - bottom_z) / 2.0
        center_z = bottom_z + half_z
        b = self._scene.create_actor_builder()
        b.add_box_collision(half_size=[half_xy, half_xy, half_z])
        plat = b.build_static(name="table_platform")
        plat.set_pose(sapien.Pose([center_xy[0], center_xy[1], center_z]))

    defcls._load_actors = patched
    defcls._platform_installed = True
    return defcls.__name__


@app.function(image=image, volumes={"/data": vol}, timeout=900)
def verify_platform2():
    """v1-table fix: spoon/carrot/stack objects should now rest at ~table height
    (~0.87), vary per episode, and render visibly. Default obj_init_z (=0.87)."""
    import os, numpy as np, hashlib
    from PIL import Image
    import sapien.core as sapien
    _force_offscreen()
    patched_cls = _install_platform()
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    os.makedirs("/data/debug_plat2", exist_ok=True)
    out = {"patched_class": patched_cls}
    INIT_Z = 0.95   # spawn objects ABOVE the platform top (0.87) so they drop onto it
    for task in ["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube"]:
        env, cfg = _build_official(task)
        rows = []
        box_rest = None
        for ep in [0, 1, 2, 4, 8, 16]:
            ret = _reset_official(env, cfg, ep, init_z=INIT_Z)
            obs = ret[0] if isinstance(ret, tuple) else ret
            u = env.unwrapped
            sn = getattr(u, "scene_name", "?")
            objs = u.episode_objs
            sz = round(float(objs[0].pose.p[2]), 4)
            tz = round(float(objs[1].pose.p[2]), 4)
            plat = [a for a in u._scene.get_all_actors() if a.name == "table_platform"]
            plat_z = round(float(plat[0].pose.p[2]), 4) if plat else None
            # DIAGNOSTIC (ep0 only): drop a FRESH dynamic box onto the platform at
            # the object xy. If it rests ~0.87, static-box collision works and the
            # object fall-through is object-specific; if it tunnels, the box
            # platform itself doesn't collide in this build.
            if ep == 0:
                ox, oy = float(objs[0].pose.p[0]), float(objs[0].pose.p[1])
                bb = u._scene.create_actor_builder()
                bb.add_box_collision(half_size=[0.02, 0.02, 0.02], density=300)
                tb = bb.build(name="probe_box")
                tb.set_pose(sapien.Pose([ox, oy, 1.05]))
                for _ in range(300):
                    u._scene.step()
                box_rest = round(float(tb.pose.p[2]), 4)
            img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
            if ep in (0, 8):
                Image.fromarray(img).save(f"/data/debug_plat2/{task}_ep{ep}.png")
            rows.append({"ep": ep, "src_z": sz, "tgt_z": tz,
                         "plat_present": bool(plat), "plat_z": plat_z,
                         "img_hash": hashlib.md5(img.tobytes()).hexdigest()[:8]})
        out[task] = {"scene_name": sn, "freshbox_rest_z": box_rest, "rows": rows,
                     "n_unique_img": len({r["img_hash"] for r in rows})}
    vol.commit()
    return out


@app.local_entrypoint()
def vplat2():
    import json
    print(json.dumps(verify_platform2.remote(), indent=2, default=str))


@app.function(image=image, timeout=600)
def dump_octo_src():
    """Dump SimplerEnv's Octo policy wrapper + main inference loop source so we
    can build a goal-image-conditioned variant + correct rollout/action logic."""
    import os, glob
    out = {}
    root = "/root/SimplerEnv"
    wanted = [
        "simpler_env/policies/octo/octo_model.py",
        "simpler_env/main_inference.py",
        "simpler_env/evaluation/maniskill2_evaluator.py",
        "simpler_env/utils/env/env_builder.py",
    ]
    for rel in wanted:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            with open(p) as f:
                out[rel] = f.read()
        else:
            # glob fallback
            hits = glob.glob(os.path.join(root, "**", os.path.basename(rel)), recursive=True)
            out[rel] = f"NOT FOUND; hits={hits}"
    return out


@app.local_entrypoint()
def octosrc():
    import json, os
    res = dump_octo_src.remote()
    os.makedirs("/tmp/octo_src", exist_ok=True)
    for rel, content in res.items():
        safe = rel.replace("/", "__")
        with open(f"/tmp/octo_src/{safe}", "w") as f:
            f.write(content)
        print(f"WROTE /tmp/octo_src/{safe}  ({len(content)} bytes)")


@app.function(image=image, timeout=300)
def probe_leafcls():
    """Find the leaf env class + the class that defines _load_actors, WITHOUT
    building an env (registry lookup only)."""
    out = {}
    try:
        from mani_skill2_real2sim.utils.registration import REGISTERED_ENVS
        out["registry_keys_sample"] = [k for k in REGISTERED_ENVS
                                       if "Spoon" in k or "Eggplant" in k][:5]
        spec = REGISTERED_ENVS["PutSpoonOnTableClothInScene-v0"]
        leaf = spec.cls
        out["leaf"] = leaf.__name__
        out["mro"] = [c.__name__ for c in leaf.__mro__]
        for c in leaf.__mro__:
            if "_load_actors" in c.__dict__:
                out["load_actors_defcls"] = c.__name__
                out["load_actors_module"] = c.__module__
                break
    except Exception as e:
        import traceback
        out["err"] = traceback.format_exc()[-1500:]
    return out


@app.local_entrypoint()
def leafcls():
    import json
    print(json.dumps(probe_leafcls.remote(), indent=2, default=str))


@app.function(image=image, volumes={"/data": vol}, timeout=300)
def fetch_debug(relpath: str):
    with open(f"/data/{relpath}", "rb") as f:
        return f.read()


@app.local_entrypoint()
def getdbg():
    import os
    os.makedirs("/tmp/simpler_dbg", exist_ok=True)
    for rp in ["debug_plat2/widowx_spoon_on_towel_ep0.png",
               "debug_plat2/widowx_spoon_on_towel_ep8.png",
               "debug_plat2/widowx_stack_cube_ep0.png"]:
        data = fetch_debug.remote(rp)
        out = f"/tmp/simpler_dbg/{os.path.basename(rp)}"
        with open(out, "wb") as f:
            f.write(data)
        print("WROTE", out, len(data))
