"""
Environment research for rebuilding SIMPLER on a known-good SAPIEN.

The PyPI sapien==2.2.2 wheel on nvidia/cuda:12.8.1-devel-ubuntu24.04 has shown 4
independent broken-ABI defects (pinocchio FK segfault, get_qpos garbage, Pose
quaternion math, bridge-object collision). Goal: find the exact env SimplerEnv
authors use, and gather ABI-relevant facts (glibc / libstdc++ / sapien build),
to design a base image where sapien's compiled code behaves correctly.
"""
import modal

app = modal.App("simpler-env-research")

# the CURRENT (broken) image, to introspect itself + dump SimplerEnv setup files
CUDA_TAG = "12.8.1-devel-ubuntu24.04"
cur = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install("git", "wget")
    .run_commands("python -m pip install --upgrade pip 'setuptools<70' wheel")
    .pip_install("numpy==1.24.4")
    .pip_install("sapien==2.2.2")
    .run_commands(
        "git clone https://github.com/simpler-env/SimplerEnv /root/SimplerEnv --recurse-submodules",
    )
)


@app.function(image=cur, timeout=900)
def introspect():
    import os, glob, subprocess, sys, json
    out = {}
    out["python"] = sys.version
    # glibc
    try:
        out["ldd_version"] = subprocess.run(["ldd", "--version"], capture_output=True,
                                            text=True).stdout.splitlines()[0]
    except Exception as e:
        out["ldd_version"] = repr(e)
    # libstdc++ GLIBCXX versions available
    libcxx = glob.glob("/usr/lib/x86_64-linux-gnu/libstdc++.so*") + \
        glob.glob("/lib/x86_64-linux-gnu/libstdc++.so*")
    out["system_libstdcxx"] = libcxx
    try:
        st = subprocess.run(f"strings {libcxx[0]} | grep GLIBCXX_3.4 | tail -5",
                            shell=True, capture_output=True, text=True).stdout
        out["system_GLIBCXX_top"] = st.strip().splitlines()
    except Exception as e:
        out["system_GLIBCXX_top"] = repr(e)
    # sapien install + its bundled .so and their lib deps
    import sapien
    sp = os.path.dirname(sapien.__file__)
    out["sapien_version"] = getattr(sapien, "__version__", "?")
    out["sapien_dir"] = sp
    sos = glob.glob(os.path.join(sp, "**", "*.so*"), recursive=True)
    out["sapien_sos"] = [os.path.relpath(s, sp) for s in sos][:40]
    # does the core pybind so bundle/link libstdc++? check ldd of the main .so
    core = [s for s in sos if "pysapien" in s or "core" in s.lower()]
    if core:
        try:
            ld = subprocess.run(["ldd", core[0]], capture_output=True, text=True).stdout
            out["pysapien_ldd"] = ld[:2000]
            out["pysapien_so"] = os.path.relpath(core[0], sp)
        except Exception as e:
            out["pysapien_ldd"] = repr(e)
    # bundled libstdc++ inside sapien?
    out["sapien_bundled_libstdcxx"] = [os.path.relpath(s, sp) for s in sos
                                       if "stdc++" in s]
    return out


@app.function(image=cur, timeout=900)
def dump_simpler_setup():
    """Dump SimplerEnv's own install instructions / requirements / any Dockerfile."""
    import os, glob
    out = {}
    root = "/root/SimplerEnv"
    # README install section
    for name in ["README.md", "README.rst"]:
        p = os.path.join(root, name)
        if os.path.isfile(p):
            with open(p) as f:
                out[name] = f.read()[:8000]
    # requirements / env / setup / Dockerfile anywhere
    for pat in ["requirements*.txt", "environment*.yml", "setup.py", "setup.cfg",
                "pyproject.toml", "Dockerfile*", "*.dockerfile"]:
        for f in glob.glob(os.path.join(root, "**", pat), recursive=True):
            try:
                with open(f) as fh:
                    out[os.path.relpath(f, root)] = fh.read()[:4000]
            except Exception as e:
                out[os.path.relpath(f, root)] = repr(e)
    return out


# ---------------------------------------------------------------------------
# Candidate KNOWN-GOOD image: Ubuntu 20.04 (glibc 2.31) matching the official
# ManiSkill2_real2sim Dockerfile, with lavapipe (kisak-mesa) for headless render.
# ---------------------------------------------------------------------------
good = (
    modal.Image.from_registry("nvidia/cuda:11.3.1-devel-ubuntu20.04", add_python="3.10")
    # ubuntu20.04 base: tzdata prompts interactively and hangs the build -> force
    # noninteractive + preset UTC before any apt.
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .run_commands(
        "ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo Etc/UTC > /etc/timezone",
    )
    # focal stock Mesa is 21.2.6 -> lavapipe Vulkan 1.1, lacks external_semaphore_fd
    # that svulkan2 needs. oibaf PPA ships newer Mesa BUILT FOR focal (keeps glibc
    # 2.31 so the sapien ABI stays intact) -> newer lavapipe with the fd extensions.
    .apt_install("software-properties-common", "gpg-agent", "ca-certificates")
    .run_commands(
        "add-apt-repository -y ppa:oibaf/graphics-drivers",
        "apt-get update",
    )
    .apt_install(
        "git", "wget", "bzip2", "libglib2.0-0", "libgl1-mesa-glx", "libglx-mesa0",
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
        # focal lavapipe ICD is named lvp_icd.x86_64.json (not lvp_icd.json)
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
        "XDG_RUNTIME_DIR": "/tmp",
        "DISPLAY": "",
    })
)

CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


@app.function(image=good, timeout=600, gpu="T4")
def validate_phys_norender():
    """Test the broken-ABI symptoms WITHOUT a renderer (no Vulkan needed): the
    Pose-math quaternion bug + a headless physics box-drop. If these are fixed on
    20.04/glibc2.31, the ABI is good and render is the only remaining problem."""
    import sys
    def mark(msg):
        print("PHYSVAL>> " + msg, flush=True)
    mark("start import numpy")
    import numpy as np
    mark("import sapien.core")
    import sapien.core as sapien
    out = {}
    mark("sapien imported ok")

    # (1) Pose math: cur*delta*cur.inv() must be UNIT (was norm 4.0 on 24.04)
    cur = sapien.Pose(p=[0.25, -0.07, 0.22])
    delta = sapien.Pose([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    bracket = cur * delta * cur.inv()
    out["pose_bracket_qnorm"] = round(float(np.linalg.norm(bracket.q)), 5)
    out["pose_math_ok"] = bool(abs(np.linalg.norm(bracket.q) - 1.0) < 1e-3)
    # accumulate like the controller (was geometric explosion on 24.04)
    prev = sapien.Pose(p=[0.25, -0.07, 0.22], q=[1.0, 0.0, 0.0, 0.0])
    for _ in range(4):
        prev = bracket * prev
    out["accum4_p"] = [round(float(x), 3) for x in prev.p]
    out["accum4_qnorm"] = round(float(np.linalg.norm(prev.q)), 5)
    mark("pose math done qnorm=%s" % out["pose_bracket_qnorm"])

    # (2) headless physics (no renderer): drop a box on a ground, must rest
    mark("create Engine")
    engine = sapien.Engine()
    mark("create_scene")
    scene = engine.create_scene()
    scene.set_timestep(1.0 / 500.0)
    mark("add_ground")
    scene.add_ground(0.0)
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=[0.05, 0.05, 0.05])
    box = b.build(name="box")
    box.set_pose(sapien.Pose([0, 0, 1.0]))
    mark("stepping physics 400")
    for _ in range(400):
        scene.step()
    out["box_rest_z"] = round(float(box.pose.p[2]), 4)
    out["box_collision_ok"] = bool(0.02 < box.pose.p[2] < 0.08)
    mark("physics done box_rest_z=%s" % out["box_rest_z"])

    # (3) get_qpos on a real articulation (was constant garbage on 24.04)
    mark("articulation test start")
    try:
        import os, glob
        from mani_skill2_real2sim import PACKAGE_ASSET_DIR
        urdfs = glob.glob(os.path.join(str(PACKAGE_ASSET_DIR), "**", "*widowx*", "**", "*.urdf"), recursive=True) \
            or glob.glob(os.path.join(str(PACKAGE_ASSET_DIR), "**", "*.urdf"), recursive=True)
        out["urdf_found"] = urdfs[:3]
        mark("load urdf %s" % (urdfs[0] if urdfs else "NONE"))
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(urdfs[0])
        qset = np.zeros(robot.dof); qset[:min(6, robot.dof)] = [0.1, -0.3, 0.2, 0.0, 0.4, -0.1][:min(6, robot.dof)]
        robot.set_qpos(qset)
        mark("get_qpos")
        q = np.asarray(robot.get_qpos(), dtype=float)
        out["get_qpos"] = [round(float(x), 3) for x in q[:8]]
        out["get_qpos_matches_set"] = bool(np.allclose(q, qset, atol=1e-3))
        # pinocchio FK (segfaulted on 24.04)
        mark("create_pinocchio_model")
        pm = robot.create_pinocchio_model()
        mark("compute_forward_kinematics")
        pm.compute_forward_kinematics(q)
        out["pinocchio_fk_ok"] = True
        mark("pinocchio FK done")
    except Exception as e:
        import traceback
        out["articulation_test_err"] = traceback.format_exc()[-1500:]
    mark("RETURNING")
    return out


@app.local_entrypoint()
def physval():
    import json
    print(json.dumps(validate_phys_norender.remote(), indent=2, default=str))


@app.function(image=good, timeout=600)
def probe_vulkan_2004():
    """What Vulkan/lavapipe is actually available on the 20.04 image? (no sapien
    renderer -> no segfault). Find the ICD, Mesa version, lavapipe lib."""
    import os, glob, subprocess
    out = {}
    for d in ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"]:
        out[f"ls {d}"] = (sorted(os.listdir(d)) if os.path.isdir(d) else "MISSING")
    # cat any lvp/lavapipe ICD
    for f in glob.glob("/usr/share/vulkan/icd.d/*lvp*") + glob.glob("/usr/share/vulkan/icd.d/*"):
        try:
            with open(f) as fh:
                out[f"cat {os.path.basename(f)}"] = fh.read()[:400]
        except Exception as e:
            out[f"cat {os.path.basename(f)}"] = repr(e)
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=60).stdout[:1500]
        except Exception as e:
            return repr(e)
    out["apt_policy_mesa_vulkan"] = sh("apt-cache policy mesa-vulkan-drivers 2>&1")
    out["find_lvp_so"] = sh("find / -iname '*lvp*' 2>/dev/null; find / -iname '*lavapipe*' 2>/dev/null")
    out["ldconfig_vulkan"] = sh("ldconfig -p | grep -i 'vulkan\\|lvp' | head")
    out["vk_device"] = sh("VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json vulkaninfo 2>&1 | grep -iE 'deviceName|apiVersion|driverName|deviceType' | head -20")
    out["vk_fd_ext"] = sh("VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json vulkaninfo 2>&1 | grep -iE 'external_semaphore_fd|external_fence_fd' | head")
    out["glxinfo_mesa"] = sh("apt-cache policy libgl1-mesa-dri 2>&1 | head -4")
    return out


@app.local_entrypoint()
def vulkan():
    import json
    print(json.dumps(probe_vulkan_2004.remote(), indent=2, default=str))


@app.function(image=good, timeout=1800)
def validate_2004():
    """On Ubuntu 20.04: do the 4 broken-ABI defects disappear? Test render,
    object collision (spoon rests), get_qpos (non-garbage), Pose math (unit quat)."""
    import numpy as np
    import sapien.core as sapien
    _orig = sapien.SapienRenderer
    sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
    out = {}
    from simpler_env.utils.env.env_builder import build_maniskill2_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    from mani_skill2_real2sim import ASSET_DIR

    env = build_maniskill2_env(
        "PutSpoonOnTableClothInScene-v0", obs_mode="rgbd", robot="widowx",
        sim_freq=500, control_mode=CONTROL_MODE, control_freq=5,
        max_episode_steps=60, scene_name="bridge_table_1_v1",
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=str(ASSET_DIR / "real_inpainting/bridge_real_eval_1.png"))
    obs, _ = env.reset(options={
        "robot_init_options": {"init_xy": np.array([0.147, 0.028]),
                               "init_rot_quat": [0, 0, 0, 1]},
        "obj_init_options": {"episode_id": 0}})
    u = env.unwrapped

    # (render)
    img = np.asarray(get_image_from_maniskill2_obs_dict(env, obs)).astype("uint8")
    out["render_img_shape"] = list(img.shape)

    # (object collision) spoon should rest on the table ~0.87, NOT fall to -1.05
    out["spoon_obj_z"] = round(float(u.episode_objs[0].pose.p[2]), 4)
    out["towel_obj_z"] = round(float(u.episode_objs[1].pose.p[2]), 4)
    out["objects_rest_on_table"] = bool(u.episode_objs[0].pose.p[2] > 0.5)

    # (get_qpos) should be varied, not constant garbage
    q = np.asarray(u.agent.robot.get_qpos(), dtype=float)
    out["get_qpos"] = [round(float(x), 4) for x in q]
    out["get_qpos_is_garbage"] = bool(np.allclose(q, q[0]))

    # (Pose math) cur * delta * cur.inv() must be a UNIT quaternion
    cur = sapien.Pose(p=[0.25, -0.07, 0.22])
    delta = sapien.Pose([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    bracket = cur * delta * cur.inv()
    out["pose_bracket_qnorm"] = round(float(np.linalg.norm(bracket.q)), 5)
    out["pose_math_ok"] = bool(abs(np.linalg.norm(bracket.q) - 1.0) < 1e-3)
    return out


@app.function(image=good, timeout=900)
def validate_pin_2004():
    """Does SAPIEN's BUNDLED pinocchio FK segfault on 20.04? (separate fn: a
    segfault crashes the worker, so isolate it from validate_2004)."""
    import numpy as np
    import sapien.core as sapien
    sapien.SapienRenderer = (lambda f: lambda *a, **k: f(*a, **{**k, "offscreen_only": True}))(sapien.SapienRenderer)
    import simpler_env
    env = simpler_env.make("widowx_put_eggplant_in_basket")
    env.reset()
    robot = env.unwrapped.agent.robot
    pm = robot.create_pinocchio_model()
    q = np.asarray(robot.get_qpos(), dtype=float)
    pm.compute_forward_kinematics(q)  # this segfaulted on 24.04
    pose = pm.get_link_pose(robot.get_links().__len__() - 1)
    return {"pinocchio_fk_ok": True,
            "ee_pose_p": [round(float(x), 4) for x in pose.p]}


@app.local_entrypoint()
def validate():
    import json
    print("===== VALIDATE on Ubuntu 20.04 =====")
    print(json.dumps(validate_2004.remote(), indent=2, default=str))


@app.local_entrypoint()
def validate_pin():
    import json
    try:
        print(json.dumps(validate_pin_2004.remote(), indent=2, default=str))
    except Exception as e:
        print("pinocchio FK FAILED/segfault:", repr(e))


@app.local_entrypoint()
def main():
    import json
    print("===== INTROSPECT (current broken env) =====")
    print(json.dumps(introspect.remote(), indent=2, default=str))


@app.local_entrypoint()
def setup():
    import json
    d = dump_simpler_setup.remote()
    for k, v in d.items():
        print(f"\n========== {k} ==========")
        print(v)
