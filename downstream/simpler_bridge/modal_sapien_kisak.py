"""
SAPIEN-on-lavapipe, attempt with NEWER Mesa (kisak PPA) + validation layer diag.

Ubuntu 24.04's stock mesa-vulkan-drivers lavapipe lacks the device-level
VK_KHR_external_semaphore_fd / VK_KHR_external_fence_fd extensions that SAPIEN's
svulkan2 requests at vkCreateDevice -> ErrorExtensionNotPresent. Newer Mesa
(>=24.2, from ppa:kisak/kisak-mesa) adds fd external semaphore/fence to lavapipe.

This run: (a) upgrades Mesa, (b) dumps the FULL lavapipe device-extension list,
(c) enables the Khronos validation layer + loader debug so the exact missing
extension name is printed if device creation still fails, (d) tries the real
SIMPLER bridge env.

Run:  modal run simpler-eval/modal_sapien_kisak.py
"""
import modal

app = modal.App("sapien-kisak")

CUDA_TAG = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install("software-properties-common", "gpg-agent", "ca-certificates")
    # Newer Mesa (lavapipe with external_semaphore_fd / external_fence_fd)
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
        "vulkan-validationlayers",  # Khronos validation -> names missing extension
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
    # SAPIEN 2.2.2's bundled pinocchio compute_forward_kinematics segfaults on
    # Modal -> use standalone pinocchio as the kinematics backend instead.
    .pip_install("pin")
    .env({
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json",
        "XDG_RUNTIME_DIR": "/tmp",
        "DISPLAY": "",
    })
)


def _p(*a):
    """flushed print so the last line survives a SIGSEGV"""
    print(*a, flush=True)


@app.function(image=image, timeout=1800)
def diag():
    import os, sys, subprocess, traceback, faulthandler
    faulthandler.enable()  # dump python stack on SIGSEGV

    def sh(cmd):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
            return (r.stdout + r.stderr).strip()
        except Exception as e:
            return f"ERR {e!r}"

    _p("=== STEP0 mesa/vulkan ===")
    _p(sh("vulkaninfo 2>/dev/null | grep -iE 'driverName|deviceName|driverInfo' | head -4"))
    _p("=== fd exts ===")
    _p(sh("vulkaninfo 2>/dev/null | grep -iE 'external_semaphore|external_fence' | sort -u"))

    os.environ["MS2_RENDERER_LOG_LEVEL"] = "info"

    _p("=== STEP1 import sapien ===")
    import sapien.core as sapien
    _p("sapien", getattr(sapien, "__version__", "?"))

    _p("=== STEP2 SapienRenderer(offscreen_only=True) ===")
    r = sapien.SapienRenderer(offscreen_only=True)
    _p("renderer OK", type(r))

    _p("=== STEP3 minimal sapien scene + offscreen camera render ===")
    try:
        engine = sapien.Engine()
        engine.set_renderer(r)
        scene = engine.create_scene()
        scene.set_timestep(1 / 100)
        scene.add_ground(0.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        cam = scene.add_camera("cam", 128, 128, 1.0, 0.01, 10.0)
        import numpy as np
        cam.set_pose(sapien.Pose([-1, 0, 0.5]))
        scene.step()
        scene.update_render()
        cam.take_picture()
        rgba = cam.get_float_texture("Color")  # HxWx4
        _p("minimal render OK shape", rgba.shape, "mean", float(np.asarray(rgba).mean()))
    except Exception:
        _p("minimal render FAILED\n" + traceback.format_exc()[-2000:])

    _p("=== STEP4 simpler_env.make ===")
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    task = "widowx_put_eggplant_in_basket"
    env = simpler_env.make(task)
    _p("make OK")

    # inspect shader/render config the bridge env uses (RT vs rasterization)
    try:
        u = env.unwrapped
        _p("shader_dir:", getattr(u, "shader_dir", "?"),
           "| render_config:", {k: getattr(sapien.render_config, k, None)
                                for k in ("camera_shader_dir", "viewer_shader_dir",
                                          "rt_samples_per_pixel", "rt_use_denoiser")})
    except Exception:
        _p("shader inspect err\n" + traceback.format_exc()[-800:])

    _p("=== STEP5 env.reset() (render happens here) ===")
    reset_ret = env.reset()
    _p("reset OK")
    obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

    _p("=== STEP6 get image ===")
    import numpy as np
    img = get_image_from_maniskill2_obs_dict(env, obs)
    _p("image shape", getattr(img, "shape", None), "min/max/mean",
       int(np.min(img)), int(np.max(img)), float(np.mean(img)))
    instruction = env.unwrapped.get_language_instruction()
    _p("instruction:", instruction)

    _p("=== STEP7 step x5 ===")
    last_info = {}
    for _ in range(5):
        step_ret = env.step(env.action_space.sample() * 0.0)
        last_info = step_ret[-1]
    _p("info_keys", list(last_info.keys()), "success", last_info.get("success", "MISSING"))
    _p("=== SIMPLER OK ===")
    return "SIMPLER OK"


# --- Forward-kinematics segfault isolation -------------------------------------
# env.step() -> controller compute_fk -> SAPIEN pinocchio binding segfaults.
# Run several FK strategies in ISOLATED subprocesses so one segfault (rc -11)
# doesn't kill the others; report which succeed.

_FK_CHILD = r'''
import sys, traceback, numpy as np
import sapien.core as sapien
_orig = sapien.SapienRenderer
sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
import simpler_env
env = simpler_env.make("widowx_put_eggplant_in_basket")
env.reset()
robot = env.unwrapped.agent.robot
qpos = robot.get_qpos()
print("QPOS dtype", qpos.dtype, "shape", qpos.shape, flush=True)
mode = sys.argv[1]
try:
    if mode == "pin_f32":
        pm = robot.create_pinocchio_model()
        pm.compute_forward_kinematics(np.asarray(qpos))
        print("LINKPOSE", pm.get_link_pose(0), flush=True)
    elif mode == "pin_f64":
        pm = robot.create_pinocchio_model()
        pm.compute_forward_kinematics(np.ascontiguousarray(qpos, dtype=np.float64))
        print("LINKPOSE", pm.get_link_pose(0), flush=True)
    elif mode == "native":
        # SAPIEN-native FK: read live link poses (no pinocchio)
        links = robot.get_links()
        print("NLINKS", len(links), "ee", links[-1].get_pose(), flush=True)
    elif mode == "getpose_only":
        # does the model deref-crash WITHOUT compute_forward_kinematics?
        pm = robot.create_pinocchio_model()
        print("model created", flush=True)
        print("LINKPOSE", pm.get_link_pose(0), flush=True)
    elif mode == "fresh64":
        pm = robot.create_pinocchio_model()
        q = np.zeros(robot.dof, dtype=np.float64)
        pm.compute_forward_kinematics(q)
        print("LINKPOSE", pm.get_link_pose(0), flush=True)
    elif mode.startswith("size:"):
        n = int(mode.split(":")[1])
        pm = robot.create_pinocchio_model()
        q = np.zeros(n, dtype=np.float64)
        pm.compute_forward_kinematics(q)
        print("SIZE_OK", n, "ee", pm.get_link_pose(13), flush=True)
    elif mode == "step":
        import numpy as np
        env.step(env.action_space.sample() * 0.0)
        print("STEP_OK", flush=True)
    print("RESULT_OK", mode, flush=True)
except Exception:
    print("RESULT_EXC", mode, traceback.format_exc()[-1500:], flush=True)
'''


def _run_fk_modes(modes=("native", "pin_f32", "pin_f64", "step"), extra_env=None):
    import os, subprocess
    os.makedirs("/tmp/fk", exist_ok=True)
    with open("/tmp/fk/child.py", "w") as f:
        f.write(_FK_CHILD)
    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
    env["MS2_RENDERER_LOG_LEVEL"] = "warn"
    if extra_env:
        env.update(extra_env)
    results = {}
    for mode in modes:
        r = subprocess.run(["python", "/tmp/fk/child.py", mode],
                           capture_output=True, text=True, timeout=300, env=env)
        tail = (r.stdout or "")[-1200:] + "||STDERR||" + (r.stderr or "")[-400:]
        verdict = "SEGFAULT" if r.returncode in (-11, 139) else (
            "OK" if "RESULT_OK" in (r.stdout or "") else "EXC/other")
        results[mode] = {"rc": r.returncode, "verdict": verdict, "tail": tail}
    return results


@app.function(image=image, timeout=1200)
def fk_probe():
    return _run_fk_modes()


@app.function(image=image, gpu="L4", timeout=1200)
def fk_probe_gpu():
    # Hypothesis: SAPIEN's bundled pinocchio segfaults on Modal's CPU-only host
    # CPU class (AVX/Eigen). GPU containers run on a newer host CPU; pinocchio
    # may work there. Render still forced onto lavapipe via VK_ICD_FILENAMES.
    return _run_fk_modes()


@app.function(image=image, cpu=2, timeout=1500)
def fk_probe_size():
    # Scan qpos sizes to discover pinocchio's expected nq (continuous joints make
    # nq>dof; passing dof over-reads -> segfault). The size that doesn't crash = nq.
    return _run_fk_modes(modes=tuple(f"size:{n}" for n in
                                     (8, 9, 10, 11, 12, 13, 14, 15, 16, 20)))


@app.function(image=image, cpu=4, timeout=1500)
def fk_probe_omp():
    # Probe model-validity + threading hypotheses:
    #  getpose_only -> does the model deref-crash even WITHOUT compute_fk?
    #  fresh64      -> fresh correctly-sized float64 array
    #  *_omp        -> single-threaded Eigen/OpenMP/BLAS (container reports many cpus)
    out = {}
    out["default"] = _run_fk_modes(modes=("getpose_only", "fresh64", "pin_f32", "step"))
    out["omp1"] = _run_fk_modes(
        modes=("getpose_only", "pin_f32", "step"),
        extra_env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                   "EIGEN_DONT_PARALLELIZE": "1"})
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(diag.remote(), indent=2, default=str))


@app.local_entrypoint()
def fk():
    import json
    print(json.dumps(fk_probe.remote(), indent=2, default=str))


@app.local_entrypoint()
def fkgpu():
    import json
    print(json.dumps(fk_probe_gpu.remote(), indent=2, default=str))


@app.local_entrypoint()
def fkomp():
    import json
    print(json.dumps(fk_probe_omp.remote(), indent=2, default=str))


@app.local_entrypoint()
def fksize():
    import json
    print(json.dumps(fk_probe_size.remote(), indent=2, default=str))


# --- Pinocchio model inspector (does NOT call the segfaulting FK) --------------
_PIN_INSPECT = r'''
import os, glob, subprocess, numpy as np
import sapien.core as sapien
_orig = sapien.SapienRenderer
sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
import simpler_env
env = simpler_env.make("widowx_put_eggplant_in_basket")
env.reset()
robot = env.unwrapped.agent.robot
print("ARTIC dof", robot.dof, "qpos", robot.get_qpos().shape, flush=True)
pm = robot.create_pinocchio_model()
print("PM type", type(pm), flush=True)
print("PM attrs", [a for a in dir(pm) if not a.startswith("_")], flush=True)
for m in ("dof","nq","nv"):
    try:
        v = getattr(pm, m)
        print("PM", m, v() if callable(v) else v, flush=True)
    except Exception as e:
        print("PM", m, "ERR", repr(e), flush=True)
# how many links/joints does pinocchio think there are
for m in ("get_link_names","get_joint_names"):
    try:
        f = getattr(pm, m); print("PM", m, f(), flush=True)
    except Exception as e:
        print("PM", m, "ERR", repr(e), flush=True)
sapd = os.path.dirname(sapien.__file__)
sos = glob.glob(sapd + "/**/*.so*", recursive=True)
print("SAPIEN SOS", [os.path.basename(s) for s in sos], flush=True)
# ldd the core sapien pyd/so for missing/odd deps
core = [s for s in sos if "pysapien" in s.lower() or "core" in s.lower()]
for c in core[:2]:
    r = subprocess.run(["ldd", c], capture_output=True, text=True)
    miss = [l for l in (r.stdout+r.stderr).splitlines() if "not found" in l]
    print("LDD", os.path.basename(c), "MISSING", miss, flush=True)
print("INSPECT_DONE", flush=True)
'''


@app.function(image=image, timeout=900)
def pin_inspect():
    import os, subprocess
    os.makedirs("/tmp/pin", exist_ok=True)
    with open("/tmp/pin/child.py", "w") as f:
        f.write(_PIN_INSPECT)
    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
    env["MS2_RENDERER_LOG_LEVEL"] = "warn"
    r = subprocess.run(["python", "/tmp/pin/child.py"],
                       capture_output=True, text=True, timeout=400, env=env)
    return {"rc": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-1000:]}


@app.local_entrypoint()
def pin():
    import json
    print(json.dumps(pin_inspect.remote(), indent=2, default=str))


# --- Validate STANDALONE pinocchio FK vs SAPIEN-native link poses -------------
_PIN_VALIDATE = r'''
import numpy as np, glob, traceback
import sapien.core as sapien
_orig = sapien.SapienRenderer
sapien.SapienRenderer = lambda *a, **k: _orig(*a, **{**k, "offscreen_only": True})
import simpler_env
env = simpler_env.make("widowx_put_eggplant_in_basket")
env.reset()
robot = env.unwrapped.agent.robot

links = robot.get_links()
names = [l.get_name() for l in links]
ee_link = links[names.index("ee_gripper_link")]
root_link = links[0]
ajoints = robot.get_active_joints()
ajoint_names = [j.get_name() for j in ajoints]
dof = robot.dof
print("ROOT_LINK", names[0], "DOF", dof, flush=True)
print("ACTIVE_JOINTS", ajoint_names, flush=True)

import pinocchio as pin
print("PIN_VERSION", pin.__version__, flush=True)
urdf = glob.glob("/root/SimplerEnv/**/wx250s.urdf", recursive=True)[0]
model = pin.buildModelFromUrdf(urdf)
data = model.createData()
print("PIN_NQ", model.nq, "NV", model.nv, "NJOINTS", model.njoints, flush=True)
print("PIN_JOINTS", [model.names[i] for i in range(model.njoints)], flush=True)
fid = model.getFrameId("ee_gripper_link")

qmap = []
for s_idx, jn in enumerate(ajoint_names):
    if model.existJointName(jn):
        jid = model.getJointId(jn); qmap.append((s_idx, model.idx_qs[jid]))

def pin_fk(qfull):
    q = pin.neutral(model)
    for s_idx, qi in qmap:
        q[qi] = qfull[s_idx]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[fid]
    quat = pin.Quaternion(oMf.rotation)
    return np.array(oMf.translation), np.array([quat.w, quat.x, quat.y, quat.z])

# CONTROLLED configs: set qpos ourselves, read live poses (ground truth), compare
test_qs = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015, 0.015]),
    np.array([0.1, -0.3, 0.2, 0.0, 0.4, -0.1, 0.02, 0.02]),
    np.array([-0.4, 0.5, -0.2, 0.3, -0.3, 0.6, 0.01, 0.01]),
]
for ti, qt in enumerate(test_qs):
    robot.set_qpos(qt)
    got = np.asarray(robot.get_qpos(), dtype=np.float64)
    ee_world = ee_link.get_pose(); root_world = root_link.get_pose()
    ee_at_base = root_world.inv() * ee_world
    np_, nq_ = np.array(list(ee_at_base.p)), np.array(list(ee_at_base.q))
    pp_, pq_ = pin_fk(qt)
    dp = np.linalg.norm(np_ - pp_)
    dq = min(np.linalg.norm(nq_ - pq_), np.linalg.norm(nq_ + pq_))
    print(f"TEST{ti} setqpos_ok={np.allclose(got, qt, atol=1e-4)} "
          f"NATIVE_P={np_.round(4).tolist()} PIN_P={pp_.round(4).tolist()} "
          f"POS_DIFF={dp:.5f} QUAT_DIFF={dq:.5f}", flush=True)
print("VALIDATE_DONE", flush=True)
'''


@app.function(image=image, timeout=900)
def pin_validate():
    import os, subprocess
    os.makedirs("/tmp/pv", exist_ok=True)
    with open("/tmp/pv/child.py", "w") as f:
        f.write(_PIN_VALIDATE)
    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
    env["MS2_RENDERER_LOG_LEVEL"] = "warn"
    r = subprocess.run(["python", "/tmp/pv/child.py"],
                       capture_output=True, text=True, timeout=400, env=env)
    return {"rc": r.returncode, "stdout": r.stdout[-3500:], "stderr": r.stderr[-1000:]}


@app.local_entrypoint()
def pv():
    import json
    print(json.dumps(pin_validate.remote(), indent=2, default=str))
