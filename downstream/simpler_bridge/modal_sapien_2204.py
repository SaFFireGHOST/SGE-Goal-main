"""
Test SAPIEN pinocchio + render on Ubuntu 22.04 (glibc 2.35) instead of 24.04.

On 24.04 (glibc 2.39) SAPIEN 2.2.2's bundled pinocchio compute_forward_kinematics
segfaults on BOTH cpu and gpu hosts (render itself is fine). SAPIEN 2.2.2 (2023)
predates glibc 2.39; 22.04 (glibc 2.35) is much closer to its build era. kisak-mesa
PPA still provides a recent lavapipe with the fd external semaphore/fence
extensions svulkan2 needs.

Run:  modal run simpler-eval/modal_sapien_2204.py::fk
"""
import modal

app = modal.App("sapien-2204")

CUDA_TAG = "12.8.1-devel-ubuntu22.04"

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
    elif mode == "step":
        env.step(env.action_space.sample() * 0.0)
        print("STEP_OK", flush=True)
    print("RESULT_OK", mode, flush=True)
except Exception:
    print("RESULT_EXC", mode, traceback.format_exc()[-1500:], flush=True)
'''


def _run_fk_modes():
    import os, subprocess
    os.makedirs("/tmp/fk", exist_ok=True)
    with open("/tmp/fk/child.py", "w") as f:
        f.write(_FK_CHILD)
    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
    env["MS2_RENDERER_LOG_LEVEL"] = "warn"
    out = {}
    out["glibc"] = subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    results = {}
    for mode in ["pin_f32", "step"]:
        r = subprocess.run(["python", "/tmp/fk/child.py", mode],
                           capture_output=True, text=True, timeout=400, env=env)
        verdict = "SEGFAULT" if r.returncode in (-11, 139) else (
            "OK" if "RESULT_OK" in (r.stdout or "") else "EXC/other")
        results[mode] = {"rc": r.returncode, "verdict": verdict,
                         "tail": (r.stdout or "")[-800:] + "||" + (r.stderr or "")[-300:]}
    out["fk"] = results
    return out


@app.function(image=image, timeout=1200)
def fk_probe():
    return _run_fk_modes()


@app.local_entrypoint()
def fk():
    import json
    print(json.dumps(fk_probe.remote(), indent=2, default=str))
