"""
SIMPLER-on-Modal smoke test (v2).

Proves SAPIEN/Vulkan headless rendering + a WidowX/Bridge task work in a Modal
GPU container BEFORE we invest in Octo + goal-image generation.

ManiSkill2_real2sim is a git SUBMODULE of SimplerEnv (NOT a PyPI package), so we
install it from source.

Run:  modal run simpler-eval/modal_simpler_smoke.py
"""
import modal

app = modal.App("simpler-smoke")

CUDA_TAG = "12.8.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install(
        "git", "wget", "bzip2", "libglib2.0-0", "libgl1", "libglx-mesa0",
        "libvulkan1", "vulkan-tools", "libvulkan-dev",
        "libegl1", "libgles2", "libglvnd-dev", "libosmesa6-dev", "ffmpeg",
        "build-essential", "g++", "cmake",
        # lavapipe = Mesa CPU Vulkan (Modal exposes GPU for compute only, not graphics)
        "mesa-vulkan-drivers", "libgl1-mesa-dri",
        # X11 client libs SAPIEN/Vulkan link against
        "libx11-6", "libxext6", "libxrender1", "libxi6", "libxfixes3",
        "libxxf86vm1", "libxrandr2", "libxinerama1", "libxcursor1",
        "libxcb1", "libxau6", "libxdmcp6", "libsm6", "libice6", "libxkbcommon0",
    )
    .env({"CC": "gcc", "CXX": "g++"})
    # NVIDIA Vulkan ICD so SAPIEN finds the GPU for offscreen rendering
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d",
        'printf \'{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}\' > /usr/share/vulkan/icd.d/nvidia_icd.json',
    )
    .run_commands("python -m pip install --upgrade pip 'setuptools<70' wheel")
    .pip_install("numpy==1.24.4", "scipy", "pillow", "opencv-python-headless", "mediapy")
    # SAPIEN 2.x line used by ManiSkill2_real2sim / SIMPLER
    .pip_install("sapien==2.2.2")
    # Prebuilt ruckig wheel so maniskill's editable install doesn't try to
    # compile it from source (which fails without clang in the CUDA image).
    .pip_install("ruckig")
    # Clone SimplerEnv + its ManiSkill2_real2sim submodule, install from source.
    .run_commands(
        "git clone https://github.com/simpler-env/SimplerEnv /root/SimplerEnv --recurse-submodules",
        # install the real2sim maniskill fork (pulls gymnasium, transforms3d, trimesh, ...)
        "cd /root/SimplerEnv/ManiSkill2_real2sim && python -m pip install -e .",
        # install the simpler_env wrapper itself WITHOUT its policy deps (tf/jax)
        "cd /root/SimplerEnv && python -m pip install -e . --no-deps",
    )
    .env({
        # Force Mesa lavapipe (CPU Vulkan); Modal GPU is compute-only (no graphics).
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "XDG_RUNTIME_DIR": "/tmp",
        "DISPLAY": "",
    })
)


# NOTE: no gpu= here on purpose. SAPIEN's svulkan2 renderer, when it sees a CUDA
# device, requests Vulkan<->CUDA interop fd extensions (external_semaphore_fd /
# external_fence_fd) that headless lavapipe (CPU Vulkan) cannot provide ->
# ErrorExtensionNotPresent. On a GPU-less container there is no CUDA, so svulkan2
# skips interop and renders fine on lavapipe. Phase A (obs dump) needs no GPU.
@app.function(image=image, timeout=1800)
def smoke():
    import os, traceback
    out = {}

    # 1) Vulkan visibility
    try:
        import subprocess
        vk = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=120)
        out["vulkaninfo_rc"] = vk.returncode
        txt = (vk.stdout or "") + (vk.stderr or "")
        # surface the GPU line(s)
        out["vulkan_gpu"] = [l.strip() for l in txt.splitlines() if "deviceName" in l or "GPU id" in l][:4]
    except Exception as e:
        out["vulkaninfo_err"] = repr(e)

    # 1b) lavapipe device extensions (so we can see if a needed ext is missing)
    try:
        import subprocess
        ext = subprocess.run(["vulkaninfo"], capture_output=True, text=True, timeout=120)
        etxt = (ext.stdout or "") + (ext.stderr or "")
        wanted = ("swapchain", "external_memory", "external_semaphore", "external_fence",
                  "VK_KHR_external", "timeline", "descriptor_indexing", "ray_tracing", "acceleration")
        out["lavapipe_exts"] = sorted({
            l.strip().split()[0] for l in etxt.splitlines()
            if "VK_" in l and any(w.lower() in l.lower() for w in wanted)
        })[:40]
    except Exception as e:
        out["lavapipe_exts_err"] = repr(e)

    # 2) Direct ManiSkill2_real2sim env (isolates SAPIEN from simpler_env packaging)
    try:
        import mani_skill2_real2sim  # noqa
        out["maniskill_import"] = "ok"
    except Exception:
        out["maniskill_import"] = traceback.format_exc()[-1500:]

    # 2b) Force SAPIEN renderer to offscreen-only: lavapipe is headless (no
    # swapchain/WSI), so the default on-screen renderer requests a device
    # extension lavapipe lacks -> ErrorExtensionNotPresent. Patch the class so
    # ManiSkill's `sapien.SapienRenderer(**kwargs)` always gets offscreen_only.
    try:
        import sapien.core as sapien_core
        _orig_renderer = sapien_core.SapienRenderer

        def _offscreen_renderer(*a, **k):
            k.setdefault("offscreen_only", True)
            return _orig_renderer(*a, **k)

        sapien_core.SapienRenderer = _offscreen_renderer
        out["sapien_patch"] = "offscreen_only forced"
    except Exception:
        out["sapien_patch"] = traceback.format_exc()[-1000:]

    # 3) simpler_env convenience API
    try:
        import numpy as np
        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        out["n_tasks"] = len(getattr(simpler_env, "ENVIRONMENTS", []))
        task = "widowx_put_eggplant_in_basket"
        env = simpler_env.make(task)
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        instruction = env.get_language_instruction()
        img = get_image_from_maniskill2_obs_dict(env, obs)
        out["task"] = task
        out["instruction"] = instruction
        out["obs_image_shape"] = list(getattr(img, "shape", []))
        # save a render to the volume-less /tmp and report basic stats (proves rendering)
        out["img_min_max_mean"] = [int(img.min()), int(img.max()), float(img.mean())]
        last_info = {}
        for _ in range(5):
            action = env.action_space.sample() * 0.0
            step_ret = env.step(action)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, last_info = step_ret
            else:
                obs, reward, done, last_info = step_ret
        out["info_keys"] = list(last_info.keys())
        out["success_field"] = last_info.get("success", "MISSING")
        out["status"] = "SIMPLER OK"
    except Exception:
        out["status"] = "SIMPLER FAILED"
        out["trace"] = traceback.format_exc()[-3000:]
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(smoke.remote(), indent=2, default=str))
