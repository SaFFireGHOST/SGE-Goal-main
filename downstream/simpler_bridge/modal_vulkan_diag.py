"""Probe what Vulkan/graphics the Modal GPU container exposes (no SAPIEN)."""
import modal

app = modal.App("vulkan-diag")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.10")
    .apt_install(
        "libvulkan1", "vulkan-tools", "libvulkan-dev", "pciutils", "findutils",
        "mesa-vulkan-drivers", "libgl1-mesa-dri",  # lavapipe CPU Vulkan fallback
        # full X11 client lib set the NVIDIA GLX/Vulkan ICD links against
        "libx11-6", "libxext6", "libxrender1", "libxi6", "libxfixes3",
        "libxxf86vm1", "libxrandr2", "libxinerama1", "libxcursor1",
        "libxcb1", "libxau6", "libxdmcp6", "libsm6", "libice6", "libxkbcommon0",
    )
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d",
        'printf \'{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}\' > /usr/share/vulkan/icd.d/nvidia_icd.json',
    )
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json"})
)


@app.function(image=image, gpu="L4", timeout=600)
def diag():
    import subprocess, os
    def sh(cmd):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            return (r.stdout + r.stderr).strip()
        except Exception as e:
            return f"ERR {e!r}"
    res = {}
    res["ldd_before"] = sh("ldd /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>&1 | grep -i 'not found' || echo 'no missing deps'")
    res["ldconfig"] = sh("ldconfig 2>&1 | tail -3; echo rc=$?")
    res["ldd_after"] = sh("ldd /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>&1 | grep -i 'not found' || echo 'no missing deps'")
    res["nm_vk_icd"] = sh("nm -D /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null | grep -i vk_icd || echo 'no vk_icd symbols'")
    res["dev_nvidia"] = sh("ls -la /dev/nvidia* 2>&1")
    res["dev_dri"] = sh("ls -la /dev/dri/ 2>&1")
    res["smi_graphics"] = sh("nvidia-smi -q 2>&1 | grep -iE 'Graphics|Display|Product Name|Driver Model' | head")
    res["vk_loader_debug"] = sh("ldconfig; VK_LOADER_DEBUG=error,warn VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary 2>&1 | tail -8")
    res["icd_dir_now"] = sh("ls /usr/share/vulkan/icd.d/ 2>&1")
    res["find_lavapipe"] = sh("find / -iname '*lvp_icd*' -o -iname '*lavapipe*' 2>/dev/null | head")
    res["vulkaninfo_lavapipe"] = sh("LVP=$(find / -iname 'lvp_icd*.json' 2>/dev/null | head -1); echo using=$LVP; VK_ICD_FILENAMES=$LVP vulkaninfo --summary 2>&1 | grep -iE 'deviceName|llvmpipe|driverName|ERROR|Found no' | head -10")
    res["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    res["env_caps"] = os.environ.get("NVIDIA_DRIVER_CAPABILITIES")
    res["env_icd"] = os.environ.get("VK_ICD_FILENAMES")
    res["libGLX_nvidia"] = sh("find / -name 'libGLX_nvidia.so*' 2>/dev/null | head")
    res["libEGL_nvidia"] = sh("find / -name 'libEGL_nvidia.so*' 2>/dev/null | head")
    res["libnvidia_glvkspirv"] = sh("find / -name 'libnvidia-glvkspirv*' 2>/dev/null | head")
    res["libvulkan"] = sh("find / -name 'libvulkan.so*' 2>/dev/null | head")
    res["nvidia_libs_sample"] = sh("ls /usr/lib/x86_64-linux-gnu/ | grep -iE 'nvidia|vulkan|GLX|EGL' | head -40")
    res["glvnd_egl_dir"] = sh("ls -la /usr/share/glvnd/egl_vendor.d/ 2>/dev/null")
    res["icd_dir"] = sh("ls -la /usr/share/vulkan/icd.d/ 2>/dev/null")
    res["vulkaninfo_summary"] = sh("vulkaninfo --summary 2>&1 | head -40")
    return res


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(diag.remote(), indent=2, default=str))
