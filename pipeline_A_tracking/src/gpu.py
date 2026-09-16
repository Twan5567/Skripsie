"""GPU acceleration setup, shared across the pipeline.

This machine has an AMD Radeon RX 7900 XT — there is no NVIDIA GPU, so CUDA
is not an option anywhere in this codebase. GPU acceleration instead comes
from OpenCL, via Mesa's rusticl driver (package: mesa-opencl-icd). rusticl
is disabled by default upstream and only activates when RUSTICL_ENABLE names
the Gallium driver to use; for this GPU that is "radeonsi". The env var must
be set before cv2 is imported (OpenCV probes for OpenCL platforms at import
time), which is why this module has to run first.

Call init_opencv_opencl() once at the top of any script that uses OpenCV,
before doing any cv2 work. Wrap arrays you want processed on the GPU in
cv2.UMat(...); OpenCV's Transparent API (T-API) dispatches UMat operations
to OpenCL automatically and falls back to the CPU for anything unsupported.

Scope of GPU acceleration in this pipeline, by component:
- OpenCV (ArUco detection, image ops): OpenCL via rusticl, handled here.
- COLMAP sparse feature extraction/matching: already GPU-accelerated via
  SiftGPU over OpenGL (RADV) — no CUDA needed, nothing to change.
- COLMAP dense reconstruction (patch_match_stereo): hard-requires CUDA
  upstream; there is no ROCm/OpenCL backend for it. Not GPU-accelerable on
  this hardware — see README for the OpenMVS (CPU) fallback.
- Open3D: the PyPI wheel is CPU-only on Linux for non-NVIDIA GPUs; no AMD
  GPU backend is available through pip.
"""
import os

_opencv_opencl_ready = False


def init_opencv_opencl(enable: bool = True) -> bool:
    """Enable OpenCL (rusticl) for OpenCV. Must be called before `import cv2`
    anywhere in the process. Returns True if the GPU is active and in use.
    """
    global _opencv_opencl_ready
    os.environ.setdefault("RUSTICL_ENABLE", "radeonsi")

    import cv2

    if not cv2.ocl.haveOpenCL():
        _opencv_opencl_ready = False
        return False

    cv2.ocl.setUseOpenCL(enable)
    _opencv_opencl_ready = enable and cv2.ocl.useOpenCL()
    return _opencv_opencl_ready


def report() -> str:
    """Human-readable summary of what's GPU-accelerated right now."""
    import cv2

    lines = []
    if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
        dev = cv2.ocl.Device.getDefault()
        lines.append(f"OpenCV OpenCL: ON ({dev.vendorName()} {dev.name()})")
    else:
        lines.append("OpenCV OpenCL: OFF")
    lines.append("COLMAP sparse (SiftGPU/OpenGL): ON (verify with `colmap -h`)")
    lines.append("COLMAP dense (patch_match_stereo): unavailable — CUDA-only, no NVIDIA GPU")
    lines.append("Open3D: CPU only (no AMD GPU backend in the pip wheel)")
    return "\n".join(lines)


if __name__ == "__main__":
    ok = init_opencv_opencl()
    print(report())
    raise SystemExit(0 if ok else 1)
