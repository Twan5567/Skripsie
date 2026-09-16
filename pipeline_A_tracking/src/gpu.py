import os

_opencv_opencl_ready = False


def init_opencv_opencl(enable: bool = True) -> bool:

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
