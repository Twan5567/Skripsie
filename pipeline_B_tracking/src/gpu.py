import os


# Turn on OpenCL in OpenCV (the AMD GPU through Mesa's rusticl). Call it before
# anything imports cv2. Returns whether OpenCL is on.
def init_opencv_opencl(enable: bool = True) -> bool:
    os.environ.setdefault("RUSTICL_ENABLE", "radeonsi")

    import cv2

    if not cv2.ocl.haveOpenCL():
        return False
    cv2.ocl.setUseOpenCL(enable)
    return enable and cv2.ocl.useOpenCL()


# one line per accelerator, for checking the setup by hand (python gpu.py)
def report() -> str:
    import cv2

    if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
        dev = cv2.ocl.Device.getDefault()
        ocl = f"OpenCV OpenCL: ON ({dev.vendorName()} {dev.name()})"
    else:
        ocl = "OpenCV OpenCL: OFF"
    return "\n".join([ocl,
                      "COLMAP sparse (SiftGPU/OpenGL): ON (verify with `colmap -h`)",
                      "COLMAP dense (patch_match_stereo): unavailable — CUDA-only, no NVIDIA GPU",
                      "Open3D: CPU only (no AMD GPU backend in the pip wheel)"])


if __name__ == "__main__":
    ok = init_opencv_opencl()
    print(report())
    raise SystemExit(0 if ok else 1)
