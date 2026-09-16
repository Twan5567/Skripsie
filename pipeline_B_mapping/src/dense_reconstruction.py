"""Dense reconstruction via OpenMVS (Docker), using COLMAP's sparse model as input.

COLMAP's own dense reconstruction (patch_match_stereo) hard-requires CUDA, which
isn't available on this AMD GPU (see README). OpenMVS does dense reconstruction
on CPU instead, run here via the openmvs/openmvs-ubuntu Docker image.

Pipeline:
  1. colmap image_undistorter   — undistort images + sparse model to the PINHOLE
     camera model (OpenMVS only supports PINHOLE).
  2. colmap model_converter     — convert the undistorted sparse model to TXT
     (InterfaceCOLMAP only reads COLMAP's TXT export).
  3. InterfaceCOLMAP (Docker)   — convert COLMAP's TXT export into OpenMVS's
     .mvs scene format.
  4. DensifyPointCloud (Docker) — the actual CPU dense reconstruction stage,
     producing scene_dense.mvs / scene_dense.ply.

Every stage runs under a hard memory ceiling (systemd scope for the native
colmap calls, Docker's --memory for the OpenMVS calls) so a runaway process
gets killed on its own instead of triggering the system OOM killer against
the whole desktop session — that's what crashed the machine (and separately,
VS Code) last time. Run this from a plain terminal, not VS Code's integrated
terminal: heavy commands launched there inherit VS Code's own snap memory
cgroup, so a memory spike can take down VS Code itself, not just the command.

Usage:
    python src/dense_reconstruction.py \
        --sparse colmap/sparse/0 --images frames/SangaCup --workspace colmap/dense/SangaCup \
        --memory-limit 8G --max-threads 4
"""
import argparse
import os
import subprocess
from pathlib import Path

OPENMVS_IMAGE = "openmvs/openmvs-ubuntu"
OPENMVS_BIN = "/usr/local/bin/OpenMVS"


def run(cmd: list[str], memory_limit: str) -> None:
    """Run a native (non-Docker) command inside a systemd scope with a hard
    memory ceiling, so it gets killed on its own if it exceeds memory_limit
    instead of triggering the system-wide OOM killer.
    """
    scoped_cmd = [
        "systemd-run", "--user", "--scope", "--quiet",
        "-p", f"MemoryMax={memory_limit}",
        "-p", "MemorySwapMax=0",
        "--",
    ] + cmd
    print("+", " ".join(cmd), f"(memory-capped at {memory_limit})")
    subprocess.run(scoped_cmd, check=True)


def run_openmvs(workspace: Path, tool: str, tool_args: list[str], memory_limit: str) -> None:
    """Run an OpenMVS CLI tool inside Docker, with `workspace` mounted at /workspace
    and used as the working directory, so relative paths in tool_args resolve
    against it. Runs as the current host user so output files aren't root-owned,
    and under a hard --memory cap so it can't take down the whole system.
    """
    cmd = [
        "docker", "run", "--rm",
        "--memory", memory_limit,
        "--memory-swap", memory_limit,
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{workspace.resolve()}:/workspace",
        "-w", "/workspace",
        OPENMVS_IMAGE,
        f"{OPENMVS_BIN}/{tool}",
    ] + tool_args
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", type=Path, required=True, help="COLMAP sparse model folder, e.g. colmap/sparse/0")
    ap.add_argument("--images", type=Path, required=True, help="original images used to build the sparse model")
    ap.add_argument("--workspace", type=Path, required=True, help="output folder for undistorted images + dense reconstruction")
    ap.add_argument("--memory-limit", default="12G", help="hard memory ceiling per stage (default: 8G)")
    ap.add_argument("--max-threads", type=int, default=max(1, os.cpu_count() // 2), help="threads for DensifyPointCloud (default: half of nproc)")
    ap.add_argument("--resolution-level", type=int, default=2, help="downscale images this many halvings before densifying; higher = less memory (default: 2)")
    ap.add_argument("--number-views", type=int, default=3, help="views used per depth-map; lower = less memory (default: 3)")
    args = ap.parse_args()

    undistorted = args.workspace / "undistorted"
    undistorted.mkdir(parents=True, exist_ok=True)

    print("running image_undistorter...")
    run([
        "colmap", "image_undistorter",
        "--image_path", str(args.images),
        "--input_path", str(args.sparse),
        "--output_path", str(undistorted),
        "--output_type", "COLMAP",
    ], args.memory_limit)

    print("converting undistorted sparse model to TXT...")
    run([
        "colmap", "model_converter",
        "--input_path", str(undistorted / "sparse"),
        "--output_path", str(undistorted / "sparse"),
        "--output_type", "TXT",
    ], args.memory_limit)

    print("running InterfaceCOLMAP (OpenMVS, via Docker)...")
    run_openmvs(undistorted, "InterfaceCOLMAP", ["-i", ".", "-o", "scene.mvs", "--image-folder", "images"], args.memory_limit)

    print("running DensifyPointCloud (OpenMVS, via Docker)...")
    run_openmvs(undistorted, "DensifyPointCloud", [
        "-i", "scene.mvs", "-o", "scene_dense.mvs",
        "--resolution-level", str(args.resolution_level),
        "--number-views", str(args.number_views),
        "--max-threads", str(args.max_threads),
    ], args.memory_limit)

    print(f"done — dense point cloud at {undistorted}/scene_dense.ply")


if __name__ == "__main__":
    main()
