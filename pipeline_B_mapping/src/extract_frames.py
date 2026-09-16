import argparse
import subprocess
from pathlib import Path





def extract_video(video_path: Path, out_root: Path, fps: int) -> Path:
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps}", str(out_dir / "frame_%06d.png")]
    subprocess.run(cmd, check=True)

    return out_dir



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("frames"))
    ap.add_argument("--fps", type=int,required=True)
    args = ap.parse_args()

    out_dir = extract_video(args.video, args.out_dir, args.fps)
    n = len(list(out_dir.glob("*.png")))
    print(f"{n} frames written to {out_dir}/")

if __name__ == "__main__":
    main()