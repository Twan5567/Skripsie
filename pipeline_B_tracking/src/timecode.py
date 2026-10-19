import subprocess

import numpy as np


def clip_start_seconds(video: str, fps: float) -> float | None:
    """Seconds of day of frame 0, from the `tmcd` timecode. None if absent."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream_tags=timecode",
             "-of", "csv=p=0", video],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tc = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    parts = tc.replace(";", ":").split(":")
    if len(parts) != 4:
        return None
    h, m, s, f = (int(p) for p in parts)
    
    return h * 3600 + m * 60 + s + f / round(fps)


def frame_times(start: float | None, n: int, stride: int, fps: float) -> np.ndarray | None:
    """Global time of each analysed frame (every `stride`-th), or None."""
    if start is None:
        return None
    return start + np.arange(n) * stride / fps


def fmt_tod(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
