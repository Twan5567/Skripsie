import subprocess

import numpy as np


# Time of day (s) of the clip's first frame, from the GoPro timecode track (tmcd,
# HH:MM:SS:FF). Both cameras scan the same QR timecode, so this puts their frames on
# one clock. None if the clip has no timecode.
def clip_start_seconds(video: str, fps: float) -> float | None:
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
    h, m, s, f = map(int, parts)
    return h * 3600 + m * 60 + s + f / round(fps)


# global time of each analysed frame
def frame_times(start: float | None, n: int, stride: int, fps: float) -> np.ndarray | None:
    return None if start is None else start + np.arange(n) * stride / fps


# seconds of day -> HH:MM:SS.sss
def fmt_tod(t: float) -> str:
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
