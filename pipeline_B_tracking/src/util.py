import numpy as np


def banner(title: str) -> str:
    # the "====\nTITLE\n====" section header that floor, world, calcache, fuse and
    # track_ground each built by hand
    return f"\n{'=' * 68}\n{title}\n{'=' * 68}"


def runs(mask: np.ndarray) -> list[np.ndarray]:
    # indices of each run of consecutive True values; was written out in
    # smooth_xy, kalman_smooth and the dual overlay
    idx = np.flatnonzero(mask)
    return np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
