"""Pure geometry: undistortion, square-error metric, camera-from-homography.

Nothing here reads video or prints; every function is a mapping from arrays
to arrays so it can be unit-tested and reused by the mapping side later.
"""
import cv2
import numpy as np


def undistort(pts: np.ndarray, c: np.ndarray, s: float, k1: float, k2: float):
    """Division-model radial undistortion about image centre `c`.

    Works in pure image space, so it needs no focal length — which matters
    because the focal length is itself recovered later FROM the undistorted
    homography. `s` normalises radius so k1/k2 stay order-1.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    d = (p - c) / s
    r2 = (d ** 2).sum(axis=1)
    f = 1.0 + k1 * r2 + k2 * r2 ** 2
    return (c + (p - c) / f[:, None]).reshape(np.shape(pts))


def square_error(med_u: dict, anchor: int, S: float):
    """RMS mm error of every marker's sides and diagonals against a true square.

    The anchor is pinned to S by construction, so this measures how well the
    REST of the floor agrees — which is exactly what lens distortion breaks.
    Diagonals are included because they are scale-aware in a different way
    than the sides and stop a uniform-shrink degeneracy.
    """
    Hm, _ = cv2.findHomography(med_u[anchor],
                               np.array([[0, 0], [S, 0], [S, S], [0, S]], float))
    if Hm is None:
        return 1e9, {}, None
    errs, per = [], {}
    for k, q in med_u.items():
        w = cv2.perspectiveTransform(q.reshape(-1, 1, 2), Hm).reshape(4, 2)
        sides = [np.linalg.norm(w[j] - w[(j + 1) % 4]) for j in range(4)]
        diags = [np.linalg.norm(w[0] - w[2]), np.linalg.norm(w[1] - w[3])]
        per[k] = float(np.mean(sides))
        errs += [x - S for x in sides]
        errs += [x - S * np.sqrt(2) for x in diags]
    return float(np.sqrt(np.mean(np.square(errs)))), per, Hm


def camera_from_homography(Hf2i: np.ndarray, c: np.ndarray):
    """Recover focal length and camera centre from one floor->image homography.

    Assumes square pixels and the principal point at the image centre, which
    leaves focal length as the only unknown in K. Zhang's two orthonormality
    constraints on the rotation columns then each give an independent estimate
    of f; their agreement is a free check that the pinhole assumption holds.
    Returns (f, camera_xyz_in_floor_mm, f_from_each_constraint).
    """
    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]], float)
    Hc = T @ Hf2i                      # floor -> centred image
    h1, h2, h3 = Hc[:, 0], Hc[:, 1], Hc[:, 2]
    den1 = h1[2] * h2[2]
    f2a = -(h1[0] * h2[0] + h1[1] * h2[1]) / den1 if abs(den1) > 1e-12 else np.nan
    den2 = h2[2] ** 2 - h1[2] ** 2
    f2b = ((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / den2 \
        if abs(den2) > 1e-12 else np.nan
    cand = [x for x in (f2a, f2b) if np.isfinite(x) and x > 0]
    if not cand:
        return None, None, (f2a, f2b)
    f = float(np.sqrt(np.mean(cand)))
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]], float)
    A = np.linalg.inv(K) @ Hc
    # Sign of lambda is free in a homography; fix it so the floor lies in
    # front of the camera (positive depth), else the recovered centre comes
    # out mirrored through the plane and the height is negative.
    lam = 1.0 / np.linalg.norm(A[:, 0])
    r1, r2 = lam * A[:, 0], lam * A[:, 1]
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)        # nearest true rotation
    R = U @ Vt
    t = lam * A[:, 2]
    C = -R.T @ t
    # The floor frame built from ArUco corner order (TL,TR,BR,BL) is
    # left-handed against a z-up world, because image y runs downward. That
    # makes r3 = r1 x r2 point INTO the floor and C_z come out negative with
    # the right magnitude. Height is the distance to the plane either way.
    C = np.array([C[0], C[1], abs(C[2])])
    return f, C, (np.sqrt(f2a) if f2a > 0 else np.nan,
                  np.sqrt(f2b) if f2b > 0 else np.nan)


def apply_H(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    flat = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(flat, Hm).reshape(np.shape(pts))
