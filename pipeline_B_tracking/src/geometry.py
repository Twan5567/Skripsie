# Lens and homography maths for the tracking pipeline. Two lens models: the division
# model (k1, k2 about the image centre, fitted from the floor sheets) and OpenCV's
# (K, D from a ChArUco calibration).
import cv2
import numpy as np


def side_lengths(q: np.ndarray) -> np.ndarray:
    return np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)


# division model: distorted -> undistorted pixels; r is normalised by s (half the
# image diagonal)
def undistort(pts: np.ndarray, c: np.ndarray, s: float, k1: float, k2: float):
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    r2 = (((p - c) / s) ** 2).sum(axis=1)
    f = 1.0 + k1 * r2 + k2 * r2 ** 2
    return (c + (p - c) / f[:, None]).reshape(np.shape(pts))


# inverse of undistort, by fixed-point iteration (there is no closed form)
def distort(pts: np.ndarray, c: np.ndarray, s: float, k1: float, k2: float, iters: int = 12):
    du = (np.asarray(pts, dtype=np.float64).reshape(-1, 2) - c) / s
    dd = du.copy()
    for _ in range(iters):
        r2 = (dd ** 2).sum(axis=1)
        dd = du * (1.0 + k1 * r2 + k2 * r2 ** 2)[:, None]
    return (c + dd * s).reshape(np.shape(pts))


# OpenCV model: distorted pixels -> undistorted pixels of the ideal camera P
def cv_undistort(pts: np.ndarray, K: np.ndarray, D: np.ndarray, P: np.ndarray):
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-8)
    return cv2.undistortPoints(p, K, D, R=None, P=P, criteria=crit).reshape(np.shape(pts))


# inverse of cv_undistort: undistorted pixels of P -> distorted pixels
def cv_distort(pts: np.ndarray, K: np.ndarray, D: np.ndarray, P: np.ndarray):
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = np.c_[(p - P[:2, 2]) / P[0, 0], np.ones(len(p))]
    q, _ = cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return q.reshape(np.shape(pts))


# How square the sheets come out. Map the anchor sheet onto an exact S x S square,
# map every sheet through the same homography and take the RMS error of all sides
# and diagonals (mm). Also returns each sheet's mean side and the homography.
def square_error(med_u: dict, anchor: int, S: float):
    Hm, _ = cv2.findHomography(med_u[anchor], np.array([[0, 0], [S, 0], [S, S], [0, S]], float))

    if Hm is None:
        return 1e9, {}, None
    errs, per = [], {}
    for k, q in med_u.items():
        w = apply_H(Hm, q)
        sides = side_lengths(w)
        diags = np.linalg.norm(w[:2] - w[2:], axis=1)       # 0-2 and 1-3
        per[k] = float(np.mean(sides))
        errs += [sides - S, diags - S * np.sqrt(2)]
    return float(np.sqrt(np.mean(np.square(np.concatenate(errs))))), per, Hm


# Focal length and camera position from a floor -> image homography. The two
# orthonormality constraints on the rotation columns (Zhang) each give f^2. C is the
# camera centre in floor mm; C[2] is its height. f_known skips the estimate when the
# lens is calibrated. Returns (f, C, (f from constraint 1, f from constraint 2)).
def camera_from_homography(Hf2i: np.ndarray, c: np.ndarray, f_known: float | None = None):

    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]], float)
    Hc = T @ Hf2i
    h1, h2 = Hc[:, 0], Hc[:, 1]
    den1 = h1[2] * h2[2]
    f2a = -(h1[0] * h2[0] + h1[1] * h2[1]) / den1 if abs(den1) > 1e-12 else np.nan
    den2 = h2[2] ** 2 - h1[2] ** 2
    f2b = ((h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)) / den2 \
        if abs(den2) > 1e-12 else np.nan
    cand = [x for x in (f2a, f2b) if np.isfinite(x) and x > 0]


    if f_known is not None:
        f = float(f_known)
    elif not cand:
        return None, None, (f2a, f2b)
    else:
        f = float(np.sqrt(np.mean(cand)))

    # H ~ K [r1 r2 t]: remove K, scale so r1 is a unit vector, then make R a true
    # rotation (nearest rotation by SVD)
    A = np.linalg.inv(np.diag([f, f, 1.0])) @ Hc
    lam = 1.0 / np.linalg.norm(A[:, 0])
    r1, r2 = lam * A[:, 0], lam * A[:, 1]
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, np.cross(r1, r2)], axis=1))
    R = U @ Vt
    C = -R.T @ (lam * A[:, 2])
    # the homography's sign is arbitrary; the camera is above the floor
    C = np.array([C[0], C[1], abs(C[2])])
    return f, C, (np.sqrt(f2a) if f2a > 0 else np.nan, np.sqrt(f2b) if f2b > 0 else np.nan)


# apply a homography to 2-D points of any shape (..., 2)
def apply_H(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    flat = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(flat, Hm).reshape(np.shape(pts))
