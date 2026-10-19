
import cv2
import numpy as np


def undistort(pts: np.ndarray, c: np.ndarray, s: float, k1: float, k2: float):

    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    d = (p - c) / s
    r2 = (d ** 2).sum(axis=1)
    f = 1.0 + k1 * r2 + k2 * r2 ** 2
    return (c + (p - c) / f[:, None]).reshape(np.shape(pts))


def distort(pts: np.ndarray, c: np.ndarray, s: float, k1: float, k2: float, iters: int = 12):


    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    du = (p - c) / s
    dd = du.copy()

    for _ in range(iters):
        r2 = (dd ** 2).sum(axis=1)
        dd = du * (1.0 + k1 * r2 + k2 * r2 ** 2)[:, None]
    return (c + dd * s).reshape(np.shape(pts))


def cv_undistort(pts: np.ndarray, K: np.ndarray, D: np.ndarray, P: np.ndarray):

    p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-8)
    q = cv2.undistortPoints(p, K, D, R=None, P=P, criteria=crit)
    return q.reshape(np.shape(pts))


def cv_distort(pts: np.ndarray, K: np.ndarray, D: np.ndarray, P: np.ndarray):

    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = np.c_[(p - P[:2, 2]) / P[0, 0], np.ones(len(p))]
    q, _ = cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return q.reshape(np.shape(pts))


def lens_undistort(fl, pts: np.ndarray):

    if getattr(fl, "K", None) is not None:
        return cv_undistort(pts, fl.K, fl.D, fl.P)
    return undistort(pts, fl.img_c, fl.norm_s, fl.k1, fl.k2)


def lens_distort(fl, pts: np.ndarray):

    if getattr(fl, "K", None) is not None:
        return cv_distort(pts, fl.K, fl.D, fl.P)
    return distort(pts, fl.img_c, fl.norm_s, fl.k1, fl.k2)


def square_error(med_u: dict, anchor: int, S: float):

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


def camera_from_homography(Hf2i: np.ndarray, c: np.ndarray, f_known: float | None = None):

    T = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]], float)
    Hc = T @ Hf2i                      
    h1, h2, h3 = Hc[:, 0], Hc[:, 1], Hc[:, 2]
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
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]], float)
    A = np.linalg.inv(K) @ Hc

    
    lam = 1.0 / np.linalg.norm(A[:, 0])
    r1, r2 = lam * A[:, 0], lam * A[:, 1]
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)        
    R = U @ Vt
    t = lam * A[:, 2]
    C = -R.T @ t


    C = np.array([C[0], C[1], abs(C[2])])
    return f, C, (np.sqrt(f2a) if f2a > 0 else np.nan,
                  np.sqrt(f2b) if f2b > 0 else np.nan)


def apply_H(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    flat = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(flat, Hm).reshape(np.shape(pts))
