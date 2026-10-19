"""
  python generate_print_sheets.py                 # everything, A4 + A3
  python generate_print_sheets.py --only gcp --paper A4
"""
from __future__ import annotations

import argparse
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from aruco_config import (
    ARUCO_DICT,
    ARUCO_DICT_NAME,
    CHARUCO_DICT,
    CHARUCO_DICT_NAME,
    GCP_MARKER_IDS,
    GCP_SIDE_MM_A3,
    GCP_SIDE_MM_A4,
    SCALE_BAR_MARKER_IDS,
    VEHICLE_MARKER_IDS,
    VEHICLE_SIDE_MM,
)

MM = 1.0 / 25.4  # mm -> inch
PAPER_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}
# Dictionary grid size in modules, excluding the marker's own 1-module border.
DICT_MODULES = {"DICT_4X4_50": 4, "DICT_5X5_100": 5}


# --------------------------------------------------------------------------
# page primitives
# --------------------------------------------------------------------------
def new_page(w_mm: float, h_mm: float):
    """A figure whose data coordinates are millimetres, y increasing downward."""
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, w_mm)
    ax.set_ylim(h_mm, 0.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def draw_marker(ax, dict_obj, dict_name: str, marker_id: int, x: float, y: float, side: float):
    """Draw an ArUco marker as vector rectangles (no rasterisation, no blur).

    (x, y) is the top-left corner of the black square; `side` is its full
    outer edge length in mm, border module included.
    """
    n = DICT_MODULES[dict_name] + 2  # + 1-module black border on each side
    bits = cv2.aruco.generateImageMarker(dict_obj, marker_id, n, 1) > 127
    m = side / n
    # White ground first, so a black module never bleeds over the page.
    ax.add_patch(Rectangle((x, y), side, side, facecolor="white", edgecolor="none", zorder=1))
    for r in range(n):
        for c in range(n):
            if not bits[r, c]:
                ax.add_patch(
                    Rectangle(
                        (x + c * m, y + r * m), m, m,
                        facecolor="black", edgecolor="none", linewidth=0, zorder=2,
                    )
                )


def draw_scale_check(ax, x: float, y: float, length: float = 100.0):
    """A bar of exactly `length` mm — the print-scale self-test."""
    ax.plot([x, x + length], [y, y], color="black", lw=1.2, solid_capstyle="butt")
    for xx in (x, x + length):
        ax.plot([xx, xx], [y - 3.0, y + 3.0], color="black", lw=1.2)
    for i in range(1, 10):  # 10 mm ticks
        xx = x + i * length / 10.0
        ax.plot([xx, xx], [y - 1.6, y], color="black", lw=0.6)
    ax.text(x + length + 4.0, y + 1.2, f"{length:.1f} mm", fontsize=8,
            weight="bold", va="center", ha="left")
    ax.text(x, y + 7.5,
            f"SCALE CHECK — measure the bar above. If it is not {length:.1f} mm the sheet printed "
            "at the wrong scale; reprint at 100%.",
            fontsize=7, va="baseline", ha="left")


def draw_corner_ticks(ax, x: float, y: float, side: float, gap: float, arm: float):
    """Registration ticks aligned with the marker edges, kept `gap` mm clear of
    the black square so the white quiet zone stays unbroken."""
    for cx in (x, x + side):
        ax.plot([cx, cx], [y - gap - arm, y - gap], color="black", lw=0.5)
        ax.plot([cx, cx], [y + side + gap, y + side + gap + arm], color="black", lw=0.5)
    for cy in (y, y + side):
        ax.plot([x - gap - arm, x - gap], [cy, cy], color="black", lw=0.5)
        ax.plot([x + side + gap, x + side + gap + arm], [cy, cy], color="black", lw=0.5)


def _text_mm(label: str, fontsize: float) -> float:
    """Rough advance width of `label` in mm — good enough to place a write-on rule."""
    return len(label) * fontsize * 0.352778 * 0.58


def field(ax, x: float, y: float, label: str, width: float = 55.0, fontsize: float = 8.0):
    ax.text(x, y, label, fontsize=fontsize, va="baseline", ha="left")
    x0 = x + _text_mm(label, fontsize) + 2.0
    ax.plot([x0, x0 + width], [y + 1.0, y + 1.0], color="black", lw=0.5)


def field_block(ax, fields: list[str], page_w: float, page_h: float, top_limit: float, margin: float = 18.0, ncols: int = 2, step: float = 6.8):
    """Lay write-on fields out in columns anchored to the BOTTOM of the page.

    Anchoring upward from the bottom means a sheet can never silently run its
    last line off the paper — it raises instead.
    """
    rows = -(-len(fields) // ncols)
    col_w = (page_w - 2 * margin) / ncols
    first = page_h - 12.0 - step * (rows - 1)
    if first < top_limit + 4.0:
        raise ValueError(f"field block does not fit: notes end {top_limit:.1f} mm, "
                         f"fields would start {first:.1f} mm")
    for n, f in enumerate(fields):
        cx = margin + (n % ncols) * col_w
        w = max(22.0, col_w - _text_mm(f, 8.0) - 8.0)
        field(ax, cx, first + (n // ncols) * step, f, width=w)


# --------------------------------------------------------------------------
# marker sheets
# --------------------------------------------------------------------------
def marker_sheet(paper: str, dict_obj, dict_name: str, marker_id: int, side: float, role: str, notes: list[str], fields: list[str]):
    w, h = PAPER_MM[paper]
    quiet = side / (DICT_MODULES[dict_name] + 2)  # 1 module — the ArUco minimum

    x = (w - side) / 2.0
    y = max(quiet, 25.0)
    if x < quiet:
        raise ValueError(f"{side} mm marker leaves < 1 module quiet zone on {paper}")

    fig, ax = new_page(w, h)
    draw_marker(ax, dict_obj, dict_name, marker_id, x, y, side)
    draw_corner_ticks(ax, x, y, side, gap=quiet + 3.0, arm=6.0)
    if role.startswith("VEHICLE TAG"):
        # Cut line sits exactly one module outside the black square, so the
        # trimmed tag keeps its full quiet zone. FRONT sits above that line.
        ax.add_patch(Rectangle((x - quiet, y - quiet), side + 2 * quiet, side + 2 * quiet,
                               fill=False, edgecolor="black", lw=0.6, linestyle=(0, (4, 3))))
        ax.text(w / 2.0, max(y - quiet - 4.0, 6.0), "FRONT  ▲", fontsize=13, weight="bold",
                ha="center", va="bottom")

    ty = y + side + quiet + 12.0
    ax.text(w / 2.0, ty, f"ID {marker_id:02d}", fontsize=26, weight="bold", ha="center", va="baseline")
    ty += 8.0
    ax.text(w / 2.0, ty, role, fontsize=10, weight="bold", ha="center", va="baseline")
    ty += 6.5
    ax.text(w / 2.0, ty, f"{dict_name}   |   nominal side {side:.1f} mm (outer edge of black square)",
            fontsize=8, ha="center", va="baseline")

    ty += 9.0
    draw_scale_check(ax, 18.0, ty)

    ty += 16.0
    for n in notes:
        ax.text(18.0, ty, n, fontsize=7, ha="left", va="baseline")
        ty += 4.6

    field_block(ax, fields, w, h, top_limit=ty)
    return fig


def gcp_sheet(paper: str, marker_id: int, side: float):
    return marker_sheet(
        paper, ARUCO_DICT, ARUCO_DICT_NAME, marker_id, side,
        "GROUND CONTROL POINT  —  Branch B (track mapping)",
        [
            "Glue flat to rigid board. A curled sheet biases the corner positions and the scale with it.",
            "Matte only — no gloss lamination. Specular highlights under hall lighting kill detection.",
            "Position is measured to the CENTRE of the black square, in the track's local frame.",
        ],
        ["Measured side (mm):", "Date placed:", "X (m):", "Y (m):", "Z above floor (m):"],
    )


def vehicle_sheet(paper: str, marker_id: int, side: float):
    return marker_sheet(
        paper, ARUCO_DICT, ARUCO_DICT_NAME, marker_id, side,
        "VEHICLE MARKER  —  Branch A (trajectory tracking)",
        [
            "Mount rigidly and level on the car deck, square to the chassis axes, +X along the forward axis.",
            "MEASURE THE HEIGHT of the marker plane above the floor. The homography maps one plane; a marker",
            "sitting h above the ground projects with a parallax error that grows with distance from the camera",
            "nadir, and h is the only thing that lets you correct it afterwards.",
        ],
        ["Measured side (mm):", "Height above floor (mm):", "Offset from rear axle (mm):", "Yaw offset (deg):"],
    )


def car_sheet(paper: str, marker_id: int, side: float):
    """One big vehicle tag. Replaces the two-tag front/back arrangement.

    The September tags failed (4-16% detection) for mounting reasons, not
    detection ones: no white margin, tilted on the sloped deck, ~45 px in
    frame. This sheet is the largest tag that fits A4 with a full one-module
    quiet zone, and the notes are the mounting rules, printed where they
    will be read.
    """
    return marker_sheet(
        paper, ARUCO_DICT, ARUCO_DICT_NAME, marker_id, side,
        "VEHICLE TAG  —  the ONE tag on the car (Branch A)",
        [
            "CUT ON THE DASHED LINE, NOT AT THE BLACK EDGE — the white margin is the quiet zone. Without it",
            "the black border merges into the chassis and the detector cannot see the tag (Sept: 4-16%).",
            "Mount FLAT and LEVEL on a raised plate above the deck, never on the sloped bodywork. FRONT forward.",
            "Measure tag height above the floor and offset from the rear axle — track_ground.py needs both.",
        ],
        ["Measured side (mm):", "Height above floor (mm):", "Offset from rear axle (mm):", "Date mounted:"],
    )


def scalebar_sheet(paper: str, marker_id: int, side: float):
    return marker_sheet(
        paper, ARUCO_DICT, ARUCO_DICT_NAME, marker_id, side,
        "SCALE BAR ENDPOINT  —  independent metric check",
        [
            "Mount IDs 30 and 31 on one rigid bar (aluminium extrusion / straight timber), coplanar.",
            "Measure centre-to-centre with a laser distance meter or steel tape, twice, and average.",
            "This single long baseline is the cross-check on the scale the GCP network solves for —",
            "keep the bar in frame during the capture, lying in the track plane.",
        ],
        ["Measured side (mm):", "Bar centre-centre (mm):", "Repeat measurement (mm):"],
    )


# --------------------------------------------------------------------------
# ChArUco calibration board
# --------------------------------------------------------------------------
# Landscape boards, sized so the board plus a ~60 mm footer fits the sheet.
CHARUCO = {
    #        squares (x, y), square mm, marker mm
    "A4": ((8, 5), 25.0, 18.5),
    "A3": ((9, 6), 35.0, 26.0),
}


def charuco_sheet(paper: str):
    (sx, sy), sq, mk = CHARUCO[paper]
    ph, pw = PAPER_MM[paper]  # landscape: swap
    board = cv2.aruco.CharucoBoard((sx, sy), sq, mk, CHARUCO_DICT)

    # Integer pixels per square => every edge lands exactly on a pixel boundary,
    # so the board stays crisp regardless of the resampling the printer does.
    px = 480
    img = board.generateImage((sx * px, sy * px), 0, 1)

    bw, bh = sx * sq, sy * sq
    x0, y0 = (pw - bw) / 2.0, 12.0
    if x0 < 8.0 or y0 + bh > ph - 72.0:
        raise ValueError(f"ChArUco {paper} board does not fit the page")

    fig, ax = new_page(pw, ph)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(x0, x0 + bw, y0 + bh, y0),
              interpolation="nearest", zorder=1)

    ty = y0 + bh + 12.0
    ax.text(pw / 2.0, ty, "ChArUco CALIBRATION BOARD", fontsize=16, weight="bold",
            ha="center", va="baseline")
    ty += 7.0
    ax.text(pw / 2.0, ty,
            f"{CHARUCO_DICT_NAME}   |   {sx} x {sy} squares   |   square {sq:.1f} mm   |   marker {mk:.1f} mm",
            fontsize=8.5, ha="center", va="baseline")
    ty += 9.0
    draw_scale_check(ax, 18.0, ty)
    ty += 15.0
    for n in [
        f"Different dictionary from the track markers ({ARUCO_DICT_NAME}) on purpose — this board can never be mistaken for a GCP.",
        "Mount dead flat on glass, MDF or foamboard. A board that flexes produces a distortion model that is wrong everywhere.",
        "Calibrate once per camera AND per capture setting — resolution, FOV/lens mode or digital zoom each change the intrinsics.",
        "~30 stills: fill the frame, then the corners, then strong 30-45 deg tilts in both axes. Tilt is what separates focal length from distortion.",
    ]:
        ax.text(18.0, ty, n, fontsize=7, ha="left", va="baseline")
        ty += 4.6
    ty += 3.0
    for i, f in enumerate(["Camera / settings:", "Measured square (mm):", "Date:"]):
        field(ax, 18.0 + i * (pw - 40.0) / 3.0, ty, f, width=(pw - 40.0) / 3.0 - 46.0)
    if ty > ph - 6.0:
        raise ValueError(f"ChArUco footer overflows {paper}: {ty:.1f} mm")
    return fig


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "print"))
    ap.add_argument("--paper", default="A4,A3", help="comma list: A4, A3")
    ap.add_argument("--only", default="gcp,vehicle,car,scalebar,charuco", help="comma list of sheet types")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    papers = [p.strip().upper() for p in args.paper.split(",") if p.strip()]
    want = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    written: list[str] = []

    def emit(name: str, figs):
        path = os.path.join(args.outdir, name)
        with PdfPages(path) as pdf:
            for f in figs:
                pdf.savefig(f)
                plt.close(f)
        written.append(path)

    for paper in papers:
        gcp_side = GCP_SIDE_MM_A4 if paper == "A4" else GCP_SIDE_MM_A3
        if "gcp" in want:
            emit(f"GCP_{ARUCO_DICT_NAME}_{paper}_{gcp_side:.0f}mm_id00-{GCP_MARKER_IDS[-1]:02d}.pdf",
                 [gcp_sheet(paper, i, gcp_side) for i in GCP_MARKER_IDS])
        if "vehicle" in want and paper == "A4":
            # Two sizes: 100 mm fits a RoboRacer deck, 150 mm if the deck allows
            # it — bigger marker, better corner precision from the overhead cam.
            for s in (VEHICLE_SIDE_MM, 150.0):
                emit(f"VEHICLE_{ARUCO_DICT_NAME}_A4_{s:.0f}mm_id20-21.pdf",
                     [vehicle_sheet(paper, i, s) for i in VEHICLE_MARKER_IDS])
        if "car" in want and paper == "A4":
            emit(f"CAR_TAG_{ARUCO_DICT_NAME}_A4_{GCP_SIDE_MM_A4:.0f}mm_id20.pdf",
                 [car_sheet(paper, VEHICLE_MARKER_IDS[0], GCP_SIDE_MM_A4)])
        if "scalebar" in want and paper == "A4":
            emit(f"SCALEBAR_{ARUCO_DICT_NAME}_A4_{GCP_SIDE_MM_A4:.0f}mm_id30-31.pdf",
                 [scalebar_sheet(paper, i, GCP_SIDE_MM_A4) for i in SCALE_BAR_MARKER_IDS])
        if "charuco" in want:
            emit(f"CHARUCO_{CHARUCO_DICT_NAME}_{paper}.pdf", [charuco_sheet(paper)])

    # Survey template — the ONLY place the real-world GCP coordinates live.
    # Umeyama consumes exactly these rows, so the header is the schema.
    csv_path = os.path.join(args.outdir, "gcp_survey_template.csv")
    with open(csv_path, "w") as fh:
        fh.write("marker_id,role,nominal_side_mm,measured_side_mm,x_m,y_m,z_m,"
                 "meas_method,use_as,notes\n")
        for i in GCP_MARKER_IDS:
            fh.write(f"{i},gcp,,,,,,laser,fit,\n")
        for i in SCALE_BAR_MARKER_IDS:
            fh.write(f"{i},scalebar,,,,,,laser,check,\n")
        for i in VEHICLE_MARKER_IDS:
            fh.write(f"{i},vehicle,,,,,,,exclude,mounted on car - not a GCP\n")
    written.append(csv_path)

    print(f"\nWrote {len(written)} file(s) to {os.path.abspath(args.outdir)}:")
    for p in written:
        print("  " + os.path.basename(p))
    print("\nPRINT AT 100% / 'Actual size'. Disable 'Fit to page' and 'Shrink oversized pages'.")
    print("Then measure the 100.0 mm bar on each sheet before mounting anything.")


if __name__ == "__main__":
    main()
