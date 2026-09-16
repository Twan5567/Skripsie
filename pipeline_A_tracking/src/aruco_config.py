"""Single source of truth for ArUco marker parameters.

Import from this module everywhere a script needs the dictionary, marker
IDs, or marker side length — never hardcode these values elsewhere, so
switching between the test print and the full-size set is a one-line change.

Current status: validating detection + paint/print workflow on ONE
scaled-down test marker before printing the full set of 8 at intended size.
"""
import cv2

# --- Dictionary -------------------------------------------------------
# Must match the physical markers actually on the track. Confirmed 2026-08:
# DICT_4X4_50 (NOT DICT_5X5_50 — the DICT_5X5_50 paper-sheet approach from
# earlier Phase 0 notes is abandoned; the STL plates were always 4X4_50).
ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))

# --- Marker IDs ---------------------------------------------------------
# Single-marker validation print (current phase).
TEST_MARKER_IDS = [0]

# Full set for the actual capture, printed later at full size.
FULL_MARKER_IDS = list(range(8))

# --- Marker side length (mm) --------------------------------------------
# TEST_MARKER_SIDE_MM feeds any scaling/pose-estimation code during this
# validation phase. It starts at the nominal STL dimension — MEASURE THE
# ACTUAL PRINTED MARKER WITH CALIPERS once painted and replace this value.
# Do not hardcode marker size anywhere else; import it from here.
TEST_MARKER_SIDE_MM = 100.0  # TODO: replace with caliper-measured value

# Full-size markers for the real capture (per files/aruco_4x4_50_id*_250mm_
# plate280mm.stl). Not yet in use — update when the full set is printed and
# measured.
FULL_MARKER_SIDE_MM = 250.0  # TODO: replace with caliper-measured value

# --- Printed A4 marker sheet (mm) ----------------------------------------
# The Tracking/ captures use the A4 sheet rendered by generate_markers.py,
# NOT either of the STL plates above. Its black square is fixed by the page
# layout in that script: (A4_W_PX - 2*MARGIN_PX) = 2480 - 300 = 2180 px at
# 300 DPI, i.e. 2180/300 in = 184.57 mm. Validated against Slide_700_take1
# (700 mm tape ground truth): this value gives 708 mm via ground-plane
# homography, +1.2%. Using 250.0 here instead yields 1039 mm (+48%).
# Printer scaling can still shift this a percent or two — measure the
# printed square with calipers and replace the value.
A4_SHEET_SIDE_MM = 184.57  # TODO: replace with caliper-measured value
