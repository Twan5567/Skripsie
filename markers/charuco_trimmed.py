
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
from generate_print_sheets import CHARUCO, PAPER_MM, new_page, CHARUCO_DICT, PdfPages, plt

(sx, sy), sq, mk = CHARUCO["A4"]
ph, pw = PAPER_MM["A4"]
board = cv2.aruco.CharucoBoard((sx, sy), sq, mk, CHARUCO_DICT)
px = 480


img = board.generateImage((sx * px, sy * px), 0, 1)
bw, bh = sx * sq, sy * sq
x0, y0 = (pw - bw) / 2.0, (ph - bh) / 2.0
fig, ax = new_page(pw, ph)


ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(x0, x0 + bw, y0 + bh, y0),
          interpolation="nearest", zorder=1)

gap, arm = 5.0, 6.0   


for cx in (x0, x0 + bw):
    ax.plot([cx, cx], [y0 - gap - arm, y0 - gap], color="black", lw=0.5)
    ax.plot([cx, cx], [y0 + bh + gap, y0 + bh + gap + arm], color="black", lw=0.5)
for cy in (y0, y0 + bh):
    ax.plot([x0 - gap - arm, x0 - gap], [cy, cy], color="black", lw=0.5)
    ax.plot([x0 + bw + gap, x0 + bw + gap + arm], [cy, cy], color="black", lw=0.5)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "print", "CHARUCO_DICT_5X5_100_A4_trimmed.pdf")
with PdfPages(out) as pdf:
    pdf.savefig(fig)
print(out, f"board {bw:.0f} x {bh:.0f} mm at ({x0:.1f}, {y0:.1f}) on {pw:.0f} x {ph:.0f} mm")
