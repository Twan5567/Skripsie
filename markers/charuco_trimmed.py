import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_print_sheets import PAPER_MM, PdfPages, charuco_size, draw_charuco, draw_corner_ticks, new_page

ph, pw = PAPER_MM["A4"]
bw, bh = charuco_size("A4")
x0, y0 = (pw - bw) / 2.0, (ph - bh) / 2.0
fig, ax = new_page(pw, ph)
draw_charuco(ax, "A4", x0, y0)
draw_corner_ticks(ax, x0, y0, bw, bh, gap=5.0, arm=6.0)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "print", "CHARUCO_DICT_5X5_100_A4_trimmed.pdf")
with PdfPages(out) as pdf:
    pdf.savefig(fig)
print(out, f"board {bw:.0f} x {bh:.0f} mm at ({x0:.1f}, {y0:.1f}) on {pw:.0f} x {ph:.0f} mm")
