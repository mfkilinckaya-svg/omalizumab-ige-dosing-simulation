"""Graphical abstract for PAI: 9:6 landscape, 400 dpi."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from dose_tables import TABLES, row_index, col_index, mg4w

C = {"blue": "#2a78d6", "orange": "#eb6834", "ink": "#1f1f1e", "ink2": "#5c5b56", "grid": "#e6e5e0", "band": "#fbe3d8"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"]})
t = TABLES["KUB"]; W = 55; col = int(col_index(t, W))
x = np.linspace(20, 395, 2000)
step = np.array([mg4w(t, int(r), col) for r in row_index(t, x)])
ramp = 0.016 * W * x

fig = plt.figure(figsize=(9, 6))
ax = fig.add_axes([0.08, 0.13, 0.55, 0.68])
ax.plot(x, step, color=C["blue"], lw=2.6, label="Dosing table, 55 kg (Turkish SmPC)")
true, lo, hi = 95, 95 * 0.806, 95 * 1.24
ax.axvspan(lo, hi, color=C["band"], zorder=0)
ax.axvline(true, color=C["ink"], lw=1.2)
ax.annotate("", xy=(hi, 420), xytext=(lo, 420), arrowprops=dict(arrowstyle="<->", color=C["orange"], lw=1.5))
ax.text(hi + 4, 405, "95% of reported values\nfor a child with true IgE 95", ha="left", va="bottom", fontsize=8.5, color=C["orange"])
ax.annotate("150 mg", xy=(85, 150), xytext=(40, 215), fontsize=9, color=C["blue"], arrowprops=dict(arrowstyle="-", color=C["blue"], lw=0.8))
ax.annotate("300 mg", xy=(160, 300), xytext=(175, 360), fontsize=9, color=C["blue"], arrowprops=dict(arrowstyle="-", color=C["blue"], lw=0.8))
ax.set_xlim(20, 395); ax.set_ylim(0, 520)
ax.set_xlabel("Pretreatment total IgE (IU/mL)"); ax.set_ylabel("Omalizumab dose per 4 weeks (mg), 55 kg")
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
ax.grid(True, color=C["grid"], lw=0.6)

fig.text(0.08, 0.92, "One measurement, one dose", fontsize=17, fontweight="bold", color=C["ink"])
fig.text(0.08, 0.865, "Routine total IgE error moves children across the steps of the omalizumab dosing table",
         fontsize=10.5, color=C["ink2"])
for y, big, small in ((0.70, "1 in 8", "eligible children assigned a\ndifferent dose or interval\n(analytical CV 11%)"),
                      (0.45, "2.3%", "eligible adolescents measured\n<30 IU/mL and excluded\nfrom reimbursed treatment"),
                      (0.20, "↓ 25%", "fewer misclassifications when\nborderline values are repeated\nand averaged")):
    fig.text(0.69, y, big, fontsize=24, fontweight="bold", color=C["orange"] if y > 0.3 else C["blue"])
    fig.text(0.69, y - 0.105, small, fontsize=9.5, color=C["ink"], va="bottom")
fig.text(0.08, 0.03, "NHANES 2005–2006 children; Turkish SmPC and FDA dosing tables; simulation of analytical error.",
         fontsize=7.5, color=C["ink2"])
for ext in ("png", "tiff"):
    fig.savefig(f"results/graphical_abstract.{ext}", dpi=400)
print("ok")
