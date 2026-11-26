#!/usr/bin/env python3
"""Supplementary Figure S1: check of the implemented Lowe 2009 binding model.
A: typical-patient profiles (70 kg, baseline IgE 150 IU/mL) after 150 mg and 300 mg every 4 weeks.
B: free IgE at the midpoint of weeks 20-24 for every cell of the Turkish/European dosing table,
   evaluated at the cell's upper IgE and upper weight limits (the least favourable point in the cell)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.integrate import solve_ivp
import pkpd_v2 as P

def profile(wt, ige_iu, dose, every_w, t_end=168.0):
    p = P.params(wt, ige_iu * P.IU2NG)
    y = np.array([0.0, 0.0, ige_iu * P.IU2NG / P.MWE * p["vx"]])
    tau = 7.0 * every_w; T, F, X = [], [], []
    for i, t0 in enumerate(np.arange(0.0, t_end, tau)):
        y[0] += dose; t1 = min(t0 + tau, t_end)
        tt = np.linspace(t0, t1, 60)
        s = solve_ivp(P.rhs, (t0, t1), y, args=(p,), method="LSODA", rtol=1e-8, atol=1e-10, t_eval=tt)
        for k in range(len(s.t)):
            xc = P.binding(s.y[1, k], s.y[2, k], p)
            T.append(s.t[k] / 7); F.append((s.y[2, k] - xc) / p["vx"] * P.MWE)
            X.append(((s.y[1, k] - xc) / p["vx"] + xc / p["vc"]) * P.MWX / 1000)  # total omalizumab, ug/mL
        y = s.y[:, -1].copy()
    return np.array(T), np.array(F), np.array(X)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_axes([0.06, 0.18, 0.36, 0.73])
for dose, col in ((150, "#E07B39"), (300, "#2B6CB0")):
    T, F, X = profile(70, 150, dose, 4)
    ax.plot(T, F, color=col, lw=1.8, label=f"{dose} mg every 4 weeks")
ax.axhline(14, color="k", ls="--", lw=0.9); ax.text(21.2, 15.2, "14 ng/mL", fontsize=8)
ax.axhline(25, color="grey", ls=":", lw=0.9); ax.text(21.2, 26.5, "25 ng/mL", fontsize=8, color="grey")
ax.set_yscale("log"); ax.set_ylim(3, 500); ax.set_xlim(0, 24)
ax.set_xlabel("Weeks"); ax.set_ylabel("Free IgE (ng/mL)")
ax.set_title("A. Typical patient, 70 kg, baseline IgE 150 IU/mL", loc="left", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper right")

t = P.TABLES["KUB"]; ige_e, wt_e = np.asarray(t["ige"], float), np.asarray(t["wt"], float)
M = np.full((len(t["rows"]), len(wt_e) - 1), np.nan)
for r in range(len(t["rows"])):
    for c in range(len(wt_e) - 1):
        cl = P.cell(t, r, c)
        if cl is None: continue
        M[r, c] = P.free_mid(wt_e[c + 1], ige_e[r + 1], cl[0], cl[1])[0]
ax2 = fig.add_axes([0.53, 0.18, 0.36, 0.73])
bounds = [0, 8, 10, 12, 14, 16, 20, 25, 40]
cmap = ListedColormap(["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"])
im = ax2.imshow(M, origin="lower", aspect="auto", cmap=cmap, norm=BoundaryNorm(bounds, cmap.N))
for r in range(M.shape[0]):
    for c in range(M.shape[1]):
        if not np.isnan(M[r, c]):
            ax2.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center", fontsize=6.5,
                     color="white" if M[r, c] < 10 or M[r, c] > 20 else "black")
ax2.set_xticks(range(len(wt_e) - 1)); ax2.set_xticklabels([f"{int(a)}–{int(b)}" for a, b in zip(wt_e[:-1], wt_e[1:])], rotation=45, fontsize=7)
ax2.set_yticks(range(len(ige_e) - 1)); ax2.set_yticklabels([f"{int(a)}–{int(b)}" for a, b in zip(ige_e[:-1], ige_e[1:])], fontsize=7)
ax2.set_xlabel("Body weight (kg)"); ax2.set_ylabel("Baseline total IgE (IU/mL)")
ax2.set_title("B. Free IgE at the upper corner of each table cell", loc="left", fontsize=9)
cb = fig.colorbar(im, cax=fig.add_axes([0.905, 0.18, 0.015, 0.73])); cb.set_label("Free IgE, weeks 20–24 midpoint (ng/mL)")
fig.savefig("results_v2/FigS1_pkpd_check.png", dpi=600)
print("cells:", np.sum(~np.isnan(M)), "max", np.nanmax(M), "median", np.nanmedian(M), "share>14", np.nanmean(M > 14), "share>25", np.nanmean(M > 25))
