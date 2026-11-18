"""Figure 2 (clinician-oriented).
A: 200 simulated repeat measurements of the same sample (CV 11%) for two adolescents (55 kg, Turkish table).
B: proportion of eligible children assigned a different regimen by analytical CV (NHANES populations)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
from core import child_outcomes, TABLES

rng = np.random.default_rng(7)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#1f1f1e", "ink2": "#5c5b56", "grid": "#e6e5e0"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "savefig.dpi": 600, "legend.frameon": False})
CV = 0.11; S = np.sqrt(np.log1p(CV ** 2))

fig = plt.figure(figsize=(7.2, 5.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.45, wspace=0.25)
cases = [(32, 30, "below", "classified ineligible", "A1  True IgE 32 IU/mL → 150 mg every 4 weeks"),
         (95, 100, "above", "given 300 mg (double dose)", "A2  True IgE 95 IU/mL → 150 mg every 4 weeks")]
for k, (true, border, side, wrong_lab, title) in enumerate(cases):
    ax = fig.add_subplot(gs[0, k])
    m = true * np.exp(rng.normal(0, S, 200)); y = rng.uniform(-1, 1, 200)
    wrong = m < border if side == "below" else m > border
    ax.scatter(m[~wrong], y[~wrong], s=10, color=C["blue"], alpha=0.75, lw=0, label="correct regimen")
    ax.scatter(m[wrong], y[wrong], s=10, color=C["orange"], alpha=0.9, lw=0, label=wrong_lab)
    ax.axvline(true, color=C["ink"], lw=1.3); ax.axvline(border, color=C["ink2"], lw=1, ls="--")
    p = norm.cdf(np.log(border / true) / S) if side == "below" else 1 - norm.cdf(np.log(border / true) / S)
    left = border < true
    ax.text(true + (0.01 if left else -0.01) * true, 1.25, "true value", ha="left" if left else "right", fontsize=7.5, color=C["ink"])
    ax.text(border + (-0.01 if left else 0.01) * true, 1.25, f"border {border}", ha="right" if left else "left", fontsize=7.5, color=C["ink2"])
    ax.set_ylim(-1.3, 2.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(true * 0.7, true * 1.38); ax.set_xlabel("Reported total IgE (IU/mL)")
    ax.set_title(title, fontsize=8.3, loc="left", color=C["ink"])
    ax.text(0.5, 0.97, f"{100*p:.0f}% of results: {wrong_lab}", transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=C["orange"], fontweight="bold")

# B
ige = pd.read_sas("data/AL_IGE_D.xpt")[["SEQN", "LBXIGE"]]; demo = pd.read_sas("data/DEMO_D.xpt")[["SEQN", "RIDAGEYR", "WTMEC2YR"]]
bmx = pd.read_sas("data/BMX_D.xpt")[["SEQN", "BMXWT"]]; mcq = pd.read_sas("data/MCQ_D.xpt")[["SEQN", "MCQ010"]]
al = pd.read_sas("data/AL_IGE_D.xpt")[["SEQN", "LBXIF1", "LBXIF2", "LBXF13"]]
d = ige.merge(demo, on="SEQN").merge(bmx, on="SEQN").merge(mcq, on="SEQN", how="left").merge(al, on="SEQN").dropna(subset=["LBXIGE", "BMXWT"])
d = d[d.RIDAGEYR < 18]; asth = d.MCQ010 == 1; food = (d[["LBXIF1", "LBXIF2", "LBXF13"]] >= 0.35).any(axis=1)
pops = [("Asthma 12–17 y (Turkish table)", (d.RIDAGEYR >= 12) & asth, "KUB", C["blue"], "-"),
        ("Asthma 6–11 y (Turkish table)", d.RIDAGEYR.between(6, 11) & asth, "KUB", C["aqua"], "--"),
        ("Food-sensitized 1–17 y (FDA food-allergy table)", food, "FDA_T4", C["orange"], "-.")]
ax = fig.add_subplot(gs[1, :]); cvs = np.linspace(0.02, 0.20, 19)
for lab, mask, tk, col, ls in pops:
    g = d[mask]; w = g.WTMEC2YR.to_numpy(); ys = []
    for cv in cvs:
        o = child_outcomes(TABLES[tk], g.LBXIGE.to_numpy(), g.BMXWT.to_numpy(), cv); e = o[:, 0] == 1
        ys.append(100 * np.average(o[e, 2], weights=w[e]))
    ax.plot(100 * cvs, ys, color=col, ls=ls, lw=1.7, label=lab)
for x, lab in ((5, "nephelometry\n~4–5%"), (11, "ImmunoCAP\n~8–11%")):
    ax.axvline(x, color=C["ink2"], lw=0.7, ls=":"); ax.text(x + 0.25, 0.8, lab, fontsize=7, color=C["ink2"], va="bottom")
ax.set_xlabel("Analytical CV of total IgE (%)"); ax.set_ylabel("Eligible children with a\ndifferent regimen (%)")
ax.set_title("B  Effect of assay imprecision in the NHANES populations", fontsize=8.5, loc="left", color=C["ink"])
ax.grid(True, color=C["grid"], lw=0.6); ax.set_ylim(0, 22); ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.0))
for ext in ("png", "pdf"): fig.savefig(f"results/Fig2_clinician.{ext}", bbox_inches="tight")
print("ok")
