"""Figures v2 (18 Nov 2026): Fig 1 heat map, Fig 2 clinician figure, graphical abstract, Turkish scenario (supplement).
Uses analysis_v2 (NHANES concentration-dependent CV profile; revised asthma populations)."""
import io, contextlib, importlib.util
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import norm

spec = importlib.util.spec_from_file_location("a2", "analysis_v2.py"); A = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(A)
from dose_tables import TABLES, row_index, col_index, mg4w
OUT = A.OUT; d = A.d
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#1f1f1e", "ink2": "#5c5b56", "grid": "#e6e5e0", "band": "#fbe3d8"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "savefig.dpi": 600, "legend.frameon": False})

def p_inel(x, n=1, cv="profile"):
    s = A.s_of(cv, np.asarray(x, float), 0.0, n); return norm.cdf(np.log(30 / np.asarray(x, float)) / s)
nums = {"p32_single": float(p_inel([32])[0]), "p32_mean2": float(p_inel([32], 2)[0]),
        "p35_single": float(p_inel([35])[0])}

# ---------------------------------------------------------------- Figure 1
t = TABLES["KUB"]
ig = np.exp(np.linspace(np.log(30), np.log(1500), 400)); wg = np.linspace(20.01, 150, 300)
R = np.full((len(ig), len(wg)), np.nan)
for j, w in enumerate(wg):
    o = A.child_outcomes(t, ig, np.full(len(ig), w), "profile")
    R[:, j] = np.where(o[:, 0] == 1, o[:, 1] + o[:, 2], np.nan)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
cmap = ListedColormap(["#f3f6fb", "#c9daf3", "#8fb5ea", "#4f8ddf", "#1f5fb8", "#0d3a78"])
bnd = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.51]
im = ax.pcolormesh(wg, ig, R, cmap=cmap, norm=BoundaryNorm(bnd, cmap.N), shading="auto")
for e in t["ige"][1:-1]: ax.axhline(e, color="white", lw=0.5)
for e in t["wt"][1:-1]: ax.axvline(e, color="white", lw=0.5)
g = d[A.POPS["A"][1]]
ax.scatter(g.BMXWT, g.LBXIGE, s=7, c=C["orange"], edgecolors="none", alpha=0.85, label="NHANES adolescents 12–17 y,\ncurrent asthma + aeroallergen sensitization")
ax.set_yscale("log"); ax.set_ylim(30, 1500); ax.set_xlim(20, 150)
ax.set_yticks(t["ige"]); ax.set_yticklabels([str(x) for x in t["ige"]], fontsize=6.5)
ax.set_xticks(t["wt"]); ax.set_xticklabels([str(x) for x in t["wt"]], fontsize=7)
ax.set_xlabel("Body weight (kg)"); ax.set_ylabel("True pretreatment total IgE (IU/mL)"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, ticks=bnd); cb.set_ticklabels(["0", "5", "10", "20", "30", "40", "50"])
cb.set_label("P(different regimen or ineligible), %")
ax.set_title("Turkish KÜB dosing table: misclassification probability (NHANES assay CV profile, 8.4–8.7%)", fontsize=8.5, color=C["ink"])
ax.legend(loc="upper right", fontsize=6.5, markerscale=2)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"Fig1_KUB_heatmap.{ext}")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
rng = np.random.default_rng(7)
fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.5, wspace=0.3)
cases = [(32, 30, "below", "classified ineligible", "A1  True IgE 32 IU/mL → 150 mg every 4 weeks"),
         (95, 100, "above", "given 300 mg (double dose)", "A2  True IgE 95 IU/mL → 150 mg every 4 weeks")]
for k, (true, border, side, wrong_lab, title) in enumerate(cases):
    ax = fig.add_subplot(gs[0, k]); S = float(A.s_of("profile", np.array([true]))[0])
    m = true * np.exp(rng.normal(0, S, 200)); y = rng.uniform(-1, 1, 200)
    wrong = m < border if side == "below" else m > border
    ax.scatter(m[~wrong], y[~wrong], s=10, color=C["blue"], alpha=0.75, lw=0)
    ax.scatter(m[wrong], y[wrong], s=10, color=C["orange"], alpha=0.9, lw=0)
    ax.vlines(true, -1.3, 1.15, color=C["ink"], lw=1.3); ax.vlines(border, -1.3, 1.15, color=C["ink2"], lw=1, ls="--")
    p = norm.cdf(np.log(border / true) / S) if side == "below" else 1 - norm.cdf(np.log(border / true) / S)
    left = border < true
    ax.text(true + (0.01 if left else -0.01) * true, 1.25, "true value", ha="left" if left else "right", fontsize=7.5)
    ax.text(border + (-0.01 if left else 0.01) * true, 1.25, f"border {border}", ha="right" if left else "left", fontsize=7.5, color=C["ink2"])
    ax.set_ylim(-1.3, 2.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(true * 0.72, true * 1.33); ax.set_xlabel("Reported total IgE (IU/mL)")
    ax.set_title(title, fontsize=8.2, loc="left")
    ax.text(0.5, 0.97, f"{100*p:.0f}% of results: {wrong_lab}", transform=ax.transAxes, ha="center", va="top", fontsize=8, color=C["orange"], fontweight="bold")
# B: probability of false ineligibility vs true IgE
ax = fig.add_subplot(gs[1, 0]); xs = np.linspace(30.01, 40, 300)
ax.plot(xs, 100 * p_inel(xs), color=C["orange"], lw=1.8, label="single measurement")
ax.plot(xs, 100 * p_inel(xs, 2), color=C["blue"], lw=1.8, ls="--", label="mean of two measurements")
ax.plot(xs, 100 * p_inel(xs, 1, 0.05), color=C["aqua"], lw=1.4, ls=":", label="single, CV 5%")
ax.set_xlabel("True total IgE (IU/mL)"); ax.set_ylabel("Probability of being\nmeasured <30 IU/mL (%)")
ax.set_title("B  Exclusion risk just above 30 IU/mL", fontsize=8.5, loc="left"); ax.set_xlim(30, 40); ax.set_ylim(0, 50)
ax.grid(True, color=C["grid"], lw=0.6); ax.legend(fontsize=7)
# C: different regimen vs CV (revised populations)
ax = fig.add_subplot(gs[1, 1]); cvs = np.linspace(0.02, 0.20, 19)
for key, lab, col, ls in (("A", "Asthma 12–17 y (KÜB)", C["blue"], "-"), ("B", "Asthma 6–11 y (KÜB)", C["aqua"], "--"),
                          ("E", "Food-sensitized 1–17 y (FDA T4)", C["orange"], "-.")):
    ax.plot(100 * cvs, [100 * A.run(key, cv=c)["diff"] for c in cvs], color=col, ls=ls, lw=1.6, label=lab)
for x_, lab, ha, dx in ((5, "nephelometry", "right", -0.3), (8.5, "NHANES\nImmunoCAP", "left", 0.3)):
    ax.axvline(x_, color=C["ink2"], lw=0.7, ls=":"); ax.text(x_ + dx, 23.5, lab, fontsize=6.5, color=C["ink2"], ha=ha, va="top")
ax.set_xlabel("Analytical CV of total IgE (%)"); ax.set_ylabel("Eligible children with a\ndifferent regimen (%)")
ax.set_title("C  Effect of assay imprecision", fontsize=8.5, loc="left"); ax.grid(True, color=C["grid"], lw=0.6)
ax.set_ylim(0, 26); ax.legend(fontsize=6.5, loc="lower right")
for ext in ("png", "pdf"): fig.savefig(OUT / f"Fig2_clinician.{ext}", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- graphical abstract
MA = A.run("A")
t = TABLES["KUB"]; W = 55; col = int(col_index(t, W)); x = np.linspace(20, 395, 2000)
step = np.array([mg4w(t, int(r), col) if r >= 0 else 0 for r in row_index(t, x)])
fig = plt.figure(figsize=(9, 6)); ax = fig.add_axes([0.08, 0.13, 0.55, 0.68])
ax.plot(x, step, color=C["blue"], lw=2.6, label="Dosing table, 55 kg (Turkish KÜB)")
ax.plot(x, 0.016 * W * x, color=C["ink2"], lw=1, ls="--", label="Formula requirement (0.016 × kg × IgE)")
S95 = float(A.s_of("profile", np.array([95]))[0]); lo, hi = 95 * np.exp(-1.96 * S95), 95 * np.exp(1.96 * S95)
ax.axvspan(lo, hi, color=C["band"], zorder=0); ax.axvline(95, color=C["ink"], lw=1.2)
ax.text(hi + 4, 420, "95% of reported values\nfor true IgE 95", fontsize=8.5, color=C["orange"])
ax.axvspan(20, 30, color="#eeeeee", zorder=0); ax.text(22, 470, "<30:\nnot\neligible", fontsize=7.5, color=C["ink2"])
ax.set_xlim(20, 395); ax.set_ylim(0, 520); ax.grid(True, color=C["grid"], lw=0.6)
ax.set_xlabel("Pretreatment total IgE (IU/mL)"); ax.set_ylabel("Omalizumab dose per 4 weeks (mg), 55 kg")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.text(0.08, 0.92, "One measurement, one dose", fontsize=17, fontweight="bold", color=C["ink"])
fig.text(0.08, 0.865, "Routine total IgE error and the steps of the omalizumab dosing table", fontsize=10.5, color=C["ink2"])
items = ((0.70, f"{100*nums['p32_single']:.0f}%", "of children with true IgE\n32 IU/mL measured <30\nand excluded", C["orange"]),
         (0.45, f"1 in {round(1/MA['diff'])}", "eligible adolescents given a\ndifferent dose or interval,\nnever below the formula need", C["blue"]),
         (0.20, f"{100*nums['p32_mean2']:.0f}%", "exclusion risk at 32 IU/mL\nwhen a value near 30\nis repeated and averaged", C["aqua"]))
for y, big, small, colr in items:
    fig.text(0.69, y, big, fontsize=24, fontweight="bold", color=colr)
    fig.text(0.69, y - 0.105, small, fontsize=9.5, color=C["ink"], va="bottom")
fig.text(0.08, 0.03, "NHANES 2005–2006 children with current asthma and aeroallergen sensitization; NHANES assay CV profile (8.4–8.7%).",
         fontsize=7.5, color=C["ink2"])
for ext in ("png", "tiff"): fig.savefig(OUT / f"graphical_abstract.{ext}", dpi=400)
plt.close(fig)

# ---------------------------------------------------------------- Turkish scenario (supplement)
from neyzi_lms import sample_weight
RNG = np.random.default_rng(20261118)
bars = pd.read_csv("digit/altan_ige_bars.csv")
bars["sigma"] = (np.log(bars.p975) - np.log(bars["median"])) / 1.959964
bars["age"] = 1 + 17 * (np.arange(len(bars)) + 0.5) / len(bars)
rows = []
for lab, lo_, hi_ in (("12–17 y", 12, 17), ("6–11 y", 6, 11)):
    b = bars[(bars.age >= lo_) & (bars.age < hi_ + 1)]
    ige_ = np.concatenate([np.exp(RNG.normal(np.log(r["median"]), r.sigma, 4000)) for _, r in b.iterrows()])
    ages = np.repeat(np.floor(b.age).astype(int).to_numpy(), 4000); wt_ = np.empty_like(ige_)
    for a in np.unique(ages):
        mm = ages == a; wt_[mm] = sample_weight(a, mm.sum(), RNG)
    for cvlab, cv in (("NHANES profile", "profile"), ("5% (nephelometry)", 0.05), ("11%", 0.11)):
        o = A.child_outcomes(TABLES["KUB"], ige_, wt_, cv); e = o[:, 0] == 1; band = e & (ige_ >= 30) & (ige_ < 40)
        rows.append({"Age": lab, "CV": cvlab, "Eligible %": round(100 * e.mean(), 1), "Eligible with IgE 30–40 %": round(100 * band.sum() / e.sum(), 1),
                     "False ineligibility %": round(100 * o[e, 1].mean(), 1), "False inelig., IgE 30–40 %": round(100 * o[band, 1].mean(), 1),
                     "Different regimen %": round(100 * o[e, 2].mean(), 1), "False eligibility %": round(100 * o[~e, 6].mean(), 1)})
TS = pd.DataFrame(rows); TS.to_csv(OUT / "S_turkish_scenario.csv", index=False)
print(nums); print(TS.to_string(index=False)); print("MA diff", MA["diff"], "1 in", round(1 / MA["diff"]))
