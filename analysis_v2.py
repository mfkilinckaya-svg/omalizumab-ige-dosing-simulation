#!/usr/bin/env python3
"""
Total IgE measurement error and omalizumab dosing in children – revised analysis (v2, 18 Nov 2026).

Changes from v1 (reviewer comments):
- Asthma populations: current asthma (MCQ010 = 1 and MCQ035 = 1) AND sensitization to >=1 perennial
  aeroallergen (sIgE >= 0.35 kU/L to D. pteronyssinus, D. farinae, cat, dog, German cockroach or Alternaria).
  Sensitivity: ever asthma (v1 definition); current asthma + attack in past year (MCQ040 = 1).
- Main analytical CV: concentration-dependent profile from the NHANES 2005-2006 total IgE QC data
  (Laboratory Procedure Manual AL_IGE_D, section 19): pooled CV 8.6% at ~24, 8.4% at ~102, 8.7% at ~438 kU/L,
  linear interpolation on log concentration, constant outside the control range. Constant CVs 5/8/11/15% as
  sensitivity analyses.
- Primary outcome: false ineligibility at the 30 IU/mL threshold (and table limits); also reported for
  children with true IgE 30-40 IU/mL. Secondary: different regimen; dose below 80% of the formula dose
  (0.016 mg x kg x IU/mL per 4 weeks); composite "clinically important misclassification".
- Bias +/-5, 10, 20%.
- 95% CIs: Rao-Wu rescaling bootstrap of PSUs within strata (SDMVSTRA/SDMVPSU), 1000 replicates.
- Repeat strategies simulated (Monte Carlo): repeat-all, conditional repeat within +/-20% of a boundary,
  and a round-up rule.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from dose_tables import TABLES, row_index, col_index, cell, mg4w

D = Path("data"); OUT = Path("results_v2"); OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261118)
NBOOT = 1000

# ---------------------------------------------------------------- CV profile (NHANES QC, pooled by level)
QC = pd.DataFrame({"mean": [24.485, 103.95, 442.359, 23.489, 100.824, 433.404],
                   "cv": [7.6, 8.1, 8.7, 9.8, 8.7, 8.8], "n": [142, 142, 142, 104, 104, 104],
                   "level": ["L", "M", "H", "L", "M", "H"]})
PROF = (QC.groupby("level", sort=False)
          .apply(lambda g: pd.Series({"conc": np.average(g["mean"], weights=g.n),
                                      "cv": np.sqrt(np.average(g.cv ** 2, weights=g.n)) / 100})))
PROF = PROF.sort_values("conc")
def cv_profile(x):
    lx = np.log(np.asarray(x, float))
    return np.interp(lx, np.log(PROF.conc.values), PROF.cv.values)
CVP = "profile"

def cv_of(cv, x):
    return cv_profile(x) if isinstance(cv, str) else np.full(np.shape(x), cv, float)

def s_of(cv, x, cvi=0.0, n=1):
    c = cv_of(cv, x)
    return np.sqrt((np.log1p(c ** 2) + np.log1p(cvi ** 2)) / n)

# ---------------------------------------------------------------- data
ige = pd.read_sas(D / "AL_IGE_D.xpt"); demo = pd.read_sas(D / "DEMO_D.xpt")
bmx = pd.read_sas(D / "BMX_D.xpt"); mcq = pd.read_sas(D / "MCQ_D.xpt")
FOOD = ["LBXIF1", "LBXIF2", "LBXF13"]
PEREN = ["LBXID1", "LBXID2", "LBXIE1", "LBXIE5", "LBXII6", "LBXIM6"]
d = (ige[["SEQN", "LBXIGE"] + FOOD + PEREN]
     .merge(demo[["SEQN", "RIDAGEYR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]], on="SEQN")
     .merge(bmx[["SEQN", "BMXWT"]], on="SEQN", how="left")
     .merge(mcq[["SEQN", "MCQ010", "MCQ035", "MCQ040"]], on="SEQN", how="left"))
d = d[d.LBXIGE.notna() & d.BMXWT.notna() & (d.RIDAGEYR < 18)].copy()
d["ever"] = d.MCQ010 == 1
d["current"] = d.ever & (d.MCQ035 == 1)
d["aero"] = (d[PEREN] >= 0.35).any(axis=1)
d["cand"] = d.current & d.aero
d["attack"] = d.cand & (d.MCQ040 == 1)
d["food_sens"] = (d[FOOD] >= 0.35).any(axis=1)
ado, kid = d.RIDAGEYR >= 12, d.RIDAGEYR.between(6, 11)

POPS = {
    "A": ("Asthma 12–17 y, Turkish KÜB", ado & d.cand, "KUB"),
    "B": ("Asthma 6–11 y, Turkish KÜB", kid & d.cand, "KUB"),
    "C": ("Asthma 6–11 y, FDA Table 2", kid & d.cand, "FDA_T2"),
    "D": ("Asthma 12–17 y, FDA Table 1", ado & d.cand, "FDA_T1"),
    "E": ("Food-sensitized 1–17 y, FDA Table 4", d.food_sens, "FDA_T4"),
    "A_ever": ("Asthma ever 12–17 y, Turkish KÜB (v1)", ado & d.ever, "KUB"),
    "A_attack": ("Asthma 12–17 y + attack past year, Turkish KÜB", ado & d.attack, "KUB"),
}

# ---------------------------------------------------------------- analytic per-child outcomes
def row_probs(t, x, cv, b=0.0, cvi=0.0, n=1):
    x = np.asarray(x, float); e = np.r_[0.0, t["ige"], np.inf]
    s = s_of(cv, x, cvi, n)
    with np.errstate(divide="ignore"):
        z = (np.log(e)[None, :] - np.log(x * (1 + b))[:, None]) / s[:, None]
    z[:, 0], z[:, -1] = -np.inf, np.inf
    return np.diff(norm.cdf(z), axis=1)

COLS = ["elig", "false_inel", "diff", "higher", "lower", "interval", "false_elig", "under80", "under80_true", "below30"]
def child_outcomes(t, x, wt, cv, b=0.0, cvi=0.0, n=1):
    P = row_probs(t, x, cv, b, cvi, n)
    rows = np.arange(-1, len(t["rows"]) + 1)
    tr, co = row_index(t, x), col_index(t, wt)
    form = 0.016 * np.asarray(wt) * np.asarray(x)
    out = np.zeros((len(x), len(COLS)))
    for i in range(len(x)):
        c = int(co[i]); tc = cell(t, int(tr[i]), c)
        if tc is None:
            out[i, 6] = sum(P[i, k] for k, r in enumerate(rows) if cell(t, int(r), c) is not None)
            continue
        out[i, 0] = 1; tm = mg4w(t, int(tr[i]), c)
        out[i, 8] = float(tm < 0.8 * form[i])
        out[i, 9] = P[i, 0]
        for k, r in enumerate(rows):
            p = P[i, k]
            if p < 1e-12: continue
            mc = cell(t, int(r), c)
            if mc is None:
                out[i, 1] += p; continue
            mm = mg4w(t, int(r), c)
            if mm < 0.8 * form[i]: out[i, 7] += p
            if mc == tc: continue
            out[i, 2] += p; out[i, 3] += p * (mm > tm); out[i, 4] += p * (mm < tm); out[i, 5] += p * (mc[1] != tc[1])
    return out

def wmean(v, w):
    return float(np.sum(v * w) / np.sum(w)) if np.sum(w) > 0 else np.nan

def summarize(o, w, x):
    e = o[:, 0] == 1; band = e & (x >= 30) & (x < 40)
    r = {k: wmean(o[e, j], w[e]) for j, k in enumerate(COLS) if k not in ("elig", "false_elig")}
    r["false_elig"] = wmean(o[~e, 6], w[~e])
    r["false_inel_30_40"] = wmean(o[band, 1], w[band])
    r["composite"] = r["false_inel"] + r["under80"]
    r["share_30_40"] = wmean(band[e].astype(float), w[e])
    return r

def raowu(g):
    """Rao-Wu rescaled bootstrap weights (one PSU drawn of two per stratum, weight x2)."""
    w = g.WTMEC2YR.to_numpy().copy(); st, ps = g.SDMVSTRA.to_numpy(), g.SDMVPSU.to_numpy()
    mult = np.zeros(len(g))
    for s in np.unique(d.SDMVSTRA):
        psus = np.unique(d.SDMVPSU[d.SDMVSTRA == s]); nh = len(psus)
        pick = RNG.choice(psus, nh - 1, replace=True)
        for p in psus:
            mult[(st == s) & (ps == p)] = (nh / (nh - 1)) * np.sum(pick == p)
    return w * mult

def run(key, cv=CVP, b=0.0, cvi=0.0, n=1, boot=0, keys=("false_inel", "diff", "under80", "composite", "false_inel_30_40")):
    lab, mask, tk = POPS[key]; g = d[mask]; t = TABLES[tk]
    x = g.LBXIGE.to_numpy(); o = child_outcomes(t, x, g.BMXWT.to_numpy(), cv, b, cvi, n)
    s = summarize(o, g.WTMEC2YR.to_numpy(), x); s["n"] = len(g); s["n_elig"] = int(o[:, 0].sum())
    s["n_30_40"] = int(((o[:, 0] == 1) & (x >= 30) & (x < 40)).sum())
    if boot:
        bs = {k: [] for k in keys}
        for _ in range(boot):
            ww = raowu(g); r = summarize(o, ww, x)
            for k in keys: bs[k].append(r[k])
        for k in keys: s[k + "_ci"] = tuple(np.nanpercentile(bs[k], [2.5, 97.5]))
    return s

pct = lambda v: round(100 * v, 1)
def ci(s, k): a, b = s[k + "_ci"]; return f"{100*a:.1f}–{100*b:.1f}"

# ---------------------------------------------------------------- Table 1
t1 = []
for key in ["A", "B", "C", "D", "E"]:
    lab, mask, tk = POPS[key]; g = d[mask]; t = TABLES[tk]
    tc = [cell(t, int(r), int(c)) for r, c in zip(row_index(t, g.LBXIGE), col_index(t, g.BMXWT))]
    el = np.array([v is not None for v in tc]); x = g.LBXIGE
    t1.append({"Population": lab, "n": len(g), "Median age": g.RIDAGEYR.median(), "Median weight": round(g.BMXWT.median(), 1),
               "Median IgE (IQR)": f"{x.median():.0f} ({x.quantile(.25):.0f}–{x.quantile(.75):.0f})",
               "IgE <30, n": int((x < 30).sum()), "Eligible, n (%)": f"{el.sum()} ({100*el.mean():.0f})",
               "Eligible with IgE 30–40, n": int((el & (x >= 30) & (x < 40)).sum())})
T1 = pd.DataFrame(t1)

# ---------------------------------------------------------------- Table 2 main (profile CV, bootstrap)
t2 = []; MAIN = {}
for key in ["A", "B", "C", "D", "E"]:
    s = run(key, boot=NBOOT); MAIN[key] = s
    t2.append({"Population": POPS[key][0], "Eligible n": s["n_elig"],
               "False ineligibility % (95% CI)": f"{pct(s['false_inel'])} ({ci(s,'false_inel')})",
               "  of which below 30 IU/mL %": pct(s["below30"]),
               "False inelig., true IgE 30–40 % (95% CI)": f"{pct(s['false_inel_30_40'])} ({ci(s,'false_inel_30_40')})" if s["n_30_40"] else "–",
               "n with IgE 30–40": s["n_30_40"],
               "False eligibility (true IgE ineligible) %": pct(s["false_elig"]),
               "Different regimen % (95% CI)": f"{pct(s['diff'])} ({ci(s,'diff')})",
               "Higher dose %": pct(s["higher"]), "Lower dose %": pct(s["lower"]), "Interval change %": pct(s["interval"]),
               "Dose <80% formula % (at true IgE)": f"{pct(s['under80'])} ({pct(s['under80_true'])})",
               "Composite % (95% CI)": f"{pct(s['composite'])} ({ci(s,'composite')})"})
T2 = pd.DataFrame(t2)

# ---------------------------------------------------------------- Table 3 sensitivity (pop A)
SENS = ([("Analytical CV (constant)", f"{int(100*c)}%", dict(cv=c)) for c in (0.05, 0.08, 0.11, 0.15)] +
        [("Bias (profile CV)", f"{int(100*b):+d}%", dict(b=b)) for b in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20)] +
        [("Within-subject biological variation (hypothetical)", f"{int(100*c)}%", dict(cvi=c)) for c in (0.10, 0.20, 0.30)])
t3 = [{"Scenario": "Base case", "Value": "NHANES CV profile (8.4–8.7%)", **{k: pct(MAIN["A"][k]) for k in ("false_inel", "false_inel_30_40", "diff", "under80", "false_elig")}}]
for grp, lab, kw in SENS:
    s = run("A", **kw)
    t3.append({"Scenario": grp, "Value": lab, **{k: pct(s[k]) for k in ("false_inel", "false_inel_30_40", "diff", "under80", "false_elig")}})
for key in ("A_ever", "A_attack"):
    s = run(key)
    t3.append({"Scenario": "Population definition", "Value": f"{POPS[key][0]} (n eligible {s['n_elig']})", **{k: pct(s[k]) for k in ("false_inel", "false_inel_30_40", "diff", "under80", "false_elig")}})
T3 = pd.DataFrame(t3).rename(columns={"false_inel": "False inelig. %", "false_inel_30_40": "False inelig., IgE 30–40 %",
                                      "diff": "Different regimen %", "under80": "Dose <80% formula %", "false_elig": "False elig. %"})

# ---------------------------------------------------------------- repeat strategies (Monte Carlo, pop A and E)
K = 4000
def strategies(key, cvi=0.0):
    lab, mask, tk = POPS[key]; g = d[mask]; t = TABLES[tk]
    x = g.LBXIGE.to_numpy(); wt = g.BMXWT.to_numpy(); w = g.WTMEC2YR.to_numpy()
    s = s_of(CVP, x, cvi)
    edges = np.asarray(t["ige"], float)
    co = col_index(t, wt); trow = row_index(t, x)
    res = {}
    z1 = RNG.standard_normal((len(x), K)); z2 = RNG.standard_normal((len(x), K))
    m1 = x[:, None] * np.exp(s[:, None] * z1); m2 = x[:, None] * np.exp(s[:, None] * z2)
    near = (np.abs(np.log(m1[..., None] / edges[None, None, :])) < np.log(1.2)).any(axis=2)
    gm = np.sqrt(m1 * m2)
    near30 = np.abs(np.log(m1 / 30.0)) < np.log(1.1)
    def bump(m):
        r = row_index(t, m.ravel()).reshape(m.shape)
        up = np.zeros_like(m, bool)
        for k in range(len(edges) - 1):
            hi = edges[k + 1]
            up |= (r == k) & (m >= hi / 1.2) & (k + 1 < len(t["rows"]))
        return np.where(up, r + 1, r)
    plans = {"Single measurement": row_index(t, m1.ravel()).reshape(m1.shape),
             "Repeat all, use mean": row_index(t, gm.ravel()).reshape(m1.shape),
             "Repeat if within ±20% of a border": row_index(t, np.where(near, gm, m1).ravel()).reshape(m1.shape),
             "Repeat if within ±10% of 30 IU/mL": row_index(t, np.where(near30, gm, m1).ravel()).reshape(m1.shape),
             "Round up if within 20% below a row border": bump(m1)}
    for name, R in plans.items():
        fi = np.zeros(len(x)); df = np.zeros(len(x)); u80 = np.zeros(len(x)); dmg = np.zeros(len(x)); el = np.zeros(len(x), bool)
        form = 0.016 * wt * x
        for i in range(len(x)):
            c = int(co[i]); tc = cell(t, int(trow[i]), c)
            if tc is None: continue
            el[i] = True; tm = mg4w(t, int(trow[i]), c)
            rr, cnt = np.unique(R[i], return_counts=True)
            for r_, n_ in zip(rr, cnt):
                p = n_ / K; mc = cell(t, int(r_), c)
                if mc is None: fi[i] += p; continue
                mm = mg4w(t, int(r_), c)
                if mc != tc: df[i] += p
                if mm < 0.8 * form[i]: u80[i] += p
                dmg[i] += p * (mm - tm)
        rep = {"Single measurement": 0.0, "Repeat all, use mean": 1.0,
               "Repeat if within ±20% of a border": near.mean(axis=1),
               "Repeat if within ±10% of 30 IU/mL": near30.mean(axis=1), "Round up if within 20% below a row border": 0.0}[name]
        rep = np.broadcast_to(rep, x.shape)
        res[name] = {"Repeat tests per 100 eligible": round(100 * wmean(rep[el], w[el]), 1),
                     "False inelig. %": round(100 * wmean(fi[el], w[el]), 2), "Different regimen %": pct(wmean(df[el], w[el])),
                     "Dose <80% formula %": pct(wmean(u80[el], w[el])),
                     "Mean change, mg per 4 weeks": round(wmean(dmg[el], w[el]), 1)}
    return pd.DataFrame(res).T.reset_index().rename(columns={"index": "Strategy"})
T5 = pd.concat([strategies("A").assign(Population=POPS["A"][0]),
                strategies("E").assign(Population=POPS["E"][0]),
                strategies("A", cvi=0.20).assign(Population=POPS["A"][0] + ", CV_I 20%")])

# ---------------------------------------------------------------- Table 4 clinical examples (profile CV)
def scenario(age, wt, x, tk="KUB", cv=CVP):
    t = TABLES[tk]; c = int(col_index(t, wt)); P = row_probs(t, [x], cv)[0]
    rows = np.arange(-1, len(t["rows"]) + 1); dist = {}
    for k, r in enumerate(rows):
        mc = cell(t, int(r), c); key = "Ineligible" if mc is None else f"{mc[0]} mg every {mc[1]} weeks"
        dist[key] = dist.get(key, 0) + P[k]
    tc = cell(t, int(row_index(t, x)), c); tk_ = "Ineligible" if tc is None else f"{tc[0]} mg every {tc[1]} weeks"
    alt = sorted(((k, v) for k, v in dist.items() if k != tk_), key=lambda z: -z[1])[0]
    return {"Age": age, "Weight": wt, "True IgE": x, "CV at this level %": round(100 * float(cv_of(cv, [x])[0]), 1),
            "Correct regimen": tk_, "Correct assigned %": round(100 * dist[tk_]), "Most likely alternative": f"{alt[0]} ({100*alt[1]:.0f}%)",
            "Formula dose, mg/4 wk": round(0.016 * wt * x)}
T4 = pd.DataFrame([scenario(14, 55, 32), scenario(14, 55, 35), scenario(14, 55, 95), scenario(15, 65, 105),
                   scenario(16, 85, 290), scenario(9, 35, 620)])

# ---------------------------------------------------------------- IgE–weight correlation (assumption check)
corr = []
for a in range(6, 18):
    g = d[d.RIDAGEYR == a]; r, p = spearmanr(g.LBXIGE, g.BMXWT); corr.append({"age": a, "n": len(g), "rho": round(r, 2)})
CORR = pd.DataFrame(corr)

with pd.ExcelWriter(OUT / "omalizumab_tables_v2.xlsx") as xw:
    for nm, df in (("T1", T1), ("T2_main", T2), ("T3_sens", T3), ("T4_examples", T4), ("T5_repeat", T5), ("CV_profile", PROF.reset_index()), ("IgE_weight_corr", CORR)):
        df.to_excel(xw, sheet_name=nm, index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 50)
with open(OUT / "summary_v2.txt", "w") as f:
    f.write(f"CV profile:\n{PROF}\n\n")
    for nm, df in (("T1", T1), ("T2", T2), ("T3", T3), ("T4", T4), ("T5", T5), ("CORR", CORR)):
        f.write(f"===== {nm}\n{df.to_string(index=False)}\n\n")
    f.write("MAIN raw:\n" + "\n".join(f"{k}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()) for k, v in MAIN.items()))
print(open(OUT / "summary_v2.txt").read())
