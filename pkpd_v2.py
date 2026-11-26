#!/usr/bin/env python3
"""
Free IgE consequence of measurement-error-driven regimen changes (BJCP revision, 26 Nov 2026).

Uses the published mechanism-based omalizumab-IgE binding model of Lowe et al. (Br J Clin Pharmacol
2009;68:61-76), typical-individual parameters (Table 3, as coded in nlmixr2lib Lowe_2009_omalizumab.R):
apparent CL/V of free omalizumab, free IgE and complex with allometric weight effects (ref 70 kg), baseline-IgE
effects on CL_E, IgE production and Kd (ref 365 ng/mL), Kd = Kd0 x (total omalizumab / total IgE)^alpha,
instantaneous equilibrium binding, first-order SC absorption. 1 IU/mL = 2.42 ng/mL.

For each eligible child (true IgE, weight) and each regimen that measurement error can assign, free IgE is
simulated for 24 weeks; the outcome is free IgE at the midpoint of weeks 20-24 (day 154), compared with
the 14 ng/mL target of Lowe 2009 (sensitivity: 25 ng/mL, the original design target).
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp

src = open("analysis_v2.py", encoding="utf-8").read()
src = src[: src.index("pct = lambda")]
G = {}
exec(compile(src, "analysis_v2_setup", "exec"), G)
d, POPS, TABLES = G["d"], G["POPS"], G["TABLES"]
row_probs, row_index, col_index, cell, CVP, raowu = (G[k] for k in
    ("row_probs", "row_index", "col_index", "cell", "CVP", "raowu"))

# ---------------------------------------------------------------- Lowe 2009 typical parameters
KA, CLX, CLE, CLC, VX, VC, RE_UG, KD0, ALPHA = 0.458, 0.208, 3.85, 0.832, 9.33, 6.31, 1220.0, 1.81, 0.0902
MWX, MWE, IU2NG = 150.0, 190.07, 2.42

def params(wt, ige_ng):
    w, e = wt / 70.0, ige_ng / 365.0
    return dict(ka=KA, clx=CLX * w ** 1.00, cle=CLE * w ** 0.499 * e ** 0.372, clc=CLC * w ** 0.671,
                vx=VX * w ** 0.828, vc=VC * w ** 0.549, r=(RE_UG / MWE) * w ** 0.491 * e ** 0.594,
                kd0=KD0 * e ** 0.115)

def binding(xt, et, p):
    if xt <= 0:
        return 0.0
    kd = p["kd0"] * (xt / et) ** ALPHA
    s = xt + et + kd * p["vx"] * p["vx"] / p["vc"]
    return 0.5 * (s - np.sqrt(max(s * s - 4 * xt * et, 0.0)))

def rhs(t, y, p):
    dep, xt, et = y
    xc = binding(xt, et, p)
    cfx, cfe, cc = (xt - xc) / p["vx"], (et - xc) / p["vx"], xc / p["vc"]
    return [-p["ka"] * dep,
            p["ka"] * dep * 1000.0 / MWX - p["clx"] * cfx - p["clc"] * cc,
            p["r"] - p["cle"] * cfe - p["clc"] * cc]

def free_ige(p, xt, et):
    xc = binding(xt, et, p)
    return (et - xc) / p["vx"] * MWE  # ng/mL

T_MID, T_END = 154.0, 168.0
_cache = {}
def free_mid(wt, ige_iu, dose, every_w):
    """Free IgE (ng/mL) at day 154 (midpoint of weeks 20-24); also mean over days 140-168."""
    key = (round(wt, 2), round(ige_iu, 3), dose, every_w)
    if key in _cache:
        return _cache[key]
    ige_ng = ige_iu * IU2NG
    p = params(wt, ige_ng)
    y = np.array([0.0, 0.0, ige_ng / MWE * p["vx"]])
    tau = 7.0 * every_w
    times = np.arange(0.0, T_END, tau)
    grid = np.linspace(140.0, 168.0, 57)
    vals, mid = [], None
    for i, t0 in enumerate(times):
        y[0] += dose
        t1 = times[i + 1] if i + 1 < len(times) else T_END
        ev = sorted(set([g for g in grid if t0 <= g <= t1] + ([T_MID] if t0 < T_MID <= t1 else [])))
        sol = solve_ivp(rhs, (t0, t1), y, args=(p,), method="LSODA", rtol=1e-8, atol=1e-10,
                        t_eval=ev if ev else None, dense_output=False)
        for tt, yy in zip(sol.t, sol.y.T):
            f = free_ige(p, yy[1], yy[2])
            if abs(tt - T_MID) < 1e-9 and mid is None:
                mid = f
            if 140.0 <= tt <= 168.0:
                vals.append(f)
        y = sol.y[:, -1].copy()
    _cache[key] = (mid, float(np.mean(vals)))
    return _cache[key]

# ---------------------------------------------------------------- per-child outcomes
def child_pk(t, x, wt, cv=CVP, b=0.0, target=14.0):
    """Returns per child: eligible, free IgE (true regimen), P(miss target | treated, assigned regimen),
    P(miss target AND true regimen meets it), P(excluded), expected ratio free IgE assigned/true."""
    P = row_probs(t, x, cv, b)
    rows = np.arange(-1, len(t["rows"]) + 1)
    tr, co = row_index(t, x), col_index(t, wt)
    out = []
    for i in range(len(x)):
        c = int(co[i]); tc = cell(t, int(tr[i]), c)
        if tc is None:
            out.append(dict(elig=0)); continue
        f_true, _ = free_mid(wt[i], x[i], tc[0], tc[1])
        miss_true = f_true > target
        pm = pnew = pex = pdiff = pdiff_new = 0.0; worst = 1.0
        for k, r in enumerate(rows):
            pk = P[i, k]
            if pk < 1e-9:
                continue
            mc = cell(t, int(r), c)
            if mc is None:
                pex += pk; continue
            f, _ = free_mid(wt[i], x[i], mc[0], mc[1]) if mc != tc else (f_true, None)
            m = f > target
            pm += pk * m
            pnew += pk * (m and not miss_true)
            if mc != tc:
                pdiff += pk; pdiff_new += pk * (m and not miss_true); worst = max(worst, f / f_true)
        out.append(dict(elig=1, f_true=f_true, miss_true=float(miss_true), p_miss=pm, p_new_miss=pnew,
                        p_excl=pex, p_diff=pdiff, p_diff_new=pdiff_new, worst_ratio=worst))
    return pd.DataFrame(out)

def wm(v, w):
    return float(np.sum(v * w) / np.sum(w))

def summarize(o, w):
    e = o.elig.to_numpy() == 1; oe, we = o[e], w[e]
    return dict(n_elig=int(e.sum()),
                true_miss=wm(oe.miss_true, we),            # target missed even with the true-IgE regimen
                p_miss=wm(oe.p_miss, we),                  # treated and target missed, reported regimen
                p_new_miss=wm(oe.p_new_miss, we),          # missed only because of measurement error
                p_diff=wm(oe.p_diff, we),
                med_f_true=float(np.median(oe.f_true)),
                max_ratio=float(oe.worst_ratio.max()))

def run(key, b=0.0, target=14.0, boot=0):
    lab, mask, tk = POPS[key]; g = d[mask]
    o = child_pk(TABLES[tk], g.LBXIGE.to_numpy(), g.BMXWT.to_numpy(), b=b, target=target)
    s = summarize(o, g.WTMEC2YR.to_numpy()); s["pop"] = lab; s["bias"] = b; s["target"] = target
    if boot:
        bs = []
        for _ in range(boot):
            bs.append(summarize(o, raowu(g))["p_new_miss"])
        s["p_new_miss_ci"] = tuple(np.percentile(bs, [2.5, 97.5]))
    return s, o, g

if __name__ == "__main__":
    OUT = Path("results_v2"); rows = []
    s, oA, gA = run("A", boot=1000); rows.append(s)
    for key in ("D",):
        rows.append(run(key)[0])
    for b in (-0.10, 0.10):
        rows.append(run("A", b=b)[0])
    rows.append(run("A", target=25.0)[0]); rows.append(run("D", target=25.0)[0])
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(res.to_string())
    res.to_csv(OUT / "pkpd_free_ige.csv", index=False)
    # sanity check: typical 70 kg / 150 IU/mL patient on table dose (300 mg q4w per Turkish table)
    print("check 70 kg, 150 IU/mL, 300 mg q4w:", free_mid(70, 150, 300, 4))
    print("check 70 kg, 150 IU/mL, 150 mg q4w:", free_mid(70, 150, 150, 4))
    print("check 70 kg, 150 IU/mL, no drug:", 150 * IU2NG)
