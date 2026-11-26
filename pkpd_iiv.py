#!/usr/bin/env python3
"""
Sensitivity analysis for pkpd_v2.py: interindividual variability (IIV) from Lowe et al. 2009, Table 3
(omega^2: ka 2.01; CL_X 0.162, V 0.0901, cov 0.103; CL_E 0.0479; CL_C 0.0649, R 0.0701, cov -0.0101;
V_C 0.0519; Kd 0.0991). For each eligible adolescent, NDRAW virtual patients share the child's true IgE and
weight; each virtual patient is simulated with the true-IgE regimen and with every regimen measurement error
can assign. Also reports the fold change in free IgE after a one-row step down, and results excluding
adolescents below the model's lower weight limit (39 kg).
"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
import pkpd_v2 as P

NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 100
RNG = np.random.default_rng(20261126)
OM = {"ka": 2.01, "cle": 0.0479, "vc": 0.0519, "kd0": 0.0991}
B1 = np.array([[0.162, 0.103], [0.103, 0.0901]])       # clx, vx
B2 = np.array([[0.0649, -0.0101], [-0.0101, 0.0701]])  # clc, r

def draw_etas(n):
    e = {k: RNG.normal(0, np.sqrt(v), n) for k, v in OM.items()}
    b1 = RNG.multivariate_normal([0, 0], B1, n); b2 = RNG.multivariate_normal([0, 0], B2, n)
    e["clx"], e["vx"], e["clc"], e["r"] = b1[:, 0], b1[:, 1], b2[:, 0], b2[:, 1]
    return pd.DataFrame(e)

def sim(wt, ige_iu, dose, every_w, eta):
    p = P.params(wt, ige_iu * P.IU2NG)
    for k, v in eta.items():
        p[k] = p[k] * np.exp(v)
    # run the same integration as pkpd_v2.free_mid but with individual parameters
    ige_ng = ige_iu * P.IU2NG
    y = np.array([0.0, 0.0, ige_ng / P.MWE * p["vx"]])
    tau = 7.0 * every_w; times = np.arange(0.0, P.T_END, tau); mid = None
    from scipy.integrate import solve_ivp
    for i, t0 in enumerate(times):
        y[0] += dose
        t1 = times[i + 1] if i + 1 < len(times) else P.T_END
        ev = [P.T_MID] if t0 < P.T_MID <= t1 else None
        sol = solve_ivp(P.rhs, (t0, t1), y, args=(p,), method="LSODA", rtol=1e-7, atol=1e-9, t_eval=ev)
        if ev:
            mid = P.free_ige(p, sol.y[1, 0], sol.y[2, 0])
            sol = solve_ivp(P.rhs, (t0, t1), y, args=(p,), method="LSODA", rtol=1e-7, atol=1e-9)
        y = sol.y[:, -1].copy()
    return mid

def child_job(args):
    i, wt, x, tc, alts, etas = args
    res = []
    for _, eta in etas.iterrows():
        e = eta.to_dict()
        ft = sim(wt, x, tc[0], tc[1], e)
        out = {"i": i, "f_true": ft}
        pm14 = pm25 = 0.0; fold = []
        for (reg, pk, lower) in alts:
            f = sim(wt, x, reg[0], reg[1], e)
            pm14 += pk * (f > 14 and ft <= 14); pm25 += pk * (f > 25 and ft <= 25)
            if lower: fold.append(f / ft)
        out.update(new14=pm14, new25=pm25, fold=np.mean(fold) if fold else np.nan)
        res.append(out)
    return res

def main():
    lab, mask, tk = P.POPS["A"]; g = P.d[mask]; t = P.TABLES[tk]
    x, wt, w = g.LBXIGE.to_numpy(), g.BMXWT.to_numpy(), g.WTMEC2YR.to_numpy()
    Pr = P.row_probs(t, x, P.CVP, 0.0)
    rows = np.arange(-1, len(t["rows"]) + 1)
    tr, co = P.row_index(t, x), P.col_index(t, wt)
    jobs, meta = [], []
    for i in range(len(x)):
        c = int(co[i]); tc = P.cell(t, int(tr[i]), c)
        if tc is None: continue
        alts = []
        for k, r in enumerate(rows):
            if Pr[i, k] < 1e-6: continue
            mc = P.cell(t, int(r), c)
            if mc is None or mc == tc: continue
            lower = mc[0] * (4 // mc[1]) < tc[0] * (4 // tc[1])
            alts.append((mc, Pr[i, k], lower))
        jobs.append((i, wt[i], x[i], tc, alts, draw_etas(NDRAW)))
        meta.append((i, w[i], wt[i]))
    with Pool(2) as pool:
        out = [r for rr in pool.map(child_job, jobs) for r in rr]
    df = pd.DataFrame(out)
    m = pd.DataFrame(meta, columns=["i", "w", "wt"])
    per = df.groupby("i").agg(new14=("new14", "mean"), new25=("new25", "mean"),
                              miss_true14=("f_true", lambda v: np.mean(v > 14)),
                              fold=("fold", "median")).reset_index().merge(m, on="i")
    def wm(v, ww): return float(np.sum(v * ww) / np.sum(ww))
    res = {}
    for lab2, sub in (("all", per), ("wt>=39", per[per.wt >= 39])):
        res[lab2] = dict(n=len(sub), new14=wm(sub.new14, sub.w), new25=wm(sub.new25, sub.w),
                         miss_true14=wm(sub.miss_true14, sub.w))
    folds = df.fold.dropna()
    res["fold_iiv"] = dict(median=float(folds.median()), p5=float(folds.quantile(0.05)), p95=float(folds.quantile(0.95)))
    print(res)
    per.to_csv("results_v2/pkpd_iiv_perchild.csv", index=False)
    pd.Series(res).to_json("results_v2/pkpd_iiv_summary.json")

if __name__ == "__main__":
    main()
