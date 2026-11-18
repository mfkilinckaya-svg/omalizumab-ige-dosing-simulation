#!/usr/bin/env python3
"""
Total IgE ölçüm hatası ve çocuklarda omalizumab doz kararı – ana analiz.
M. F. Kilinckaya ve ark. (taslak, 17 Kas 2026)

Girdi : data/AL_IGE_D.xpt, DEMO_D.xpt, BMX_D.xpt, MCQ_D.xpt (NHANES 2005–2006, yaş 1–17)
        dose_tables.py (Türkiye KÜB 04.07.2026; FDA etiketi 2/2024 Tablo 1, 2, 4)
Çıktı : results/ (Excel tablolar, şekiller, özet)

Model (INR makalesiyle aynı): ölçülen = gerçek × (1 + bias) × exp(e),
  e ~ N(0, s²), s² = ln(1+CV_A²) + ln(1+CV_I²); n ölçümün (farklı günlerde) geometrik ortalamasında s²/n.
"Gerçek" = çocuğun NHANES'te ölçülen total IgE'si (homeostatik ayar noktası kabul edilir).
Olasılıklar analitik olarak hesaplanır (her çocuk için ölçülen IgE'nin düşeceği tablo satırları).
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from dose_tables import TABLES, row_index, col_index, cell, mg4w

D = Path("data"); OUT = Path("results"); OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261117)
CV_MAIN = 0.11            # HMR Ålesund, Phadia 1000, 24 kU/L (EQS 33424 v1.6)

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#1f1f1e", "ink2": "#5c5b56", "grid": "#e6e5e0"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": C["grid"],
                     "grid.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300})

# ------------------------------------------------------------------ veri
ige = pd.read_sas(D / "AL_IGE_D.xpt"); demo = pd.read_sas(D / "DEMO_D.xpt")
bmx = pd.read_sas(D / "BMX_D.xpt"); mcq = pd.read_sas(D / "MCQ_D.xpt")
FOOD = {"LBXIF1": "yumurta akı (f1)", "LBXIF2": "süt (f2)", "LBXF13": "yer fıstığı (f13)"}
d = (ige[["SEQN", "LBXIGE"] + list(FOOD)]
     .merge(demo[["SEQN", "RIDAGEYR", "RIAGENDR", "WTMEC2YR"]], on="SEQN")
     .merge(bmx[["SEQN", "BMXWT"]], on="SEQN", how="left")
     .merge(mcq[["SEQN", "MCQ010"]], on="SEQN", how="left"))
d = d[d.LBXIGE.notna() & d.BMXWT.notna() & (d.RIDAGEYR < 18)].copy()
d["asthma"] = d.MCQ010 == 1
d["food_sens"] = (d[list(FOOD)] >= 0.35).any(axis=1)

POPS = {  # ad: (maske, tablo)
    "A. TR geri ödeme – 12–17 y astım (KÜB)": ((d.RIDAGEYR >= 12) & d.asthma, "KUB"),
    "B. 6–11 y astım (KÜB; TR endikasyon dışı)": (d.RIDAGEYR.between(6, 11) & d.asthma, "KUB"),
    "C. 6–11 y astım (FDA Tablo 2)": (d.RIDAGEYR.between(6, 11) & d.asthma, "FDA_T2"),
    "D. 12–17 y astım (FDA Tablo 1)": ((d.RIDAGEYR >= 12) & d.asthma, "FDA_T1"),
    "E. 1–17 y besin duyarlanması (FDA Tablo 4)": (d.food_sens, "FDA_T4"),
}

# ------------------------------------------------------------------ olasılık çekirdeği
def s_tot(cv, cvi=0.0, n=1):
    return float(np.sqrt((np.log1p(cv ** 2) + np.log1p(cvi ** 2)) / n))

def row_probs(t, x, cv, b=0.0, cvi=0.0, n=1):
    """(len(x), len(rows)+2) olasılık; sütun 0 = alt sınır altı, son = üst sınır üstü."""
    e = np.r_[0.0, t["ige"], np.inf]
    with np.errstate(divide="ignore"):
        z = (np.log(e)[None, :] - np.log(np.asarray(x) * (1 + b))[:, None]) / s_tot(cv, cvi, n)
    z[:, 0], z[:, -1] = -np.inf, np.inf
    return np.diff(norm.cdf(z), axis=1)

def child_outcomes(t, ige_true, wt, cv, b=0.0, cvi=0.0, n=1):
    """Her çocuk için: uygun mu (gerçek), P(yanlış uygun değil), P(farklı rejim), P(yüksek), P(düşük), P(sıklık), P(yanlış uygun), mg farkı beklenen."""
    P = row_probs(t, ige_true, cv, b, cvi, n)
    rows = np.arange(-1, len(t["rows"]) + 1)
    tr, co = row_index(t, ige_true), col_index(t, wt)
    out = np.zeros((len(ige_true), 8))
    for i in range(len(ige_true)):
        tc, c = cell(t, int(tr[i]), int(co[i])), int(co[i])
        if tc is None:
            out[i, 6] = sum(P[i, k] for k, r in enumerate(rows) if cell(t, int(r), c) is not None)
            continue
        out[i, 0] = 1; tm = mg4w(t, int(tr[i]), c)
        for k, r in enumerate(rows):
            mc = cell(t, int(r), c); p = P[i, k]
            if p < 1e-12 or mc == tc: continue
            if mc is None: out[i, 1] += p; continue
            mm = mg4w(t, int(r), c)
            out[i, 2] += p; out[i, 3] += p * (mm > tm); out[i, 4] += p * (mm < tm)
            out[i, 5] += p * (mc[1] != tc[1]); out[i, 7] += p * abs(mm - tm) / tm
    return out

def summarize(o, w):
    e = o[:, 0] == 1; we, wn = w[e], w[~e]
    m = lambda j, mask, ww: float(np.sum(o[mask, j] * ww) / np.sum(ww)) if ww.size else np.nan
    return {"yanlis_uygun_degil": m(1, e, we), "farkli_rejim": m(2, e, we), "yuksek": m(3, e, we),
            "dusuk": m(4, e, we), "siklik": m(5, e, we), "yanlis_uygun": m(6, ~e, wn),
            "herhangi_hata_uygunlarda": m(1, e, we) + m(2, e, we)}

def run(pop, cv, b=0.0, cvi=0.0, n=1, boot=0):
    mask, tk = POPS[pop]; g = d[mask]; t = TABLES[tk]
    o = child_outcomes(t, g.LBXIGE.to_numpy(), g.BMXWT.to_numpy(), cv, b, cvi, n)
    w = g.WTMEC2YR.to_numpy(); s = summarize(o, w)
    s.update({"n_toplam": len(g), "n_uygun": int((o[:, 0] == 1).sum())})
    if boot:
        bs = []
        for _ in range(boot):
            ix = RNG.integers(0, len(g), len(g)); bs.append(summarize(o[ix], w[ix])["farkli_rejim"])
        s["farkli_rejim_GA"] = tuple(np.nanpercentile(bs, [2.5, 97.5]))
    return s

def fmt(s, key): return round(100 * s[key], 1)

# ------------------------------------------------------------------ Tablo 1: popülasyon özellikleri
t1 = []
for pop, (mask, tk) in POPS.items():
    g = d[mask]; t = TABLES[tk]; tc = [cell(t, int(r), int(c)) for r, c in zip(row_index(t, g.LBXIGE), col_index(t, g.BMXWT))]
    el = np.array([x is not None for x in tc])
    t1.append({"Popülasyon": pop, "Tablo": tk, "n": len(g), "Yaş medyan": g.RIDAGEYR.median(),
               "Kilo medyan (kg)": round(g.BMXWT.median(), 1), "IgE medyan (IU/mL)": round(g.LBXIGE.median(), 1),
               "IgE P25–P75": f"{g.LBXIGE.quantile(.25):.0f}–{g.LBXIGE.quantile(.75):.0f}",
               "Tabloya göre uygun n (%)": f"{el.sum()} ({100*el.mean():.0f}%)",
               "4 haftada bir n": sum(1 for x in tc if x and x[1] == 4), "2 haftada bir n": sum(1 for x in tc if x and x[1] == 2)})
T1 = pd.DataFrame(t1)

# ------------------------------------------------------------------ Tablo 2: ana sonuçlar (CV senaryoları, bootstrap GA)
t2 = []
for pop in POPS:
    for cv in (0.05, 0.08, 0.11, 0.15):
        s = run(pop, cv, boot=500 if cv == CV_MAIN else 0)
        ga = s.get("farkli_rejim_GA")
        t2.append({"Popülasyon": pop, "n uygun": s["n_uygun"], "CV_A %": int(100 * cv),
                   "Farklı doz rejimi %": fmt(s, "farkli_rejim"),
                   "%95 GA (bootstrap)": "" if ga is None else f"{100*ga[0]:.1f}–{100*ga[1]:.1f}",
                   "1000 uygun çocukta": round(1000 * s["farkli_rejim"]),
                   "Daha yüksek doz %": fmt(s, "yuksek"), "Daha düşük doz %": fmt(s, "dusuk"),
                   "Sıklık değişir %": fmt(s, "siklik"),
                   "Uygun → yanlış 'uygun değil' %": fmt(s, "yanlis_uygun_degil"),
                   "Uygun değil → yanlış 'uygun' %": fmt(s, "yanlis_uygun")})
T2 = pd.DataFrame(t2)

# ------------------------------------------------------------------ Tablo 3: duyarlılık (pop A ve B, CV %11 temelinde)
t3 = []
SENS = ([("Sapma", f"{int(100*b):+d}%", dict(cv=CV_MAIN, b=b)) for b in (-0.10, -0.05, 0.0, 0.05, 0.10)] +
        [("CV_I (varsayımsal)", f"{int(100*ci)}%", dict(cv=CV_MAIN, cvi=ci)) for ci in (0.0, 0.10, 0.20, 0.30)] +
        [("Ölçüm sayısı", "2 ölçüm ort. (yalnız analitik)", dict(cv=CV_MAIN, n=2)),
         ("Ölçüm sayısı", "2 ölçüm ort. + CV_I %20", dict(cv=CV_MAIN, cvi=0.20, n=2)),
         ("Ölçüm sayısı", "1 ölçüm + CV_I %20 (karşılaştırma)", dict(cv=CV_MAIN, cvi=0.20, n=1))])
for pop in list(POPS)[:2] + [list(POPS)[4]]:
    for grp, lab, kw in SENS:
        s = run(pop, **kw)
        t3.append({"Popülasyon": pop, "Senaryo": grp, "Değer": lab, "Farklı doz rejimi %": fmt(s, "farkli_rejim"),
                   "1000 uygun çocukta": round(1000 * s["farkli_rejim"]),
                   "Uygun → yanlış 'uygun değil' %": fmt(s, "yanlis_uygun_degil"),
                   "Uygun değil → yanlış 'uygun' %": fmt(s, "yanlis_uygun")})
T3 = pd.DataFrame(t3)

# ------------------------------------------------------------------ Tablo 4: klinik senaryolar (KÜB)
def scenario(age, wt, ige_true, t="KUB", cv=CV_MAIN):
    tt = TABLES[t]; c = int(col_index(tt, wt)); P = row_probs(tt, [ige_true], cv)[0]
    rows = np.arange(-1, len(tt["rows"]) + 1); dist = {}
    for k, r in enumerate(rows):
        mc = cell(tt, int(r), c); key = "uygun değil" if mc is None else f"{mc[0]} mg / {mc[1]} haftada bir"
        dist[key] = dist.get(key, 0) + P[k]
    tc = cell(tt, int(row_index(tt, ige_true)), c)
    true_key = "uygun değil" if tc is None else f"{tc[0]} mg / {tc[1]} haftada bir"
    others = "; ".join(f"{k}: %{100*v:.0f}" for k, v in sorted(dist.items(), key=lambda z: -z[1]) if k != true_key and v >= 0.01)
    return {"Yaş": age, "Kilo (kg)": wt, "Gerçek IgE (IU/mL)": ige_true, "Doğru doz": true_key,
            "Doğru doza düşme %": round(100 * dist.get(true_key, 0), 1), "Diğer olasılıklar (≥%1)": others}
T4 = pd.DataFrame([scenario(14, 55, 32), scenario(14, 55, 95), scenario(15, 65, 105), scenario(16, 85, 290),
                   scenario(13, 45, 410), scenario(17, 120, 1450), scenario(8, 28, 190), scenario(9, 35, 620)])

with pd.ExcelWriter(OUT / "omalizumab_tablolar.xlsx") as xw:
    T1.to_excel(xw, sheet_name="T1_populasyon", index=False); T2.to_excel(xw, sheet_name="T2_ana_sonuc", index=False)
    T3.to_excel(xw, sheet_name="T3_duyarlilik", index=False); T4.to_excel(xw, sheet_name="T4_senaryolar", index=False)

# ------------------------------------------------------------------ Şekil 1: KÜB ısı haritası (hücre bazında risk, CV %11)
t = TABLES["KUB"]
ig = np.exp(np.linspace(np.log(30), np.log(1500), 400)); wg = np.linspace(20.01, 150, 300)
R = np.full((len(ig), len(wg)), np.nan)
for j, w in enumerate(wg):
    o = child_outcomes(t, ig, np.full(len(ig), w), CV_MAIN)
    R[:, j] = np.where(o[:, 0] == 1, o[:, 1] + o[:, 2], np.nan)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
cmap = ListedColormap(["#f3f6fb", "#c9daf3", "#8fb5ea", "#4f8ddf", "#1f5fb8", "#0d3a78"])
bnd = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.51]
im = ax.pcolormesh(wg, ig, R, cmap=cmap, norm=BoundaryNorm(bnd, cmap.N), shading="auto")
for e in t["ige"][1:-1]: ax.axhline(e, color="white", lw=0.5)
for e in t["wt"][1:-1]: ax.axvline(e, color="white", lw=0.5)
g = d[POPS["A. TR geri ödeme – 12–17 y astım (KÜB)"][0]]
ax.scatter(g.BMXWT, g.LBXIGE, s=6, c=C["orange"], edgecolors="none", alpha=0.8, label="NHANES: 12–17 y, asthma")
ax.set_yscale("log"); ax.set_ylim(30, 1500); ax.set_xlim(20, 150)
ax.set_yticks(t["ige"]); ax.set_yticklabels([str(x) for x in t["ige"]], fontsize=6.5)
ax.set_xticks(t["wt"]); ax.set_xticklabels([str(x) for x in t["wt"]], fontsize=7)
ax.set_xlabel("Body weight (kg)"); ax.set_ylabel("True pretreatment total IgE (IU/mL)"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, ticks=bnd); cb.set_ticklabels(["0", "5", "10", "20", "30", "40", "50"])
cb.set_label("P(different regimen or ineligible), %")
ax.set_title("Turkish SmPC dosing table: misclassification probability, analytical CV 11%", fontsize=9, color=C["ink"])
ax.legend(loc="upper right", fontsize=7, markerscale=2)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"Fig1_KUB_isi_haritasi.{ext}")
plt.close(fig)

# ------------------------------------------------------------------ Şekil 2: CV'ye göre farklı rejim (popülasyonlar)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cvs = np.linspace(0.02, 0.20, 19)
for (pop, col, ls) in zip(list(POPS)[:2] + [list(POPS)[4]], [C["blue"], C["aqua"], C["orange"]], ["-", "--", "-."]):
    ax.plot(100 * cvs, [100 * run(pop, cv)["farkli_rejim"] for cv in cvs], color=col, ls=ls, lw=1.6,
            label={"A": "Asthma 12–17 y (Turkish SmPC)", "B": "Asthma 6–11 y (Turkish SmPC)", "E": "Food-sensitized 1–17 y (FDA Table 4)"}[pop[0]])
ax.axvline(100 * CV_MAIN, color=C["ink2"], lw=0.8, ls=":"); ax.text(100 * CV_MAIN + 0.3, 1, "CV 11%", fontsize=7, color=C["ink2"])
ax.set_xlabel("Analytical CV of total IgE (%)"); ax.set_ylabel("Eligible children with a\ndifferent dosing regimen (%)")
ax.legend(fontsize=7.5); fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(OUT / f"Fig2_CV_egrisi.{ext}")
plt.close(fig)

# ------------------------------------------------------------------ özet
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 60)
with open(OUT / "ozet.txt", "w") as f:
    for name, df in (("T1", T1), ("T2", T2), ("T3", T3), ("T4", T4)):
        f.write(f"===== {name}\n{df.to_string(index=False)}\n\n")
print(open(OUT / "ozet.txt").read())
