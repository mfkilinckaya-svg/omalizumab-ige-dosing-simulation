"""Türk senaryosu: Altan ve ark. AJCP 2025 Ek Şekil (IgE, 1–18 y; nefelometrik) sayısallaştırıldı →
her çubuk için log-normal (mu = ln medyan, sigma = (ln P97,5 − ln medyan)/1,96). 108 çubuk 1–18 yaşa eşit aralıklı kabul edildi.
Kilo: (a) NHANES 2005–06 aynı yaştaki çocukların ampirik kilosu (ABD); (b) Neyzi 2015 LMS (Türk) – tablo bekleniyor.
Aynı yaşta IgE ve kilo bağımsız kabul edildi. Doz tablosu: Türkiye KÜB.
"""
import numpy as np, pandas as pd
from core import child_outcomes, TABLES
from neyzi_lms import sample_weight
RNG = np.random.default_rng(20261118)
bars = pd.read_csv("data/altan2025_ige_bars_digitized.csv")
bars["sigma"] = (np.log(bars.p975) - np.log(bars["median"])) / 1.959964
bars["age"] = 1 + 17 * (np.arange(len(bars)) + 0.5) / len(bars)
nh = pd.read_sas("data/BMX_D.xpt")[["SEQN", "BMXWT"]].merge(pd.read_sas("data/DEMO_D.xpt")[["SEQN", "RIDAGEYR"]], on="SEQN").dropna()
nh = nh[nh.RIDAGEYR < 18]

def sample(lo, hi, per_bar=4000, weights='NHANES'):
    b = bars[(bars.age >= lo) & (bars.age < hi + 1)]
    ige = np.concatenate([np.exp(RNG.normal(np.log(r["median"]), r.sigma, per_bar)) for _, r in b.iterrows()])
    ages = np.repeat(np.floor(b.age).astype(int).to_numpy(), per_bar)
    wt = np.empty_like(ige)
    for a in np.unique(ages):
        m = ages == a
        if weights == 'NHANES':
            wt[m] = RNG.choice(nh[nh.RIDAGEYR == a].BMXWT.to_numpy(), m.sum())
        else:
            wt[m] = sample_weight(a, m.sum(), RNG)
    return ige, wt

rows = []
for (lab, lo, hi), wsrc in [(x, w) for x in (("12–17 y (TR geri ödeme yaşı)", 12, 17), ("6–11 y", 6, 11)) for w in ("Neyzi (TR)", "NHANES (ABD)")]:
    ige, wt = sample(lo, hi, weights='NHANES' if wsrc.startswith('NHANES') else 'NEYZI')
    for cv in (0.05, 0.08, 0.11, 0.15):
        o = child_outcomes(TABLES["KUB"], ige, wt, cv)
        e = o[:, 0] == 1
        rows.append({"Yaş": lab, "Kilo kaynağı": wsrc, "Kilo medyan": round(float(np.median(wt)),1), "Simüle çocuk": len(ige), "Tabloya göre uygun %": round(100 * e.mean(), 1), "CV %": int(100 * cv),
                     "Farklı doz rejimi %": round(100 * o[e, 2].mean(), 1),
                     "Uygun → yanlış 'uygun değil' %": round(100 * o[e, 1].mean(), 1),
                     "Uygun değil → yanlış 'uygun' %": round(100 * o[~e, 6].mean(), 1),
                     "IgE 30–40 arası (uygunlar içinde) %": round(100 * ((ige[e] >= 30) & (ige[e] <= 40)).mean(), 1)})
T = pd.DataFrame(rows); T.to_csv("results/T5_turk_senaryosu.csv", index=False)
pd.set_option("display.width", 250); print(T.to_string(index=False))
print("\nmedyan IgE 12–17:", bars[bars.age >= 12]["median"].median().round(1), " sigma:", bars[bars.age >= 12].sigma.median().round(2))
