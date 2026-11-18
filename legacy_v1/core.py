import numpy as np
from scipy.stats import norm
from dose_tables import TABLES, row_index, col_index, cell, mg4w

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

