"""Omalizumab doz tabloları – kaynak belgelerden sayfa görüntüleri üzerinden hücre hücre aktarıldı (17 Kas 2026).
Hücre: (doz mg, sıklık hafta: 4 veya 2); None = uygulama yok / doz önerisi için veri yok.
IgE satırları: ilk satır [alt, üst], diğerleri (alt, üst]. Kilo sütunları: ilk sütun [alt, üst], diğerleri (alt, üst].

KUB    : Türkiye Xolair 150 mg PFS KÜB (yenilenme 04.07.2026), 4.2 Tablo 2+3 – astım (≥6 yaş) ve NP'li KRS. EMA SmPC ile aynı yapı.
FDA_T1 : FDA etiketi (2/2024, Ref ID 5330389) Tablo 1 – astım ≥12 yaş.
FDA_T2 : FDA Tablo 2 – astım 6–<12 yaş.
FDA_T4 : FDA Tablo 4 – IgE aracılı besin alerjisi ≥1 yaş. (≥20 kg sütunları KUB ile hücre hücre aynı → çapraz doğrulama.)
"""
import numpy as np
_ = None; Q4, Q2 = 4, 2

def q4(*v): return [(x, Q4) for x in v]
def q2(*v): return [(x, Q2) for x in v]
def no(n): return [None] * n

TABLES = {}

TABLES["KUB"] = dict(
    ige=[30, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1500],
    wt=[20, 25, 30, 40, 50, 60, 70, 80, 90, 125, 150],
    rows=[
        q4(75, 75, 75, 150, 150, 150, 150, 150, 300, 300),
        q4(150, 150, 150, 300, 300, 300, 300, 300, 450, 600),
        q4(150, 150, 225, 300, 300, 450, 450, 450, 600) + q2(375),
        q4(225, 225, 300, 450, 450, 450, 600, 600) + q2(450, 525),
        q4(225, 300, 450, 450, 600, 600) + q2(375, 375, 525, 600),
        q4(300, 300, 450, 600, 600) + q2(375, 450, 450, 600) + no(1),
        q4(300) + q2(225) + q4(450, 600) + q2(375, 450, 450, 525) + no(2),
        q2(225, 225, 300, 375, 450, 450, 525, 600) + no(2),
        q2(225, 225, 300, 375, 450, 525, 600) + no(3),
        q2(225, 300, 375, 450, 525, 600) + no(4),
        q2(225, 300, 375, 450, 600) + no(5),
        q2(300, 300, 450, 525, 600) + no(5),
        q2(300, 375, 450, 525) + no(6),
        q2(300, 375, 525, 600) + no(6),
    ])

TABLES["FDA_T1"] = dict(
    ige=[30, 100, 200, 300, 400, 500, 600, 700],
    wt=[30, 60, 70, 90, 150],
    rows=[
        q4(150, 150, 150, 300),
        q4(300, 300, 300) + q2(225),
        q4(300) + q2(225, 225, 300),
        q2(225, 225, 300) + no(1),
        q2(300, 300, 375) + no(1),   # 375 q2w — FDA Table 1 görseliyle teyit edildi (18.11.2026)
        q2(300, 375) + no(2),
        q2(375) + no(3),
    ])

TABLES["FDA_T2"] = dict(
    ige=[30, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300],
    wt=[20, 25, 30, 40, 50, 60, 70, 80, 90, 125, 150],
    rows=[
        q4(75, 75, 75, 150, 150, 150, 150, 150, 300, 300),
        q4(150, 150, 150, 300, 300, 300, 300, 300) + q2(225, 300),
        q4(150, 150, 225, 300, 300) + q2(225, 225, 225, 300, 375),
        q4(225, 225, 300) + q2(225, 225, 225, 300, 300) + no(2),
        q4(225, 300) + q2(225, 225, 300, 300, 375, 375) + no(2),
        q4(300, 300) + q2(225, 300, 300, 375) + no(4),
        q4(300) + q2(225, 225, 300, 375) + no(5),
        q2(225, 225, 300, 375) + no(6),
        q2(225, 225, 300, 375) + no(6),
        q2(225, 300, 375) + no(7),
        q2(225, 300, 375) + no(7),
        q2(300, 300) + no(8),
        q2(300, 375) + no(8),
    ])

TABLES["FDA_T4"] = dict(
    ige=[30, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1500, 1850],
    wt=[10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 125, 150],
    rows=[
        q4(75, 75, 75, 75, 75, 75, 150, 150, 150, 150, 150, 300, 300),
        q4(75, 75, 75, 150, 150, 150, 300, 300, 300, 300, 300, 450, 600),
        q4(75, 75, 150, 150, 150, 225, 300, 300, 450, 450, 450, 600) + q2(375),
        q4(150, 150, 150, 225, 225, 300, 450, 450, 450, 600, 600) + q2(450, 525),
        q4(150, 150, 225, 225, 300, 450, 450, 600, 600) + q2(375, 375, 525, 600),
        q4(150, 150, 225, 300, 300, 450, 600, 600) + q2(375, 450, 450, 600) + no(1),
        q2(150, 150) + q4(225, 300) + q2(225) + q4(450, 600) + q2(375, 450, 450, 525) + no(2),
        q2(150, 150, 150, 225, 225, 300, 375, 450, 450, 525, 600) + no(2),
        q2(150, 150, 150, 225, 225, 300, 375, 450, 525, 600) + no(3),
        q2(150, 150, 225, 225, 300, 375, 450, 525, 600) + no(4),
        q2(150, 150, 225, 225, 300, 375, 450, 600) + no(5),
        q2(150, 150, 225, 300, 300, 450, 525, 600) + no(5),
        q2(150, 225, 225, 300, 375, 450, 525) + no(6),
        q2(150, 225, 300, 300, 375, 525, 600) + no(6),
        no(1) + q2(225, 300, 375, 450, 600) + no(7),
    ])

for k, t in TABLES.items():
    assert len(t["rows"]) == len(t["ige"]) - 1, k
    assert all(len(r) == len(t["wt"]) - 1 for r in t["rows"]), k

# çapraz doğrulama: FDA_T4 (≥20 kg sütunları, 30–1500 satırları) == KUB
_t4 = [r[3:] for r in TABLES["FDA_T4"]["rows"][:14]]
assert _t4 == TABLES["KUB"]["rows"], "FDA_T4 ≥20 kg ile KÜB uyuşmuyor"


def row_index(t, x):
    """-1: alt sınırın altında; len(rows): üst sınırın üstünde."""
    e = np.asarray(t["ige"], float); x = np.asarray(x, float)
    r = np.searchsorted(e, x, side="left") - 1
    r = np.where(x == e[0], 0, r); r = np.where(x < e[0], -1, r)
    return np.where(x > e[-1], len(t["rows"]), r)


def col_index(t, w):
    e = np.asarray(t["wt"], float); w = np.asarray(w, float)
    c = np.searchsorted(e, w, side="left") - 1
    c = np.where(w == e[0], 0, c)
    return np.where((w < e[0]) | (w > e[-1]), -1, c)


def cell(t, r, c):
    if r < 0 or r >= len(t["rows"]) or c < 0: return None
    return t["rows"][r][c]


def mg4w(t, r, c):
    x = cell(t, r, c)
    return 0 if x is None else x[0] * (4 // x[1])
