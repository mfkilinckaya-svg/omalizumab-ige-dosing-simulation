# One measurement, one dose: total IgE measurement error and omalizumab dosing in children

Code and transcribed dosing tables for the simulation study:

> Kilinckaya L, Kilinckaya MF, Topcu DI, Doğru M. *One measurement, one dose: total IgE measurement error and omalizumab dosing in children.* (submitted to Pediatric Allergy and Immunology)

Omalizumab eligibility, dose and dosing interval are read from a table of pretreatment total IgE and body weight. This repository estimates how often routine analytical error in total IgE changes the eligibility decision or the assigned regimen, compared with the child's true IgE, and whether the assigned dose falls below the pharmacological (formula) requirement.

## Contents

| File | Purpose |
| --- | --- |
| `dose_tables.py` | Omalizumab dosing tables transcribed cell by cell: Turkish SmPC (Kısa Ürün Bilgisi, revised 4 July 2026, Tables 2–3) and FDA prescribing information (February 2024, Tables 1, 2, 4). Automatic cross-check: the FDA food-allergy table equals the Turkish table for weights ≥20 kg. |
| `analysis_v2.py` | Main analysis (manuscript Tables 1–3, Supplementary Tables S2–S3). |
| `pkpd_v2.py` | Free IgE consequence of regimen changes, using the published omalizumab–IgE binding model of Lowe et al. 2009 (typical-patient parameters) → `results_v2/pkpd_free_ige.csv`. |
| `figs_v2.py` | Figures 1–2, graphical abstract and Supplementary Table S1 (Turkish scenario). |
| `neyzi_lms.py` | Turkish weight-for-age LMS parameters (Neyzi et al. 2015, Table 5, with the published 2026 correction for 8-year-old boys). |
| `data/altan2025_ige_bars_digitized.csv` | Median and 2.5th/97.5th percentiles of total IgE in 108 age bins (1–18 years), digitized from the published supplementary figure of Altan et al. 2025. |
| `results_v2/` | Output tables (`omalizumab_tables_v2.xlsx`, `summary_v2.txt`) and figures. |
| `legacy_v1/` | First-draft analysis (all children with ever-asthma, constant CV 11%), kept for transparency. |

## Methods in brief

- **Populations (NHANES 2005–2006):** children with current physician-diagnosed asthma (MCQ010, MCQ035) and sensitization to at least one perennial aeroallergen (specific IgE ≥0.35 kU/L to *D. pteronyssinus*, *D. farinae*, cat, dog, German cockroach or *Alternaria*); food-sensitized children (egg, milk, peanut). Survey weights are applied.
- **Measurement error:** reported IgE = true IgE × (1 + bias) × exp(ε), ε ~ N(0, s²), s² = ln(1 + CV_A²) + ln(1 + CV_I²); for the mean of n measurements, s²/n. Probabilities of each table row are computed analytically.
- **Main analytical CV:** concentration-dependent profile from the NHANES total IgE quality-control data (laboratory procedure manual AL_IGE_D): 8.6% at 24, 8.4% at 103, 8.7% at 439 kU/L, interpolated on log concentration.
- **Outcomes (label discordance):** discordant exclusion (primary; eligible by true IgE, excluded by reported IgE), different regimen, dose below 80% and 100% of the formula requirement (0.016 × kg × IU/mL per 4 weeks), discordant inclusion.
- **Free IgE (adolescents):** Lowe et al. 2009 binding model (Br J Clin Pharmacol 68:61–76), 24 weeks of dosing, free IgE at the midpoint of weeks 20–24; targets 14 ng/mL (Lowe 2009) and 25 ng/mL (design target of the dosing table). Outcome: free IgE above target with the reported regimen but not with the true regimen.
- **Confidence intervals:** Rao–Wu rescaling bootstrap of primary sampling units within strata (1000 replicates).
- **Repeat-testing strategies:** Monte Carlo (4000 measurement pairs per child).

## Data

NHANES 2005–2006 files are public and are not redistributed here. Download into `data/`:

- `AL_IGE_D.xpt`, `DEMO_D.xpt`, `BMX_D.xpt`, `MCQ_D.xpt` from https://wwwn.cdc.gov/nchs/nhanes/ (cycle 2005–2006).

## Running

```bash
pip install -r requirements.txt
python analysis_v2.py   # → results_v2/omalizumab_tables_v2.xlsx, summary_v2.txt
python figs_v2.py       # → results_v2/Fig1, Fig2, graphical abstract, S_turkish_scenario.csv
python pkpd_v2.py       # → results_v2/pkpd_free_ige.csv (free IgE analysis)
```

Random seeds are fixed.

## Sources

- FDA. XOLAIR (omalizumab) prescribing information, February 2024.
- Novartis. Xolair 150 mg Kısa Ürün Bilgisi (Turkish SmPC), revised 4 July 2026.
- CDC. NHANES 2005–2006 laboratory procedure manual: allergen-specific and total IgE (AL_IGE_D).
- Altan TK, Karakoyun İ, Çolak A, Topcu DI. Am J Clin Pathol 2025;164:870-878. doi:10.1093/ajcp/aqaf111
- Neyzi O, et al. J Clin Res Pediatr Endocrinol 2015;7:280-293. doi:10.4274/jcrpe.2183

## Use of AI

Claude (Opus 5.5; Anthropic), a large language model, was used to draft the analysis code (Python) and figures, and to draft and edit the manuscript text.

## License

MIT (see `LICENSE`).
