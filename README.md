# Total IgE measurement error and omalizumab dosing in children

Code and transcribed dosing tables for the simulation study:

> Kilinckaya L, Kilinckaya MF, Topcu DI, Doğru M. *Total IgE measurement error and omalizumab dosing in children: a simulation study.* (submitted to Pediatric Allergy and Immunology)

Omalizumab dose and dosing interval are read from a table of pretreatment total IgE and body weight. This repository estimates how often routine analytical error in total IgE assigns a child a different regimen, or a different eligibility decision, than the true IgE would.

## Contents

| File | Purpose |
| --- | --- |
| `dose_tables.py` | Omalizumab dosing tables transcribed cell by cell: Turkish SmPC (KÜB, revised 4 July 2026, Tables 2–3) and FDA prescribing information (February 2024, Tables 1, 2, 4). Includes an automatic cross-check: the FDA food-allergy table equals the Turkish table for weights ≥20 kg. |
| `core.py` | Measurement error model and child-level outcome probabilities (analytical, no Monte Carlo). |
| `omal_analysis.py` | Main analysis on NHANES 2005–2006 children: Tables 1–4 and Figure 1. |
| `tr_scenario.py` | Turkish scenario: IgE distribution digitized from Altan et al. (2025) with Turkish (Neyzi 2015) or NHANES body weights. |
| `neyzi_lms.py` | Turkish weight-for-age LMS parameters (Neyzi et al. 2015, Table 5, with the authors' 2026 correction for 8-year-old boys). |
| `fig2_clinician.py` | Figure 2: repeated-measurement illustration and effect of analytical CV. |
| `data/altan2025_ige_bars_digitized.csv` | Median and 2.5th/97.5th percentiles of total IgE in 108 age bins (1–18 years), digitized from the published supplementary figure of Altan et al. 2025. |
| `results/` | Output tables and figures. |

## Measurement error model

Reported IgE = true IgE × (1 + bias) × exp(ε), with ε ~ N(0, s²) and s² = ln(1 + CV_A²) + ln(1 + CV_I²); for the mean of n measurements, s²/n. For each child, the probability of falling into each IgE row of the dosing table is computed analytically for the child's weight column.

## Data

NHANES 2005–2006 files are public and are not redistributed here. Download into `data/`:

- `AL_IGE_D.xpt`, `DEMO_D.xpt`, `BMX_D.xpt`, `MCQ_D.xpt` from https://wwwn.cdc.gov/nchs/nhanes/ (cycle 2005–2006).

## Running

```bash
pip install -r requirements.txt
python omal_analysis.py      # main analysis → results/omalizumab_tablolar.xlsx, Fig1
python tr_scenario.py        # Turkish scenario → results/T5_turk_senaryosu.csv
python fig2_clinician.py     # Figure 2
```

Tested with Python 3.11, NumPy 2.5, SciPy 1.18, pandas 3.0, matplotlib 3.11. Random seeds are fixed.

## Sources

- FDA. XOLAIR (omalizumab) prescribing information, February 2024.
- Novartis. Xolair 150 mg Kısa Ürün Bilgisi (Turkish SmPC), revised 4 July 2026.
- Altan TK, Karakoyun İ, Çolak A, Topcu DI. Am J Clin Pathol 2025;164:870-878. doi:10.1093/ajcp/aqaf111
- Neyzi O, et al. J Clin Res Pediatr Endocrinol 2015;7:280-293. doi:10.4274/jcrpe.2183

## Use of AI

Code drafting and transcription checks were assisted by Claude (Anthropic). All code, tables and results were reviewed by the authors.

## License

MIT (see `LICENSE`).
