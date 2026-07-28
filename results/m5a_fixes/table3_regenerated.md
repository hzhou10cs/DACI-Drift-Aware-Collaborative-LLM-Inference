# Table 3, regenerated — paper vs BEFORE vs AFTER

All figures computed from the traces in one pass (§6a.4). 30 seeds (42–71) per cell.

## qwen3-14b

| scheme | paper TTLT | BEFORE TTLT | AFTER TTLT | P99 TPOT b→a | Ovhd b→a | #Rec b→a | accept% b→a | **a-changed% b→a** |
|---|---|---|---|---|---|---|---|---|
| SDA | 425.16 | 407.1±55.7 | 416.9±69.9 | 41.4→42.6 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.000→**0.000** |
| RT | 488.17 | 504.8±49.4 | 395.0±50.5 | 44.1→36.7 | 94.94→31.38 | 10.43→3.77 | 1.39→0.50 | 1.391→**0.373** |
| FM | 416.5 | 396.0±42.5 | 376.1±31.5 | 38.9→37.1 | 19.35→14.29 | 2.87→2.50 | 0.38→0.33 | 0.276→**0.164** |
| DACI | 371.89 | 370.5±44.8 | 372.1±45.1 | 36.7→36.9 | 6.18→5.95 | 2.43→2.37 | 0.32→0.32 | 0.000→**0.000** |

**DACI's TTLT lead, regenerated:**

| vs | BEFORE | AFTER |
|---|---|---|
| SDA | +8.99% | +10.74% |
| RT | +26.62% | +5.79% |
| FM | +6.44% | +1.05% |

**§5.2 prose vs regenerated (§0g):**

| quantity | prose says | Table 3 as printed | AFTER (regenerated) |
|---|---|---|---|
| DACI TTLT | 367.8 | 371.89 | 372.09 |
| FM TTLT | 437.5 | 416.5 | 376.06 |
| RT TTLT | 488.2 | 488.17 | 394.96 |
| DACI lead vs FM | 16.0% | 10.7% | **1.05%** |
| DACI lead vs RT | 25.0% | 23.8% | **5.79%** |
| RT overhead | 85.3 s | 90.29 s | 31.38 s |
| DACI overhead | 8.2 s | 7.69 s | 5.95 s |

## gemma3-4b

| scheme | paper TTLT | BEFORE TTLT | AFTER TTLT | P99 TPOT b→a | Ovhd b→a | #Rec b→a | accept% b→a | **a-changed% b→a** |
|---|---|---|---|---|---|---|---|---|
| SDA | — | 95.4±12.6 | 95.4±12.6 | 10.0→10.0 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.000→**0.000** |
| RT | — | 113.3±45.6 | 110.0±8.3 | 11.4→9.5 | 9.44→22.29 | 2.43→6.13 | 0.32→0.82 | 0.324→**0.760** |
| FM | — | 117.1±43.9 | 100.3±5.0 | 13.2→9.1 | 5.11→12.22 | 1.27→3.33 | 0.17→0.44 | 0.169→**0.444** |
| DACI | — | 95.9±11.5 | 95.9±11.5 | 10.6→10.6 | 0.21→0.21 | 0.27→0.27 | 0.04→0.04 | 0.000→**0.000** |

**DACI's TTLT lead, regenerated:**

| vs | BEFORE | AFTER |
|---|---|---|
| SDA | -0.54% | -0.54% |
| RT | +15.38% | +12.78% |
| FM | +18.10% | +4.40% |

## llama-3.2-8b

| scheme | paper TTLT | BEFORE TTLT | AFTER TTLT | P99 TPOT b→a | Ovhd b→a | #Rec b→a | accept% b→a | **a-changed% b→a** |
|---|---|---|---|---|---|---|---|---|
| SDA | — | 178.1±44.7 | 178.1±44.7 | 18.8→18.8 | 0.00→0.00 | 0.00→0.00 | 0.00→0.00 | 0.000→**0.000** |
| RT | — | 167.8±38.1 | 149.5±25.2 | 15.9→14.4 | 15.63→13.71 | 2.93→4.40 | 0.39→0.59 | 0.391→**0.324** |
| FM | — | 147.9±36.8 | 144.1±17.5 | 15.0→14.9 | 6.06→7.48 | 1.57→1.53 | 0.21→0.20 | 0.151→**0.169** |
| DACI | — | 148.3±32.9 | 148.3±32.9 | 15.0→15.0 | 3.56→3.56 | 2.43→2.43 | 0.32→0.32 | 0.000→**0.000** |

**DACI's TTLT lead, regenerated:**

| vs | BEFORE | AFTER |
|---|---|---|
| SDA | +16.75% | +16.75% |
| RT | +11.62% | +0.78% |
| FM | -0.27% | -2.89% |


wrote results/m5a_fixes/table3_regenerated.csv
wrote results/m5a_fixes/table3_regenerated.tex

## Counter availability (§5b.4)

| counter | available | why |
|---|---|---|
| #reconfigs | yes | summary.csv |
| acceptance rate | yes | per-window `accepted` |
| **a-changed rate** | **yes** | per-window `a`; this is the "has the baseline gone inert" test |
| pool size | **no** | `control/mechanism.py` post-dates these runs |
| placements enumerated | **no** | same |
