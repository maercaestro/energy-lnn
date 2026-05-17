# EB-LNN vs LNN vs LSTM vs PID — Benchmark Report

Evaluated on the PINN-augmented industrial furnace dataset.
PID is a controller-style baseline (P / PI / PID gains tuned by
validation grid search; no learnable neural parameters).

Each neural model is trained with multiple random seeds; values are reported as mean $\pm$ std.

## Table 1 — Prediction Accuracy (physical units)

| Model | Runs | Temp RMSE (°C) | O₂ RMSE (%) | Temp MAE (°C) | O₂ MAE (%) | Temp R² | O₂ R² |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EBLNN | 3 | 0.2273 $\pm$ 0.0308 | 0.7479 $\pm$ 0.0344 | 0.1577 $\pm$ 0.0221 | 0.4903 $\pm$ 0.0461 | 0.999699 $\pm$ 0.000093 | 0.959939 $\pm$ 0.005571 |
| LNN | 3 | 0.2046 $\pm$ 0.0210 | 0.7167 $\pm$ 0.0096 | 0.1500 $\pm$ 0.0187 | 0.4468 $\pm$ 0.0117 | 0.999758 $\pm$ 0.000054 | 0.963369 $\pm$ 0.001417 |
| LSTM | 3 | 0.2138 $\pm$ 0.0196 | **0.7154 $\pm$ 0.0095** | 0.1555 $\pm$ 0.0183 | **0.4415 $\pm$ 0.0088** | 0.999738 $\pm$ 0.000040 | **0.963504 $\pm$ 0.001339** |
| PID | 3 | **0.0000 $\pm$ 0.0000** | 3.7468 $\pm$ 0.0817 | **0.0000 $\pm$ 0.0000** | 2.4739 $\pm$ 0.0555 | **1.000000 $\pm$ 0.000000** | -0.000309 $\pm$ 0.000151 |

## Table 2 — Disturbance Robustness (absolute perturbed RMSE)

Realistic = process-level disturbances (sensor dropout, feature scaling, extrapolation).
Synthetic = sensor-level disturbances (Gaussian noise, spike injection, temporal shuffle).
Lower = more robust.

| Model | Realistic Temp | Realistic O₂ | Synthetic Temp | Synthetic O₂ |
| :--- | ---: | ---: | ---: | ---: |
| EBLNN | **3.9015 $\pm$ 0.7080** | 2.0746 $\pm$ 0.1341 | 11.8280 $\pm$ 2.8454 | 3.5058 $\pm$ 0.6544 |
| LNN | 5.0397 $\pm$ 0.1074 | 2.0205 $\pm$ 0.0619 | 7.7253 $\pm$ 0.1840 | 2.0388 $\pm$ 0.0777 |
| LSTM | 5.0367 $\pm$ 0.1612 | **2.0052 $\pm$ 0.0291** | **6.8234 $\pm$ 0.0789** | **1.7809 $\pm$ 0.0494** |
| PID | 6.2318 $\pm$ 0.0247 | 3.7468 $\pm$ 0.0817 | 13.6748 $\pm$ 0.0368 | 3.7468 $\pm$ 0.0817 |

## Table 3 — Safety Stability (violation rates under disturbance)

Critical = O₂ < 1.5% or Temp > 500 °C.

| Model | Clean Critical | Clean Total | Noise Critical | Noise Total | Overall Critical | Overall Total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| EBLNN | **0.0000 $\pm$ 0.0000** | 0.0491 $\pm$ 0.0010 | 0.0004 $\pm$ 0.0004 | 0.0581 $\pm$ 0.0128 | 0.0002 $\pm$ 0.0003 | 0.0396 $\pm$ 0.0034 |
| LNN | 0.0000 $\pm$ 0.0000 | 0.0480 $\pm$ 0.0018 | 0.0120 $\pm$ 0.0111 | 0.0424 $\pm$ 0.0108 | 0.0111 $\pm$ 0.0094 | 0.0409 $\pm$ 0.0093 |
| LSTM | 0.0000 $\pm$ 0.0000 | 0.0483 $\pm$ 0.0010 | 0.0020 $\pm$ 0.0017 | 0.0385 $\pm$ 0.0042 | 0.0064 $\pm$ 0.0089 | 0.0405 $\pm$ 0.0082 |
| PID | 0.0000 $\pm$ 0.0000 | **0.0000 $\pm$ 0.0000** | **0.0000 $\pm$ 0.0000** | **0.0000 $\pm$ 0.0000** | **0.0000 $\pm$ 0.0000** | **0.0000 $\pm$ 0.0000** |

## Table 4 — Composite Ranking

Weighted score combining accuracy (50%), disturbance robustness (30% — realistic 2×, absolute RMSE), and safety (20%).
Lower score = better overall.

| Rank | Model | Composite Score |
| ---: | :--- | ---: |
| 1 | LSTM | 0.3698 |
| 2 | EBLNN | 0.4016 |
| 3 | LNN | 0.4550 |
| 4 | PID | 0.4900 |

## Table 5 — Model Efficiency

Parameter count, convergence speed, inference latency, and accuracy-per-parameter.

| Model | Parameters | Best Epoch | Epochs Run | Wall Time (s) | Latency (ms/sample) | Throughput (samples/s) | RMSE / 10K Params |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EBLNN | 176,899 | 81 $\pm$ 49 | 101 $\pm$ 49 | 44692.0 $\pm$ 21840.1 | 0.428 $\pm$ 0.056 | 2360 $\pm$ 288 | 0.0129 |
| LNN | 152,066 | 98 $\pm$ 30 | 118 $\pm$ 30 | 2369.9 $\pm$ 580.7 | 0.402 $\pm$ 0.052 | 2514 $\pm$ 302 | 0.0135 |
| LSTM | 201,474 | 109 $\pm$ 8 | 129 $\pm$ 8 | 1276.0 $\pm$ 182.9 | 0.153 $\pm$ 0.006 | 6532 $\pm$ 255 | 0.0106 |
| PID | **0** | **1 $\pm$ 0** | 1 $\pm$ 0 | 11.8 $\pm$ 5.1 | **0.010 $\pm$ 0.003** | **105494 $\pm$ 42810** | nan |

## Table 6 — Seed Consistency (Coefficient of Variation)

Lower CoV = more reproducible across random seeds.

| Model | Temp RMSE CoV | O₂ RMSE CoV | Noise Deg Temp CoV | Overall Deg Temp CoV |
| :--- | ---: | ---: | ---: | ---: |
| EBLNN | 13.54% | 4.61% | 35.92% | 32.02% |
| LNN | 10.26% | 1.34% | **7.93%** | 9.45% |
| LSTM | **9.18%** | **1.33%** | 11.46% | **8.63%** |
| PID | 29.82% | 2.18% | 34.19% | 34.29% |

## Table 7 — Per-Disturbance Category Breakdown (Temp RMSE degradation)

Mean degradation ratio per perturbation category. Lower = more robust.

| Model | Noise | Dropout | Spike | Shuffle | Extrapolation | Scaling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| EBLNN | 57.4464 $\pm$ 20.6375 | **14.8003 $\pm$ 2.0715** | 54.5532 $\pm$ 19.3087 | **32.2406 $\pm$ 4.0800** | **53.5545 $\pm$ 22.9688** | **8.2414 $\pm$ 1.2455** |
| LNN | 38.6462 $\pm$ 3.0630 | 16.9985 $\pm$ 1.9153 | 37.7362 $\pm$ 3.4402 | 35.3531 $\pm$ 3.7725 | 88.7339 $\pm$ 9.4319 | 9.5179 $\pm$ 0.8769 |
| LSTM | **30.3877 $\pm$ 3.4821** | 15.9500 $\pm$ 1.3322 | **34.3998 $\pm$ 3.2996** | 33.9496 $\pm$ 3.2262 | 82.0474 $\pm$ 6.6631 | 9.9985 $\pm$ 0.7854 |
| PID | 742391.1992 $\pm$ 253856.3882 | 194131.9588 $\pm$ 69124.4487 | 1585988.9622 $\pm$ 538228.7966 | 528564.9136 $\pm$ 179771.8073 | 1927871.7070 $\pm$ 657568.5729 | 145599.1589 $\pm$ 51843.3286 |

## Table 8 — Per-Disturbance Category Breakdown (O₂ RMSE degradation)

| Model | Noise | Dropout | Spike | Shuffle | Extrapolation | Scaling |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| EBLNN | 5.2480 $\pm$ 1.2784 | 2.9671 $\pm$ 0.3398 | 4.9868 $\pm$ 1.0456 | 1.2805 $\pm$ 0.0578 | 4.9436 $\pm$ 0.5351 | 2.0430 $\pm$ 0.2123 |
| LNN | 3.1374 $\pm$ 0.1705 | 3.1282 $\pm$ 0.1188 | 2.8797 $\pm$ 0.1546 | 1.2853 $\pm$ 0.0352 | 4.5718 $\pm$ 0.2929 | 2.1405 $\pm$ 0.0815 |
| LSTM | 2.6357 $\pm$ 0.1247 | 3.1622 $\pm$ 0.0472 | 2.6439 $\pm$ 0.1016 | 1.2967 $\pm$ 0.0167 | 4.5193 $\pm$ 0.2614 | 2.1092 $\pm$ 0.0771 |
| PID | **1.0000 $\pm$ 0.0000** | **1.0000 $\pm$ 0.0000** | **1.0000 $\pm$ 0.0000** | **1.0000 $\pm$ 0.0000** | **1.0000 $\pm$ 0.0000** | **1.0000 $\pm$ 0.0000** |

## Table 9 — Extreme Stress Tests (absolute perturbed RMSE)

Realistic refinery failure scenarios. Lower = more robust.

| Model | Multi-Sensor Drop Temp | Multi-Sensor Drop O₂ | Sensor Drift Temp | Sensor Drift O₂ | Stuck Sensor Temp | Stuck Sensor O₂ | Oscillation Temp | Oscillation O₂ | Intermittent Temp | Intermittent O₂ | Combined Attack Temp | Combined Attack O₂ | Extreme Extrap (5–10σ) Temp | Extreme Extrap (5–10σ) O₂ | Extreme Noise (3–5σ) Temp | Extreme Noise (3–5σ) O₂ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EBLNN | 9.1046 $\pm$ 0.1432 | **3.6553 $\pm$ 0.0952** | 7.2949 $\pm$ 1.3085 | 3.0458 $\pm$ 0.1663 | 1.7474 $\pm$ 0.0384 | 1.2680 $\pm$ 0.0585 | 7.3560 $\pm$ 0.8583 | 2.7264 $\pm$ 0.0889 | 2.2380 $\pm$ 0.0553 | 1.6258 $\pm$ 0.0571 | 16.9597 $\pm$ 3.0252 | 5.2979 $\pm$ 0.7421 | **16.3301 $\pm$ 3.8666** | 5.9677 $\pm$ 0.8823 | 22.8736 $\pm$ 2.2193 | 7.0364 $\pm$ 0.1970 |
| LNN | 9.3677 $\pm$ 0.1807 | 3.7179 $\pm$ 0.1184 | 3.9480 $\pm$ 0.0623 | 2.1013 $\pm$ 0.0735 | 1.8004 $\pm$ 0.0443 | 1.2855 $\pm$ 0.0131 | 4.3176 $\pm$ 0.1860 | 1.9472 $\pm$ 0.0636 | 2.3449 $\pm$ 0.0658 | 1.6691 $\pm$ 0.0293 | **12.3501 $\pm$ 0.1516** | 4.0073 $\pm$ 0.0581 | 27.3742 $\pm$ 1.0074 | 4.2200 $\pm$ 0.3789 | **21.1501 $\pm$ 0.2523** | 4.7778 $\pm$ 0.2888 |
| LSTM | 9.1117 $\pm$ 0.2020 | 3.7232 $\pm$ 0.0947 | **3.8520 $\pm$ 0.3363** | **1.8017 $\pm$ 0.0239** | 1.8421 $\pm$ 0.0114 | **1.1452 $\pm$ 0.0801** | 3.7814 $\pm$ 0.1313 | **1.6767 $\pm$ 0.0689** | 1.8499 $\pm$ 0.0353 | **1.4222 $\pm$ 0.0335** | 12.4811 $\pm$ 0.0452 | 3.9296 $\pm$ 0.0186 | 24.5877 $\pm$ 0.2525 | 4.4589 $\pm$ 0.1666 | 21.4373 $\pm$ 0.2183 | 4.2617 $\pm$ 0.2412 |
| PID | **8.8205 $\pm$ 0.1355** | 3.7468 $\pm$ 0.0817 | 5.7572 $\pm$ 0.0087 | 3.7468 $\pm$ 0.0817 | **1.4506 $\pm$ 0.0201** | 3.7468 $\pm$ 0.0817 | **3.7286 $\pm$ 0.0056** | 3.7468 $\pm$ 0.0817 | **1.6585 $\pm$ 0.0249** | 3.7468 $\pm$ 0.0817 | 24.0202 $\pm$ 0.2207 | **3.7468 $\pm$ 0.0817** | 98.8691 $\pm$ 0.1492 | **3.7468 $\pm$ 0.0817** | 52.7786 $\pm$ 0.1190 | **3.7468 $\pm$ 0.0817** |

## Table 10 — Safety Under Extreme Conditions (critical violation rate)

| Model | Multi-Sensor Drop | Sensor Drift | Stuck Sensor | Combined Attack | Overall Extreme |
| :--- | ---: | ---: | ---: | ---: | ---: |
| EBLNN | 0.0000 $\pm$ 0.0000 | 0.0153 $\pm$ 0.0262 | **0.0000 $\pm$ 0.0000** | 0.0010 $\pm$ 0.0011 | 0.0042 $\pm$ 0.0069 |
| LNN | 0.0015 $\pm$ 0.0015 | 0.0191 $\pm$ 0.0178 | 0.0001 $\pm$ 0.0001 | 0.0073 $\pm$ 0.0056 | 0.0083 $\pm$ 0.0073 |
| LSTM | **0.0000 $\pm$ 0.0000** | 0.0016 $\pm$ 0.0015 | 0.0000 $\pm$ 0.0000 | 0.0008 $\pm$ 0.0004 | 0.0016 $\pm$ 0.0006 |
| PID | 0.0000 $\pm$ 0.0000 | **0.0000 $\pm$ 0.0000** | 0.0000 $\pm$ 0.0000 | **0.0003 $\pm$ 0.0000** | **0.0001 $\pm$ 0.0000** |

---

Lower is better for RMSE, MAE, degradation ratio, violation rates, composite score, CoV, and latency.
Higher is better for R² and throughput.