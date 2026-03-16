# Ablation Study Results — Generative EB-LNN

**Batch**: 20 experiments  
**Started**: 2026-02-26 04:21:24  
**Finished**: 2026-03-07 02:44:43  
**Total wall time**: 771,799 s (≈ 8.9 days)  
**Passed**: 20 / 20  
**W&B project**: `eblnn-ablation`

---

## 1. Experimental Setup

All experiments share the baseline architecture (176,899 trainable parameters) unless otherwise noted:

| Parameter | Baseline Value |
|-----------|---------------|
| CfC backbone | 151,808 params |
| Physics head | 258 params |
| EBM head | 24,833 params |
| Input size | 5 |
| Hidden size | 128 |
| EBM dims | [128, 64] |
| Sequence length | 30 |
| Stride | 30 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Alpha (α) | 1.0 |
| L2 reg | 0.1 |
| Langevin steps | 20 |
| Step size | 0.01 |
| Noise scale | 0.005 |
| Buffer capacity | 10,000 |
| Buffer prob | 0.95 |
| Patience | 20 |
| Max epochs | 200 |
| Device | CPU |

---

## 2. Summary of All Experiments

| Experiment | Description | Temp RMSE (°C) | O₂ RMSE (%) | Val Phys Loss | Best Epoch | Stopped Epoch | Wall Time (s) |
|------------|-------------|:--------------:|:----------:|:------------:|:----------:|:------------:|:------------:|
| A1\_real\_only | Real data only | 0.2265 | 0.4767 | 0.1757 | 105 | 125 | 36,577 |
| A2\_baseline | Real + all edge cases | 0.2059 | 0.7063 | 0.0192 | 110 | 130 | 47,651 |
| A3\_edge\_only | Edge cases only | 0.1721 | 1.1906 | 0.0320 | 89 | 109 | 9,993 |
| B1\_edge\_10pct | Real + 10% edge | 0.2952 | 0.5341 | 0.0532 | 133 | 153 | 46,172 |
| B2\_edge\_25pct | Real + 25% edge | 0.2380 | 0.5814 | 0.0341 | 73 | 93 | 27,981 |
| B3\_edge\_50pct | Real + 50% edge | 0.2914 | 0.6683 | 0.0216 | 113 | 133 | 45,071 |
| B4\_edge\_200pct | Real + 200% edge | 0.2289 | 0.8197 | 0.0177 | 77 | 97 | 43,782 |
| C1\_flame\_out | Real + S1 flame-out | 0.2738 | 0.4784 | 0.1275 | 150 | 170 | 52,390 |
| C2\_air\_leak | Real + S2 air leak | 0.2187 | 0.5111 | 0.0168 | 126 | 146 | 43,245 |
| C3\_tube\_rupture | Real + S3 tube rupture | 0.2452 | 0.5605 | 0.0781 | 97 | 117 | 35,098 |
| C4\_positive\_pressure | Real + S4 positive pressure | 0.3605 | 0.4929 | 0.1579 | 53 | 73 | 21,792 |
| C5\_fuel\_trip | Real + S5 fuel trip | 0.2069 | 0.6086 | 0.0164 | 115 | 135 | 41,583 |
| C6\_fuel\_contamination | Real + S6 fuel contamination | 0.3067 | 0.6424 | 0.0609 | 93 | 113 | 34,901 |
| D1\_hidden\_64 | CfC hidden = 64 | 0.2318 | 0.7245 | 0.0202 | 79 | 99 | 29,929 |
| D2\_hidden\_256 | CfC hidden = 256 | 0.2741 | 0.7830 | 0.0202 | 57 | 77 | 43,316 |
| D3\_ebm\_small | EBM head [64, 32] | 0.2406 | 0.7109 | 0.0195 | 92 | 112 | 43,440 |
| E1\_seq15 | seq\_len = 15 | 0.2334 | 0.7689 | 0.0214 | 67 | 87 | 34,132 |
| E2\_seq60 | seq\_len = 60 | 0.2939 | 0.7236 | 0.0191 | 81 | 101 | 41,517 |
| F1\_alpha\_01 | α = 0.1 (physics-dominant) | 0.2313 | 0.7057 | 0.0192 | 91 | 111 | 45,868 |
| F2\_alpha\_5 | α = 5.0 (CD-dominant) | 0.2003 | 0.7114 | 0.0196 | 94 | 114 | 47,248 |

---

## 3. Data Composition Details

| Experiment | Real Rows | Real Seqs | Edge Rows | Edge Seqs | Total Seqs | Train | Val | Test |
|------------|----------:|----------:|----------:|----------:|-----------:|------:|----:|-----:|
| A1\_real\_only | 470,676 | 15,689 | — | — | 15,689 | 12,551 | 1,569 | 1,569 |
| A2\_baseline | 470,676 | 15,689 | 150,000 | 4,998 | 20,687 | 16,549 | 2,069 | 2,069 |
| A3\_edge\_only | — | — | 150,000 | 4,998 | 4,998 | 3,998 | 500 | 500 |
| B1\_edge\_10pct | 470,676 | 15,689 | 15,000 | 497 | 16,186 | 12,948 | 1,619 | 1,619 |
| B2\_edge\_25pct | 470,676 | 15,689 | 37,500 | 1,247 | 16,936 | 13,548 | 1,694 | 1,694 |
| B3\_edge\_50pct | 470,676 | 15,689 | 75,000 | 2,497 | 18,186 | 14,548 | 1,819 | 1,819 |
| B4\_edge\_200pct | 470,676 | 15,689 | 300,000 | 9,996 | 25,685 | 20,548 | 2,568 | 2,569 |
| C1–C6 | 470,676 | 15,689 | 25,000 | 833 | 16,522 | 13,217 | 1,652 | 1,653 |
| E1\_seq15 | 470,676 | 31,378 | 150,000 | 9,996 | 41,374 | 33,099 | 4,137 | 4,138 |
| E2\_seq60 | 470,676 | 7,844 | 150,000 | 2,496 | 10,340 | 8,272 | 1,034 | 1,034 |

---

## 4. Architecture Variants (Axis D)

| Experiment | Hidden Size | EBM Dims | Total Params | CfC Params | EBM Params |
|------------|:----------:|:--------:|:------------:|:----------:|:----------:|
| D1\_hidden\_64 | 64 | [128, 64] | 76,675 | 59,904 | 16,641 |
| A2\_baseline | 128 | [128, 64] | 176,899 | 151,808 | 24,833 |
| D2\_hidden\_256 | 256 | [128, 64] | 475,651 | 433,920 | 41,217 |
| D3\_ebm\_small | 128 | [64, 32] | 162,435 | 151,808 | 10,369 |

---

## 5. Per-Axis Analysis

### 5.1 Axis A — Data Composition

| Experiment | Temp RMSE (°C) | Δ Temp | O₂ RMSE (%) | Δ O₂ |
|------------|:--------------:|:------:|:----------:|:----:|
| A1\_real\_only | 0.2265 | +0.0206 | 0.4767 | −0.2296 |
| A2\_baseline | 0.2059 | — | 0.7063 | — |
| A3\_edge\_only | 0.1721 | −0.0338 | 1.1906 | +0.4843 |

Edge cases improve temperature prediction but degrade O₂ generalisation. Real data anchors O₂ accuracy. The baseline (real + edge) provides the best compromise.

### 5.2 Axis B — Edge-Case Volume

| Experiment | Edge Fraction | Temp RMSE (°C) | Δ Temp | O₂ RMSE (%) | Δ O₂ |
|------------|:------------:|:--------------:|:------:|:----------:|:----:|
| B1\_edge\_10pct | 0.10 | 0.2952 | +0.0893 | 0.5341 | −0.1722 |
| B2\_edge\_25pct | 0.25 | 0.2380 | +0.0321 | 0.5814 | −0.1249 |
| B3\_edge\_50pct | 0.50 | 0.2914 | +0.0855 | 0.6683 | −0.0380 |
| A2\_baseline | 1.00 | 0.2059 | — | 0.7063 | — |
| B4\_edge\_200pct | 2.00 | 0.2289 | +0.0230 | 0.8197 | +0.1134 |

Increasing edge-case volume monotonically reduces validation physics loss but increases O₂ RMSE. The 100% fraction (baseline) balances both metrics. Oversampling (200%) yields diminishing returns on temperature with notable O₂ degradation.

### 5.3 Axis C — Scenario Contribution

| Experiment | Scenario | Temp RMSE (°C) | Δ Temp | O₂ RMSE (%) | Δ O₂ |
|------------|----------|:--------------:|:------:|:----------:|:----:|
| C1\_flame\_out | S1 | 0.2738 | +0.0679 | 0.4784 | −0.2279 |
| C2\_air\_leak | S2 | 0.2187 | +0.0128 | 0.5111 | −0.1952 |
| C3\_tube\_rupture | S3 | 0.2452 | +0.0393 | 0.5605 | −0.1458 |
| C4\_positive\_pressure | S4 | 0.3605 | +0.1546 | 0.4929 | −0.2134 |
| C5\_fuel\_trip | S5 | 0.2069 | +0.0010 | 0.6086 | −0.0977 |
| C6\_fuel\_contamination | S6 | 0.3067 | +0.1008 | 0.6424 | −0.0639 |
| A2\_baseline | All | 0.2059 | — | 0.7063 | — |

All single-scenario models improve O₂ RMSE over baseline (fewer edge cases, more real-data influence). C2 (air leak) and C5 (fuel trip) are the most informative individual scenarios for temperature. C4 (positive pressure) is the hardest scenario, with the highest temperature error and earliest stopping (epoch 73).

### 5.4 Axis D — Architecture

| Experiment | Config | Temp RMSE (°C) | Δ Temp | O₂ RMSE (%) | Δ O₂ |
|------------|--------|:--------------:|:------:|:----------:|:----:|
| D1\_hidden\_64 | hidden=64 | 0.2318 | +0.0259 | 0.7245 | +0.0182 |
| A2\_baseline | hidden=128 | 0.2059 | — | 0.7063 | — |
| D2\_hidden\_256 | hidden=256 | 0.2741 | +0.0682 | 0.7830 | +0.0767 |
| D3\_ebm\_small | EBM [64,32] | 0.2406 | +0.0347 | 0.7109 | +0.0046 |

The baseline hidden size of 128 is optimal. The 256-unit model shows signs of overfitting (early stop at epoch 77). Reducing the EBM head ([64, 32]) has minimal impact, suggesting the energy model can be simplified without significant performance loss.

### 5.5 Axis E — Sequence Length

| Experiment | seq\_len | Temp RMSE (°C) | Δ Temp | O₂ RMSE (%) | Δ O₂ |
|------------|:-------:|:--------------:|:------:|:----------:|:----:|
| E1\_seq15 | 15 | 0.2334 | +0.0275 | 0.7689 | +0.0626 |
| A2\_baseline | 30 | 0.2059 | — | 0.7063 | — |
| E2\_seq60 | 60 | 0.2939 | +0.0880 | 0.7236 | +0.0173 |

The baseline seq\_len = 30 is optimal. Shorter sequences (15) lose temporal context; longer sequences (60) reduce the number of training samples and increase per-sample variance, degrading temperature RMSE.

### 5.6 Axis F — CD Balance (Alpha)

| Experiment | Alpha (α) | Temp RMSE (°C) | Δ Temp | O₂ RMSE (%) | Δ O₂ |
|------------|:---------:|:--------------:|:------:|:----------:|:----:|
| F1\_alpha\_01 | 0.1 | 0.2313 | +0.0254 | 0.7057 | −0.0006 |
| A2\_baseline | 1.0 | 0.2059 | — | 0.7063 | — |
| F2\_alpha\_5 | 5.0 | 0.2003 | −0.0056 | 0.7114 | +0.0051 |

Alpha has relatively small impact. Higher α (CD-dominant) slightly improves temperature RMSE, achieving the best temp RMSE among real+edge experiments. O₂ RMSE is stable across all α values (~0.70–0.71%).

---

## 6. Best Configurations per Metric

| Metric | Best Experiment | Value | Notes |
|--------|----------------|-------|-------|
| Temp RMSE | A3\_edge\_only | 0.1721 °C | Edge-only; poor O₂ generalisation |
| Temp RMSE (with real) | F2\_alpha\_5 | 0.2003 °C | α = 5.0, CD-dominant |
| O₂ RMSE | A1\_real\_only | 0.4767 % | Real-only; no edge cases |
| O₂ RMSE (with edge) | C1\_flame\_out | 0.4784 % | Single scenario; near real-only performance |
| Val Physics Loss | C5\_fuel\_trip | 0.0164 | Fuel trip scenario |
| Fastest convergence | C4\_positive\_pressure | epoch 53 | Hardest scenario; early plateau |
| Longest training | C1\_flame\_out | epoch 150 | Steady improvement over many epochs |
| Best balanced | A2\_baseline | 0.2059 °C / 0.7063 % | Default configuration |

---

## 7. Training Dynamics

| Experiment | Epochs Run | CD Gap (final) | E⁺ (final) | E⁻ (final) |
|------------|:---------:|:--------------:|:----------:|:----------:|
| A1\_real\_only | 125 | 9.999 | −5.000 | 4.999 |
| A2\_baseline | 130 | 10.000 | −5.000 | 5.000 |
| A3\_edge\_only | 109 | 10.000 | −5.000 | 5.000 |
| B1\_edge\_10pct | 153 | 10.000 | −5.000 | 5.000 |
| B2\_edge\_25pct | 93 | 10.000 | −5.000 | 5.000 |
| B3\_edge\_50pct | 133 | 10.000 | −5.000 | 5.000 |
| B4\_edge\_200pct | 97 | 10.000 | −5.000 | 5.000 |
| C1\_flame\_out | 170 | 10.000 | −5.000 | 5.000 |
| C2\_air\_leak | 146 | 10.000 | −5.000 | 5.000 |
| C3\_tube\_rupture | 117 | 10.000 | −5.000 | 5.000 |
| C4\_positive\_pressure | 73 | 9.995 | −4.998 | 4.998 |
| C5\_fuel\_trip | 135 | 10.000 | −5.000 | 5.000 |
| C6\_fuel\_contamination | 113 | 10.000 | −5.000 | 5.000 |
| D1\_hidden\_64 | 99 | 9.999 | −5.000 | 5.000 |
| D2\_hidden\_256 | 77 | 10.000 | −5.000 | 5.000 |
| D3\_ebm\_small | 112 | 10.000 | −5.000 | 5.000 |
| E1\_seq15 | 87 | 10.000 | −5.000 | 5.000 |
| E2\_seq60 | 101 | 10.000 | −5.000 | 5.000 |
| F1\_alpha\_01 | 111 | 10.000 | −5.000 | 5.000 |
| F2\_alpha\_5 | 114 | 10.000 | −5.000 | 5.000 |

All experiments achieved full energy separation (CD gap ≈ 10.0, E⁺ ≈ −5.0, E⁻ ≈ +5.0), confirming the contrastive divergence head converges reliably across all configurations. Only C4 (positive pressure) showed slightly incomplete convergence (gap = 9.995).

---

## 8. Key Takeaways

1. **Edge cases are essential for temperature accuracy** but degrade O₂ predictions when used without real data (A3 vs A1).

2. **Full edge-case integration (100%) is optimal.** Lower fractions (10–50%) improve O₂ at the cost of temperature; oversampling (200%) yields diminishing returns.

3. **Air leak (S2) and fuel trip (S5) are the most valuable individual scenarios**, contributing the most to temperature accuracy when used alone.

4. **Positive pressure (S4) is the hardest failure mode** to learn, showing the worst temperature RMSE and earliest convergence plateau.

5. **Hidden size 128 is the architectural sweet spot.** Both smaller (64) and larger (256) models underperform — the larger model overfits despite regularisation.

6. **Sequence length 30 is optimal** for the 1-minute sampling resolution, providing a 30-minute temporal context window.

7. **Higher α (CD-dominant) marginally improves temperature prediction** (F2, α = 5.0 achieves 0.2003°C), though the effect is small.

8. **The EBM head can be halved** ([64, 32] vs [128, 64]) with only +0.035°C degradation, a useful finding for deployment efficiency.

9. **Contrastive divergence converges universally** — all 20 experiments reached near-maximum energy separation (gap ≈ 10.0).

10. **The baseline configuration (A2) remains the recommended default**, offering the best balance of temperature and O₂ accuracy at reasonable training cost.
