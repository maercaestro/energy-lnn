# Causal Discovery Analysis on the Augmented Dataset

**Script:** `dataset/causal_discovery_dataset.py`  
**Input:** `merged_eblnn_dataset.csv` (real furnace data + 6 synthetic edge-case scenarios, ~620k rows total)  
**Output:** 5 plots saved to `plots_causality/`, results saved to `plots_causality/causal_discovery_results.json`

---

## 1. Purpose

This analysis verifies that the merged dataset (real + synthetic edge cases) retains genuine **structural causality** matching known furnace physics — not just spurious correlations. If the synthetic edge cases had distorted the causal structure (e.g., decoupled air-fuel ratio from O₂), the EB-LNN model would have learned wrong physics. Confirming causal structure is preserved validates the dataset augmentation as physically coherent.

---

## 2. Variables Analysed

| Role | Columns |
|---|---|
| Input features | `fuel_flow`, `air_fuel_ratio`, `current_temp`, `inflow_temp`, `inflow_rate` |
| Target outputs | `next_temp`, `next_excess_o2` |

All 7 columns are analysed jointly. A subsample of **50,000 rows** (seed=42) is used for computational tractability.

---

## 3. Ground-Truth Causal Edges (from Furnace Physics)

These are the expected causal relationships based on first-principles combustion physics:

| Cause | Effect | Physical Reasoning |
|---|---|---|
| `fuel_flow` | `next_temp` | Direct energy balance: more fuel → more heat |
| `fuel_flow` | `current_temp` | Energy input heats the process fluid |
| `air_fuel_ratio` | `next_excess_o2` | Mass balance: excess air = excess O₂ in flue gas |
| `inflow_temp` | `next_temp` | Inlet temperature propagates to outlet |
| `inflow_rate` | `next_temp` | More mass flow → less ΔT (heat dilution) |
| `current_temp` | `next_temp` | Thermal inertia of the furnace |
| `fuel_flow` ↔ `air_fuel_ratio` | (coupled) | Combustion control couples fuel and air supply |

---

## 4. Three Complementary Causal Methods

### Method 1 — Partial Correlation DAG (Constraint-Based)

Removes all shared variance due to other variables. Uses the **precision matrix** (inverse of the correlation matrix):

$$\rho_{\text{partial}}(i,j) = \frac{-P_{ij}}{\sqrt{P_{ii} \cdot P_{jj}}}$$

Each pair is tested with a two-sided t-test:

$$t = \rho_{\text{partial}} \cdot \sqrt{\frac{n - k - 2}{1 - \rho_{\text{partial}}^2}}$$

where $n$ = number of observations, $k$ = number of conditioning variables.

P-values are corrected for multiple comparisons using **Benjamini-Hochberg FDR** at $\alpha = 0.01$.

- **Output:** Undirected skeleton (shows *which* pairs are directly related, not direction)
- **Significant edges found:** 19

---

### Method 2 — Cross-Sectional Granger Causality (Regression-Based)

For every directed pair $X \to Y$, tests whether knowing $X$ improves prediction of $Y$ beyond all other variables:

- **Restricted model:** $Y \sim \text{all variables except } X$
- **Full model:** $Y \sim \text{all variables including } X$
- **F-test** on residual sum of squares improvement

$$F = \frac{(RSS_R - RSS_F) / 1}{RSS_F / (n - p_F - 1)}$$

Score reported as $-\log_{10}(p)$ — higher means stronger causal signal. Threshold: $-\log_{10}(p) > 2$ (i.e., $p < 0.01$).

- **Output:** Directed matrix
- **Significant directed edges found:** 16

---

### Method 3 — Transfer Entropy (Information-Theoretic, Nonlinear)

Approximates conditional mutual information using **quantile binning** (10 bins per variable):

$$TE(X \to Y) \approx H(Y \mid Z) - H(Y \mid Z, X)$$

where $Z$ = all other variables. Conditional entropy is estimated via groupby counting over binned values. Captures **nonlinear dependencies** that linear regression misses.

Threshold: $TE > 0.01$ bits.

- **Output:** Directed matrix in bits
- **Significant directed edges found:** 10

---

## 5. Consensus DAG

A directed edge $X \to Y$ is accepted into the final causal graph if **≥ 2 out of 3 methods agree**.

$$\text{score}(X \to Y) = \mathbb{1}[\text{Partial Corr}] + \mathbb{1}[\text{Granger}] + \mathbb{1}[\text{Transfer Entropy}]$$

Edge is kept if score ≥ 2.

- **Total consensus directed edges:** 18

---

## 6. Results

| Metric | Value |
|---|---|
| Sample size used | 50,000 |
| Significance level (α) | 0.01 |
| Partial correlation edges | 19 |
| Granger causality edges | 16 |
| Transfer entropy edges | 10 |
| Consensus edges (≥ 2/3) | 18 |
| Ground-truth edges recovered | 3 / 7 (43%) |

### Ground-Truth Recovery Detail

| Edge | Consensus Score | Status |
|---|---|---|
| `fuel_flow` → `next_temp` | 1 / 3 | Not recovered at the consensus threshold |
| `fuel_flow` → `current_temp` | 1 / 3 | Not recovered at the consensus threshold |
| `air_fuel_ratio` → `next_excess_o2` | 2 / 3 | Recovered |
| `inflow_temp` → `next_temp` | 1 / 3 | Not recovered at the consensus threshold |
| `inflow_rate` → `next_temp` | 1 / 3 | Not recovered at the consensus threshold |
| `current_temp` → `next_temp` | 3 / 3 | Recovered by all methods |
| `fuel_flow` ↔ `air_fuel_ratio` | 2 / 3 | Recovered as a coupled relationship |

The analysis therefore recovers 3 of 7 prespecified physical relationships at the strict two-method threshold. The strongest result is `current_temp` → `next_temp`, which is supported by all three methods and is consistent with thermal inertia. The recovery of `air_fuel_ratio` → `next_excess_o2` also supports the expected combustion mass-balance relationship. Finally, the recovered coupling between fuel flow and air-fuel ratio is consistent with coordinated combustion control.

The remaining four expected temperature relationships each receive one method vote. They should be reported as detected signals that did not meet the conservative consensus criterion, rather than as evidence that the physical relationships are absent.

---

## 7. Output Plots

| File | Description |
|---|---|
| `plots_causality/partial_correlation_dag.png` | Partial correlation matrix heatmap + significant edges after FDR correction |
| `plots_causality/granger_causality.png` | Directed $-\log_{10}(p)$ Granger matrix (row causes column) |
| `plots_causality/transfer_entropy.png` | Directed transfer entropy matrix in bits (row → column) |
| `plots_causality/consensus_dag.png` | Heatmap of method agreement count per directed edge (0–3) |
| `plots_causality/causal_dag_network.png` | Final consensus DAG as NetworkX graph: inputs (blue, left) → targets (orange, right) |

---

## 8. How to Run

```bash
cd /path/to/energy-lnn/dataset

# Default: 50k subsample, merged dataset
python causal_discovery_dataset.py

# All rows (slow)
python causal_discovery_dataset.py --n_samples 0

# Custom data path
python causal_discovery_dataset.py --data merged_eblnn_dataset.csv --n_samples 100000
```

---

## 9. Interpretation for Thesis

Using a 50,000-row, seed-42 subsample, the analysis recovered 3 of 7 prespecified furnace-physics relationships (43%) at the strict 2/3 consensus threshold. This provides partial, rather than complete, structural validation of the augmented dataset. In particular, the all-method recovery of `current_temp` → `next_temp` and the consensus recovery of `air_fuel_ratio` → `next_excess_o2` show that the data retain key thermal-inertia and air-excess relationships after edge-case augmentation.

The 43% recovery rate reflects two important limitations:

1. **Cross-sectional data vs. time series:** True Granger causality requires temporal lags. The cross-sectional approximation used here is less powerful for detecting lagged causal effects (e.g., `inflow_rate → next_temp` requires a time lag to appear).

2. **Feature compression:** The 5 EBLNN columns are derived from 7 raw features. `air_fuel_ratio` is a composite of `FGFlow`, `DraftP`, and `OP_Damper`, which compresses some causal pathways.

Accordingly, this analysis should be presented as a robustness and plausibility check, not as definitive causal identification. The three recovered relationships are the primary validation signal; the four one-vote relationships motivate a follow-up analysis using explicitly lagged furnace time series and the original, uncompressed process variables.
