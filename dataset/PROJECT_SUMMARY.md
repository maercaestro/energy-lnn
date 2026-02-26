# Physics-Informed Data Augmentation for Furnace Anomaly Detection

## Project Summary — Phase 1: PINN Training & Edge-Case Generation

---

## 1. Problem Statement

We have **470,676 rows** of real furnace operational data collected at 1-minute intervals. This data covers **normal steady-state operations only** — no dangerous events (flame-outs, tube ruptures, fuel trips) are present.

**Challenge:** To train an Energy-Based Liquid Neural Network (EB-LNN) for anomaly detection, the model needs to learn what dangerous states look like. Without edge-case data, the energy landscape cannot distinguish "safe" from "unsafe."

**Solution:** Train a Physics-Informed Neural Network (PINN) on real data, then use its learned physics parameters to generate synthetic edge-case scenarios via analytical energy and mass balance equations.

---

## 2. Data Pipeline

### 2.1 Raw Data → Cleaned Data

| Stage | Rows | File |
|-------|------|------|
| Raw collected | 525,600 | `DataCollected.csv` |
| After cleaning (outlier removal, NaN handling) | **470,676** (89.6% retained) | `cleaned_furnace_data.csv` |

**10 columns:** Date, InletT, OutletT, DraftP, OP_Damper, InletFlow, ExcessO2, FGFlow, FGPressure, Bridgewall

### 2.2 Real Data Statistics

| Variable | Mean | Std | Min | Max | Unit |
|----------|------|-----|-----|-----|------|
| InletT | 288.1 | 6.67 | 263.9 | 305.0 | °C |
| OutletT | 323.8 | 5.07 | 304.1 | 335.2 | °C |
| ΔT | 35.7 | — | 19.5 | 52.5 | °C |
| ExcessO2 | 3.75 | 0.79 | 0.98 | 6.68 | % |
| FGFlow | 1398.0 | 208.9 | 721.0 | 1976.6 | Nm³/hr |
| InletFlow | 368.4 | 42.8 | 245.3 | 442.4 | kbbl/day |
| DraftP | -4.40 | 1.89 | -10.8 | +1.6 | mmH₂O |
| Bridgewall | 658.0 | 27.4 | 556.7 | 723.2 | °C |

### 2.3 Key Observation: Missing Edge Cases

| Dangerous Condition | Rows in Real Data | Coverage |
|---------------------|-------------------|----------|
| ExcessO2 < 1% (flame-out risk) | 6 | 0.00% |
| OutletT > 340°C (overheating) | 0 | 0.00% |
| DraftP > 0 (positive pressure) | ~1,500 | 0.32% |
| Bridgewall > 720°C (refractory damage) | 13 | 0.00% |
| FGFlow < 800 (fuel trip) | 1,166 | 0.25% |

---

## 3. PINN v2 Architecture & Training

### 3.1 Model Design

```
FurnacePINN_v2
├── Main MLP:     7 → [64→Tanh] × 4 → 2   (predicts OutletT, ExcessO2)
├── Air Sub-Net:  3 → [32→Tanh→16→Tanh→1→Softplus]  (DraftP, OP_Damper, FGFlow → air supply)
├── θ_eff:        learnable scalar (η·Cp, softplus-constrained)
└── AFR_stoich:   learnable scalar (stoichiometric air-fuel ratio)
```

**7 inputs:** InletT, InletFlow, FGFlow, DraftP, FGPressure, Bridgewall, OP_Damper  
**2 outputs:** OutletT, ExcessO2

### 3.2 Physics Constraints (Embedded in Loss)

**Energy Balance:**

$$Q_{in} = \dot{m}_{fuel} \cdot LHV \approx \dot{m}_{proc} \cdot \theta_{eff} \cdot (T_{out} - T_{in})$$

**Mass Balance (O₂):**

$$O_2 = 21 \cdot \frac{\lambda - 1}{\lambda}, \quad \lambda = \frac{\dot{m}_{air}}{\dot{m}_{fuel} \cdot AFR_s}$$

**Unit conversions built in:**
- InletFlow: kbbl/day → kg/s (× 1.564)
- FGFlow: Nm³/hr → kg/s (× 0.72 / 3600)
- LHV = 50,000 kJ/kg (natural gas)

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | Normalised MSE (data) + physics residuals |
| Target normalisation | Z-score (critical — model collapsed without it) |
| Warmup | 20 epochs (data-only, no physics) |
| Physics ramp | Linear 0 → 0.1 over 100 epochs after warmup |
| Optimizer | Adam, separate param groups (MLP: wd=1e-5, physics: wd=0) |
| Early stopping | Patience 300, data-only val loss |
| Gradient clipping | max_norm = 1.0 |

### 3.4 Training Results

| Metric | Value |
|--------|-------|
| **RMSE OutletT** | **1.33 °C** |
| **R² OutletT** | **0.898** |
| RMSE ExcessO2 | 0.58% |
| R² ExcessO2 | 0.355 |
| MAE OutletT | 1.09 °C |
| MAE ExcessO2 | 0.41% |
| Learned θ_eff | 0.680 (physically reasonable) |
| Learned AFR_stoich | 17.47 (near theoretical 17.2) |
| Best epoch | 26 / 3000 |

### 3.5 Bugs Fixed During Development

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| Model collapse | OutletT stuck at 147°C | No target normalisation (outputs ~324, initial pred ~0) | Z-score targets |
| θ_eff divergence | θ → -113, RMSE 176°C | Unconstrained parameter | Softplus constraint |
| Physics spike | Loss 200+ at warmup end | Hard step warmup→physics | Linear ramp over 100 epochs |
| θ_eff killed during warmup | θ → ln(2), stuck | weight_decay applied to physics params | Separate Adam param groups |
| λ gradient death | O2 stuck, no learning | Hard clamp `torch.clamp(λ, 1.01, 5.0)` | Soft bound: `1.01 + softplus(λ - 1.01)` |
| Energy balance wrong | E ≈ 1.0 permanently | InletFlow assumed kg/hr, actually kbbl/day | Unit conversion: × 1.564 |

---

## 4. Edge-Case Generation

### 4.1 Approach

Instead of using the PINN's MLP (which can only interpolate within training distribution), we use the **analytical physics equations** with the PINN's **learned parameters** (θ_eff, AFR_stoich) and **learned air supply sub-network** to generate physically consistent edge cases.

### 4.2 Generation Pipeline

```
Scenario Definition (LHS bounds per feature)
    ↓
Latin Hypercube Sampling (25,000 samples per scenario)
    ↓
Analytical Physics Prediction
    ├── Energy balance → OutletT
    └── Mass balance → ExcessO2
    ↓
AR(1) Coloured Noise (autocorrelated, not white noise)
    ↓
Mahalanobis Distance Flagging (confidence: high/medium/low)
    ↓
5-Column EBLNN Derivation (fuel_flow, AFR, temp, T_in, flow_in)
```

### 4.3 Why AR(1) Noise, Not White Noise?

Real plant signals are **autocorrelated** — if a thermocouple reads 0.3°C high this minute, it's still reading high next minute. White noise (i.i.d. Gaussian) gives unrealistically smooth or static-fuzz patterns.

$$\epsilon_t = \phi \cdot \epsilon_{t-1} + \sigma\sqrt{1-\phi^2} \cdot w_t, \quad w_t \sim \mathcal{N}(0,1)$$

AR(1) parameters estimated per column from real data residuals (60-min rolling mean subtracted):

| Variable | φ (autocorrelation) | σ_innovation |
|----------|---------------------|--------------|
| OutletT | 0.895 | 0.128 |
| ExcessO2 | 0.901 | 0.080 |
| FGFlow | 0.738 | 6.212 |
| OP_Damper | 0.982 | 0.086 |

### 4.4 Mahalanobis Distance (Confidence Flag)

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

Measures how far each synthetic point is from the real data distribution (accounting for correlations and variance). Used to flag the reliability of generated data.

| Flag | Threshold | Meaning |
|------|-----------|---------|
| High | D_M ≤ 3 | Close to real data, highly reliable |
| Medium | 3 < D_M ≤ 5 | Moderate extrapolation |
| Low | D_M > 5 | Far OOD, physics-only reliability |

### 4.5 Six Edge-Case Scenarios

| # | Scenario | Key Perturbation | OutletT (°C) | ExcessO2 (%) | Physical Meaning |
|---|----------|-------------------|-------------|-------------|-------------------|
| S1 | **Flame-Out** | FGFlow ↑↑, OP_Damper ↓, DraftP → 0 | 310.6 ± 11.0 | 5.71 ± 0.49 | Rich combustion, CO risk, explosion hazard |
| S2 | **Air Leak** | OP_Damper ↑↑, DraftP ↓↓ | 296.0 ± 10.2 | 14.54 ± 0.72 | Excess cold air dilutes flame |
| S3 | **Tube Rupture** | InletFlow ↓↓, FGFlow ↑, Bridgewall ↑ | 311.6 ± 10.3 | 8.38 ± 1.15 | Process flow loss, overheating |
| S4 | **Positive Pressure** | DraftP → +3, OP_Damper ↓ | 300.1 ± 10.8 | 5.12 ± 0.35 | Blowback through observation ports |
| S5 | **Fuel Trip** | FGFlow → 0–500 | 286.5 ± 10.1 | 17.59 ± 2.34 | Emergency shutoff, furnace cooling |
| S6 | **Fuel Contamination** | FGFlow effective ↓, FGPressure ↓ | 292.5 ± 10.2 | 10.89 ± 1.83 | Lower LHV, hidden underfiring |

### 4.6 Scenario Visualisation

| Plot | File |
|------|------|
| All scenarios overview | `plots_edge_cases/overview_all_scenarios.png` |
| Phase space (FGFlow vs ΔT) | `plots_edge_cases/phase_space_FGFlow_DeltaT.png` |
| Per-scenario details | `plots_edge_cases/S1_flame_out.png` through `S6_fuel_contamination.png` |

---

## 5. Output Datasets

| File | Rows | Columns | Size | Purpose |
|------|------|---------|------|---------|
| `cleaned_furnace_data.csv` | 470,676 | 10 (7 features + 2 targets + Date) | ~55 MB | Cleaned real data |
| `edge_cases_v2.csv` | 150,000 | 7 features + 2 targets + metadata | 43.1 MB | Full edge-case data |
| `real_furnace_eblnn.csv` | 470,676 | 5 EBLNN inputs + 2 targets + labels | 38.8 MB | Real data in EBLNN format |
| `edge_cases_v2_eblnn.csv` | 150,000 | 5 EBLNN inputs + 2 targets + labels | 22.0 MB | Edge cases in EBLNN format |
| **`merged_eblnn_dataset.csv`** | **620,676** | 5 + 2 + 2 metadata | **79.8 MB** | **Final EBLNN training set** |

### 5.1 EBLNN Feature Mapping (7 → 5 columns)

| EBLNN Column | Derived From | Conversion |
|-------------|-------------|------------|
| `fuel_flow` | FGFlow | × ρ_fuel / 3600 → kg/s |
| `air_fuel_ratio` | DraftP, OP_Damper, FGFlow | Via PINN's learned air_net |
| `current_temp` | OutletT | Direct (°C) |
| `inflow_temp` | InletT | Direct (°C) |
| `inflow_rate` | InletFlow | × 1.564 → kg/s |

---

## 6. Proposed Ablation Studies

### 6.1 Data Ablations (A)

| ID | Training Data | Purpose |
|----|--------------|---------|
| A1 | Real only (470k) | Baseline — no augmentation |
| A2 | Real + normal PINN augment (470k + 150k) | Does more normal data help? |
| A3 | Real + edge cases (470k + 150k) | **Main method** |
| A4 | Real + random OOD (470k + 150k) | Is physics-guided better than blind? |
| A5 | Edge cases only (150k) | Can synthetic-only work? |

### 6.2 Physics Fidelity (B)

| ID | Target Generation | Purpose |
|----|-------------------|---------|
| B1 | PINN MLP inference | MLP extrapolation (O2 ≈ 5.1% everywhere — failed) |
| B2 | Analytical physics | **Current method** — wider, realistic extremes |
| B3 | Hybrid (MLP when close, physics when far) | Blend both |

### 6.3 Noise Model (C)

| ID | Method | Purpose |
|----|--------|---------|
| C1 | No noise | Smooth signals only |
| C2 | White noise (i.i.d.) | No autocorrelation |
| C3 | AR(1) coloured noise | **Current method** — realistic |

### 6.4 Scenario Contribution (D)

| ID | Method | Purpose |
|----|--------|---------|
| D1 | All 6 scenarios | Full method |
| D2 | Leave-one-out × 6 | Which scenario matters most? |
| D3 | Single scenario × 6 | Can one failure mode generalise? |

### 6.5 Volume (E)

| ID | Edge Rows | Ratio |
|----|-----------|-------|
| E1 | 30k (5k each) | 6% |
| E3 | 150k (25k each) | 32% |
| E5 | 600k (100k each) | 127% |

### 6.6 Evaluation Metrics

- **Prediction:** RMSE, MAE, R² on held-out real + edge test sets
- **Energy landscape:** ΔE = E_edge − E_normal (separation between safe/unsafe)
- **Anomaly detection:** AUROC using energy score
- **Causal:** DAG structure stability across ablations

**Total: ~36 training runs**

---

## 7. Next Steps

1. **EB-LNN architecture update** — expand input pipeline for 5-column real furnace features
2. **Ablation training harness** — systematic runner for all experiments
3. **EB-LNN training** — on `merged_eblnn_dataset.csv`
4. **Energy landscape analysis** — visualise safe vs dangerous regions
5. **DAG extraction** — causal structure from trained model
6. **Paper writing** — results, ablation tables, visualisations

---

## 8. Repository Structure

```
energy-lnn/
├── dataset/
│   ├── cleaned_furnace_data.csv        # Real data (470k rows)
│   ├── edge_cases_v2.csv               # Synthetic edge cases (150k rows)
│   ├── merged_eblnn_dataset.csv        # Final merged dataset (620k rows)
│   ├── train_pinn_v2.py                # PINN training script
│   ├── generate_edge_cases.py          # Edge-case generator (physics-only)
│   ├── generate_pinn_v2.py             # Normal augmentation generator
│   ├── plot_edge_cases.py              # Visualisation
│   ├── checkpoints_v2/                 # Trained PINN model
│   │   ├── best_pinn_v2.pth
│   │   ├── scaler_v2.pkl
│   │   └── test_metrics_v2.json
│   └── plots_edge_cases/              # Generated figures
├── pilot-study/                        # EB-LNN pilot (synthetic data)
└── literature/
```
