# Fresh Food Supply Chain Optimization
### Data-Driven Inventory Optimization & Weather-Aware Procurement for Perishable Retail

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/LightGBM-Demand%20Forecasting-green?style=flat-square" />
  <img src="https://img.shields.io/badge/PuLP-Stochastic%20MILP-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Stochastic%20Programming-Two--Stage-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/CVaR-Risk%20Management-red?style=flat-square" />
</p>

<p align="center">
  <b>−28.5% inventory cost</b> &nbsp;·&nbsp; <b>−36.3% food waste</b> &nbsp;·&nbsp; <b>VSS 37.5%</b> &nbsp;·&nbsp; <b>EVPI 11.8%</b>
</p>

---

## Overview

This project builds an end-to-end optimization pipeline for fresh food retail supply chains, addressing two core objectives:

1. **Inventory Optimization** — determine the profit-maximizing order quantity for each SKU by recovering true demand from censored observations, forecasting it with a hybrid ML architecture, and solving the newsvendor problem nonparametrically via Sample Average Approximation (SAA).

2. **Weather-Aware Procurement Optimization** — automatically select suppliers, allocate order quantities, and generate delivery routes while explicitly hedging against weather-driven supply disruptions through two-stage stochastic programming.

The domain context is Vietnamese fresh food retail, with procurement optimization grounded in real Da Nang geography and 10 years of historical weather data from the Open-Meteo ERA5 API.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data](#2-data)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [References](#5-references)
6. [Installation](#6-installation)

---

## 1. Problem Statement

Fresh food retail operates under three structural inefficiencies that standard supply chain tools fail to address simultaneously.

**Censored demand.** Stockout events systematically suppress sales records: when a product is unavailable, zero is recorded regardless of actual demand. Any model trained on raw sales data inherits this downward bias, producing conservative replenishment decisions that perpetuate the shortage cycle. The magnitude is quantifiable — the volume-weighted correlation between stockout frequency and observed sales (ρ) measures −0.57 in the FreshRetailNet-50K dataset before correction.

**Non-Gaussian inventory risk.** Standard ERP safety stock formulas assume forecast errors follow a normal distribution. Fresh food demand is empirically heavy-tailed and right-skewed, making the Gaussian safety factor structurally inaccurate — either insufficient during demand spikes or excessive during slow periods, both costly for perishables with shelf lives measured in days.

**Weather-blind procurement.** Procurement decisions for fresh food are made days in advance under significant weather uncertainty. In Da Nang, typhoon probability reaches 28% in October, and a Level 4 flood event renders coastal seafood and agricultural suppliers inaccessible while simultaneously reducing vehicle payload capacity by 40%. Deterministic planning using expected weather conditions fails catastrophically in these tail scenarios — the expected cost of ignoring weather uncertainty is quantified at VSS = 37.5% in this study.

---

## 2. Data

### Demand Data: FreshRetailNet-50K

| Property | Value |
|----------|-------|
| Source | HuggingFace: `Dingdong-Inc/FreshRetailNet-50K` (Wang et al., 2025) |
| Scale | ~50,000 store-product pairs |
| Granularity | Hourly sales amounts + hourly stock status per SKU per store |
| Features | Sales, discount rate, holiday flag, precipitation, temperature |
| Key challenge | Stockout-censored observations: zero sales during out-of-stock hours do not reflect true demand |

The hourly stock status column is the critical signal for identifying censored observations and triggering the reconstruction algorithm.

### Supply Chain Network: Synthetically Generated

The supply chain network used for procurement optimization is synthetically generated to reflect Da Nang's actual geography and commercial structure. The generation pipeline (`src/data_pipeline/`) produces:

- **6 suppliers**: 5 specialized (seafood ports Thọ Quang and Nam Ô, meat supplier in Hòa Khánh industrial zone, vegetable farms in Hòa Vang, fruit supplier) + 1 general wholesale market
- **2 distribution centers**: Hải Châu (central) and Liên Chiểu (north)
- **6 retail stores** across city districts
- **Product catalog**: 10 SKUs across 4 categories (seafood, vegetables, meat, fruit), with unit costs, margins, MOQs, and shelf lives generated via a risk-aware enrichment model — high-volatility SKUs are assigned high-margin profiles where $C_u \gg C_o$, making lost sales economically costlier than overstock

### Weather Data: Da Nang ERA5 (2014–2023)

Historical daily weather (precipitation, temperature, wind speed) is fetched from the Open-Meteo ERA5 API and cached locally. Seasonal distributions are fitted per season:

| Variable | Distribution | Season |
|----------|-------------|--------|
| Rainfall | Gamma(α, β) with zero-day mass | Dry / Monsoon |
| Temperature | Normal(μ, σ) | Dry / Monsoon |
| Wind speed | Weibull(k, λ) | Dry / Monsoon |

Monsoon season (September–December) is the primary risk period: mean rainfall 9.9 mm/day with typhoon probability up to 28% in October.

---

## 3. Methodology

### Phase 1 — Inventory Optimization

**Demand Reconstruction.** Raw sales data is censored: any observation where stock status indicates a stockout period is an incomplete record of true demand. The reconstruction algorithm uses intra-day CDF patterns — derived from non-stockout days at multiple aggregation levels (SKU-store down to global fallback) with shrinkage blending by data availability — to impute true daily demand from the partial pre-stockout sales signal:

$$
\hat{D} = \frac{\sum_{h \leq h^{\ast}} y_{h}}{\hat{F}(h^{\ast})}
$$

where h^{\ast}$ is the last available (non-stockout) hour and $\hat{F}(h^{\ast})$ is the estimated cumulative demand fraction at that hour. A floor constraint prevents unbounded correction in edge cases.

**Demand Forecasting.** The pipeline applies automatic segmentation: SKU-store pairs meeting minimum sales volume and data density thresholds are routed to a LightGBM direct multi-horizon model (one model per forecast horizon $h \in \{1, \ldots, 7\}$ days); remaining intermittent/sparse items use a 7-day Simple Moving Average. This ensures 100% SKU coverage for downstream planning regardless of data quality. The direct multi-step architecture avoids error accumulation from recursive single-step forecasting.

**SAA Inventory Planning.** Given forecast point estimates and the empirical distribution of historical forecast residuals, the order quantity is determined by the Newsvendor critical ratio:

$$SL^* = \frac{C_u}{C_u + C_o}$$

Rather than assuming a parametric (Gaussian) error distribution, Sample Average Approximation (SAA) uses the empirical quantile of scaled residuals directly:

$$Q_{\text{SAA}} = \hat{y} \cdot \left(1 + q_{SL^*}(\tilde{\varepsilon})\right)$$

where $\tilde{\varepsilon}$ is the pooled empirical scaled-error distribution from historical forecast residuals, pooled by risk group for robustness on sparse SKUs. The SAA benchmark compares against two baselines: **Naive** (order = forecast, no safety stock) and **Normal** (parametric Gaussian safety stock, $Q = \hat{y} + Z_\alpha \cdot \sigma_{\text{RMSE}}$).

---

### Phase 2 — Weather-Aware Procurement Optimization

**Weather Scenario Generation.** Ten representative scenarios are generated from fitted historical distributions via Latin Hypercube Sampling (LHS) followed by Fast Forward Selection (FFS) to minimize Wasserstein distance from the full 600-sample distribution. A constrained probability blending step ($\alpha = 0.40$) corrects for systematic underrepresentation of typhoon scenarios in FFS output, using WMO Da Nang climatological targets as the blending anchor.

Each scenario maps to operational parameters: vehicle speed reduction factor, payload capacity factor, spoilage multiplier, and supplier accessibility by subtype:

| Severity | Seafood Suppliers | Vegetable Farms | Meat Suppliers | General Wholesale |
|----------|:-----------------:|:---------------:|:--------------:|:-----------------:|
| L1–L3 | ✓ | ✓ | ✓ | ✓ |
| L4 (Heavy rain) | ✗ | ✗ | ✓ | ✓ |
| L5 (Typhoon) | ✗ | ✗ | ✗ | ✓ |

The L4 accessibility split reflects Da Nang geography: coastal seafood ports and rural farms flood before the inland industrial zone.

**Two-Stage Stochastic MILP.** The procurement problem is formulated as an extensive-form two-stage stochastic MILP (Birge & Louveaux, 2011):

*Stage 1 — before weather realization:*

$$\min_{x, y} \sum_{s,p} c_{sp} \cdot x_{sp} + \sum_{s,p} f_s \cdot y_{sp}$$

where $x_{sp}$ is order quantity (continuous) and $y_{sp}$ is binary supplier activation for fixed-cost charging. Non-anticipativity is enforced structurally: Stage 1 variables carry no scenario index $k$.

*Stage 2 — after scenario k realizes:*

$$\min_{e_k,\, u_k} \sum_p \left(2c_p \cdot e_{k,p} + \pi_k \cdot c_p \cdot u_{k,p}\right)$$

where $e_{k,p}$ is emergency procurement (at 2× cost) and $u_{k,p}$ is unmet demand (at penalty $\pi_k$). Demand satisfaction under scenario $k$ uses only accessible suppliers:

$$\sum_{s:\, a_{k,s}=1} x_{sp} + e_{k,p} + u_{k,p} \geq D_p \qquad \forall k, p$$

This constraint — accessible supply set varies by scenario while $x_{sp}$ remains fixed — is the core non-anticipativity-preserving formulation, preserving linearity while correctly modeling supply disruptions.

**Heterogeneous Fleet VRP.** Routing is co-optimized with procurement in the extensive form. The fleet (adapted from Patel et al., 2024) includes 8 vehicles across 4 types with weather-adjusted payload and operational limits. Refrigerated trucks reduce perishable spoilage by 65% (cold-chain Arrhenius benefit). Subtour elimination uses MTZ constraints.

**CVaR Risk Extension.** The objective extends to a risk-adjusted formulation (Rockafellar & Uryasev, 2000):

$$\min Z = (1-\lambda)\cdot\mathbb{E}[\text{cost}] + \lambda\cdot\text{CVaR}_\alpha$$

with $\lambda \in [0,1]$ controlling risk aversion. Auxiliary variables $\eta$ (VaR threshold) and $\zeta_k = \max(\text{cost}_k - \eta,\, 0)$ linearize CVaR within the MILP.

**Validation.** Model correctness is verified through the stochastic programming ordering property (Birge & Louveaux, 2011):

$$WS \leq RP \leq EEV$$

where $WS = \sum_k p_k \cdot \text{OPT}_k$ (wait-and-see, $K$ separate deterministic solves), $RP$ is the stochastic solution, and $EEV = \sum_k p_k \cdot \text{cost}_k(x^*_{EV})$ (deterministic EV solution evaluated stochastically). All three values are computed with their mathematically correct definitions.

---

## 4. Results

### Phase 1 — Inventory Optimization

**Demand reconstruction successfully decoupled true demand from stockout bias.** The volume-weighted correlation between stockout frequency and observed sales dropped from ρ = −0.57 (raw) to ρ = 0.07 post-reconstruction — near-zero, confirming that the censoring bias was effectively removed. Global WAPE on a random censoring test was 27.38%, consistent with the FreshRetailNet-50K benchmark paper.

**Forecasting achieved WAPE 40% on the test set**, competitive for fresh produce under real-world noise and seasonal volatility (industry benchmarks: 35–60%).

**SAA outperformed both baselines on all metrics:**

| Policy | Total Cost | Cost Savings | Waste Reduction | Service Level |
|--------|:----------:|:------------:|:---------------:|:-------------:|
| Naive | $20,399,969 | — | — | 88.8% |
| Normal (Gaussian) | $15,092,468 | −26.0% | −33.5% | 81.9% |
| **SAA (data-driven)** | **$14,584,390** | **−28.5%** | **−36.3%** | **81.6%** |

The 2.5% cost advantage of SAA over the Gaussian baseline reflects the structural benefit of using the empirical error distribution rather than a normal approximation. The service level reduction from 88.8% (Naive) to 81.6% (SAA) represents the elimination of economically unprofitable demand: the last 7.2% of sales satisfied under the Naive policy requires disproportionate inventory holding relative to the margin captured.

---

### Phase 2 — Weather-Aware Procurement Optimization

**The stochastic solution provides substantial value over deterministic planning.** VSS = 37.5% (2.21B VND): a deterministic plan evaluated across the actual weather distribution performs 37.5% worse than the stochastic solution. This places the problem in the high-stochastic-value category where ignoring weather uncertainty generates significant economic harm.

| Metric | Value |
|--------|------:|
| RP — stochastic solution | 3,691,403,494 VND |
| EEV — deterministic evaluated stochastically | 5,903,012,685 VND |
| **VSS = EEV − RP** | **2,211,609,192 VND (37.47%)** |
| WS — wait-and-see | 3,255,626,126 VND |
| **EVPI = RP − WS** | **435,777,368 VND (11.81%)** |
| Ordering property WS ≤ RP ≤ EEV | ✓ PASS |

**EVPI = 11.8%** quantifies the maximum rational investment in a perfect weather forecasting system per planning cycle — providing a concrete ROI ceiling for weather monitoring infrastructure decisions.

**The tail risk profile is severe.** CVaR-90% reaches 175.2% above expected cost, driven by the typhoon scenario (9.5% probability in monsoon season):

| Risk Metric | Value |
|-------------|------:|
| Expected cost | 3,691,403,494 VND |
| VaR-90% | 5,283,359,949 VND |
| CVaR-90% | 10,159,802,874 VND |
| **CVaR premium** | **175.2%** |

A CVaR premium of 175.2% means the worst 10% of scenarios cost 2.75× more than the expected case — not marginal variance, but a supply failure event. This result provides quantitative justification for CVaR-aware procurement ($\lambda > 0$) over risk-neutral optimization in food security contexts.

---

## 5. References

| Paper | Role |
|-------|------|
| Wang et al. (2025) — *FreshRetailNet-50K*, arXiv:2505.16319 | Benchmark dataset + demand reconstruction foundation |
| Lau & Lau (1996) — *EJOR 92(2)* | Censored demand reconstruction methodology |
| Huber et al. (2019) — *EJOR 278(3)* | SAA newsvendor / data-driven inventory planning |
| Patel et al. (2024) — *Sustainability 16(1)* | Stochastic MILP + VRP structure for perishables |
| Birge & Louveaux (2011) — *Springer* | Two-stage SP theory, VSS, EVPI definitions |
| Rockafellar & Uryasev (2000) — *Journal of Risk 2(3)* | CVaR formulation |
| Heitsch & Römisch (2003) — *COAP 24(2–3)* | Fast Forward Selection scenario reduction |
| McKay, Beckman & Conover (1979) — *Technometrics 21(2)* | Latin Hypercube Sampling |
| Miller, Tucker & Zemlin (1960) — *JACM 7(4)* | MTZ subtour elimination |

---

## 6. Installation

```bash
pip install pandas numpy lightgbm scipy pulp scikit-learn
pip install datasets   # HuggingFace for FreshRetailNet-50K
pip install requests   # Open-Meteo weather API
```

### Repository Structure

```
├── config/settings.py                    # Hyperparameters and paths
├── src/
│   ├── data_pipeline/                    # Preprocessing, catalog, network generation
│   ├── demand/
│   │   ├── reconstruction.py             # Censored demand recovery
│   │   └── forecasting.py                # Hybrid LightGBM + SMA pipeline
│   ├── inventory/
│   │   ├── planner.py                    # SAA inventory planning
│   │   └── evaluator.py                  # Policy benchmarking
│   ├── optimization/
│   │   ├── extensive_form_optimizer.py   # Two-stage MILP + integrated VRP
│   │   ├── stochastic_procurement.py     # Stochastic procurement formulation
│   │   └── cvar_procurement.py           # CVaR risk-averse extension
│   ├── weather/
│   │   ├── weather_data.py               # ERA5 API + distribution fitting
│   │   ├── scenario_generator.py         # LHS + FFS scenario generation
│   │   └── manual_scenarios.py           # Pre-defined Da Nang scenarios
│   └── evaluation/
│       └── vss_evpi_calculator.py        # VSS / EVPI / CVaR validation
└── scripts/
    └── run_stochastic_optimization.py    # Full procurement pipeline entry point
```

### Quick Start

```bash
# Phase 1 — demand recovery, forecasting, inventory benchmarking
python -c "
from src.data_pipeline.preprocessor import FrnPreprocessor
from src.demand.reconstruction import DemandReconstructor
from src.demand.forecasting import DemandForecaster
from src.inventory.evaluator import PolicyEvaluator

FrnPreprocessor().run()
DemandReconstructor().run()
DemandForecaster().run()
PolicyEvaluator().run()    # prints Naive / Normal / SAA scorecard
"

# Phase 2 — stochastic procurement optimization
python scripts/run_stochastic_optimization.py
# Select: 2 (Monsoon season)
```

---

## Skills & Technologies

`Python` · `Supply Chain Optimization` · `Operations Research` · `Demand Forecasting` · `LightGBM` · `Time Series Analysis` · `Mixed-Integer Linear Programming (MILP)` · `Two-Stage Stochastic Programming` · `Sample Average Approximation (SAA)` · `Newsvendor Problem` · `Vehicle Routing Problem (VRP)` · `Conditional Value-at-Risk (CVaR)` · `Scenario Generation` · `Latin Hypercube Sampling` · `Inventory Management` · `Perishable Goods Logistics` · `Weather Uncertainty Modeling` · `PuLP` · `Pandas` · `NumPy` · `SciPy`


