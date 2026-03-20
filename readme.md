# Fresh Food Supply Chain Optimization
### End-to-End Demand Intelligence & Weather-Aware Procurement for Perishable Retail

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/LightGBM-ML%20Forecasting-green?style=flat-square" />
  <img src="https://img.shields.io/badge/PuLP-MILP%20Optimization-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Stochastic%20Programming-Two--Stage-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Research%20Complete-brightgreen?style=flat-square" />
</p>

<p align="center">
  <b>28.5% cost reduction</b> · <b>36.3% waste reduction</b> · <b>VSS 37.5%</b> · <b>EVPI 11.8%</b>
</p>

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Overview](#2-project-overview)
3. [System Architecture](#3-system-architecture)
4. [Dataset](#4-dataset)
5. [Module 1 — Demand Reconstruction](#5-module-1--demand-reconstruction)
6. [Module 2 — Demand Forecasting](#6-module-2--demand-forecasting)
7. [Module 3 — Inventory Planning](#7-module-3--inventory-planning)
8. [Module 4 — Stochastic Procurement Optimization](#8-module-4--stochastic-procurement-optimization)
9. [Integrated Results](#9-integrated-results)
10. [Scientific Basis & References](#10-scientific-basis--references)
11. [Installation & Usage](#11-installation--usage)

---

## 1. Problem Statement

The fresh food supply chain is one of the most operationally complex and financially punishing domains in retail logistics. Three structural inefficiencies converge to create a system that is simultaneously wasteful and understocked:

**The Stockout Bias Problem.** In fresh food retail, periods of high demand often coincide with stockout events — the product is unavailable precisely when customers want it most. Historical sales records therefore systematically underestimate true demand. Any forecasting or inventory model trained on raw sales data inherits this censoring bias, producing conservative replenishment decisions that perpetuate the shortage cycle.

**The Distribution Assumption Problem.** Classical inventory management (as implemented in most ERP systems) assumes forecast errors follow a Gaussian distribution. Fresh food demand is empirically heavy-tailed, right-skewed, and strongly seasonal. A normal distribution approximation systematically underestimates tail risk, generating safety stock levels that are either insufficient during demand spikes or excessive during slow periods — both costly outcomes for perishables with shelf lives measured in days.

**The Uncertainty Blind Spot.** Procurement and routing decisions in fresh food logistics are made days in advance under significant weather uncertainty. A typhoon that closes a coastal supplier, or heavy rain that reduces vehicle capacity by 40%, can cascade into total supply failure. Deterministic planning — using expected weather conditions — ignores this uncertainty entirely and fails catastrophically in tail scenarios.

This project addresses all three problems through a unified, end-to-end pipeline: recovering true demand from censored observations, forecasting it with a hybrid ML architecture, optimizing inventory policy with distribution-free methods, and finally solving procurement and routing as a two-stage stochastic program with explicit weather scenario modeling.

The domain context is Vietnamese fresh food retail — specifically Da Nang, a city with a pronounced monsoon season (September–December) during which typhoon probabilities reach 28% in October and supply disruptions are structurally predictable.

---

## 2. Project Overview

This project delivers a four-module optimization pipeline that transforms raw transactional retail data into actionable procurement and routing plans, explicitly accounting for demand uncertainty, distribution shape, and weather risk.

| Module | Problem | Method | Key Result |
|--------|---------|--------|------------|
| **1. Demand Reconstruction** | Censored demand from stockouts | Hierarchical Statistical CDF Reconstruction | WAPE 27.4%, ρ reduced to 0.07 |
| **2. Demand Forecasting** | Volatile fresh food demand | Hybrid LightGBM + SMA segmentation | WAPE 40% on test set |
| **3. Inventory Planning** | Non-Gaussian demand distribution | Newsvendor + Sample Average Approximation (SAA) | −28.5% cost, −36.3% waste vs. naive |
| **4. Procurement Optimization** | Supply & weather uncertainty | Two-stage stochastic MILP + Weather-aware VRP | VSS 37.5%, EVPI 11.8% |

**Two primary business objectives:**
- **Inventory Intelligence:** Determine the optimal order quantity for each SKU by learning the true empirical error distribution, not assuming it.
- **Procurement Automation:** Automatically select suppliers, allocate order quantities, and generate delivery routes while hedging against weather-driven supply disruptions.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REPO 1 — Demand Intelligence                  │
│                                                                   │
│  FreshRetailNet-50K                                               │
│  (Hourly store-product sales + stock status)                      │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐    True Demand    ┌─────────────────────┐   │
│  │  Module 1       │ ─────────────────► │  Module 2           │   │
│  │  Demand Recon.  │                   │  Demand Forecasting  │   │
│  │  Hierarchical   │                   │  LightGBM + SMA      │   │
│  │  CDF / Shrinkage│                   │  Hybrid Pipeline     │   │
│  └─────────────────┘                   └──────────┬──────────┘   │
│                                                    │ Forecast +   │
│                                                    │ Residuals    │
│                                                    ▼              │
│                                         ┌─────────────────────┐  │
│                                         │  Module 3           │  │
│                                         │  Inventory Planning  │  │
│                                         │  SAA Newsvendor      │  │
│                                         └──────────┬──────────┘  │
└────────────────────────────────────────────────────│─────────────┘
                                                      │ Order Qty
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  REPO 2 — Procurement Optimization               │
│                                                                   │
│  Da Nang Weather Data ──► Scenario Generation (LHS + FFS)        │
│  (Open-Meteo API, 2014–2023)   10 scenarios, probability-corrected│
│                          │                                        │
│                          ▼                                        │
│           ┌──────────────────────────────┐                       │
│           │  Module 4                    │                       │
│           │  Two-Stage Stochastic MILP   │                       │
│           │  ┌──────────┐ ┌───────────┐ │                       │
│           │  │ Stage 1  │ │ Stage 2   │ │                       │
│           │  │ Procure  │ │ Recourse  │ │                       │
│           │  │ (x, y)   │ │ (e, u)    │ │                       │
│           │  └──────────┘ └───────────┘ │                       │
│           │  + Weather-Aware VRP         │                       │
│           │  + Heterogeneous Fleet       │                       │
│           └──────────────────────────────┘                       │
│                          │                                        │
│                          ▼                                        │
│           Supplier selection + Order allocation + Route plan      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Dataset

### FreshRetailNet-50K

All demand intelligence modules are built on **FreshRetailNet-50K** (Ding et al., 2024), a publicly available benchmark dataset specifically designed for fresh food retail research.

| Property | Value |
|----------|-------|
| Source | HuggingFace: `Dingdong-Inc/FreshRetailNet-50K` |
| Scale | ~50,000 store-product pairs |
| Granularity | Hourly sales + stock status per SKU per store |
| Features | Sales amount, discount, holiday flag, precipitation, temperature |
| Coverage | Multiple stores across product categories (vegetables, fruits, meat, seafood) |
| Challenge | Heavy censoring: stockout hours record zero sales regardless of true demand |

The hourly stock status column (`hours_stock_status`) is the critical signal used in Module 1 to identify censored observations and trigger the reconstruction algorithm.

**Preprocessing pipeline** (`src/data_pipeline/preprocessor.py`) handles memory-efficient loading, feature engineering (trigonometric day-of-week encoding, promo flagging), hourly array parsing, and coverage-based SKU filtering. A heuristic stockout flag detector automatically identifies whether status=0 or status=1 indicates out-of-stock conditions.

### Weather Data (Module 4)

Da Nang historical weather (2014–2023) is fetched from the Open-Meteo ERA5 API and cached locally. Seasonal distributions (Gamma for rainfall, Normal for temperature, Weibull for wind) are fitted per season and used to generate representative scenarios via Latin Hypercube Sampling.

| Season | Characteristics | Typhoon Risk |
|--------|----------------|--------------|
| Dry (Feb–Aug) | Mean temp 28.4°C, low rainfall | < 3% |
| Monsoon (Sep–Dec) | Mean rainfall 9.9mm/day, high variance | Up to 28% in October |

---

## 5. Module 1 — Demand Reconstruction

### The Problem: Censored Demand

When a product is out of stock, sales are recorded as zero regardless of actual customer demand. This creates **right-censored observations** — the true demand is unobserved, and its value is known only to be at least equal to actual sales. Training any model on these censored values produces systematically biased (downward) demand estimates.

The magnitude of this bias is measurable: the volume-weighted correlation between stockout fraction and observed sales (ρ_raw) was **−0.57**, confirming that high-stockout SKUs are severely underrepresented in raw sales data.

### Method: Hierarchical Statistical Reconstruction

The reconstruction follows a 4-tier CDF hierarchy that exploits the structure of intra-day demand patterns:

```
L1 (SKU × Store × Weekday × Promo)  ← most specific
L2 (SKU × Weekday × Promo)
L3 (Global × Weekday)
L4 (Global fallback)
```

For each SKU-day observation, a **shrinkage blending** weight is computed from data availability and distributional stability (Kolmogorov-Smirnov drift statistic). The blended CDF is then used to reconstruct the true daily demand from partial (pre-stockout) hourly observations:

```
D̂ = (Σ sales up to stockout hour) / CDF(last_available_hour)
```

A floor constraint (`CDF_MIN_CLIP = 0.15`) prevents unbounded upward correction in pathological cases.

### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Global WAPE (random censoring test) | **27.38%** | Matches benchmark paper performance |
| Post-reconstruction correlation (ρ) | **0.07** | Near-zero: demand successfully decoupled from stockout bias |

The reduction from ρ = −0.57 to ρ = 0.07 is the primary validation signal. The reconstructed demand is used as the clean training target for all downstream modules.

---

## 6. Module 2 — Demand Forecasting

### Hybrid Segmentation Architecture

Not all SKUs are equally forecastable. Fresh food includes both high-volume, stable products (forecastable with ML) and intermittent, sparse items (better served by simple moving averages). Applying ML universally wastes compute and degrades accuracy on sparse items.

The pipeline automatically segments all SKU-store pairs:

| Segment | Criteria | Method | Coverage |
|---------|----------|--------|----------|
| **ML branch** | Mean sales ≥ 1.5 AND data density ≥ 70% | LightGBM direct multi-horizon | High-volume items |
| **SMA branch** | Otherwise | 7-day Simple Moving Average | Intermittent / sparse items |

This segmentation ensures 100% SKU coverage for downstream inventory planning regardless of data quality.

### LightGBM Multi-Horizon Forecasting

The ML branch implements a **direct multi-horizon** approach — one model per forecast horizon h ∈ {1, …, 7} — rather than recursive single-step prediction, which would accumulate errors over the horizon.

**Feature engineering:**
- 28-day lag window (sequential lags + rolling mean/std over 7 days)
- Trigonometric day-of-week and month-of-year encoding
- Promotional binary and intensity features
- Average temperature (correlated with fresh food demand in Da Nang)
- Store and product category embeddings

**Training protocol:** Strict temporal split (80% train / 10% validation / 10% test) with early stopping on validation RMSE. Log1p transformation for target normalization with bias-corrected back-transformation.

### Results

| Metric | Value | Context |
|--------|-------|---------|
| WAPE on test set | **40%** | Competitive for fresh produce (industry benchmark: 35–60%) |
| Coverage | **100% of SKUs** | Full coverage via ML + SMA fallback |

A WAPE of 40% on fresh food — which is subject to weather effects, promotions, and high intrinsic volatility — is consistent with published results (Fildes et al., 2019; Bojer & Meldgaard, 2021) and represents a solid forecasting baseline for inventory planning.

---

## 7. Module 3 — Inventory Planning

### The Problem: Non-Gaussian Demand Errors

Standard safety stock formulas assume forecast errors follow a normal distribution:

```
Q = ŷ + Z_α × σ_RMSE
```

where Z_α is the normal quantile at target service level α. For fresh food, this assumption is empirically violated: error distributions are heavy-tailed, right-skewed (underforecast events are larger than overforceast events), and product-dependent. A Gaussian approximation systematically underestimates the safety buffer needed for high-percentile coverage.

### Method: Risk-Aware Catalog + SAA Newsvendor

**Economics assignment.** Each SKU is assigned an asymmetric cost structure via a risk-aware catalog:

```
Cu (underage cost) = Unit price × margin        — opportunity cost of lost sales
Co (overage cost)  = Unit cost + disposal cost  — cost of expired inventory
Target SL*         = Cu / (Cu + Co)             — critical ratio (Newsvendor formula)
```

Products with high demand volatility (CV > 0.8) are classified as `HighRisk` and assigned high-margin profiles (Cu >> Co), which pushes their optimal order quantity upward — economically correct for items where a stockout is far more costly than a small overstock.

**Sample Average Approximation (SAA).** Instead of assuming a parametric error distribution, SAA uses the empirical distribution of historical forecast residuals:

```
Q_SAA = ŷ × (1 + q_{SL*}(ε̃))
```

where ε̃ is the empirical scaled error pool, pooled by risk group for robustness on sparse SKUs. This is the nonparametric solution to the newsvendor problem — optimal under the true (unknown) demand distribution as sample size grows (Kleywegt et al., 2002).

### Benchmark Comparison

Three policies are evaluated on the same held-out test set via ex-post simulation:

| Policy | Total Cost | Cost Savings | Waste Reduction | Service Level |
|--------|-----------|-------------|----------------|---------------|
| **Naive** (zero safety stock) | $20,399,969 | 0.0% | 0.0% | 88.8% |
| **Normal** (Gaussian parametric) | $15,092,468 | 26.0% | 33.5% | 81.9% |
| **SAA** (data-driven empirical) | **$14,584,390** | **28.5%** | **36.3%** | **81.6%** |

**Key insights:**

SAA outperforms the Gaussian parametric baseline by 2.5% on total cost and 2.8 percentage points on waste reduction. The margin is modest on this relatively clean benchmark dataset; in real-world deployments with noisier data and more extreme distribution shapes, the gap widens (Bertsimas & Koduri, 2022 report 1–8% improvement over parametric methods).

The service level reduction from 88.8% (Naive) to 81.6% (SAA) reflects the elimination of **"Toxic Revenue"** — sales volume that is economically unprofitable to serve. The last 7.2% of demand satisfied under the Naive policy requires disproportionate inventory holding costs. SAA correctly identifies and cuts this unprofitable tail.

---

## 8. Module 4 — Stochastic Procurement Optimization

### Motivation: Why Stochastic Programming?

A deterministic procurement plan uses expected weather conditions. In Da Nang, this approach fails because:

- Weather uncertainty is **asymmetric**: a typhoon (10% probability in monsoon season) does not merely double costs — it can cause total supply failure for coastal and agricultural suppliers.
- The **Value of Stochastic Solution (VSS = 37.5%)** demonstrates that a deterministic plan evaluated across all weather scenarios performs nearly 40% worse than the stochastic solution. This is one of the strongest arguments for stochastic programming in the supply chain literature.

### Two-Stage Stochastic MILP

The extensive-form formulation follows Birge & Louveaux (2011):

**Stage 1 — "Here and Now" (before weather realization):**
```
min Σ_{s,p} c_{sp} · x[s,p] + Σ_{s,p} f_s · y[s,p]   (procurement + fixed order costs)
```

Decision variables `x[s,p]` (quantity from supplier s for product p) and `y[s,p]` (binary activation) are made before any weather scenario is realized, satisfying non-anticipativity.

**Stage 2 — "Wait and See" (after scenario k is realized):**
```
min Σ_p (2·c_p · e[k,p] + penalty_k · c_p · u[k,p])   (emergency procurement + unmet demand)
```

Suppliers become accessible or inaccessible per scenario based on geographic flood maps:

| Severity Level | Seafood Suppliers | Vegetable Farms | Meat Suppliers | General Wholesale |
|---------------|-------------------|-----------------|----------------|------------------|
| 1–3 (Normal–Moderate) | ✓ | ✓ | ✓ | ✓ |
| 4 (Heavy Rain) | ✗ (coastal flood) | ✗ (rural flood) | ✓ | ✓ |
| 5 (Typhoon) | ✗ | ✗ | ✗ | ✓ |

### Weather Scenario Generation

10 representative scenarios are generated via a **Latin Hypercube Sampling + Fast Forward Selection (LHS + FFS)** pipeline fitted to 10 years of Da Nang ERA5 weather data:

- **LHS** generates 600 stratified samples from fitted seasonal distributions (Gamma rainfall, Normal temperature, Weibull wind)
- **FFS** reduces to 10 scenarios minimizing Wasserstein distance from the full distribution
- **Constrained probability correction** (α = 0.40) blends FFS-derived probabilities with WMO Da Nang climatology to prevent systematic underweighting of typhoon scenarios
- **Ordering property verification** (`WS ≤ RP ≤ EEV`) is enforced as a model consistency check

### Weather-Aware VRP with Heterogeneous Fleet

Routing is formulated as a VRP integrated with procurement via the extensive form, using a heterogeneous fleet adapted from Patel et al. (2024):

| Vehicle Type | Payload | Fixed Cost (VND) | Max Severity | Refrigerated |
|-------------|---------|-----------------|-------------|-------------|
| Mini van × 3 | 300 kg | 150,000 | Level 3 | No |
| Light truck × 2 | 1,000 kg | 400,000 | Level 4 | No |
| Refrigerated truck × 2 | 1,500 kg | 700,000 | Level 4 | Yes |
| Heavy truck × 1 | 3,000 kg | 1,200,000 | Level 3 | No |

Vehicle capacities are weather-adjusted per scenario. Mini vans are disabled at flood Level 4; heavy trucks cannot operate on flooded suburban roads. Refrigerated trucks reduce spoilage rates by 65% for seafood and meat (cold-chain Arrhenius benefit).

### Validation Results (VSS / EVPI / CVaR)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RP** (stochastic solution) | 3,691,403,494 VND | Optimal under uncertainty |
| **EEV** (deterministic evaluated stochastically) | 5,903,012,685 VND | What happens if you ignore uncertainty |
| **VSS = EEV − RP** | **2,211,609,192 VND (37.47%)** | Value of using stochastic approach |
| **WS** (wait-and-see, perfect info) | 3,255,626,126 VND | Lower bound |
| **EVPI = RP − WS** | **435,777,368 VND (11.81%)** | Value of perfect weather forecasting |
| Ordering property `WS ≤ RP ≤ EEV` | ✓ PASS | Model consistency verified |

**Risk analysis (CVaR-90%):**

| Metric | Value |
|--------|-------|
| Expected cost | 3,691,403,494 VND |
| VaR-90% | 5,283,359,949 VND |
| CVaR-90% | 10,159,802,874 VND |
| **CVaR premium** | **175.2%** |
| Risk-adjusted objective (λ=0.3) | 5,631,923,308 VND |

The CVaR premium of 175.2% quantifies the fat-tail risk exposure: the worst 10% of weather scenarios cost 2.75× more than the expected scenario. This motivates CVaR-aware procurement for operators with food security mandates — the cost of ignoring tail risk is not a 10% miss, it is an existential supply failure.

---

## 9. Integrated Results

### End-to-End Pipeline Performance Summary

```
Raw sales data (censored)
        │
        ▼  [WAPE 27.4%, ρ: −0.57 → 0.07]
True demand estimates
        │
        ▼  [WAPE 40%, full SKU coverage]
7-day demand forecasts + residuals
        │
        ▼  [−28.5% cost, −36.3% waste vs. naive]
Inventory order quantities per SKU
        │
        ▼  [VSS 37.5%, EVPI 11.8%, CVaR premium 175.2%]
Supplier selection + routing plan (weather-hedged)
```

### Business Impact (illustrative, based on benchmark scale)

| Metric | Naive Baseline | This System | Improvement |
|--------|---------------|-------------|-------------|
| Inventory total cost | $20.4M | **$14.6M** | **−28.5%** |
| Food waste (units) | Baseline | **−36.3%** | Significant ESG impact |
| Procurement cost under uncertainty | 5.9B VND (EEV) | **3.7B VND (RP)** | **−37.5%** |
| Worst-case cost (typhoon scenario) | Unplanned | **Explicitly hedged** | Tail risk managed |

---

## 10. Scientific Basis & References

This project builds on and extends the following peer-reviewed work:

| Paper | Contribution to This Project |
|-------|------------------------------|
| **FreshRetailNet-50K** (Ding et al., 2024) | Benchmark dataset + demand reconstruction methodology |
| **Bertsimas & Koduri (2022)** — *Data-Driven Newsvendor* | SAA framework for inventory planning under empirical distributions |
| **Patel et al. (2024)** — *Hyperconnected Logistics* | Stochastic MILP structure for perishable procurement + VRP |
| **Birge & Louveaux (2011)** — *Introduction to Stochastic Programming* | VSS, EVPI, and two-stage SP theory |
| **Rockafellar & Uryasev (2000)** — *CVaR Optimization* | CVaR-aware objective formulation |
| **Heitsch & Römisch (2007)** — *Scenario Tree Reduction* | Fast Forward Selection for scenario reduction |
| **Kleywegt et al. (2002)** — *Sample Average Approximation* | Convergence theory for SAA newsvendor |

---

## 11. Installation & Usage

### Requirements

```bash
# Core dependencies
pip install pandas numpy lightgbm scipy pulp scikit-learn
pip install datasets  # HuggingFace for FreshRetailNet-50K
pip install requests  # Open-Meteo API for weather data
```

### Repository Structure

```
├── config/
│   └── settings.py              # All hyperparameters and paths
├── src/
│   ├── data_pipeline/
│   │   ├── preprocessor.py      # Data loading + feature engineering
│   │   ├── catalog_enricher.py  # Risk-aware product economics
│   │   └── generator.py         # Supplier network generation
│   ├── demand/
│   │   ├── reconstruction.py    # Module 1: Hierarchical CDF reconstruction
│   │   └── forecasting.py       # Module 2: Hybrid LightGBM + SMA
│   ├── inventory/
│   │   ├── planner.py           # Module 3: SAA inventory planning
│   │   └── evaluator.py         # Policy benchmarking (Naive / Normal / SAA)
│   ├── optimization/
│   │   ├── extensive_form_optimizer.py  # Module 4: Two-stage MILP + VRP
│   │   ├── stochastic_procurement.py    # Stochastic procurement model
│   │   └── cvar_procurement.py          # CVaR-extended formulation
│   ├── weather/
│   │   ├── weather_data.py       # API fetch + distribution fitting
│   │   ├── scenario_generator.py # LHS + FFS scenario generation
│   │   └── manual_scenarios.py   # Pre-defined Da Nang scenarios
│   └── evaluation/
│       └── vss_evpi_calculator.py  # VSS / EVPI / CVaR metrics
├── scripts/
│   ├── run_stochastic_optimization.py   # Full procurement pipeline
│   └── sensitivity_analysis.py          # Robustness testing
└── results/
    ├── validation_report_fixed.txt
    ├── scenario_costs_fixed.csv
    └── cvar_metrics.csv
```

### Quick Start

```python
# Step 1: Run demand reconstruction + forecasting (Repo 1)
from src.data_pipeline.preprocessor import FrnPreprocessor
from src.demand.reconstruction import DemandReconstructor
from src.demand.forecasting import DemandForecaster

preprocessor = FrnPreprocessor()
df = preprocessor.run()

reconstructor = DemandReconstructor()
df_recon = reconstructor.run()

forecaster = DemandForecaster()
metrics = forecaster.run()

# Step 2: Inventory planning with SAA
from src.inventory.evaluator import PolicyEvaluator
evaluator = PolicyEvaluator()
evaluator.run()  # Outputs Naive / Normal / SAA scorecard

# Step 3: Run stochastic procurement optimization (Repo 2)
# python scripts/run_stochastic_optimization.py
# Select season: 2 (Monsoon)
```

### Configuration

Key parameters in `config/settings.py`:

```python
# Demand reconstruction
KS_THR_SOFT = 0.65     # Shrinkage stability threshold
CDF_MIN_CLIP = 0.15    # Floor for CDF correction denominator

# Forecasting
SEQ_LEN = 28           # Lookback window (days)
HORIZON = 7            # Forecast horizon (days)
THRES_MEAN_SALES = 1.5 # ML branch eligibility threshold

# Inventory (SAA)
SAA_POOLING_LEVEL = 'risk_group'  # Pool residuals by risk category
MIN_RESIDUAL_SAMPLES = 30         # Minimum samples for local SAA

# Procurement optimization
SERVICE_LEVEL = 0.95
MAX_SOLVE_TIME_S = 900
GAP_REL = 0.02
```

---

## Skills & Technologies

`Python` · `Machine Learning` · `LightGBM` · `Demand Forecasting` · `Time Series` · `Operations Research` · `Mixed-Integer Linear Programming (MILP)` · `Stochastic Programming` · `Sample Average Approximation (SAA)` · `Newsvendor Problem` · `Vehicle Routing Problem (VRP)` · `Supply Chain Optimization` · `Inventory Management` · `PuLP` · `Pandas` · `NumPy` · `SciPy` · `HuggingFace Datasets` · `Conditional Value-at-Risk (CVaR)` · `Scenario Generation` · `Latin Hypercube Sampling` · `Perishable Goods Logistics` · `Food Waste Reduction` · `Weather Uncertainty Modeling`

---

*This project is part of an ongoing research initiative in supply chain intelligence for perishable food retail. Results are benchmarked on public datasets and validated against published literature.*