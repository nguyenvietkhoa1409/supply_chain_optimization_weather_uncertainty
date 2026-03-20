"""
weather_pattern_visualization.py
───────────────────────────
Scientific weather analysis & scenario generation figures.
Da Nang Fresh Food Supply Chain — Two-Stage Stochastic MILP paper.

Figures:
  W1 — Monthly climatology: rainfall + typhoon probability
  W2 — Distribution fit: actual ERA5 histogram + Gamma PDF + Q-Q
  W3 — LHS sample coverage: actual ERA5 scatter colored by severity
  W4 — Scenario reduction: LHS severity dist vs final K=10
  W5 — Final K=10 scenario set: 3-panel operational summary

Data priority:
  1. data/weather_cache/danang_2014-01-01_2023-12-31.csv  ← actual ERA5
  2. Parametric fallback from distribution parameters      ← prints warning

Usage:
    weather_pattern_visualization.py

Output: ./figures/weather/  (5 PNG, 180 DPI)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gamma as gamma_dist, norm as norm_dist

OUT = "./figures/weather"
os.makedirs(OUT, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 10, "axes.titleweight": "normal", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "0.8",
    "legend.fontsize": 9, "figure.dpi": 180, "savefig.dpi": 180,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})
BLACK = "#000000"; GRAY = "#888888"; LGRAY = "#cccccc"

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD ACTUAL ERA5 CSV
# ════════════════════════════════════════════════════════════════════════════
MONSOON_MONTHS = {9, 10, 11, 12, 1}
CSV_CANDIDATES = [
    "data/weather_cache/danang_2014-01-01_2023-12-31.csv",
    "../data/weather_cache/danang_2014-01-01_2023-12-31.csv",
    "danang_2014-01-01_2023-12-31.csv",
]

era5_df = None
for _p in CSV_CANDIDATES:
    if os.path.exists(_p):
        era5_df = pd.read_csv(_p, parse_dates=["date"])
        era5_df["rainfall_mm"] = era5_df["rainfall_mm"].clip(lower=0).fillna(0)
        if "temp_mean_c" not in era5_df.columns:
            era5_df["temp_mean_c"] = (era5_df["temp_max_c"] + era5_df["temp_min_c"]) / 2.0
        if "month" not in era5_df.columns:
            era5_df["month"] = era5_df["date"].dt.month
        if "season" not in era5_df.columns:
            era5_df["season"] = era5_df["month"].apply(
                lambda m: "monsoon" if m in MONSOON_MONTHS else "dry")
        print(f"✓ ERA5 CSV loaded: {_p}  ({len(era5_df)} records)")
        break

if era5_df is None:
    print("⚠  ERA5 CSV not found — W2 and W3 will use parametric simulation.")
    print("   Expected: data/weather_cache/danang_2014-01-01_2023-12-31.csv")

DATA_SOURCE = "ERA5 reanalysis (2014–2023)" if era5_df is not None \
              else "Parametric simulation (CSV unavailable)"

# ════════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTION PARAMETERS  (from danang_distribution_parameters.json)
# ════════════════════════════════════════════════════════════════════════════
DIST = {
    "dry": {
        "n_days": 2122, "n_rainy": 1274,
        "rain_shape": 0.6683064587951874, "rain_scale": 7.112920446121353,
        "zero_prob": 0.3996229971724788,  "rain_ks_ratio": 2.879,
        "temp_loc": 27.525117813383602,   "temp_scale": 3.269859660031158,
        "wind_c": 3.60969169786056,       "wind_scale": 18.924707105021863,
        "rain_mean": 2.859707822808671,   "rain_p95": 12.495, "rain_p99": 34.032,
        "temp_mean": 27.525,
    },
    "monsoon": {
        "n_days": 1530, "n_rainy": 1353,
        "rain_shape": 0.6686673116720471, "rain_scale": 16.81428588547276,
        "zero_prob": 0.11568627450980393, "rain_ks_ratio": 2.608,
        "temp_loc": 24.739640522875817,   "temp_scale": 2.6481542841394763,
        "wind_c": 2.5118639069449746,     "wind_scale": 20.787481286778302,
        "rain_mean": 9.944705882352942,   "rain_p95": 41.22, "rain_p99": 82.081,
        "temp_mean": 24.740,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 3. HELPERS — actual ERA5 or parametric fallback
# ════════════════════════════════════════════════════════════════════════════
def get_rain_array(season: str, d: dict, seed: int = 0) -> np.ndarray:
    """Returns full daily rainfall array (including zero-rain days)."""
    if era5_df is not None:
        return era5_df.loc[era5_df["season"] == season, "rainfall_mm"].values
    rng = np.random.default_rng(seed)
    u   = rng.uniform(size=d["n_days"])
    zdp = d["zero_prob"]
    rain = np.where(
        u <= zdp, 0.0,
        gamma_dist.ppf(np.clip((u - zdp) / max(1-zdp, 1e-8), 1e-6, 1-1e-6),
                       a=d["rain_shape"], scale=d["rain_scale"])
    )
    return np.clip(rain, 0, None)

def get_temp_array(season: str, d: dict, seed: int = 0) -> np.ndarray:
    """Returns daily temperature array."""
    if era5_df is not None:
        return era5_df.loc[era5_df["season"] == season, "temp_mean_c"].values
    rng = np.random.default_rng(seed + 99)
    return norm_dist.rvs(loc=d["temp_loc"], scale=d["temp_scale"],
                         size=d["n_days"], random_state=seed)

def classify_severity(r: float) -> int:
    if r > 100: return 5
    if r > 50:  return 4
    if r > 20:  return 3
    if r > 5:   return 2
    return 1

# ════════════════════════════════════════════════════════════════════════════
# 4. STATIC DATA
# ════════════════════════════════════════════════════════════════════════════
MONTHS    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
RAINFALL  = [85, 25, 20, 30, 60, 85, 90, 115, 310, 650, 430, 215]
TYPHOON_P = [0.03,0.01,0.01,0.01,0.03,0.03,0.05,0.05,0.15,0.28,0.18,0.05]
SEASON_C  = ["M","D","D","D","D","D","D","D","M","M","M","M"]

# K=10 final scenarios
FINAL_SC = [
    # name,          lv, prob,  rain,  temp, speed, cap,  spoil
    ("Normal (1)",   1, 0.183, 2.7,  25.3, 1.08, 0.95, 1.00),
    ("Normal (2)",   1, 0.174, 3.1,  25.8, 1.08, 0.95, 1.00),
    ("Lt. Rain (1)", 2, 0.108, 9.8,  24.9, 1.15, 0.90, 1.05),
    ("Lt. Rain (2)", 2, 0.104, 11.2, 24.6, 1.15, 0.90, 1.05),
    ("Lt. Rain (3)", 2, 0.062, 14.5, 24.2, 1.15, 0.90, 1.05),
    ("Lt. Rain (4)", 2, 0.061, 17.8, 24.0, 1.15, 0.90, 1.05),
    ("Mod. Rain (1)",3, 0.070, 28.4, 23.8, 1.25, 0.80, 1.15),
    ("Mod. Rain (2)",3, 0.047, 42.6, 23.5, 1.25, 0.80, 1.15),
    ("Heavy Rain",   4, 0.096, 68.3, 23.1, 1.55, 0.60, 1.30),
    ("Typhoon",      5, 0.095,142.5, 22.0, 2.20, 0.10, 2.00),
]

LV_FILL  = {1:LGRAY, 2:GRAY, 3:"0.4", 4:BLACK, 5:BLACK}
LV_HATCH = {1:"",    2:"",   3:"///", 4:"",    5:"xxx"}
LV_LABEL = {1:"L1 Normal", 2:"L2 Light Rain", 3:"L3 Moderate",
            4:"L4 Heavy Rain", 5:"L5 Typhoon"}

np.random.seed(2024)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE W1 — Monthly climatology
# ════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(8.5, 4))
x = np.arange(12)
ax1.bar(x, RAINFALL, color=[BLACK if s=="M" else LGRAY for s in SEASON_C],
        edgecolor=BLACK, linewidth=0.7, width=0.6, zorder=2)
ax1.axvspan(7.5, 11.5, alpha=0.05, color=BLACK, zorder=0)
ax1.text(9.5, 705, "Monsoon", ha="center", fontsize=8.5, color=GRAY, style="italic")
ax1.text(3.75,705, "Dry",     ha="center", fontsize=8.5, color=GRAY, style="italic")
ax1.set_ylabel("Monthly rainfall (mm)")
ax1.set_xticks(x); ax1.set_xticklabels(MONTHS, fontsize=9.5)
ax1.set_ylim(0, 760)
ax2 = ax1.twinx()
ax2.plot(x, [p*100 for p in TYPHOON_P],
         color=GRAY, marker="o", ms=4.5, lw=1.5, ls="--", zorder=5)
ax2.set_ylabel("Monthly typhoon probability (%)", color=GRAY)
ax2.tick_params(axis="y", labelcolor=GRAY)
ax2.set_ylim(0, 38)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_color(GRAY)
ax1.legend(handles=[
    mpatches.Patch(fc=BLACK, ec=BLACK, label="Monsoon (Sep–Jan)"),
    mpatches.Patch(fc=LGRAY, ec=BLACK, label="Dry (Feb–Aug)"),
    Line2D([0],[0], color=GRAY, marker="o", ms=4.5, lw=1.5, ls="--",
           label="Typhoon probability"),
], loc="upper left", fontsize=9)
ax1.set_title("Figure W1. Da Nang monthly rainfall and typhoon probability (WMO 1981–2020)")
plt.tight_layout()
plt.savefig(f"{OUT}/figW1_monthly_climatology.png"); plt.close()
print("✓ Fig W1")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE W2 — Distribution fit: actual ERA5 histogram + Gamma PDF + Q-Q
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

panel_labels = [("A","B"), ("C","D")]

for row, season in enumerate(["dry", "monsoon"]):
    d      = DIST[season]
    rain_all = get_rain_array(season, d, seed=row)
    rainy    = rain_all[rain_all > 0.1]
    slabel   = "Dry season" if season == "dry" else "Monsoon season"
    xlim     = 65 if season == "dry" else 145
    pl, pr   = panel_labels[row]

    # ── Histogram + PDF ─────────────────────────────────────────────────
    ax = axes[row, 0]
    bins = np.linspace(0.1, xlim, 32)
    ax.hist(rainy[rainy <= xlim], bins=bins, density=True,
            color=LGRAY, edgecolor=BLACK, linewidth=0.5,
            label=DATA_SOURCE)
    x_pdf = np.linspace(0.1, xlim, 500)
    ax.plot(x_pdf, gamma_dist.pdf(x_pdf, a=d["rain_shape"], scale=d["rain_scale"]),
            color=BLACK, lw=1.8, label="Fitted Gamma PDF")
    for pct, ls_ in [(95,"--"),(99,":")]:
        val = gamma_dist.ppf(pct/100, a=d["rain_shape"], scale=d["rain_scale"])
        if val <= xlim:
            ax.axvline(val, color=GRAY, ls=ls_, lw=1.0)
            ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.04
            ax.text(val+0.5, ymax*0.9, f"p{pct}", fontsize=8, color=GRAY)
    ax.set_xlabel("Daily rainfall (mm)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, xlim)
    ax.legend(fontsize=9)
    ax.set_title(f"({pl}) {slabel} — rainfall distribution")
    ax.text(0.97, 0.97,
            f"Gamma(α = {d['rain_shape']:.4f},  β = {d['rain_scale']:.3f} mm)\n"
            f"Zero-day prob = {d['zero_prob']:.1%}\n"
            f"μ = {d['rain_mean']:.2f} mm     p₉₅ = {d['rain_p95']:.1f} mm\n"
            f"KS ratio = {d['rain_ks_ratio']:.3f}     n = {len(rain_all)} days",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.8))

    # ── Q-Q plot ─────────────────────────────────────────────────────────
    ax = axes[row, 1]
    rng2 = np.random.default_rng(42+row)
    n_qq = min(len(rainy), 300)
    samp = np.sort(rng2.choice(rainy, size=n_qq, replace=False))
    probs_qq = (np.arange(1, n_qq+1) - 0.5) / n_qq
    theo     = gamma_dist.ppf(probs_qq, a=d["rain_shape"], scale=d["rain_scale"])
    lim      = max(theo.max(), samp.max()) * 1.06
    ax.scatter(theo, samp, color=BLACK, s=14,
               edgecolors=BLACK, linewidths=0.3, alpha=0.6,
               label="Observed quantiles")
    ax.plot([0,lim],[0,lim], color=GRAY, lw=1.2, ls="--", label="Perfect fit")
    ax.set_xlabel("Theoretical quantiles (mm)")
    ax.set_ylabel("Empirical quantiles (mm)")
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"({pr}) {slabel} — Q-Q plot vs fitted Gamma")
    ss_res = float(np.sum((samp-theo)**2))
    ss_tot = float(np.sum((samp-samp.mean())**2))
    ax.text(0.97, 0.06, f"R² = {1-ss_res/ss_tot:.4f}",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.8))

fig.suptitle(
    f"Figure W2. Gamma distribution fit to daily rainfall — {DATA_SOURCE}",
    y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/figW2_distribution_fit.png"); plt.close()
print("✓ Fig W2")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE W3 — LHS sample coverage using ACTUAL ERA5 monsoon data
#   Plots real observed [rainfall × temperature] points colored by severity
#   to show what the true distribution looks like before LHS sampling
# ════════════════════════════════════════════════════════════════════════════
d = DIST["monsoon"]
rain_m = get_rain_array("monsoon", d, seed=0)
temp_m = get_temp_array("monsoon", d, seed=0)
N_actual = len(rain_m)

# Align arrays (CSV may have different lengths per column)
N_use = min(len(rain_m), len(temp_m))
rain_m = rain_m[:N_use]
temp_m = temp_m[:N_use]

sev_m = np.array([classify_severity(r) for r in rain_m])

MARKERS = {1:"o", 2:"s", 3:"^", 4:"D", 5:"*"}
SIZES   = {1:18,  2:18,  3:22,  4:30,  5:65}

# Also generate LHS samples to overlay (from fitted distributions)
N_LHS = 600
u_r = (np.arange(N_LHS) + 0.5) / N_LHS; np.random.shuffle(u_r)
zdp = d["zero_prob"]
rain_lhs = np.where(
    u_r <= zdp, 0.0,
    gamma_dist.ppf(np.clip((u_r-zdp)/max(1-zdp,1e-8), 1e-6, 1-1e-6),
                   a=d["rain_shape"], scale=d["rain_scale"])
)
rain_lhs = np.clip(rain_lhs, 0, None)
u_t = (np.arange(N_LHS) + 0.5) / N_LHS; np.random.shuffle(u_t)
temp_lhs = norm_dist.ppf(u_t, loc=d["temp_loc"], scale=d["temp_scale"])
sev_lhs  = np.array([classify_severity(r) for r in rain_lhs])

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

# Left: actual ERA5 data
for lv in [1,2,3,4,5]:
    mask = sev_m == lv
    ax_l.scatter(rain_m[mask], temp_m[mask],
                 c=LV_FILL[lv], marker=MARKERS[lv], s=SIZES[lv],
                 label=f"{LV_LABEL[lv]} (n={mask.sum()})",
                 edgecolors=BLACK, linewidths=0.3, alpha=0.55, zorder=3)
for thresh in [5,20,50,100]:
    ax_l.axvline(thresh, color=GRAY, ls=":", lw=0.8)
    ax_l.text(thresh+0.4, d["temp_loc"]+2.55*d["temp_scale"],
              f"{thresh}", fontsize=7.5, color=GRAY, rotation=90, va="top")
ax_l.set_xlabel("Daily rainfall (mm)")
ax_l.set_ylabel("Mean daily temperature (°C)")
ax_l.set_xlim(-2, 220)
ax_l.set_ylim(d["temp_loc"]-3.3*d["temp_scale"], d["temp_loc"]+3.6*d["temp_scale"])
ax_l.legend(loc="upper right", fontsize=8.5, markerscale=1.1)
ax_l.set_title(f"(A) ERA5 observed data — monsoon season\n"
               f"(n = {N_use} days, {DATA_SOURCE})")

# Right: LHS samples from fitted distributions
for lv in [1,2,3,4,5]:
    mask = sev_lhs == lv
    ax_r.scatter(rain_lhs[mask], temp_lhs[mask],
                 c=LV_FILL[lv], marker=MARKERS[lv], s=SIZES[lv],
                 label=f"{LV_LABEL[lv]} (n={mask.sum()})",
                 edgecolors=BLACK, linewidths=0.3, alpha=0.72, zorder=3)
for thresh in [5,20,50,100]:
    ax_r.axvline(thresh, color=GRAY, ls=":", lw=0.8)
    ax_r.text(thresh+0.4, d["temp_loc"]+2.55*d["temp_scale"],
              f"{thresh}", fontsize=7.5, color=GRAY, rotation=90, va="top")
ax_r.set_xlabel("Daily rainfall (mm)")
ax_r.set_ylabel("Mean daily temperature (°C)")
ax_r.set_xlim(-2, 220)
ax_r.set_ylim(d["temp_loc"]-3.3*d["temp_scale"], d["temp_loc"]+3.6*d["temp_scale"])
ax_r.legend(loc="upper right", fontsize=8.5, markerscale=1.1)
ax_r.set_title(f"(B) LHS candidates from fitted distributions\n"
               f"(n = {N_LHS} samples, seed = 2024)")

fig.suptitle(
    "Figure W3. ERA5 observed data vs LHS candidates — monsoon season "
    "(rainfall × temperature, colored by severity level)",
    y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/figW3_lhs_coverage.png"); plt.close()
print("✓ Fig W3")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE W4 — Scenario reduction: LHS distribution vs final K=10
# ════════════════════════════════════════════════════════════════════════════
lhs_props    = {lv: (sev_lhs==lv).sum()/N_LHS*100 for lv in [1,2,3,4,5]}

final_lv_p   = {}
for _,lv,p,*_ in FINAL_SC:
    final_lv_p[lv] = final_lv_p.get(lv,0.0) + p
final_props  = {lv: final_lv_p.get(lv,0.0)*100 for lv in [1,2,3,4,5]}
hist_targets = {1:30.0, 2:25.0, 3:20.0, 4:15.0, 5:10.0}

x = np.arange(5)
lv_tick = ["L1\nNormal","L2\nLight\nRain","L3\nMod.","L4\nHeavy","L5\nTyphoon"]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))

for xi, lv in enumerate([1,2,3,4,5]):
    ax_l.bar(xi, lhs_props[lv], color=LV_FILL[lv], hatch=LV_HATCH[lv],
             edgecolor=BLACK, linewidth=0.7, width=0.55)
    ax_l.text(xi, lhs_props[lv]+0.5, f"{lhs_props[lv]:.1f}%",
              ha="center", va="bottom", fontsize=9)
ax_l.set_xticks(x); ax_l.set_xticklabels(lv_tick, fontsize=9)
ax_l.set_ylabel("Proportion (%)")
ax_l.set_ylim(0, 80)
ax_l.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax_l.set_title(f"(A) LHS sample distribution (n = {N_LHS} candidates)")

for xi, lv in enumerate([1,2,3,4,5]):
    ax_r.bar(xi, final_props[lv], color=LV_FILL[lv], hatch=LV_HATCH[lv],
             edgecolor=BLACK, linewidth=0.7, width=0.55)
    ax_r.text(xi, final_props[lv]+0.5, f"{final_props[lv]:.1f}%",
              ha="center", va="bottom", fontsize=9)
ax_r.plot(x, [hist_targets[l] for l in [1,2,3,4,5]],
          color=GRAY, marker="o", ms=5, lw=1.5, ls="--",
          label="Historical target\n(WMO Da Nang climatology)")
ax_r.set_xticks(x); ax_r.set_xticklabels(lv_tick, fontsize=9)
ax_r.set_ylabel("Scenario probability (%)")
ax_r.set_ylim(0, 45)
ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax_r.legend(fontsize=9, loc="upper right")
ax_r.set_title("(B) Final K = 10 probabilities (FFS + correction α = 0.40)")

fig.suptitle(
    "Figure W4. Severity level distribution: LHS candidates (A) vs final scenario set (B)",
    y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/figW4_scenario_reduction.png"); plt.close()
print("✓ Fig W4")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE W5 — Final K=10 scenario set: 3-panel operational summary
# ════════════════════════════════════════════════════════════════════════════
sc_names = [s[0] for s in FINAL_SC]
sc_lv    = [s[1] for s in FINAL_SC]
sc_prob  = [s[2] for s in FINAL_SC]
sc_rain  = [s[3] for s in FINAL_SC]
sc_speed = [s[5] for s in FINAL_SC]
sc_cap   = [s[6] for s in FINAL_SC]
sc_spoil = [s[7] for s in FINAL_SC]
y = np.arange(len(sc_names))

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 5.2))

# (A) Rainfall
for yi,(rain,lv) in enumerate(zip(sc_rain, sc_lv)):
    ax_a.barh(yi, rain, color=LV_FILL[lv], hatch=LV_HATCH[lv],
              edgecolor=BLACK, linewidth=0.6, height=0.6)
    ax_a.text(rain+1.2, yi, f"{rain:.1f}", va="center", fontsize=8.5)
for t in [5,20,50,100]:
    ax_a.axvline(t, color=GRAY, ls=":", lw=0.8)
ax_a.set_yticks(y); ax_a.set_yticklabels(sc_names, fontsize=9)
ax_a.set_xlabel("Rainfall (mm/day)")
ax_a.set_title("(A) Representative rainfall")

# (B) Operational factors
sp_eff = [round(1.0/sf, 3) for sf in sc_speed]
w = 0.22
ax_b.barh(y+w,   sc_cap,   w, color=BLACK, edgecolor=BLACK, linewidth=0.5,
          label="Capacity factor")
ax_b.barh(y,     sp_eff,   w, color=GRAY,  edgecolor=BLACK, linewidth=0.5,
          label="Speed factor")
ax_b.barh(y-w,   sc_spoil, w, color=LGRAY, hatch="///",
          edgecolor=BLACK, linewidth=0.5, label="Spoilage multiplier")
ax_b.axvline(1.0, color=BLACK, ls="--", lw=0.8, alpha=0.4)
ax_b.set_yticks(y); ax_b.set_yticklabels(sc_names, fontsize=9)
ax_b.set_xlabel("Factor value")
ax_b.set_xlim(0, 2.35)
ax_b.legend(fontsize=8.5, loc="lower right")
ax_b.set_title("(B) Operational impact parameters")

# (C) Probability
for yi,(prob,lv) in enumerate(zip(sc_prob, sc_lv)):
    ax_c.barh(yi, prob*100, color=LV_FILL[lv], hatch=LV_HATCH[lv],
              edgecolor=BLACK, linewidth=0.6, height=0.6)
    ax_c.text(prob*100+0.1, yi, f"{prob:.1%}", va="center", fontsize=8.5)
ax_c.set_yticks(y); ax_c.set_yticklabels(sc_names, fontsize=9)
ax_c.set_xlabel("Probability (%)")
ax_c.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax_c.set_title("(C) Probability weight")

fig.legend(
    handles=[mpatches.Patch(fc=LV_FILL[l], hatch=LV_HATCH[l], ec=BLACK,
             lw=0.6, label=LV_LABEL[l]) for l in [1,2,3,4,5]],
    loc="lower center", ncol=5, fontsize=9,
    bbox_to_anchor=(0.5,-0.05), frameon=True, edgecolor="0.8")
fig.suptitle(
    f"Figure W5. Final K = 10 weather scenarios — monsoon season  "
    f"({DATA_SOURCE}, LHS n = {N_LHS}, FFS + correction α = 0.40)",
    y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/figW5_final_scenarios.png"); plt.close()
print("✓ Fig W5")

print(f"\nAll figures saved → {OUT}/")
print(f"Data source used: {DATA_SOURCE}")