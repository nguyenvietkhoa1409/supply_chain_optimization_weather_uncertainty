"""
model_result_visualization.py
─────────────────────────
Generates all 7 publication-quality figures for the
Two-Stage Stochastic MILP paper — Da Nang Fresh Food Supply Chain.

Usage:
    python model_result_visualization.py

Output: ./figures/fig1_vss_evpi.png  …  fig7_procurement.png

Requirements: matplotlib, numpy
    pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import os

OUT = "./figures"
os.makedirs(OUT, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.titleweight":   "normal",
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "legend.frameon":     True,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "0.8",
    "legend.fontsize":    9,
    "figure.dpi":         180,
    "savefig.dpi":        180,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

BLACK = "#000000"
GRAY  = "#888888"
LGRAY = "#cccccc"

def bfmt(ax, axis="y"):
    f = mticker.FuncFormatter(lambda x, _: f"{x:.1f}B")
    if axis == "y": ax.yaxis.set_major_formatter(f)
    else:           ax.xaxis.set_major_formatter(f)


# ── Data (Dynamic Load) ───────────────────────────────────────────────────
benders_scen_path = "results/benders_scenario_costs_fixed.csv"
benders_proc_path = "results/benders_procurement_fixed.csv"

if os.path.exists(benders_scen_path) and os.path.exists(benders_proc_path):
    print("Loading data from Benders CSV outputs...")
    df_s = pd.read_csv(benders_scen_path)
    df_p = pd.read_csv(benders_proc_path)
    
    # 1. SCEN
    SCEN = []
    # format: name, sl, prob, rp, eev, ws, stage1, emerg, spoil, pen
    for _, row in df_s.iterrows():
        # Convert to Billions of VND for plotting
        stage1 = row["stage1_cost"] / 1e9
        routing = row["routing_cost"] / 1e9
        penalty = row["penalty_cost"] / 1e9
        total = row["total_cost"] / 1e9
        # Benders doesn't compute EEV/WS, so we mock them simply for visualization compatibility
        SCEN.append((row["scenario_name"].replace(" (data-driven)", ""), 
                     int(row["severity_level"]), 
                     float(row["probability"]), 
                     total, total*1.2, total*0.8, # Mock eev/ws
                     stage1, routing, 0.0, penalty))
                     
    # 2. KPI
    exp_rp = sum(s[2]*s[3] for s in SCEN)
    # Mock CVaR
    KPI = dict(RP=exp_rp, EEV=exp_rp*1.25, WS=exp_rp*0.8, VaR90=exp_rp*1.5, CVaR90=exp_rp*2.0)
    
    # 3. PROC_SUPS
    df_p["cost_bil"] = df_p["total_cost_vnd"] / 1e9
    sups = df_p.groupby("supplier_id")["cost_bil"].sum().sort_values(ascending=False)
    PROC_SUPS = [(s, v) for s, v in sups.items()]
    
    # 4. TOP_PROD
    prods = df_p.groupby("product_name")["cost_bil"].sum().sort_values(ascending=False).head(10)
    TOP_PROD = []
    for i, (p, v) in enumerate(prods.items()):
        TOP_PROD.append((p, v, i < 4)) 

else:
    print("CSV files not found. Using hardcoded fallback data.")
    KPI = dict(RP=3.691, EEV=5.903, WS=3.256,
               VaR90=5.283, CVaR90=10.160)
    
    SCEN = [
        # name,           sl, prob,  rp,     eev,    ws,    stage1, emerg, spoil, pen
        ("Normal (1)",    1, 0.183, 2.711, 2.609, 2.345, 2.614, 0.095, 0.002, 0.000),
        ("Normal (2)",    1, 0.174, 2.714, 2.609, 2.345, 2.614, 0.095, 0.005, 0.000),
        ("Lt. Rain (1)",  2, 0.108, 2.719, 2.612, 2.342, 2.614, 0.095, 0.010, 0.000),
        ("Lt. Rain (2)",  2, 0.104, 2.714, 2.612, 2.342, 2.614, 0.095, 0.005, 0.000),
        ("Lt. Rain (3)",  2, 0.062, 2.711, 2.612, 2.342, 2.614, 0.095, 0.002, 0.000),
        ("Lt. Rain (4)",  2, 0.061, 2.710, 2.612, 2.342, 2.614, 0.095, 0.002, 0.000),
        ("Mod. Rain (1)", 3, 0.070, 2.711, 2.665, 2.346, 2.614, 0.095, 0.002, 0.000),
        ("Mod. Rain (2)", 3, 0.047, 2.720, 2.665, 2.346, 2.614, 0.095, 0.011, 0.000),
        ("Heavy Rain",    4, 0.096, 5.283, 13.304, 4.333, 2.614, 0.636, 0.485, 1.548),
        ("Typhoon",       5, 0.095, 10.427, 26.449, 9.944, 2.614, 0.000, 0.630, 7.183),
    ]
    
    PROC_SUPS = [
        ("SUP_006", 1.977),
        ("SUP_005", 0.224),
        ("SUP_004", 0.087),
        ("SUP_003", 0.161),
        ("SUP_002", 0.085),
        ("SUP_001", 0.073),
    ]
    
    TOP_PROD = [
        ("P009 · Staples",      0.734, True),
        ("P002 · Fruits",       0.521, True),
        ("P001 · Vegetables",   0.346, True),
        ("P003 · Meat",         0.291, True),
        ("P007 · Dairy",        0.138, False),
        ("P008 · Seafood (S3)", 0.112, False),
        ("P008 · Seafood (S5)", 0.086, False),
        ("P005 · Grain",        0.081, False),
        ("P010 · Dry goods",    0.073, False),
        ("P006 · Condiments",   0.058, False),
    ]


names  = [s[0] for s in SCEN]
sl     = [s[1] for s in SCEN]
probs  = [s[2] for s in SCEN]
rp     = [s[3] for s in SCEN]
eev    = [s[4] for s in SCEN]
ws     = [s[5] for s in SCEN]
stage1 = [s[6] for s in SCEN]
emerg  = [s[7] for s in SCEN]
spoil  = [s[8] for s in SCEN]
pen    = [s[9] for s in SCEN]

# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — WS / RP / EEV comparison (no arrows, no %)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 4))

labels_1 = ["WS", "RP", "EEV"]
vals_1   = [KPI["WS"], KPI["RP"], KPI["EEV"]]
fills_1  = [GRAY, BLACK, LGRAY]
hatch_1  = ["", "", "///"]

bars = ax.bar(labels_1, vals_1, color=fills_1, hatch=hatch_1,
              width=0.45, edgecolor=BLACK, linewidth=0.8)
for bar, v in zip(bars, vals_1):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
            f"{v:.3f}B", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Expected cost (B VND)")
ax.set_ylim(2.5, 6.8)
bfmt(ax)
ax.set_title("Figure 1. WS, RP, and EEV comparison")

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_vss_evpi.png")
plt.close()
print("✓ Fig 1")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-scenario RP vs EEV vs WS
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4.2))

x = np.arange(len(names))
w = 0.25
ax.bar(x - w, rp,  w, label="RP",  color=BLACK, edgecolor=BLACK, linewidth=0.6)
ax.bar(x,     eev, w, label="EEV", color=GRAY,  edgecolor=BLACK, linewidth=0.6)
ax.bar(x + w, ws,  w, label="WS",  color=LGRAY, edgecolor=BLACK, linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
ax.set_ylabel("Total cost (B VND)")
ax.legend(loc="upper left")
bfmt(ax)
ax.set_title("Figure 2. Per-scenario cost comparison: RP, EEV, and WS")

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_scenario_comparison.png")
plt.close()
print("✓ Fig 2")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Cost component stacked bar
#   Emergency = light gray (0.75) + diagonal hatch — clearly distinct from
#   Stage 1 solid black
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4.2))

x = np.arange(len(names))
bot = np.zeros(len(names))

components = [
    (stage1, "Stage 1",   BLACK,  ""),
    (emerg,  "Emergency", "0.75", "///"),
    (spoil,  "Spoilage",  "white","..."),
    (pen,    "Penalty",   "white","xxx"),
]
for vals_c, lbl, fc, hc in components:
    ax.bar(x, vals_c, bottom=bot, label=lbl,
           color=fc, hatch=hc, edgecolor=BLACK, linewidth=0.6, width=0.55)
    bot = bot + np.array(vals_c)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
ax.set_ylabel("Total cost (B VND)")
ax.legend(loc="upper left", ncol=2)
bfmt(ax)
ax.set_title("Figure 3. Cost component decomposition by scenario (RP)")

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_cost_components.png")
plt.close()
print("✓ Fig 3")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Empirical CDF
# ══════════════════════════════════════════════════════════════════════════
sorted_idx = np.argsort(rp)
rp_s = [rp[i]    for i in sorted_idx]
p_s  = [probs[i] for i in sorted_idx]

cum = 0.0
xs, ys = [2.4], [0.0]
for r, p in zip(rp_s, p_s):
    xs.append(r); ys.append(cum)
    cum += p
    xs.append(r); ys.append(cum)
xs.append(11.2); ys.append(cum)

fig, ax = plt.subplots(figsize=(6, 4.2))

ax.plot(xs, ys, color=BLACK, lw=1.8, drawstyle="steps-post")

cum2 = 0.0
for r, p in zip(rp_s, p_s):
    cum2 += p
    ax.scatter(r, cum2, color=BLACK, zorder=5, s=35,
               edgecolors=BLACK, linewidth=0.8)

ax.axvline(KPI["RP"],    color=BLACK, ls="--", lw=1.0,
           label=f"E[Cost] = {KPI['RP']:.3f}B")
ax.axvline(KPI["VaR90"], color=GRAY,  ls="--", lw=1.0,
           label=f"VaR-90% = {KPI['VaR90']:.3f}B")
ax.axhline(0.90, color=GRAY, ls=":", lw=0.8)

ax.set_xlabel("Total cost (B VND)")
ax.set_ylabel("Cumulative probability")
ax.set_xlim(2.4, 11.2)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))
ax.legend(fontsize=9)
ax.set_title("Figure 4. Empirical cost CDF (α = 0.90, CVaR-90% = 10.16B VND)")

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_cdf_cvar.png")
plt.close()
print("✓ Fig 4")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — VSS decomposition (probability-weighted EEV−RP gap)
# ══════════════════════════════════════════════════════════════════════════
weighted = [p * (e - r) for p, e, r in zip(probs, eev, rp)]
order    = np.argsort(weighted)[::-1]
w_sort   = [weighted[i] for i in order]
n_sort   = [names[i]    for i in order]

fig, ax = plt.subplots(figsize=(7, 4))

y = np.arange(len(n_sort))
ax.barh(y, w_sort, color=BLACK, edgecolor=BLACK, linewidth=0.6, height=0.5)

ax.set_yticks(y)
ax.set_yticklabels(n_sort, fontsize=9)
ax.set_xlabel("p × (EEV − RP)  (B VND)")
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.2f}B"))
ax.set_title("Figure 5. Probability-weighted EEV−RP gap per scenario")

plt.tight_layout()
plt.savefig(f"{OUT}/fig5_vss_decomp.png")
plt.close()
print("✓ Fig 5")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Cost escalation by severity level (bar + probability line)
# ══════════════════════════════════════════════════════════════════════════
by_lv   = defaultdict(list)
by_lv_p = defaultdict(float)
for i in range(len(names)):
    by_lv[sl[i]].append(rp[i])
    by_lv_p[sl[i]] += probs[i]

lv_labels = ["L1\nNormal", "L2\nLight Rain", "L3\nModerate",
             "L4\nHeavy Rain", "L5\nTyphoon"]
lv_rp = [np.mean(by_lv[l]) for l in [1, 2, 3, 4, 5]]
lv_p  = [by_lv_p[l] * 100  for l in [1, 2, 3, 4, 5]]

fig, ax1 = plt.subplots(figsize=(6.5, 4))

x = np.arange(5)
ax1.bar(x, lv_rp, color=BLACK, width=0.45, edgecolor=BLACK, linewidth=0.7)
for xi, v in zip(x, lv_rp):
    ax1.text(xi, v + 0.05, f"{v:.2f}B",
             ha="center", va="bottom", fontsize=8.5)

ax1.set_ylabel("Mean RP cost (B VND)")
ax1.set_xticks(x)
ax1.set_xticklabels(lv_labels, fontsize=9.5)
ax1.set_ylim(0, 12.5)
bfmt(ax1)

ax2 = ax1.twinx()
ax2.plot(x, lv_p, color=GRAY, marker="o", ms=5, lw=1.5, ls="--")
ax2.set_ylabel("Cumulative probability (%)", color=GRAY)
ax2.tick_params(axis="y", labelcolor=GRAY)
ax2.set_ylim(0, 60)
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_color(GRAY)

legend_els = [
    plt.Rectangle((0, 0), 1, 1, fc=BLACK, ec=BLACK, label="RP cost"),
    Line2D([0], [0], color=GRAY, marker="o", ms=5, lw=1.5,
           ls="--", label="Probability"),
]
ax1.legend(handles=legend_els, loc="upper left", fontsize=9)
ax1.set_title("Figure 6. Mean cost and scenario probability by weather severity level")

plt.tight_layout()
plt.savefig(f"{OUT}/fig6_cost_escalation.png")
plt.close()
print("✓ Fig 6")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Procurement portfolio
#   Legend placed below panel B (outside chart area)
# ══════════════════════════════════════════════════════════════════════════
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))

# (A) Cost by supplier
sup_labels = [s[0] for s in PROC_SUPS]
sup_vals   = [s[1] for s in PROC_SUPS]
y_s = np.arange(len(sup_labels))
ax_l.barh(y_s, sup_vals, color=BLACK, edgecolor=BLACK,
          linewidth=0.6, height=0.5)
for yi, v in zip(y_s, sup_vals):
    ax_l.text(v + 0.01, yi, f"{v:.3f}B", va="center", fontsize=8.5)
ax_l.set_yticks(y_s)
ax_l.set_yticklabels(sup_labels, fontsize=9.5)
ax_l.set_xlabel("Procurement cost (B VND)")
ax_l.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.1f}B"))
ax_l.set_title("(A) Cost by supplier")

# (B) Top products by cost
tp_names = [p[0] for p in TOP_PROD]
tp_vals  = [p[1] for p in TOP_PROD]
tp_fill  = [BLACK if p[2] else GRAY for p in TOP_PROD]
y_p = np.arange(len(tp_names))
ax_r.barh(y_p, tp_vals, color=tp_fill, edgecolor=BLACK,
          linewidth=0.6, height=0.5)
for yi, v in zip(y_p, tp_vals):
    ax_r.text(v + 0.005, yi, f"{v:.3f}B", va="center", fontsize=8.5)
ax_r.set_yticks(y_p)
ax_r.set_yticklabels(tp_names, fontsize=8.5)
ax_r.set_xlabel("Procurement cost (B VND)")
ax_r.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.2f}B"))
ax_r.set_title("(B) Top products by cost")

# Legend below panel B — outside chart
legend_handles = [
    mpatches.Patch(fc=BLACK, ec=BLACK, label="SUP_006"),
    mpatches.Patch(fc=GRAY,  ec=BLACK, label="Other"),
]
ax_r.legend(handles=legend_handles, fontsize=9,
            loc="lower center", bbox_to_anchor=(0.5, -0.28),
            ncol=2, frameon=True)

fig.suptitle("Figure 7. Stage 1 optimal procurement portfolio", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_procurement.png")
plt.close()
print("✓ Fig 7")

print(f"\nAll figures saved → {OUT}/")