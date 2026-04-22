import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set working directories based on file location to make it portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "figures")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting Settings
sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_gantt_timeline():
    timeline_path = os.path.join(RESULTS_DIR, "analysis_timeline.csv")
    if not os.path.exists(timeline_path): return
    
    df = pd.read_csv(timeline_path)
    if df.empty: return

    # Focus on Normal scenario to prevent overlap
    scen = "Normal Monsoon Day"
    df_scen = df[df["scenario"] == scen].copy()
    if df_scen.empty: return

    vehicles = sorted(df_scen["vehicle"].unique())
    v_map = {v: i for i, v in enumerate(vehicles)}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw time windows correctly matching the uniform architecture
    ax.axvspan(4.0, 9.5, alpha=0.1, color='blue', label="Supplier Window (2A)")
    ax.axvspan(10.0, 13.0, alpha=0.1, color='green', label="Store Window (2B)")

    for idx, row in df_scen.iterrows():
        y = v_map[row["vehicle"]]
        start = row["arrival_h"]
        dur = max(row["departure_h"] - start, 0.05)
        
        # Color coding phase 2A vs 2B
        c = "darkblue" if row["phase"] == "2A" else "darkgreen"
        ax.barh(y, dur, left=start, height=0.4, color=c, edgecolor='w', align="center")
        ax.text(start + dur/2, y + 0.25, row["node"], ha='center', va='bottom', fontsize=8)

    ax.set_yticks(range(len(vehicles)))
    ax.set_yticklabels(vehicles)
    ax.set_xlabel("Time (Hour of Day)")
    ax.set_title(f"Vehicle Activity Timeline: {scen}", pad=20, fontweight="bold")
    
    ax.legend(loc="upper right")
    
    # Set X-axis to represent actual time (eg. 04:30)
    xticks = np.arange(3.0, 15.0, 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x):02d}:{int((x%1)*60):02d}" for x in xticks])
    ax.set_xlim([3.0, 14.5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Timeline_GanttChart.png"), dpi=300)
    plt.close()

def plot_procurement_diversification():
    proc_path = os.path.join(RESULTS_DIR, "tp_stochastic_procurement.csv")
    if not os.path.exists(proc_path): return
    
    proc = pd.read_csv(proc_path)
    if proc.empty: return
    
    pivot = proc.pivot_table(index="supplier_id", columns="product_id", values="quantity_units", aggfunc="sum").fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".1f", cbar_kws={'label': 'Ordered Quantity (units)'}, ax=ax)
    ax.set_title("Procurement Diversification Matrix (Stage 1)", pad=15, fontweight="bold")
    ax.set_xlabel("Product")
    ax.set_ylabel("Supplier")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Procurement_Heatmap.png"), dpi=300)
    plt.close()

def plot_scenario_response():
    costs_path = os.path.join(RESULTS_DIR, "tp_scenario_costs.csv")
    sl_path = os.path.join(RESULTS_DIR, "analysis_service_level.csv")
    if not os.path.exists(costs_path) or not os.path.exists(sl_path): return
    
    costs = pd.read_csv(costs_path)
    sl = pd.read_csv(sl_path)
    
    if costs.empty or sl.empty: return
    
    scenarios = ["Normal Monsoon Day", "Light Rain", "Moderate Rain", "Heavy Rain", "Tropical Storm/Typhoon"]
    scenarios = [s for s in scenarios if s in costs["scenario_name"].values]
    
    veh_data = costs.set_index("scenario_name").loc[scenarios, "n_operable_vehicles"]
    sl_data = sl.groupby("scenario")["fill_rate"].mean().loc[scenarios] * 100
    inv_data = sl.groupby("scenario")["delivered"].sum().loc[scenarios]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Subplot 1: Inventory delivered
    sns.barplot(x=scenarios, y=inv_data.values, ax=axes[0], color="skyblue")
    axes[0].set_ylabel("Total DC Transmit\nVolume (units)", fontsize=11)
    axes[0].set_title("Operational Response Under Weather Disruption", pad=15, fontweight="bold")
    
    # Subplot 2: Operable vehicles
    sns.barplot(x=scenarios, y=veh_data.values, ax=axes[1], color="salmon")
    axes[1].set_ylabel("Operable\nVehicles", fontsize=11)
    
    # Subplot 3: Fill rate
    sns.lineplot(x=scenarios, y=sl_data.values, ax=axes[2], color="forestgreen", marker="o", linewidth=2.5, markersize=8)
    axes[2].set_ylabel("Overall Service\nFill Rate (%)", fontsize=11)
    axes[2].set_ylim([0, 105])
    
    # Rotate Scenario labels
    axes[2].set_xticklabels(scenarios, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Scenario_Response_Panel.png"), dpi=300)
    plt.close()

def plot_service_level_heatmap():
    sl_path = os.path.join(RESULTS_DIR, "analysis_service_level.csv")
    if not os.path.exists(sl_path): return
    
    sl = pd.read_csv(sl_path)
    if sl.empty: return
    
    hmap_data = sl.groupby(["store", "scenario"])["fill_rate"].mean().unstack() * 100
    
    scen_order = ["Normal Monsoon Day", "Light Rain", "Moderate Rain", "Heavy Rain", "Tropical Storm/Typhoon"]
    cols = [s for s in scen_order if s in hmap_data.columns]
    hmap_data = hmap_data[cols]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(hmap_data, cmap="RdYlGn", annot=True, fmt=".1f", vmin=0, vmax=100, cbar_kws={'label': 'Average Fill Rate (%)'})
    ax.set_title("Service Fill Rate (%) across Stores and Weather Scenarios", pad=20, fontweight="bold")
    ax.set_ylabel("Store")
    ax.set_xlabel("Weather Scenario")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Service_Level_Heatmap.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating visualizations in 'figures/' ...")
    plot_gantt_timeline()
    print(" ✓ Created Timeline Gantt Chart")
    plot_procurement_diversification()
    print(" ✓ Created Procurement Heatmap")
    plot_scenario_response()
    print(" ✓ Created Scenario Response Multi-panel")
    plot_service_level_heatmap()
    print(" ✓ Created Service Fill Rate Heatmap")
