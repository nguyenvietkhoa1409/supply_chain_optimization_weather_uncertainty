#!/usr/bin/env python3
"""
verify_priority1_fix.py
=======================
Quick feasibility test after Priority 1 fix:
  - MandatoryPickup hard constraint removed
  - WASTE_PENALTY_MULTIPLIER raised to 3.5

Runs with time_limit=300s only (not full 1800s).
Confirms: does Gurobi find at least ONE feasible incumbent?

Expected output:
  Status = "Feasible" or "Optimal"
  ObjVal > 0
  SolCount > 0
"""

import os, sys, time
os.environ["GRB_LICENSE_FILE"] = r"D:\gurobi.lic"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from data_generation.fleet_config import VEHICLE_TYPES, expand_fleet, to_optimizer_fleet
from weather.manual_scenarios import ManualWeatherScenarios
from weather.scenario_adapter import get_data_driven_scenarios
from optimization.two_phase_optimizer import TwoPhaseExtensiveFormOptimizer

# ── Load data ──────────────────────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

suppliers       = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
stores          = pd.read_csv(os.path.join(data_dir, "stores.csv"))
dcs             = pd.read_csv(os.path.join(data_dir, "distribution_centers.csv"))
all_locations   = pd.read_csv(os.path.join(data_dir, "network_topology.csv"))
distance_matrix = pd.read_csv(os.path.join(data_dir, "distance_matrix.csv"), index_col=0)
products        = pd.read_csv(os.path.join(data_dir, "products.csv"))
supplier_product = pd.read_csv(os.path.join(data_dir, "supplier_product_matrix.csv"))
demand_full     = pd.read_csv(os.path.join(data_dir, "daily_demand.csv"))
daily_demand    = demand_full[demand_full["date"] == "2024-10-01"].copy()

if "requires_refrigeration" not in products.columns:
    products["requires_refrigeration"] = (products["temperature_sensitivity"] == "high")

network = dict(
    suppliers=suppliers, stores=stores, dcs=dcs,
    all_locations=all_locations, distance_matrix=distance_matrix,
)

fleet_vehicles  = expand_fleet(VEHICLE_TYPES)
fleet_optimizer = to_optimizer_fleet(fleet_vehicles)

# ── Scenarios ─────────────────────────────────────────────────────────────────
try:
    scenarios = get_data_driven_scenarios(season="monsoon", target_count=5, merge_duplicates=True)
    season_name = "Monsoon (data-driven)"
except Exception:
    scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    season_name = "Monsoon (manual)"

total_p = sum(s.probability for s in scenarios)
if abs(total_p - 1.0) > 0.01:
    for s in scenarios:
        s.probability /= total_p

print("=" * 70)
print("  PRIORITY-1 + PRIORITY-2 FIX VERIFICATION")
print("  P1: MandatoryPickup removed | WASTE_PENALTY = 3.5")
print("  P2: S1Base(global) → S1AccBase + S1AccMin per scenario")
print(f"  Scenarios: {len(scenarios)} ({season_name})")
print(f"  Time limit: 300s  (quick feasibility + quality check)")
print("=" * 70)

# ── Build & solve ──────────────────────────────────────────────────────────────
optimizer = TwoPhaseExtensiveFormOptimizer(
    network             = network,
    products_df         = products,
    supplier_product_df = supplier_product,
    demand_df           = daily_demand,
    weather_scenarios   = scenarios,
    fleet_instances     = fleet_optimizer,
    baseline_ratio      = 0.70,
    concentration_max   = 0.40,
)

t0 = time.time()
status, solution = optimizer.solve(
    time_limit    = 300,
    gap_tolerance = 0.15,    # wider gap for quick check — feasibility > optimality
)
elapsed = time.time() - t0

print("\n" + "=" * 70)
print("  RESULT")
print("=" * 70)

if status in ("Optimal", "Feasible"):
    obj = solution.get("objective_value", 0)
    sc_costs = solution.get("scenario_costs", pd.DataFrame())
    print(f"  ✅ STATUS   : {status}  ({elapsed:.1f}s)")
    print(f"  ✅ ObjVal   : {obj:,.0f} VND")
    print(f"  ✅ Fix is WORKING — model is now feasible!")
    print()
    print(f"  P1-only baseline (pre-P2): 629,078,786 VND")
    change_pct = (obj - 629_078_786) / 629_078_786 * 100
    direction = "▼ lower (better)" if change_pct < 0 else "▲ higher"
    print(f"  P1+P2 result             : {obj:,.0f} VND  ({change_pct:+.1f}%  {direction})")

    if not sc_costs.empty:
        print(f"\n  Scenario cost breakdown (P1+P2):")
        for _, row in sc_costs.iterrows():
            print(f"    [{row['scenario_name']:<30}] p={row['probability']:.2f}  "
                  f"total={row['total_cost']:>14,.0f}  "
                  f"penalty={row.get('penalty_cost',0):>12,.0f}  "
                  f"spoilage={row.get('spoilage_cost',0):>12,.0f}")

    # Check procurement split: accessible vs inaccessible
    s1 = solution.get("stage1_procurement", pd.DataFrame())
    if not s1.empty:
        print(f"\n  Stage 1 procurement: {len(s1)} supplier-product pairs")
        print(f"  Total procurement cost: {s1['cost_vnd'].sum():,.0f} VND")

        # Identify accessible suppliers in best (sev=1) scenario
        sup_subtype = dict(zip(suppliers["id"], suppliers.get("subtype", suppliers["id"])))
        best_sc = scenarios[0]
        acc_ids  = {s for s in suppliers["id"]
                    if best_sc.get_supplier_accessible(sup_subtype.get(s, "general")) == 1}

        s1["accessible"] = s1["supplier_id"].isin(acc_ids)
        acc_qty  = s1[s1["accessible"]]["quantity_units"].sum()
        inacc_qty = s1[~s1["accessible"]]["quantity_units"].sum()
        total_qty = acc_qty + inacc_qty

        print(f"\n  Accessible procurement  : {acc_qty:,.1f} units "
              f"({acc_qty/max(total_qty,1)*100:.1f}%)")
        print(f"  Inaccessible procurement: {inacc_qty:,.1f} units "
              f"({inacc_qty/max(total_qty,1)*100:.1f}%)")

        if acc_qty / max(total_qty, 1) > 0.60:
            print(f"  ✅ Accessible fraction > 60% — P2 effectively closed the loophole.")
        else:
            print(f"  ⚠️  Accessible fraction still low — consider tightening S1AccMin weights.")

    # Show inventory per scenario from routes
    sc_routes = solution.get("scenario_routes", {})
    if sc_routes:
        print(f"\n  Inventory built per scenario (from Phase 2A pickups):")
        for sc_name, data in sc_routes.items():
            inv = data.get("inventory", {})
            inv_total = sum(
                qty * float(products.set_index("id").loc[p, "weight_kg_per_unit"])
                for p, qty in inv.items()
                if p in products["id"].values and qty > 0.01
            )
            n_proc = len(data.get("procurement_routes", []))
            n_dist = len(data.get("distribution_routes", []))
            print(f"    [{sc_name:<30}] inv={inv_total:>8,.0f} kg  "
                  f"proc_routes={n_proc}  dist_routes={n_dist}")

    print("\n" + "=" * 70)
    print("  NEXT STEP: Run full 1800s solve via run_two_phase_optimization.py")
    print("=" * 70)

else:
    print(f"  ❌ STATUS   : {status}  ({elapsed:.1f}s)")
    print(f"  ❌ Priority 1+2 fix did NOT restore feasibility in 300s.")
    print(f"     Bound reached: check above for ObjBound value.")
    print()
    print("  Possible next steps:")
    print("    1. Run Gurobi IIS on single scenario via diagnose_infeasibility.py")
    print("    2. Temporarily lower baseline_ratio to 0.50 or concentration_max to 0.60")
    print("=" * 70)
