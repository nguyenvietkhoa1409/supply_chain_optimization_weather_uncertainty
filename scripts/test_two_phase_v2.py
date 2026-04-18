"""
test_two_phase_v2.py
====================
Minimal smoke-test for TwoPhaseOptimizerV2.

Uses a tiny synthetic problem (2 scenarios, 3 suppliers, 3 stores, 3 products)
that Gurobi / CBC solves in < 30 seconds — easy to inspect by hand.

Expected healthy output
-----------------------
  Phase 2A: ≥ 1 procurement vehicle dispatched in Normal scenario
  inventory[k=0, p] > 0 for all products in Normal scenario
  penalty_cost at sev=1 ≈ 0  (all demand satisfied through routing)
  penalty_cost at sev=5 > 0  (no vehicles → all unmet)
  Ordering property: WS ≤ RP ≤ EEV
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict

# ── Tiny synthetic data ────────────────────────────────────────────────────────

SUPPLIERS = pd.DataFrame([
    {"id": "S1", "name": "Seafood Co",  "subtype": "seafood",    "latitude": 16.08, "longitude": 108.22, "capacity_kg_per_day": 500, "fixed_cost_vnd": 200_000},
    {"id": "S2", "name": "Veggie Farm", "subtype": "vegetables", "latitude": 16.02, "longitude": 108.12, "capacity_kg_per_day": 400, "fixed_cost_vnd": 150_000},
    {"id": "S3", "name": "Wholesale",   "subtype": "general",    "latitude": 16.05, "longitude": 108.20, "capacity_kg_per_day": 300, "fixed_cost_vnd": 300_000},
])

DCS = pd.DataFrame([
    {"id": "DC1", "name": "HoaKhanh DC", "latitude": 16.03, "longitude": 108.15, "capacity_kg_per_day": 2000, "fixed_cost_vnd": 500_000},
])

STORES = pd.DataFrame([
    {"id": "R1", "name": "Store Hai Chau",  "latitude": 16.07, "longitude": 108.22, "demand_factor": 1.2},
    {"id": "R2", "name": "Store Son Tra",   "latitude": 16.08, "longitude": 108.25, "demand_factor": 1.0},
    {"id": "R3", "name": "Store Cam Le",    "latitude": 16.02, "longitude": 108.20, "demand_factor": 0.8},
])

PRODUCTS = pd.DataFrame([
    {"id": "P1", "name": "Fish",   "category": "seafood",   "unit_cost_vnd": 100_000, "weight_kg_per_unit": 0.5, "requires_refrigeration": True},
    {"id": "P2", "name": "Veggie", "category": "vegetable", "unit_cost_vnd":  20_000, "weight_kg_per_unit": 0.3, "requires_refrigeration": False},
    {"id": "P3", "name": "Fruit",  "category": "fruit",     "unit_cost_vnd":  30_000, "weight_kg_per_unit": 0.4, "requires_refrigeration": False},
])

SP_MATRIX = pd.DataFrame([
    {"supplier_id": "S1", "product_id": "P1", "unit_cost_vnd":  90_000, "moq_units": 5, "available": True},
    {"supplier_id": "S1", "product_id": "P2", "unit_cost_vnd":  18_000, "moq_units": 5, "available": True},
    {"supplier_id": "S2", "product_id": "P2", "unit_cost_vnd":  17_000, "moq_units": 5, "available": True},
    {"supplier_id": "S2", "product_id": "P3", "unit_cost_vnd":  28_000, "moq_units": 5, "available": True},
    {"supplier_id": "S3", "product_id": "P1", "unit_cost_vnd": 130_000, "moq_units": 8, "available": True},
    {"supplier_id": "S3", "product_id": "P2", "unit_cost_vnd":  26_000, "moq_units": 8, "available": True},
    {"supplier_id": "S3", "product_id": "P3", "unit_cost_vnd":  39_000, "moq_units": 8, "available": True},
])

# Daily demand: 3 stores × 3 products
DEMAND = pd.DataFrame([
    {"store_id": r, "product_id": p, "demand_units": d}
    for r, p, d in [
        ("R1","P1",20), ("R1","P2",30), ("R1","P3",15),
        ("R2","P1",15), ("R2","P2",25), ("R2","P3",10),
        ("R3","P1",10), ("R3","P2",20), ("R3","P3",8),
    ]
])

# Distance matrix (Haversine approximation, km)
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

all_locs = pd.concat([SUPPLIERS[["id","latitude","longitude"]],
                       DCS[["id","latitude","longitude"]],
                       STORES[["id","latitude","longitude"]]], ignore_index=True)
ids = all_locs["id"].tolist()
dm  = pd.DataFrame(0.0, index=ids, columns=ids)
for i in ids:
    ri = all_locs[all_locs["id"]==i].iloc[0]
    for j in ids:
        if i != j:
            rj = all_locs[all_locs["id"]==j].iloc[0]
            dm.loc[i,j] = _haversine(ri.latitude, ri.longitude,
                                      rj.latitude, rj.longitude)

NETWORK = dict(
    suppliers=SUPPLIERS, dcs=DCS, stores=STORES,
    all_locations=all_locs, distance_matrix=dm,
)

# ── Minimal weather scenarios ──────────────────────────────────────────────────

@dataclass
class SimpleScenario:
    scenario_id: int
    name: str
    severity_level: int
    probability: float
    speed_reduction_factor: float
    capacity_reduction_factor: float
    spoilage_multiplier: float
    supplier_accessibility: Dict = field(default_factory=lambda: {
        "seafood":1,"vegetables":1,"meat":1,"general":1})
    emergency_feasible: bool = True

    def get_supplier_accessible(self, subtype: str) -> int:
        return self.supplier_accessibility.get(subtype, 1)

SCENARIOS = [
    SimpleScenario(
        scenario_id=1, name="Normal Day", severity_level=1,
        probability=0.70, speed_reduction_factor=1.08,
        capacity_reduction_factor=0.95, spoilage_multiplier=1.00,
    ),
    SimpleScenario(
        scenario_id=2, name="Typhoon", severity_level=5,
        probability=0.30, speed_reduction_factor=2.20,
        capacity_reduction_factor=0.10, spoilage_multiplier=2.00,
        supplier_accessibility={"seafood":0,"vegetables":0,"meat":0,"general":1},
        emergency_feasible=False,
    ),
]

# ── Minimal fleet (2 vehicles) ─────────────────────────────────────────────────

FLEET = [
    {
        "vehicle_id": "van_0", "type_id": "mini_van", "name": "Van 0",
        "capacity_kg": 300, "volume_m3": 1.0,
        "fixed_cost_vnd": 150_000, "cost_per_km": 3_000, "cost_per_hour": 25_000,
        "base_speed_kmh": 50, "refrigerated": False, "spoilage_reduction": 0.0,
        "weather_capacity_factor": {1:0.95, 2:0.90, 3:0.80, 4:0.00, 5:0.00},
        "weather_speed_factor":    {1:1.00, 2:0.95, 3:0.85, 4:0.00, 5:0.00},
    },
    {
        "vehicle_id": "truck_0", "type_id": "light_truck", "name": "Truck 0",
        "capacity_kg": 1000, "volume_m3": 3.0,
        "fixed_cost_vnd": 400_000, "cost_per_km": 6_000, "cost_per_hour": 45_000,
        "base_speed_kmh": 40, "refrigerated": False, "spoilage_reduction": 0.0,
        "weather_capacity_factor": {1:0.92, 2:0.88, 3:0.75, 4:0.60, 5:0.00},
        "weather_speed_factor":    {1:1.00, 2:0.95, 3:0.88, 4:0.80, 5:0.00},
    },
]

# ── Run test ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SMOKE TEST — TwoPhaseOptimizerV2 (no emergency procurement)")
    print("=" * 70)

    # Import the new optimizer (adjust path if needed)
    try:
        from optimization.two_phase_optimizer_v2 import TwoPhaseOptimizerV2
    except ImportError:
        # Fallback: load from current directory
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "two_phase_optimizer_v2",
            pathlib.Path(__file__).parent / "two_phase_optimizer_v2.py"
        )
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        TwoPhaseOptimizerV2 = mod.TwoPhaseOptimizerV2

    opt = TwoPhaseOptimizerV2(
        network=NETWORK,
        products_df=PRODUCTS,
        supplier_product_df=SP_MATRIX,
        demand_df=DEMAND,
        weather_scenarios=SCENARIOS,
        fleet_instances=FLEET,
        baseline_ratio=0.70,
        concentration_max=0.50,   # relaxed for small test
    )

    status, sol = opt.solve(time_limit=120, gap_tolerance=0.05)

    if status not in ("Optimal", "Feasible"):
        print(f"\n❌ FAIL: solver returned {status}")
        return

    print("\n── Stage-1 Procurement ──")
    print(sol["stage1_procurement"].to_string(index=False))

    print("\n── Scenario Routes & Inventory ──")
    for sc_name, data in sol["scenario_routes"].items():
        inv_total = sum(v for v in data["inventory"].values() if v > 0)
        n_proc    = len(data["procurement_routes"])
        n_dist    = len(data["distribution_routes"])
        print(f"\n  {sc_name}:")
        print(f"    Phase 2A vehicles : {n_proc}")
        print(f"    Phase 2B vehicles : {n_dist}")
        print(f"    DC inventory (units): {inv_total:.1f}")
        for r in data["procurement_routes"]:
            print(f"      [{r['vehicle_type']}] {' → '.join(r['route'])}")
            print(f"        pickups: {r['pickups']}")

    print("\n── Scenario Costs ──")
    sc_df = sol["scenario_costs"]
    print(sc_df[["scenario_name","proc_vrp_cost","dist_vrp_cost",
                  "penalty_cost","total_cost"]].to_string(index=False))

    print("\n── Sanity Checks ──")
    normal = sc_df[sc_df["severity_level"] == 1]
    typhoon = sc_df[sc_df["severity_level"] == 5]

    routes_normal = sol["scenario_routes"].get("Normal Day", {})

    check1 = len(routes_normal.get("procurement_routes", [])) >= 1
    check2 = sum(routes_normal.get("inventory", {}).values()) > 0
    check3 = float(normal["proc_vrp_cost"].iloc[0]) > 0 if not normal.empty else False
    check4 = float(typhoon["penalty_cost"].iloc[0]) > 0 if not typhoon.empty else True
    check5 = float(normal["penalty_cost"].iloc[0]) < float(normal["total_cost"].iloc[0]) * 0.5 if not normal.empty else False

    print(f"  [{'✅' if check1 else '❌'}] Phase 2A runs in Normal scenario (n_proc≥1)")
    print(f"  [{'✅' if check2 else '❌'}] DC inventory > 0 in Normal scenario")
    print(f"  [{'✅' if check3 else '❌'}] Phase 2A VRP cost > 0 in Normal scenario")
    print(f"  [{'✅' if check4 else '❌'}] Typhoon penalty > 0 (no vehicles → unmet)")
    print(f"  [{'✅' if check5 else '❌'}] Normal penalty < 50% of total cost")

    all_pass = all([check1, check2, check3, check4, check5])
    print(f"\n{'✅ ALL CHECKS PASSED' if all_pass else '❌ SOME CHECKS FAILED'}")
    print("\nIf all pass: copy two_phase_optimizer_v2.py to src/optimization/")
    print("and update integrated_stochastic.py to use TwoPhaseOptimizerV2.")


if __name__ == "__main__":
    main()