#!/usr/bin/env python3
"""
Run Stochastic Optimization – Complete Pipeline
FIXED VERSION

Changes from original
─────────────────────
[M-1] EEV is now computed correctly via StochasticValidator.compute_eev()
[M-2] WS  is now computed correctly via StochasticValidator.compute_ws()
[M-3] Ordering check (WS ≤ RP ≤ EEV) is meaningful
[V-3] Extensive form used by default (solve_extensive_form()), not sequential
[W-3, P-1, P-3, P-5, V-1, V-4] Inherited from fixed modules

Usage
─────
    python scripts/run_stochastic_optimization.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from data_generation.network_generator import DaNangNetworkGenerator
from data_generation.product_generator import ProductCatalogGenerator
from data_generation.demand_generator import DemandPatternGenerator
from weather.manual_scenarios import ManualWeatherScenarios
from optimization.integrated_stochastic import IntegratedStochasticModel
from optimization.deterministic_baseline import DeterministicBaselineModel
from evaluation.vss_evpi_calculator import StochasticValidator


def main():
    print("=" * 80)
    print("WEATHER-AWARE STOCHASTIC OPTIMIZATION  (FIXED PIPELINE)")
    print("=" * 80)

    # ── STEP 1: Load data ────────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

    if not os.path.exists(os.path.join(data_dir, "network_topology.csv")):
        print("⚠ Data not found — generating…")
        os.system("python scripts/generate_all_data.py")

    suppliers = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    dcs = pd.read_csv(os.path.join(data_dir, "distribution_centers.csv"))
    all_locations = pd.read_csv(os.path.join(data_dir, "network_topology.csv"))
    distance_matrix = pd.read_csv(os.path.join(data_dir, "distance_matrix.csv"), index_col=0)

    network = {
        "suppliers": suppliers, "stores": stores, "dcs": dcs,
        "all_locations": all_locations, "distance_matrix": distance_matrix,
    }

    products = pd.read_csv(os.path.join(data_dir, "products.csv"))
    supplier_product = pd.read_csv(os.path.join(data_dir, "supplier_product_matrix.csv"))
    weekly_demand = pd.read_csv(os.path.join(data_dir, "weekly_demand.csv"))

    print(f"✓ {len(suppliers)} suppliers, {len(products)} products, {len(stores)} stores")

    # ── STEP 2: Weather scenarios ────────────────────────────────────────────
    season = input("\nSeason (1=Dry, 2=Monsoon): ").strip()
    if season == "1":
        scenarios = ManualWeatherScenarios.create_dry_season_scenarios()
        season_name = "Dry Season"
    else:
        scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
        season_name = "Monsoon Season"

    total_p = sum(s.probability for s in scenarios)
    if abs(total_p - 1.0) > 0.01:
        for s in scenarios:
            s.probability /= total_p

    print(f"\n✓ {len(scenarios)} {season_name} scenarios  (Σp = {sum(s.probability for s in scenarios):.3f})")
    print(ManualWeatherScenarios.get_scenario_summary_table(scenarios).to_string(index=False))

    # ── STEP 3: Solve stochastic (extensive form) ────────────────────────────
    print("\n" + "-" * 80)
    print("STEP 3: Solving Extensive Form (true two-stage MILP) [V-3 fixed]")
    print("-" * 80)

    integrated = IntegratedStochasticModel(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        weather_scenarios=scenarios,
        vehicle_config={"num_vehicles": 3, "capacity_kg": 1000,
                        "base_speed_kmh": 40, "cost_per_km": 5000,
                        "cost_per_hour": 50000, "max_route_time_hours": 10},
        risk_aversion=0.0,
        baseline_ratio=0.70,
    )

    rp_status, rp_solution = integrated.solve_extensive_form(
        time_limit=1800, gap_tolerance=0.05
    )

    if rp_status not in ("Optimal", "Feasible"):
        print(f"⚠ Extensive form failed: {rp_status}")
        return

    rp = rp_solution["objective_value"]
    print(f"\n✓ RP (Recourse Problem) = {rp:,.0f} VND")

    # ── STEP 4: Deterministic baseline → get x*_EV ──────────────────────────
    print("\n" + "-" * 80)
    print("STEP 4: Solving Deterministic Baseline (Expected Weather)")
    print("-" * 80)

    det_model = DeterministicBaselineModel(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        weather_scenarios=scenarios,
    )
    det_status, det_solution = det_model.solve(time_limit=300)

    if det_status not in ("Optimal", "Feasible"):
        print(f"⚠ Deterministic baseline failed: {det_status}")
        return

    ev_stage1 = det_solution.get("stage1_procurement", det_solution.get("procurement", pd.DataFrame()))
    print(f"✓ EV solution: {det_solution.get('deterministic_objective', det_solution['objective_value']):,.0f} VND")

    # ── STEP 5: Correct EEV [M-1 fix] ───────────────────────────────────────
    print("\n" + "-" * 80)
    print("STEP 5: Computing EEV (M-1 fix — evaluate EV solution across scenarios)")
    print("-" * 80)

    validator = StochasticValidator()
    eev, eev_breakdown = validator.compute_eev(
        ev_stage1_procurement=ev_stage1,
        scenarios=scenarios,
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        time_limit_per_scenario=120,
    )

    # ── STEP 6: Correct WS [M-2 fix] ────────────────────────────────────────
    print("\n" + "-" * 80)
    print("STEP 6: Computing WS (M-2 fix — K separate deterministic solves)")
    print("-" * 80)

    ws, ws_breakdown = validator.compute_ws(
        scenarios=scenarios,
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        time_limit_per_scenario=300,
    )

    # ── STEP 7: Report [M-3 fix] ─────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("STEP 7: Validation Report")
    print("-" * 80)

    sc_costs = rp_solution.get("scenario_costs", pd.DataFrame())

    report = validator.generate_validation_report(
        rp=rp,
        eev=eev,
        ws=ws,
        scenario_costs_rp=sc_costs,
        eev_breakdown=eev_breakdown,
        ws_breakdown=ws_breakdown,
    )
    print("\n" + report)

    # ── STEP 8: Save ─────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "validation_report_fixed.txt"), "w") as f:
        f.write(report)

    rp_solution["stage1_procurement"].to_csv(
        os.path.join(results_dir, "stochastic_procurement_fixed.csv"), index=False
    )
    sc_costs.to_csv(os.path.join(results_dir, "scenario_costs_fixed.csv"), index=False)
    eev_breakdown.to_csv(os.path.join(results_dir, "eev_breakdown.csv"), index=False)
    ws_breakdown.to_csv(os.path.join(results_dir, "ws_breakdown.csv"), index=False)

    print(f"\n✓ Results saved to {results_dir}/")
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()