#!/usr/bin/env python3
"""
run_two_phase_optimization.py
=============================================================
Two-Phase Stochastic Optimization Pipeline
Following Patel et al. (2024) architecture:

  Stage 1: Procurement MILP (supplier selection + order quantities)
  Stage 2A: Procurement VRP  —  DC → Suppliers → DC
  Stage 2B: Distribution VRP —  DC → Stores    → DC

Linked by DC inventory balance: what vehicles physically pick up
in Phase 2A becomes the available inventory for Phase 2B delivery.

Usage:
  python scripts/run_two_phase_optimization.py

Outputs (same folder as working version):
  results/tp_stochastic_procurement.csv
  results/tp_scenario_costs.csv
  results/tp_validation_report.txt
  results/tp_scenario_routes.json
"""

import os, sys, json
os.environ["GRB_LICENSE_FILE"] = r"D:\gurobi.lic"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from data_generation.fleet_config import (
    VEHICLE_TYPES, get_fleet_summary, expand_fleet,
    get_effective_capacity, to_optimizer_fleet,
)
from weather.manual_scenarios import ManualWeatherScenarios
from weather.scenario_adapter import get_data_driven_scenarios, get_historical_frequency_scenarios
from optimization.integrated_stochastic import IntegratedStochasticModel
from optimization.deterministic_baseline import DeterministicBaselineModel
from evaluation.vss_evpi_calculator import StochasticValidator

# ── Fleet ─────────────────────────────────────────────────────────────────────
FLEET_CONFIG      = VEHICLE_TYPES
VEHICLE_CONFIG_LEGACY = {
    "types": VEHICLE_TYPES,
    "max_route_time_hours": 10,
    "unmet_penalty_per_unit": 80_000,
}


def main():
    print("=" * 80)
    print("TWO-PHASE STOCHASTIC OPTIMIZATION  (Patel et al. 2024 Architecture)")
    print("  Phase 2A: DC → Suppliers → DC  (procurement VRP)")
    print("  Phase 2B: DC → Stores   → DC  (distribution VRP)")
    print("=" * 80)
    print(get_fleet_summary())

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

    suppliers       = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
    stores          = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    dcs             = pd.read_csv(os.path.join(data_dir, "distribution_centers.csv"))
    all_locations   = pd.read_csv(os.path.join(data_dir, "network_topology.csv"))
    distance_matrix = pd.read_csv(os.path.join(data_dir, "distance_matrix.csv"), index_col=0)

    network = dict(
        suppliers=suppliers, stores=stores, dcs=dcs,
        all_locations=all_locations, distance_matrix=distance_matrix,
    )

    products         = pd.read_csv(os.path.join(data_dir, "products.csv"))
    supplier_product = pd.read_csv(os.path.join(data_dir, "supplier_product_matrix.csv"))
    demand_full      = pd.read_csv(os.path.join(data_dir, "daily_demand.csv"))
    daily_demand     = demand_full[demand_full["date"] == "2024-10-01"].copy()

    # Back-fill columns if needed
    if "volume_m3_per_unit" not in products.columns:
        from data_generation.fleet_config import PRODUCT_VOLUME_M3, DEFAULT_VOLUME_M3_PER_UNIT
        products["volume_m3_per_unit"] = (
            products["name"].map(PRODUCT_VOLUME_M3).fillna(DEFAULT_VOLUME_M3_PER_UNIT)
        )
    if "requires_refrigeration" not in products.columns:
        products["requires_refrigeration"] = (
            products["temperature_sensitivity"] == "high"
        )

    fleet_vehicles  = expand_fleet(FLEET_CONFIG)
    fleet_optimizer = to_optimizer_fleet(fleet_vehicles)

    print(f"\n✓ {len(suppliers)} suppliers | {len(products)} products | "
          f"{len(stores)} stores | {len(fleet_vehicles)} vehicles")
    print(f"  Refrigerated products: {products['requires_refrigeration'].sum()}/{len(products)}")

    # ── Weather scenarios ──────────────────────────────────────────────────────
    print("\nSeason (1=Dry, 2=Monsoon [default]): ", end="")
    season_input = input().strip()
    season_name = "Dry" if season_input == "1" else "Monsoon"
    
    print("\nScenario Method (1=Data-driven LHS/FFS, 2=Historical Freq ERA5 [recommended]): ", end="")
    method_input = input().strip() or "2"
    
    if method_input == "1":
        scenarios = get_data_driven_scenarios(
            season=season_name.lower(), target_count=5, merge_duplicates=True
        )
    else:
        scenarios = get_historical_frequency_scenarios(season=season_name.lower())

    # Normalize probabilities
    total_p = sum(s.probability for s in scenarios)
    if abs(total_p - 1.0) > 0.01:
        for s in scenarios:
            s.probability /= total_p

    print(f"\n✓ {len(scenarios)} {season_name} scenarios")
    print(ManualWeatherScenarios.get_scenario_summary_table(scenarios).to_string(index=False))

    # Show vehicle availability per scenario
    print("\nVehicle availability by scenario:")
    for sc in scenarios:
        avail_types = sorted(set(
            v["type_id"] for v in fleet_vehicles
            if get_effective_capacity(v, sc)["available_bool"]
        ))
        n = sum(1 for v in fleet_vehicles
                if get_effective_capacity(v, sc)["available_bool"])
        label = ", ".join(avail_types) if avail_types else "NONE"
        print(f"  {sc.name:35s}: {n}/{len(fleet_vehicles)}  ({label})")

    # ── Solve RP (two-phase) ───────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("STEP 3: Solving Two-Phase Extensive Form")
    print("-" * 80)

    integrated = IntegratedStochasticModel(
        network              = network,
        products_df          = products,
        supplier_product_df  = supplier_product,
        demand_df            = daily_demand,
        weather_scenarios    = scenarios,
        vehicle_config       = VEHICLE_CONFIG_LEGACY,
        fleet_instances      = fleet_optimizer,
        risk_aversion        = 0.0,
        baseline_ratio       = 0.70,
    )

    rp_status, rp_solution = integrated.solve_two_phase_extensive_form(
        time_limit    = 1800,
        gap_tolerance = 0.03,
        unmet_penalty = 500_000,
    )

    if rp_status not in ("Optimal", "Feasible"):
        print(f"⚠ Two-phase extensive form failed with status: {rp_status}")
        print("  Hint: Check if solver time limit was reached and no feasible incumbent was found.")
        return

    rp = rp_solution["objective_value"]
    print(f"\n✓ RP = {rp:,.0f} VND")

    # ── Print route summary ────────────────────────────────────────────────────
    print("\nRoute Summary by Scenario:")
    sc_routes = rp_solution.get("scenario_routes", {})
    for sc_name, data in sc_routes.items():
        proc_routes = data.get("procurement_routes", [])
        dist_routes = data.get("distribution_routes", [])
        inv = data.get("inventory", {})
        inv_total_kg = sum(
            qty * products.set_index("id").loc[p, "weight_kg_per_unit"]
            for p, qty in inv.items()
            if p in products["id"].values and qty > 0
        )
        print(f"  {sc_name}:")
        print(f"    Phase 2A (procurement): {len(proc_routes)} vehicles → "
              f"DC inventory={inv_total_kg:.0f} kg")
        for r in proc_routes:
            print(f"      {r['vehicle_type']:15s}: {' → '.join(r['route'])}")
        print(f"    Phase 2B (distribution): {len(dist_routes)} vehicles")
        for r in dist_routes:
            print(f"      {r['vehicle_type']:15s}: {' → '.join(r['route'])}")

    # ── Deterministic baseline + VSS/EVPI ─────────────────────────────────────
    print("\n" + "-" * 80)
    print("STEP 4: Deterministic Baseline")
    print("-" * 80)

    det_model = DeterministicBaselineModel(
        network=network, products_df=products,
        supplier_product_df=supplier_product,
        demand_df=daily_demand, weather_scenarios=scenarios,
        fleet_instances=fleet_optimizer,
        concentration_max=0.30,
    )
    det_status, det_solution = det_model.solve(time_limit=300)
    if det_status not in ("Optimal", "Feasible"):
        print(f"⚠ Deterministic baseline failed: {det_status}")
        det_solution = None

    validator = StochasticValidator()
    sc_costs  = rp_solution.get("scenario_costs", pd.DataFrame())

    if det_solution is not None and not sc_costs.empty:
        ev_stage1 = det_solution.get(
            "stage1_procurement",
            det_solution.get("procurement", pd.DataFrame()),
        )

        print("\n" + "-" * 80)
        print("STEP 5: Computing EEV")
        eev, eev_breakdown = validator.compute_eev(
            ev_stage1, scenarios, network, products, supplier_product, daily_demand, 120,
            vehicle_config   = VEHICLE_CONFIG_LEGACY,
            fleet_instances  = fleet_optimizer,
        )

        print("\n" + "-" * 80)
        print("STEP 6: Computing WS")
        ws, ws_breakdown = validator.compute_ws(
            scenarios, network, products, supplier_product, daily_demand, 300,
            vehicle_config   = VEHICLE_CONFIG_LEGACY,
            fleet_instances  = fleet_optimizer,
        )

        report = validator.generate_validation_report(
            rp=rp, eev=eev, ws=ws,
            scenario_costs_rp=sc_costs,
            eev_breakdown=eev_breakdown,
            ws_breakdown=ws_breakdown,
        )

        # CVaR
        if not sc_costs.empty:
            cvar_list = [
                (row["probability"], row["total_cost"], row["scenario_name"])
                for _, row in sc_costs.iterrows()
            ]
            cvar_metrics = validator.compute_cvar_metrics(cvar_list, alpha=0.90)
            report += "\n" + validator.format_cvar_report_section(cvar_metrics)
    else:
        report = integrated.generate_report(rp_solution)

    print("\n" + report)

    # ── Save results ───────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    rp_solution["stage1_procurement"].to_csv(
        os.path.join(results_dir, "tp_stochastic_procurement.csv"), index=False
    )
    sc_costs.to_csv(
        os.path.join(results_dir, "tp_scenario_costs.csv"), index=False
    )
    with open(os.path.join(results_dir, "tp_validation_report.txt"), "w") as f:
        f.write(report)
    with open(
        os.path.join(results_dir, "tp_scenario_routes.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {k: {
                "inventory":            v.get("inventory", {}),
                "procurement_routes":   [
                    {**r, "route": r["route"]}
                    for r in v.get("procurement_routes", [])
                ],
                "distribution_routes":  [
                    {**r, "route": r["route"]}
                    for r in v.get("distribution_routes", [])
                ],
             }
             for k, v in sc_routes.items()},
            f, indent=2, ensure_ascii=False,
        )

    print(f"\n✓ Results saved to {results_dir}/  (prefix: tp_)")
    print("=" * 80)
    print("TWO-PHASE PIPELINE COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
