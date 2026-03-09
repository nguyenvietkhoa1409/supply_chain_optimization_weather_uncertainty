#!/usr/bin/env python3
"""
Run Stochastic Optimization – Complete Pipeline
UPDATED v2: Heterogeneous Fleet (Patel et al. adaptation)
"""

import os
import sys
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from data_generation.fleet_config import (VEHICLE_TYPES, get_fleet_summary,
                                            expand_fleet, get_effective_capacity,
                                            to_optimizer_fleet)
from weather.manual_scenarios import ManualWeatherScenarios
from weather.scenario_adapter import get_data_driven_scenarios
from optimization.integrated_stochastic import IntegratedStochasticModel
from optimization.deterministic_baseline import DeterministicBaselineModel
from evaluation.vss_evpi_calculator import StochasticValidator

# ── Fleet Configuration ────────────────────────────────────────────────────
FLEET_CONFIG = VEHICLE_TYPES   # 4 heterogeneous vehicle types for Da Nang

VEHICLE_CONFIG_LEGACY = {
    "types": VEHICLE_TYPES,
    "max_route_time_hours": 10,
    "unmet_penalty_per_unit": 80_000,
}


def build_fleet_dispatch_report(scenario_costs_df: pd.DataFrame,
                                 vehicle_dispatch: Dict) -> str:
    lines = [
        "\n5. FLEET DISPATCH ANALYSIS  (Patel et al. heterogeneous fleet)",
        "-" * 90,
        f"  {'Scenario':30s}  {'#Veh':>5}  {'Types Used':>30}  "
        f"{'Fleet Fixed':>14}  {'VRP Cost':>12}",
        "  " + "-" * 86,
    ]
    for _, row in scenario_costs_df.iterrows():
        sc_name   = row["scenario_name"]
        n_v       = int(row.get("n_vehicles_used", 0))
        fleet_fix = row.get("fleet_fixed_cost", 0)
        vrp_c     = row.get("vrp_cost", 0)
        dispatch  = vehicle_dispatch.get(sc_name, [])
        types     = {}
        for d in dispatch:
            tid = d.get("type_id", "?")
            types[tid] = types.get(tid, 0) + 1
        refrig = sum(1 for d in dispatch if d.get("refrigerated", False))
        type_str = ", ".join(f"{v}×{k}" for k, v in types.items())
        if refrig:
            type_str += f" (冷×{refrig})"
        lines.append(
            f"  {sc_name:30s}  {n_v:>5}  {type_str:>30}  "
            f"{fleet_fix:>14,.0f}  {vrp_c:>12,.0f}"
        )
    lines.append("=" * 90)
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("WEATHER-AWARE STOCHASTIC OPTIMIZATION  v2 (Heterogeneous Fleet)")
    print("=" * 80)
    print(get_fleet_summary())

    # ── STEP 1: Load data ─────────────────────────────────────────────────
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

    if not os.path.exists(os.path.join(data_dir, "network_topology.csv")):
        print("⚠ Data not found — generating…")
        os.system("python scripts/generate_all_data.py")

    suppliers       = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
    stores          = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    dcs             = pd.read_csv(os.path.join(data_dir, "distribution_centers.csv"))
    all_locations   = pd.read_csv(os.path.join(data_dir, "network_topology.csv"))
    distance_matrix = pd.read_csv(os.path.join(data_dir, "distance_matrix.csv"), index_col=0)

    network = {
        "suppliers": suppliers, "stores": stores, "dcs": dcs,
        "all_locations": all_locations, "distance_matrix": distance_matrix,
    }

    products         = pd.read_csv(os.path.join(data_dir, "products.csv"))
    supplier_product = pd.read_csv(os.path.join(data_dir, "supplier_product_matrix.csv"))
    weekly_demand    = pd.read_csv(os.path.join(data_dir, "weekly_demand.csv"))

    # Back-fill new columns for older saved data
    if "volume_m3_per_unit" not in products.columns:
        from data_generation.fleet_config import PRODUCT_VOLUME_M3, DEFAULT_VOLUME_M3_PER_UNIT
        products["volume_m3_per_unit"] = (
            products["name"].map(PRODUCT_VOLUME_M3).fillna(DEFAULT_VOLUME_M3_PER_UNIT)
        )
        print("  ⚠ volume_m3_per_unit back-filled (re-run generate_all_data.py)")

    if "requires_refrigeration" not in products.columns:
        products["requires_refrigeration"] = products["temperature_sensitivity"] == "high"
        print("  ⚠ requires_refrigeration back-filled from temperature_sensitivity")

    fleet_vehicles     = expand_fleet(FLEET_CONFIG)          # compact format (for display/checks)
    fleet_optimizer    = to_optimizer_fleet(fleet_vehicles)   # optimizer-compatible format [FIX-F3]
    print(f"\n✓ {len(suppliers)} suppliers, {len(products)} products, "
          f"{len(stores)} stores, {len(fleet_vehicles)} vehicles")
    print(f"  Refrigerated products: {products['requires_refrigeration'].sum()}/{len(products)}")

    # ── STEP 2: Weather scenarios ──────────────────────────────────────────
    season = input("\nSeason (1=Dry, 2=Monsoon): ").strip()
    if season == "1":
        scenarios   = ManualWeatherScenarios.create_dry_season_scenarios()
        season_name = "Dry Season"
    else:
        # scenarios   = ManualWeatherScenarios.create_monsoon_season_scenarios()
        # season_name = "Monsoon Season"
        scenarios = get_data_driven_scenarios(season="monsoon")
        season_name = "Monsoon Season"

    total_p = sum(s.probability for s in scenarios)
    if abs(total_p - 1.0) > 0.01:
        for s in scenarios: s.probability /= total_p

    print(f"\n✓ {len(scenarios)} {season_name} scenarios")
    print(ManualWeatherScenarios.get_scenario_summary_table(scenarios).to_string(index=False))

    # Show vehicle availability per scenario
    print("\nVehicle availability by scenario:")
    for sc in scenarios:
        avail_types = set(
            v["type_id"] for v in fleet_vehicles
            if get_effective_capacity(v, sc)["available_bool"]
        )
        n_avail = sum(1 for v in fleet_vehicles
                      if get_effective_capacity(v, sc)["available_bool"])
        print(f"  {sc.name:30s}: {n_avail}/{len(fleet_vehicles)} "
              f"({', '.join(sorted(avail_types)) if avail_types else 'NONE'})")

    # ── STEP 3: Solve RP ──────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("STEP 3: Solving Extensive Form — Heterogeneous Fleet")
    print("-" * 80)

    integrated = IntegratedStochasticModel(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        weather_scenarios=scenarios,
        vehicle_config=VEHICLE_CONFIG_LEGACY,
        fleet_instances=fleet_optimizer,         # [FIX-F1] optimizer-format fleet
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
    print(f"\n✓ RP = {rp:,.0f} VND")

    # ── STEP 4: Deterministic baseline ────────────────────────────────────
    print("\n" + "-" * 80)
    print("STEP 4: Deterministic Baseline")
    print("-" * 80)

    det_model = DeterministicBaselineModel(
        network=network, products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand, weather_scenarios=scenarios,
    )
    det_status, det_solution = det_model.solve(time_limit=300)
    if det_status not in ("Optimal", "Feasible"):
        print(f"⚠ Det baseline failed: {det_status}"); return

    ev_stage1 = det_solution.get("stage1_procurement",
                                  det_solution.get("procurement", pd.DataFrame()))

    # ── STEP 5-6: EEV and WS ──────────────────────────────────────────────
    validator = StochasticValidator()

    print("\n" + "-" * 80)
    print("STEP 5: Computing EEV")
    eev, eev_breakdown = validator.compute_eev(
        ev_stage1, scenarios, network, products, supplier_product, weekly_demand, 120
    )

    print("\n" + "-" * 80)
    print("STEP 6: Computing WS")
    ws, ws_breakdown = validator.compute_ws(
        scenarios, network, products, supplier_product, weekly_demand, 300,
        vehicle_config=VEHICLE_CONFIG_LEGACY,
        fleet_instances=fleet_optimizer,         # [FIX-F2] optimizer-format fleet
    )

    # ── STEP 7: Report ────────────────────────────────────────────────────
    sc_costs = rp_solution.get("scenario_costs", pd.DataFrame())
    report   = validator.generate_validation_report(
        rp=rp, eev=eev, ws=ws,
        scenario_costs_rp=sc_costs,
        eev_breakdown=eev_breakdown,
        ws_breakdown=ws_breakdown,
    )
    # [FIX-C1] TÍCH HỢP BÁO CÁO CVaR VÀO ĐÂY
    if not sc_costs.empty:
        # Chuyển đổi DataFrame thành list of tuples theo yêu cầu của hàm
        rp_scenario_costs_list = [
            (row["probability"], row["total_cost"], row["scenario_name"])
            for _, row in sc_costs.iterrows()
        ]
        
        cvar_metrics = validator.compute_cvar_metrics(
            rp_scenario_costs_list,
            alpha=0.90,
            lambda_weight=0.30
        )
        
        # Nối thêm phần CVaR report vào báo cáo chung
        report += "\n" + validator.format_cvar_report_section(cvar_metrics)

        # Lưu riêng file CSV cho CVaR metrics (STEP 8)
        pd.DataFrame([{
            "metric":  k,
            "value":   v,
        } for k, v in cvar_metrics.items() if isinstance(v, (int, float))]).to_csv(
            os.path.join(os.path.dirname(__file__), "..", "results", "cvar_metrics.csv"), index=False
        )
        
    vehicle_dispatch = rp_solution.get("vehicle_dispatch", {})
    if not sc_costs.empty and vehicle_dispatch:
        try:
            report += build_fleet_dispatch_report(sc_costs, vehicle_dispatch)
        except Exception:
            pass

    print("\n" + report)

    # ── STEP 8: Save ──────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "validation_report_fixed.txt"), "w") as f:
        f.write(report)

    rp_solution["stage1_procurement"].to_csv(
        os.path.join(results_dir, "stochastic_procurement_fixed.csv"), index=False)
    sc_costs.to_csv(os.path.join(results_dir, "scenario_costs_fixed.csv"), index=False)
    eev_breakdown.to_csv(os.path.join(results_dir, "eev_breakdown.csv"), index=False)
    ws_breakdown.to_csv(os.path.join(results_dir, "ws_breakdown.csv"), index=False)

    if vehicle_dispatch:
        rows = []
        for sc_name, vehs in vehicle_dispatch.items():
            for v in vehs:
                rows.append({"scenario": sc_name, **v})
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(results_dir, "fleet_dispatch_by_scenario.csv"), index=False)

    print(f"\n✓ Results saved to {results_dir}/")
    print("=" * 80)
    print("PIPELINE v2 COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()