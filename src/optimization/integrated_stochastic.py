"""
Integrated Stochastic Supply Chain Model – Dispatcher
UPDATED: fleet_instances parameter passed through to ExtensiveFormOptimizer

[FLEET]  Added fleet_instances: Optional[List] parameter.
         When provided, passes heterogeneous fleet to ExtensiveFormOptimizer.
         Backward compatible: vehicle_config still accepted as fallback.
[V-3, V-1, V-4, W-3, P-1, P-3, M-1, M-2, M-3]  All fixes inherited (unchanged).
"""

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from optimization.extensive_form_optimizer import ExtensiveFormOptimizer
from optimization.stochastic_procurement   import StochasticProcurementModel
from optimization.weather_vrp              import WeatherAwareVRP
from evaluation.vss_evpi_calculator        import StochasticValidator


class IntegratedStochasticModel:
    """
    Entry point for the complete two-stage stochastic supply chain model.

    Preferred usage: solve_extensive_form()
    Legacy usage:    solve_sequential()  (heuristic)
    """

    def __init__(
        self,
        network:             Dict,
        products_df:         pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df:           pd.DataFrame,
        weather_scenarios:   List,
        vehicle_config:      Optional[Dict] = None,
        fleet_instances:     Optional[List] = None,   # NEW [FLEET]
        risk_aversion:       float = 0.0,
        cvar_alpha:          float = 0.95,
        baseline_ratio:      float = 0.70,
    ):
        self.network             = network
        self.products_df         = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df           = demand_df
        self.scenarios           = weather_scenarios
        self.vehicle_config      = vehicle_config
        self.fleet_instances     = fleet_instances      # [FLEET]
        self.risk_aversion       = risk_aversion
        self.cvar_alpha          = cvar_alpha
        self.baseline_ratio      = baseline_ratio

        fleet_label = (f"{len(fleet_instances)} heterogeneous vehicles"
                       if fleet_instances else "legacy homogeneous config")
        print(f"IntegratedStochasticModel:")
        print(f"  Scenarios: {len(self.scenarios)}")
        print(f"  Fleet: {fleet_label}")
        print(f"  Risk aversion λ: {self.risk_aversion}")

    # ------------------------------------------------------------------
    # PRIMARY METHOD [V-3 FIX + FLEET]
    # ------------------------------------------------------------------
    def solve_extensive_form(
        self,
        time_limit:    int   = 1800,
        gap_tolerance: float = 0.05,
    ) -> Tuple[str, Dict]:
        """
        Solve using TRUE extensive form (single MILP, procurement + VRP coupled).
        Passes fleet_instances to ExtensiveFormOptimizer [FLEET].
        """
        print("\n" + "=" * 80)
        print("INTEGRATED MODEL: EXTENSIVE FORM (heterogeneous fleet)")
        print("=" * 80)

        optimizer = ExtensiveFormOptimizer(
            network              = self.network,
            products_df          = self.products_df,
            supplier_product_df  = self.supplier_product_df,
            demand_df            = self.demand_df,
            weather_scenarios    = self.scenarios,
            vehicle_config       = self.vehicle_config,
            fleet_instances      = self.fleet_instances,   # [FLEET]
            risk_aversion        = self.risk_aversion,
            cvar_alpha           = self.cvar_alpha,
            baseline_ratio       = self.baseline_ratio,
        )

        status, solution = optimizer.solve(
            time_limit=time_limit, gap_tolerance=gap_tolerance)

        if status in ("Optimal", "Feasible"):
            solution["method"]         = "extensive_form"
            solution["risk_aversion"]  = self.risk_aversion
            solution["fleet_size"]     = len(self.fleet_instances) if self.fleet_instances else 0

        return status, solution

    # ------------------------------------------------------------------
    # LEGACY HEURISTIC (kept for quick testing)
    # ------------------------------------------------------------------
    def solve_sequential(
        self,
        time_limit_procurement: int = 600,
        time_limit_vrp:         int = 300,
    ) -> Tuple[str, Dict]:
        """
        ⚠  HEURISTIC — procurement cannot account for VRP costs.
        Use solve_extensive_form() for thesis/research results.
        """
        print("\n⚠  SEQUENTIAL HEURISTIC (use solve_extensive_form() for production)")

        proc_model = StochasticProcurementModel(
            network              = self.network,
            products_df          = self.products_df,
            supplier_product_df  = self.supplier_product_df,
            demand_df            = self.demand_df,
            weather_scenarios    = self.scenarios,
            risk_aversion        = self.risk_aversion,
            baseline_ratio       = self.baseline_ratio,
        )
        proc_status, proc_sol = proc_model.solve(time_limit=time_limit_procurement)
        if proc_status not in ("Optimal", "Feasible"):
            return proc_status, {}

        routing_costs = []
        routing_solutions = {}
        for k, sc in enumerate(self.scenarios):
            vrp = WeatherAwareVRP(
                network              = self.network,
                products_df          = self.products_df,
                demand_df            = self.demand_df,
                procurement_solution = proc_sol["stage1_procurement"],
                weather_scenarios    = self.scenarios,
                vehicle_config       = self.vehicle_config,
            )
            vrp_status, vrp_sol = vrp.solve(scenario_id=k, time_limit=time_limit_vrp)
            cost = vrp_sol.get("objective_value", 0)
            routing_costs.append({
                "scenario": sc.name, "probability": sc.probability, "routing_cost": cost
            })
            routing_solutions[sc.name] = vrp_sol

        routing_df    = pd.DataFrame(routing_costs)
        exp_routing   = (routing_df["routing_cost"] * routing_df["probability"]).sum()
        total         = proc_sol["objective_value"] + exp_routing

        return "Optimal", {
            "method": "sequential_heuristic",
            "status": "Optimal",
            "objective_value": total,
            "procurement_cost": proc_sol["objective_value"],
            "expected_routing_cost": exp_routing,
            "procurement_solution": proc_sol,
            "routing_solutions": routing_solutions,
            "routing_costs_df": routing_df,
        }

    # ------------------------------------------------------------------
    def generate_report(self, solution: Dict) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("INTEGRATED STOCHASTIC SUPPLY CHAIN OPTIMIZATION REPORT")
        lines.append("=" * 80)
        method = solution.get("method", "unknown")
        if method == "extensive_form":
            fleet_sz = solution.get("fleet_size", "?")
            lines.append(f"\nMethod: Extensive Form (fleet: {fleet_sz} heterogeneous vehicles)")
        else:
            lines.append("\nMethod: Sequential Heuristic ⚠")

        sc_df = solution.get("scenario_costs")
        if sc_df is not None and not sc_df.empty:
            lines.append("\nSCENARIO COST BREAKDOWN")
            lines.append("-" * 80)
            for _, row in sc_df.iterrows():
                vrp_c = row.get("vrp_cost", 0)
                fix_c = row.get("vrp_fixed_cost", 0)
                lines.append(
                    f"  {row['scenario_name']:30s}  p={row['probability']:.2f}  "
                    f"total={row['total_cost']:>14,.0f} VND  "
                    f"vrp={vrp_c:>10,.0f} (fix={fix_c:>8,.0f})  "
                    f"ops={row.get('n_operable_vehicles', '?')}veh"
                )
        lines.append("=" * 80)
        return "\n".join(lines)