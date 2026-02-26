"""
Integrated Stochastic Supply Chain Model – Dispatcher
FIXED VERSION

Fixes applied
─────────────
[HIGH / V-3]  Sequential solve replaced by true extensive form.
    OLD: IntegratedStochasticModel.solve_sequential() → StochasticProcurementModel
         then WeatherAwareVRP per scenario (procurement fixed, routing cannot feed back).
    NEW: IntegratedStochasticModel delegates to ExtensiveFormOptimizer, which places
         both Stage 1 (procurement) and Stage 2 (routing) variables in a SINGLE MILP.
         Stage 1 procurement decisions are optimised with full knowledge of routing costs
         across all scenarios.

    solve_sequential() is kept for backward compatibility and quick testing,
    but is clearly marked as HEURISTIC / NOT RECOMMENDED for production.

[V-1, V-4, W-3, P-1, P-3, M-1, M-2, M-3 fixes]
    All inherited from ExtensiveFormOptimizer and StochasticValidator.
"""

import time
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from optimization.extensive_form_optimizer import ExtensiveFormOptimizer
from optimization.stochastic_procurement import StochasticProcurementModel
from optimization.weather_vrp import WeatherAwareVRP
from evaluation.vss_evpi_calculator import StochasticValidator


class IntegratedStochasticModel:
    """
    Entry point for the complete two-stage stochastic supply chain model.

    Preferred usage: solve_extensive_form()
    Legacy usage:    solve_sequential()  (heuristic, V-3 NOT fixed)
    """

    def __init__(
        self,
        network: Dict,
        products_df: pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        weather_scenarios: List,
        vehicle_config: Dict = None,
        risk_aversion: float = 0.0,
        cvar_alpha: float = 0.95,
        baseline_ratio: float = 0.70,
    ):
        self.network = network
        self.products_df = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df = demand_df
        self.scenarios = weather_scenarios
        self.vehicle_config = vehicle_config
        self.risk_aversion = risk_aversion
        self.cvar_alpha = cvar_alpha
        self.baseline_ratio = baseline_ratio

        print(f"IntegratedStochasticModel:")
        print(f"  Scenarios: {len(self.scenarios)}")
        print(f"  Risk aversion λ: {self.risk_aversion}")

    # ------------------------------------------------------------------
    # PRIMARY METHOD [V-3 FIX]
    # ------------------------------------------------------------------
    def solve_extensive_form(
        self,
        time_limit: int = 1800,
        gap_tolerance: float = 0.05,
    ) -> Tuple[str, Dict]:
        """
        Solve using TRUE extensive form (single MILP, procurement + VRP coupled).

        [V-3 FIX] This is the theoretically correct method.
        Stage 1 procurement is optimised accounting for all Stage 2 routing costs.

        Returns
        ───────
        (status, solution_dict)
        """
        print("\n" + "=" * 80)
        print("INTEGRATED MODEL: EXTENSIVE FORM (V-3 fixed)")
        print("=" * 80)

        optimizer = ExtensiveFormOptimizer(
            network=self.network,
            products_df=self.products_df,
            supplier_product_df=self.supplier_product_df,
            demand_df=self.demand_df,
            weather_scenarios=self.scenarios,
            vehicle_config=self.vehicle_config,
            risk_aversion=self.risk_aversion,
            cvar_alpha=self.cvar_alpha,
            baseline_ratio=self.baseline_ratio,
        )

        status, solution = optimizer.solve(
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
        )

        if status in ("Optimal", "Feasible"):
            solution["method"] = "extensive_form"
            solution["risk_aversion"] = self.risk_aversion

        return status, solution

    # ------------------------------------------------------------------
    # LEGACY METHOD (heuristic; kept for quick comparison / testing)
    # ------------------------------------------------------------------
    def solve_sequential(
        self,
        time_limit_procurement: int = 600,
        time_limit_vrp: int = 300,
    ) -> Tuple[str, Dict]:
        """
        ⚠ HEURISTIC METHOD — NOT the theoretically correct two-stage approach.

        Procurement and routing are solved separately:
          1. StochasticProcurementModel (Stage 1 + recourse without routing).
          2. WeatherAwareVRP per scenario with fixed procurement.

        Procurement cannot account for VRP infeasibility costs → suboptimal.
        Use solve_extensive_form() for research / thesis results.

        Retained for:
          - Quick feasibility checks
          - Comparison baseline
          - Testing data pipelines before running the full extensive form
        """
        print("\n" + "=" * 80)
        print("INTEGRATED MODEL: SEQUENTIAL (heuristic — use solve_extensive_form() for thesis)")
        print("=" * 80)

        # Step 1: Procurement
        print("\n--- Step 1: Stochastic Procurement (fixed Stage 1) ---")
        proc_model = StochasticProcurementModel(
            network=self.network,
            products_df=self.products_df,
            supplier_product_df=self.supplier_product_df,
            demand_df=self.demand_df,
            weather_scenarios=self.scenarios,
            risk_aversion=self.risk_aversion,
            baseline_ratio=self.baseline_ratio,
        )
        proc_status, proc_sol = proc_model.solve(time_limit=time_limit_procurement)

        if proc_status not in ("Optimal", "Feasible"):
            print(f"⚠ Procurement failed: {proc_status}")
            return proc_status, {}

        print(f"  ✓ Procurement: {proc_sol['objective_value']:,.0f} VND")

        # Step 2: VRP per scenario (finite cost guaranteed by V-1 fix)
        print("\n--- Step 2: Weather-Aware VRP (per scenario) ---")
        routing_costs = []
        routing_solutions = {}

        for k, sc in enumerate(self.scenarios):
            vrp = WeatherAwareVRP(
                network=self.network,
                products_df=self.products_df,
                demand_df=self.demand_df,
                procurement_solution=proc_sol["stage1_procurement"],
                weather_scenarios=self.scenarios,
                vehicle_config=self.vehicle_config,
            )
            vrp_status, vrp_sol = vrp.solve(scenario_id=k, time_limit=time_limit_vrp)

            # [V-1 FIX] vrp_sol["objective_value"] is always finite now
            cost = vrp_sol.get("objective_value", 0)
            routing_costs.append(
                {"scenario": sc.name, "probability": sc.probability, "routing_cost": cost}
            )
            routing_solutions[sc.name] = vrp_sol

        routing_df = pd.DataFrame(routing_costs)
        expected_routing = (routing_df["routing_cost"] * routing_df["probability"]).sum()
        total = proc_sol["objective_value"] + expected_routing

        print(f"\n  Procurement:      {proc_sol['objective_value']:>15,.0f} VND")
        print(f"  Expected routing: {expected_routing:>15,.0f} VND")
        print(f"  Total:            {total:>15,.0f} VND")

        solution = {
            "method": "sequential_heuristic",
            "status": "Optimal",
            "objective_value": total,
            "procurement_cost": proc_sol["objective_value"],
            "expected_routing_cost": expected_routing,
            "procurement_solution": proc_sol,
            "routing_solutions": routing_solutions,
            "routing_costs_df": routing_df,
            "risk_aversion": self.risk_aversion,
        }
        return "Optimal", solution

    # ------------------------------------------------------------------
    def generate_report(self, solution: Dict) -> str:
        """Generate formatted solution report."""
        lines = []
        lines.append("=" * 80)
        lines.append("INTEGRATED STOCHASTIC SUPPLY CHAIN OPTIMIZATION REPORT")
        lines.append("=" * 80)

        method = solution.get("method", "unknown")
        lines.append(
            f"\nMethod: {'Extensive Form (theoretically correct)' if method == 'extensive_form' else 'Sequential Heuristic'}"
        )
        if method == "sequential_heuristic":
            lines.append("  ⚠ Sequential heuristic — procurement does not account for routing costs.")
            lines.append("  Use solve_extensive_form() for thesis-quality results.")

        lines.append("\n1. COST BREAKDOWN")
        lines.append("-" * 80)

        if method == "extensive_form":
            sc_df = solution.get("scenario_costs", pd.DataFrame())
            if not sc_df.empty:
                s1 = sc_df["stage1_cost"].iloc[0]
                exp_vrp = (sc_df["vrp_cost"] * sc_df["probability"]).sum()
                exp_em = (sc_df["emergency_cost"] * sc_df["probability"]).sum()
                exp_sp = (sc_df["spoilage_cost"] * sc_df["probability"]).sum()
                exp_pen = (sc_df["penalty_cost"] * sc_df["probability"]).sum()
                lines.append(f"  Stage 1 procurement:    {s1:>15,.0f} VND")
                lines.append(f"  Expected VRP routing:   {exp_vrp:>15,.0f} VND")
                lines.append(f"  Expected emergency:     {exp_em:>15,.0f} VND")
                lines.append(f"  Expected spoilage loss: {exp_sp:>15,.0f} VND")
                lines.append(f"  Expected unmet penalty: {exp_pen:>15,.0f} VND")
                lines.append(f"  {'='*35}")
                lines.append(f"  Total expected cost:    {solution['objective_value']:>15,.0f} VND")
        else:
            lines.append(f"  Procurement (Stage 1):  {solution.get('procurement_cost', 0):>15,.0f} VND")
            lines.append(f"  Expected routing:       {solution.get('expected_routing_cost', 0):>15,.0f} VND")
            lines.append(f"  Total:                  {solution['objective_value']:>15,.0f} VND")

        lines.append("\n2. SCENARIO COST BREAKDOWN")
        lines.append("-" * 80)
        sc_df = solution.get("scenario_costs")
        if sc_df is not None and not sc_df.empty:
            for _, row in sc_df.iterrows():
                lines.append(
                    f"  {row['scenario_name']:30s}  p={row['probability']:.2f}  "
                    f"total={row['total_cost']:>14,.0f} VND"
                )
        elif "routing_costs_df" in solution:
            for _, row in solution["routing_costs_df"].iterrows():
                lines.append(
                    f"  {row['scenario']:30s}  p={row['probability']:.2f}  "
                    f"routing={row['routing_cost']:>14,.0f} VND"
                )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)