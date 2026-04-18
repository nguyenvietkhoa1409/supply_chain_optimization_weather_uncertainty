"""
integrated_stochastic.py — Dispatcher (updated: two-phase method added)
========================================================================
Added method:  solve_two_phase_extensive_form()
  → Calls TwoPhaseExtensiveFormOptimizer (Phase 2A + Phase 2B coupled MILP)

All original methods preserved for backward compatibility.
"""

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from optimization.extensive_form_optimizer import ExtensiveFormOptimizer
from optimization.stochastic_procurement   import StochasticProcurementModel
from optimization.weather_vrp              import WeatherAwareVRP
from evaluation.vss_evpi_calculator        import StochasticValidator

# ── Import new two-phase optimizer ────────────────────────────────────────────
try:
    from optimization.two_phase_optimizer import TwoPhaseExtensiveFormOptimizer
    _TWO_PHASE_AVAILABLE = True
except ImportError:
    _TWO_PHASE_AVAILABLE = False


class IntegratedStochasticModel:
    """
    Entry point for the complete two-stage stochastic supply chain model.

    Methods
    -------
    solve_two_phase_extensive_form()   [NEW — faithful to Patel et al. 2024]
        Phase 2A: DC → Suppliers → DC  (procurement VRP)
        Phase 2B: DC → Stores → DC     (distribution VRP)
        Linked by DC inventory balance.

    solve_extensive_form()             [ORIGINAL — DC→Stores only, working]
    solve_sequential()                 [LEGACY HEURISTIC]
    """

    def __init__(
        self,
        network:             Dict,
        products_df:         pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df:           pd.DataFrame,
        weather_scenarios:   List,
        vehicle_config:      Optional[Dict] = None,
        fleet_instances:     Optional[List] = None,
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
        self.fleet_instances     = fleet_instances
        self.risk_aversion       = risk_aversion
        self.cvar_alpha          = cvar_alpha
        self.baseline_ratio      = baseline_ratio

        fleet_label = (f"{len(fleet_instances)} heterogeneous vehicles"
                       if fleet_instances else "legacy config")
        print(f"IntegratedStochasticModel:")
        print(f"  Scenarios: {len(self.scenarios)}")
        print(f"  Fleet: {fleet_label}")
        print(f"  Risk aversion λ: {self.risk_aversion}")

    # ==========================================================================
    # NEW: Two-Phase Extensive Form (Patel et al. 2024 architecture)
    # ==========================================================================
    def solve_two_phase_extensive_form(
        self,
        time_limit:         int   = 1800,
        gap_tolerance:      float = 0.05,
        unmet_penalty:      float = 500_000.0,
        refrig_penalty:     float = 1.5,
    ) -> Tuple[str, Dict]:
        """
        Solve using TRUE two-phase extensive form.

        Phase 2A — Procurement VRP: DC → Suppliers → DC
          Vehicles physically collect from accessible suppliers.
          inventory[k,p] = actual goods collected at DC.

        Phase 2B — Distribution VRP: DC → Stores → DC
          Vehicles deliver from DC inventory to stores.
          Demand satisfied from inventory + emergency + unmet slack.

        This correctly implements Patel et al. (2024) two-stage model:
          Stage 1 = procurement decisions (x[s,p])
          Stage 2A = procurement routing (per scenario)
          Stage 2B = distribution routing (per scenario)

        Parameters
        ----------
        unmet_penalty  : VND per unit unmet — keep ≥ 5× avg product cost
        refrig_penalty : spoilage multiplier for non-ref carrying ref products (soft)
        """
        if not _TWO_PHASE_AVAILABLE:
            raise ImportError(
                "TwoPhaseExtensiveFormOptimizer not found. "
                "Ensure src/optimization/two_phase_optimizer.py is present."
            )
        if self.fleet_instances is None:
            raise ValueError(
                "fleet_instances required for two-phase mode. "
                "Pass fleet_instances=fleet_optimizer to constructor."
            )

        print("\n" + "=" * 80)
        print("INTEGRATED MODEL: TWO-PHASE EXTENSIVE FORM  (Patel et al. 2024)")
        print("  Phase 2A: DC → Suppliers → DC  (procurement VRP)")
        print("  Phase 2B: DC → Stores   → DC  (distribution VRP)")
        print("=" * 80)

        optimizer = TwoPhaseExtensiveFormOptimizer(
            network              = self.network,
            products_df          = self.products_df,
            supplier_product_df  = self.supplier_product_df,
            demand_df            = self.demand_df,
            weather_scenarios    = self.scenarios,
            fleet_instances      = self.fleet_instances,
            baseline_ratio       = self.baseline_ratio,
        )

        status, solution = optimizer.solve(
            time_limit=time_limit, gap_tolerance=gap_tolerance
        )

        if status in ("Optimal", "Feasible"):
            solution["method"]         = "two_phase_extensive_form"
            solution["risk_aversion"]  = self.risk_aversion
            solution["fleet_size"]     = len(self.fleet_instances)

        return status, solution

    # ==========================================================================
    # ORIGINAL: Extensive Form (DC→Stores, working baseline)
    # ==========================================================================
    def solve_extensive_form(
        self,
        time_limit:    int   = 1800,
        gap_tolerance: float = 0.05,
    ) -> Tuple[str, Dict]:
        """
        Original working extensive form (DC → Stores routing only).
        Supplier delivery to DC is implicit.
        Keep as baseline for VSS/EVPI comparison.
        """
        print("\n" + "=" * 80)
        print("INTEGRATED MODEL: EXTENSIVE FORM (heterogeneous fleet — original)")
        print("=" * 80)

        optimizer = ExtensiveFormOptimizer(
            network              = self.network,
            products_df          = self.products_df,
            supplier_product_df  = self.supplier_product_df,
            demand_df            = self.demand_df,
            weather_scenarios    = self.scenarios,
            vehicle_config       = self.vehicle_config,
            fleet_instances      = self.fleet_instances,
            risk_aversion        = self.risk_aversion,
            cvar_alpha           = self.cvar_alpha,
            baseline_ratio       = self.baseline_ratio,
        )

        status, solution = optimizer.solve(
            time_limit=time_limit, gap_tolerance=gap_tolerance
        )

        if status in ("Optimal", "Feasible"):
            solution["method"]        = "extensive_form"
            solution["risk_aversion"] = self.risk_aversion
            solution["fleet_size"]    = (len(self.fleet_instances)
                                          if self.fleet_instances else 0)

        return status, solution

    # ==========================================================================
    # LEGACY: Sequential heuristic
    # ==========================================================================
    def solve_sequential(
        self,
        time_limit_procurement: int = 600,
        time_limit_vrp:         int = 300,
    ) -> Tuple[str, Dict]:
        """
        Sequential heuristic (procurement then VRP separately).
        Kept for quick testing only.
        """
        print("\n⚠  SEQUENTIAL HEURISTIC — use solve_two_phase_extensive_form() for research")

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

        routing_costs     = []
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
            routing_costs.append(dict(
                scenario=sc.name, probability=sc.probability, routing_cost=cost
            ))
            routing_solutions[sc.name] = vrp_sol

        routing_df  = pd.DataFrame(routing_costs)
        exp_routing = (routing_df["routing_cost"] * routing_df["probability"]).sum()
        total       = proc_sol["objective_value"] + exp_routing

        return "Optimal", dict(
            method                  = "sequential_heuristic",
            status                  = "Optimal",
            objective_value         = total,
            procurement_cost        = proc_sol["objective_value"],
            expected_routing_cost   = exp_routing,
            procurement_solution    = proc_sol,
            routing_solutions       = routing_solutions,
            routing_costs_df        = routing_df,
        )

    # ==========================================================================
    # Report helper (works for both methods)
    # ==========================================================================
    def generate_report(self, solution: Dict) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("INTEGRATED STOCHASTIC SUPPLY CHAIN OPTIMIZATION REPORT")
        lines.append("=" * 80)
        method = solution.get("method", "unknown")
        if method == "two_phase_extensive_form":
            lines.append("\nMethod: Two-Phase Extensive Form  [Patel et al. 2024]")
            lines.append("  Phase 2A: DC → Suppliers → DC  (procurement VRP)")
            lines.append("  Phase 2B: DC → Stores → DC    (distribution VRP)")
        elif method == "extensive_form":
            lines.append("\nMethod: Extensive Form (original — DC→Stores)")
        else:
            lines.append("\nMethod: Sequential Heuristic ⚠")

        sc_df = solution.get("scenario_costs")
        if sc_df is not None and not sc_df.empty:
            lines.append("\nSCENARIO COST BREAKDOWN")
            lines.append("-" * 80)
            for _, row in sc_df.iterrows():
                proc_c = row.get("proc_vrp_cost", row.get("vrp_fixed_cost", 0))
                dist_c = row.get("dist_vrp_cost", row.get("vrp_variable_cost", 0))
                lines.append(
                    f"  {row['scenario_name']:35s}  p={row['probability']:.2f}  "
                    f"total={row['total_cost']:>14,.0f} VND  "
                    f"procVRP={proc_c:>9,.0f}  distVRP={dist_c:>9,.0f}  "
                    f"ops={row.get('n_operable_vehicles','?')}veh"
                )
        lines.append("=" * 80)
        return "\n".join(lines)
