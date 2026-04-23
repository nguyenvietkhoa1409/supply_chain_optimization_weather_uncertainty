"""
Deterministic Baseline - Solves with expected weather conditions
Used for VSS (Value of Stochastic Solution) comparison

FIXED: Now uses TwoPhaseExtensiveFormOptimizer with a single
expected scenario (probability-weighted average of all scenarios).
This guarantees the same feasible set and objective structure as RP,
so that EEV ≥ RP (ordering property) is mathematically guaranteed.
"""

import copy
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DeterministicBaselineModel:
    """
    Deterministic EV model using expected weather parameters.

    Strategy: Construct a single synthetic "expected" scenario from the
    probability-weighted average of all weather scenario attributes,
    then solve TwoPhaseExtensiveFormOptimizer with K=1 and probability=1.0.

    This is the correct EV model per Birge & Louveaux (2011) §4.1:
    the EV solution x*_EV must be feasible in the SAME model as RP.
    """

    def __init__(self,
                 network: Dict,
                 products_df: pd.DataFrame,
                 supplier_product_df: pd.DataFrame,
                 demand_df: pd.DataFrame,
                 weather_scenarios: List,
                 fleet_instances: List = None,
                 concentration_max: float = 0.30):

        self.network               = network
        self.products_df           = products_df
        self.supplier_product_df   = supplier_product_df
        self.demand_df             = demand_df
        self.scenarios             = weather_scenarios
        self.fleet_instances       = fleet_instances
        self.concentration_max     = concentration_max

        # Build expected scenario (probability-weighted average)
        self._expected_scenario = self._build_expected_scenario()

        print(f"Deterministic Baseline (EV model) Initialized:")
        sc = self._expected_scenario
        print(f"  Expected severity : {sc.severity_level:.2f}")
        print(f"  Speed factor      : {sc.speed_reduction_factor:.3f}")
        print(f"  Spoilage mult     : {sc.spoilage_multiplier:.3f}")

    def _build_expected_scenario(self):
        """
        Build a single synthetic scenario by probability-weighting all attributes.
        Accessibility is set to the expected fraction (rounded to 1 = accessible if ≥ 0.5).
        """
        import copy

        # weighted average of continuous attributes
        p_total = sum(s.probability for s in self.scenarios)
        probs   = [s.probability / p_total for s in self.scenarios]

        # Use first scenario as template, then overwrite fields
        ev_sc = copy.deepcopy(self.scenarios[0])
        ev_sc.name              = "EV_Expected"
        ev_sc.probability       = 1.0   # single scenario → full weight

        # Weighted averages
        ev_sc.severity_level         = round(sum(p * s.severity_level
                                                  for p, s in zip(probs, self.scenarios)))
        ev_sc.speed_reduction_factor = sum(p * s.speed_reduction_factor
                                           for p, s in zip(probs, self.scenarios))
        ev_sc.capacity_reduction_factor = sum(p * s.capacity_reduction_factor
                                              for p, s in zip(probs, self.scenarios))
        ev_sc.spoilage_multiplier    = sum(p * s.spoilage_multiplier
                                           for p, s in zip(probs, self.scenarios))

        # Emergency: feasible only if majority of scenarios allow it
        ev_sc.emergency_feasible = (sum(p for p, s in zip(probs, self.scenarios)
                                        if getattr(s, "emergency_feasible", True)) >= 0.5)

        # Clamp severity to a valid integer (1-5)
        ev_sc.severity_level = max(1, min(5, int(ev_sc.severity_level)))

        return ev_sc

    def solve(self, time_limit: int = 300) -> Tuple[str, Dict]:
        """
        Solve EV model using TwoPhaseExtensiveFormOptimizer with the
        single expected scenario. Returns a solution dict with key
        'stage1_procurement' compatible with compute_eev().
        """
        print("\nSolving EV deterministic model (TwoPhase, expected scenario)...")

        if self.fleet_instances is None:
            raise ValueError(
                "DeterministicBaselineModel requires fleet_instances. "
                "Pass fleet_instances= in the constructor."
            )

        from optimization.two_phase_optimizer import TwoPhaseExtensiveFormOptimizer

        optimizer = TwoPhaseExtensiveFormOptimizer(
            network              = self.network,
            products_df          = self.products_df,
            supplier_product_df  = self.supplier_product_df,
            demand_df            = self.demand_df,
            weather_scenarios    = [self._expected_scenario],   # K = 1
            fleet_instances      = self.fleet_instances,
            concentration_max    = self.concentration_max,      # same as RP
        )

        status, solution = optimizer.solve(
            time_limit    = time_limit,
            gap_tolerance = 0.05,
        )

        if status in ("Optimal", "Feasible"):
            print(f"  ✓ EV Stage-1 solved: obj = {solution.get('objective_value', 0):,.0f} VND")
        else:
            print(f"  ⚠ EV solve failed: {status}")

        return status, solution


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../..')
    
    from data_generation.network_generator import DaNangNetworkGenerator
    from data_generation.product_generator import ProductCatalogGenerator
    from data_generation.demand_generator import DemandPatternGenerator
    from weather.manual_scenarios import ManualWeatherScenarios
    
    print("="*80)
    print("DETERMINISTIC BASELINE - TEST")
    print("="*80)
    
    # Load data
    network_gen = DaNangNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=6, n_dcs=2, n_stores=5)
    
    product_gen = ProductCatalogGenerator(seed=42)
    products = product_gen.generate_products(n_products=10)
    supplier_product = product_gen.generate_supplier_product_matrix(
        network['suppliers'], products
    )
    
    demand_gen = DemandPatternGenerator(seed=42)
    daily_demand = demand_gen.generate_demand_plan(
        network['stores'], products, planning_horizon_days=7
    )
    weekly_demand = demand_gen.aggregate_to_weekly(daily_demand)
    
    scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    
    # Solve deterministic
    det_model = DeterministicBaselineModel(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        weather_scenarios=scenarios
    )
    
    status, solution = det_model.solve()
    
    if status == 'Optimal':
        print("\n" + "="*80)
        print("DETERMINISTIC SOLUTION")
        print("="*80)
        
        print(f"\nDeterministic Objective: {solution['deterministic_objective']:,.0f} VND")
        print(f"Base Objective: {solution['objective_value']:,.0f} VND")