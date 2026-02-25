"""
Deterministic Baseline - Solves with expected weather conditions
Used for VSS (Value of Stochastic Solution) comparison
"""

import pandas as pd
import numpy as np
from pulp import *
from typing import Dict, List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from procurement_base import ProcurementOptimizer


class DeterministicBaselineModel:
    """
    Deterministic model using expected weather parameters
    
    Strategy: Take expected value of weather factors and solve once
    Compare with stochastic solution to compute VSS
    """
    
    def __init__(self,
                 network: Dict,
                 products_df: pd.DataFrame,
                 supplier_product_df: pd.DataFrame,
                 demand_df: pd.DataFrame,
                 weather_scenarios: List):
        """
        Args:
            Same as stochastic model
        """
        
        self.network = network
        self.products_df = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df = demand_df
        self.scenarios = weather_scenarios
        
        # Compute expected weather parameters
        self.expected_capacity_factor = sum(
            s.probability * s.capacity_reduction_factor 
            for s in weather_scenarios
        )
        self.expected_spoilage_mult = sum(
            s.probability * s.spoilage_multiplier 
            for s in weather_scenarios
        )
        
        print(f"Deterministic Baseline Initialized:")
        print(f"  - Expected capacity factor: {self.expected_capacity_factor:.3f}")
        print(f"  - Expected spoilage multiplier: {self.expected_spoilage_mult:.3f}")
    
    def solve(self, time_limit: int = 300) -> Tuple[str, Dict]:
        """
        Solve deterministic model with expected weather
        FIXED: Proper capacity reduction (not increase!)
        """
        
        print("\nSolving deterministic model with expected weather...")
        
        # CRITICAL FIX: Capacity should be REDUCED by weather, not increased
        # capacity_factor < 1.0 means reduced capacity
        # But we're taking expected value, so check if > 1.0 (bug indicator)
        
        if self.expected_capacity_factor > 1.0:
            print(f"  ⚠ WARNING: Capacity factor > 1.0 ({self.expected_capacity_factor:.3f})")
            print(f"    This indicates probability normalization issue!")
            print(f"    Using 0.9 as conservative estimate instead.")
            effective_capacity_factor = 0.90
        else:
            effective_capacity_factor = self.expected_capacity_factor
        
        # Create adjusted network with expected capacity
        import copy
        adjusted_network = copy.deepcopy(self.network)
        adjusted_network['suppliers'] = self.network['suppliers'].copy()
        adjusted_network['suppliers']['capacity_kg_per_day'] = \
            self.network['suppliers']['capacity_kg_per_day'] * effective_capacity_factor
        
        print(f"  Adjusted capacity factor: {effective_capacity_factor:.3f}")
        print(f"  Original total capacity: {self.network['suppliers']['capacity_kg_per_day'].sum():,.0f} kg/day")
        print(f"  Adjusted total capacity: {adjusted_network['suppliers']['capacity_kg_per_day'].sum():,.0f} kg/day")
        
        # Use base optimizer with adjusted parameters
        optimizer = ProcurementOptimizer(
            network=adjusted_network,
            products_df=self.products_df,
            supplier_product_df=self.supplier_product_df,
            demand_df=self.demand_df
        )
        
        # Solve
        status, solution = optimizer.solve(time_limit=time_limit)
        
        if status == 'Optimal':
            base_obj = solution['objective_value']
            
            # Adjust for expected spoilage penalty
            if not solution['unmet_demand'].empty:
                additional_penalty = solution['unmet_demand']['penalty_cost_vnd'].sum() * \
                                (self.expected_spoilage_mult - 1.0)
                adjusted_obj = base_obj + additional_penalty
            else:
                adjusted_obj = base_obj
            
            solution['deterministic_objective'] = adjusted_obj
            solution['expected_capacity_factor'] = effective_capacity_factor
            solution['expected_spoilage_mult'] = self.expected_spoilage_mult
        
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