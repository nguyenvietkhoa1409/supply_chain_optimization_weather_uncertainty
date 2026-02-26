"""
CVaR-Based Risk-Averse Procurement - Two-Stage Stochastic MILP
Incorporates Conditional Value-at-Risk for risk-averse decision making

Objective: min (1-λ)·E[Cost] + λ·CVaR_α[Cost]

where:
- λ ∈ [0,1]: risk aversion parameter (0=risk-neutral, 1=fully risk-averse)
- α: confidence level (typically 0.90 or 0.95)
- CVaR_α: Conditional Value-at-Risk (expected cost in worst α% scenarios)
"""

import pandas as pd
import numpy as np
from pulp import *
from typing import Dict, List, Tuple
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from stochastic_procurement import StochasticProcurementModel


class CVaRProcurementModel(StochasticProcurementModel):
    """
    CVaR-based risk-averse extension of stochastic procurement
    
    Inherits from StochasticProcurementModel and adds CVaR risk measure
    """
    
    def __init__(self,
                 network: Dict,
                 products_df: pd.DataFrame,
                 supplier_product_df: pd.DataFrame,
                 demand_df: pd.DataFrame,
                 weather_scenarios: List,
                 risk_aversion: float = 0.3,
                 cvar_alpha: float = 0.90):
        """
        Args:
            risk_aversion: λ ∈ [0,1], weight on CVaR (0=risk-neutral, 1=only CVaR)
            cvar_alpha: α ∈ [0,1], confidence level (0.90 = focus on worst 10%)
        """
        
        super().__init__(network, products_df, supplier_product_df, 
                        demand_df, weather_scenarios, risk_aversion)
        
        self.cvar_alpha = cvar_alpha
        
        print(f"  - CVaR confidence level (α): {self.cvar_alpha}")
        print(f"  - Risk measure: (1-λ)·E[Cost] + λ·CVaR_α[Cost]")
        print(f"    where λ={self.risk_aversion}, α={self.cvar_alpha}")
    
    def build_model(self) -> Tuple[LpProblem, Dict]:
        """
        Build CVaR-based two-stage stochastic MILP
        
        CVaR formulation (Rockafellar & Uryasev, 2000):
        CVaR_α = min_η { η + (1/(1-α)) * E[max(Cost - η, 0)] }
        """
        
        print("\nBuilding CVaR-based stochastic MILP...")
        
        model = LpProblem("CVaR_Procurement_Weather", LpMinimize)
        
        # =================================================================
        # STAGE 1 VARIABLES (same as base model)
        # =================================================================
        print("  Creating Stage 1 variables...")
        
        x = LpVariable.dicts("stage1_procure",
                            ((s, p) for s in self.suppliers for p in self.products),
                            lowBound=0,
                            cat='Continuous')
        
        y = LpVariable.dicts("stage1_activate",
                            ((s, p) for s in self.suppliers for p in self.products),
                            cat='Binary')
        
        # =================================================================
        # STAGE 2 VARIABLES (same as base model)
        # =================================================================
        print("  Creating Stage 2 variables (per scenario)...")
        
        x_extra = LpVariable.dicts("stage2_emergency",
                                   ((k, s, p) 
                                    for k in range(len(self.scenarios))
                                    for s in self.suppliers 
                                    for p in self.products),
                                   lowBound=0,
                                   cat='Continuous')
        
        u = LpVariable.dicts("stage2_unmet",
                            ((k, p) 
                             for k in range(len(self.scenarios))
                             for p in self.products),
                            lowBound=0,
                            cat='Continuous')
        
        # =================================================================
        # CVaR VARIABLES
        # =================================================================
        print("  Creating CVaR variables...")
        
        # η: Value-at-Risk (VaR) threshold
        eta = LpVariable("cvar_eta", lowBound=0, cat='Continuous')
        
        # z[k]: Auxiliary variable for CVaR = max(Cost_k - η, 0)
        z = LpVariable.dicts("cvar_z",
                            range(len(self.scenarios)),
                            lowBound=0,
                            cat='Continuous')
        
        # =================================================================
        # OBJECTIVE FUNCTION with CVaR
        # =================================================================
        print("  Formulating CVaR objective...")
        
        # Stage 1 cost
        stage1_variable_cost = lpSum([
            self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False)
        ])
        
        stage1_fixed_cost = lpSum([
            self.supplier_fixed_cost[s] * y[s, p]
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False)
        ])
        
        # Stage 2 costs per scenario
        scenario_costs = []
        
        for k, scenario in enumerate(self.scenarios):
            prob = scenario.probability
            spoilage_mult = scenario.spoilage_multiplier
            
            # Emergency cost
            emergency_cost = lpSum([
                2.0 * self.sp_cost.get((s, p), self.product_cost[p]) * x_extra[k, s, p]
                for s in self.suppliers
                for p in self.products
                if self.sp_available.get((s, p), False)
            ])
            
            # Penalty cost
            penalty_multiplier = min(10.0, 5.0 * spoilage_mult)
            penalty_cost = lpSum([
                penalty_multiplier * self.product_cost[p] * u[k, p]
                for p in self.products
            ])
            
            scenario_cost = emergency_cost + penalty_cost
            scenario_costs.append((prob, scenario_cost))
        
        # Expected Stage 2 cost
        expected_stage2 = lpSum([prob * cost for prob, cost in scenario_costs])
        
        # CVaR calculation: η + (1/(1-α)) * E[max(Cost - η, 0)]
        cvar_term = eta + (1.0 / (1.0 - self.cvar_alpha)) * lpSum([
            prob * z[k] for k, (prob, _) in enumerate(scenario_costs)
        ])
        
        # Total objective: (1-λ)·E[Cost] + λ·CVaR
        total_cost = stage1_variable_cost + stage1_fixed_cost + expected_stage2
        
        objective = (1.0 - self.risk_aversion) * total_cost + self.risk_aversion * cvar_term
        
        model += objective, "CVaR_Objective"
        
        # =================================================================
        # CVaR CONSTRAINTS
        # =================================================================
        print("  Adding CVaR constraints...")
        
        # For each scenario: z[k] >= Cost[k] - η
        for k, (_, scenario_cost) in enumerate(scenario_costs):
            total_scenario_cost = stage1_variable_cost + stage1_fixed_cost + scenario_cost
            
            model += (
                z[k] >= total_scenario_cost - eta,
                f"CVaR_Definition_{k}"
            )
        
        # =================================================================
        # STAGE 1 & STAGE 2 CONSTRAINTS (same as base model)
        # =================================================================
        print("  Adding Stage 1 and Stage 2 constraints...")
        
        # Stage 1: Supplier capacity
        for s in self.suppliers:
            capacity = self.supplier_capacity[s]
            model += (
                lpSum([x[s, p] * self.product_weight[p] 
                      for p in self.products
                      if self.sp_available.get((s, p), False)]) <= capacity,
                f"S1_Capacity_{s}"
            )
        
        # Stage 1: MOQ logic
        M = 100000
        for s in self.suppliers:
            for p in self.products:
                if self.sp_available.get((s, p), False):
                    moq = self.sp_moq.get((s, p), 0)
                    model += (x[s, p] >= moq * y[s, p], f"S1_MOQ_lower_{s}_{p}")
                    model += (x[s, p] <= M * y[s, p], f"S1_MOQ_upper_{s}_{p}")
        
        # Stage 2: Per scenario
        for k, scenario in enumerate(self.scenarios):
            capacity_factor = scenario.capacity_reduction_factor
            severity = scenario.severity_level
            
            # Survival rates
            survival_rates = {1: 0.99, 2: 0.97, 3: 0.92, 4: 0.82, 5: 0.70}
            base_survival = survival_rates.get(severity, 0.90)
            
            # Demand satisfaction
            for p in self.products:
                demand = self.total_demand.get(p, 0)
                
                if demand > 0:
                    # Product-specific survival
                    prod_row = self.products_df[self.products_df['id'] == p].iloc[0]
                    temp_sensitivity = prod_row['temperature_sensitivity']
                    
                    if temp_sensitivity == 'high':
                        survival_rate = base_survival
                    elif temp_sensitivity == 'medium':
                        survival_rate = base_survival + (1.0 - base_survival) * 0.5
                    else:
                        survival_rate = base_survival + (1.0 - base_survival) * 0.75
                    
                    # Stage 1 effective + emergency
                    stage1_effective = lpSum([
                        x[s, p] * survival_rate
                        for s in self.suppliers
                        if self.sp_available.get((s, p), False)
                    ])
                    
                    emergency_supply = lpSum([
                        x_extra[k, s, p]
                        for s in self.suppliers
                        if self.sp_available.get((s, p), False)
                    ])
                    
                    model += (
                        stage1_effective + emergency_supply + u[k, p] >= demand,
                        f"S2_Demand_{k}_{p}"
                    )
            
            # Emergency capacity
            for s in self.suppliers:
                base_capacity = self.supplier_capacity[s]
                adjusted_capacity = base_capacity * capacity_factor
                
                model += (
                    lpSum([x_extra[k, s, p] * self.product_weight[p]
                          for p in self.products
                          if self.sp_available.get((s, p), False)]) <= adjusted_capacity * 0.8,
                    f"S2_Emergency_Capacity_{k}_{s}"
                )
        
        var_counts = model.numVariables()
        const_counts = model.numConstraints()
        
        print(f"✓ CVaR model built:")
        print(f"  - Variables: {var_counts}")
        print(f"  - Constraints: {const_counts}")
        print(f"  - Risk-averse objective with λ={self.risk_aversion}")
        
        return model, {'x': x, 'y': y, 'x_extra': x_extra, 'u': u, 'eta': eta, 'z': z}
    
    def solve(self, time_limit: int = 600, gap_tolerance: float = 0.02) -> Tuple[str, Dict]:
        """Solve CVaR model and extract solution"""
        
        model, vars_dict = self.build_model()
        
        solver = PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=gap_tolerance,
            msg=1
        )
        
        print(f"\nSolving CVaR model (λ={self.risk_aversion}, α={self.cvar_alpha})...")
        start_time = time.time()
        model.solve(solver)
        solve_time = time.time() - start_time
        
        status = LpStatus[model.status]
        print(f"\n✓ Status: {status}")
        print(f"✓ Solve time: {solve_time:.2f} seconds")
        
        if status in ['Optimal', 'Feasible']:
            obj_value = value(model.objective)
            print(f"✓ Objective value: {obj_value:,.0f} VND")
            
            # Extract CVaR metrics
            eta_value = value(vars_dict['eta'])
            print(f"✓ VaR (η): {eta_value:,.0f} VND")
            
            solution = self._extract_solution(vars_dict)
            solution['objective_value'] = obj_value
            solution['solve_time'] = solve_time
            solution['status'] = status
            solution['var_threshold'] = eta_value
            solution['risk_aversion'] = self.risk_aversion
            solution['cvar_alpha'] = self.cvar_alpha
            
            # Compute scenario costs
            solution['scenario_costs'] = self._compute_scenario_costs(vars_dict)
            
            return status, solution
        else:
            print(f"⚠ Model status: {status}")
            return status, {}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../..')
    
    from data_generation.network_generator import DaNangNetworkGenerator
    from data_generation.product_generator import ProductCatalogGenerator
    from data_generation.demand_generator import DemandPatternGenerator
    from weather.manual_scenarios import ManualWeatherScenarios
    
    print("="*80)
    print("CVaR PROCUREMENT MODEL - TEST")
    print("="*80)
    
    # Load data
    network_gen = DaNangNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=6, n_dcs=2, n_stores=6)
    
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
    
    # Test with different risk aversions
    for lambda_val in [0.0, 0.3, 0.7, 1.0]:
        print(f"\n{'='*80}")
        print(f"Testing with λ = {lambda_val}")
        print(f"{'='*80}")
        
        model = CVaRProcurementModel(
            network=network,
            products_df=products,
            supplier_product_df=supplier_product,
            demand_df=weekly_demand,
            weather_scenarios=scenarios,
            risk_aversion=lambda_val,
            cvar_alpha=0.90
        )
        
        status, solution = model.solve(time_limit=300)
        
        if status in ['Optimal', 'Feasible']:
            print(f"\n✓ Objective: {solution['objective_value']:,.0f} VND")
            print(f"✓ VaR threshold: {solution['var_threshold']:,.0f} VND")