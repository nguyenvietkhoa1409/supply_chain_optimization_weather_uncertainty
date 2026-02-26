"""
Stochastic Procurement Optimization - Two-Stage Model
Integrates weather uncertainty into procurement decisions

Stage 1: Procurement decisions (before weather known)
Stage 2: Recourse actions (after weather scenario realizes)
"""

import pandas as pd
import numpy as np
from pulp import *
from typing import Dict, List, Tuple
import time
import sys
import os

# Import base model for parameter handling
sys.path.insert(0, os.path.dirname(__file__))
from procurement_base import ProcurementOptimizer


class StochasticProcurementModel:
    """
    Two-Stage Stochastic MILP for Weather-Aware Procurement
    
    Objective: min c^T x + E[Q(x, ξ)]
    where:
    - x: Stage 1 decisions (procurement, before weather known)
    - Q(x, ξ): Stage 2 cost given scenario ξ (recourse)
    - E[·]: Expected value over weather scenarios
    
    Decision Variables:
    Stage 1 (scenario-independent):
    - x[s,p]: Base procurement quantity
    - y[s,p]: Supplier activation
    
    Stage 2 (scenario-dependent per scenario k):
    - x_extra[k,s,p]: Emergency procurement under scenario k
    - u[k,p]: Unmet demand under scenario k
    """
    
    def __init__(self,
                 network: Dict,
                 products_df: pd.DataFrame,
                 supplier_product_df: pd.DataFrame,
                 demand_df: pd.DataFrame,
                 weather_scenarios: List,
                 risk_aversion: float = 0.0):
        """
        Args:
            network: Network topology
            products_df: Product catalog
            supplier_product_df: Supplier-product matrix
            demand_df: Demand forecast
            weather_scenarios: List of WeatherScenario objects
            risk_aversion: CVaR risk parameter λ ∈ [0,1]
                          0 = risk-neutral (expected cost only)
                          1 = fully risk-averse (CVaR only)
        """
        
        self.network = network
        self.products_df = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df = demand_df
        self.scenarios = weather_scenarios
        self.risk_aversion = risk_aversion
        
        # Extract key data
        self.suppliers = network['suppliers']['id'].tolist()
        self.products = products_df['id'].tolist()
        
        # Create lookups (reuse from base model)
        self._create_lookup_dicts()
        
        # Aggregate demand
        self.total_demand = demand_df.groupby('product_id')['demand_units'].sum().to_dict()
        
        print(f"Stochastic Model Initialized:")
        print(f"  - Suppliers: {len(self.suppliers)}")
        print(f"  - Products: {len(self.products)}")
        print(f"  - Weather scenarios: {len(self.scenarios)}")
        print(f"  - Risk aversion (λ): {self.risk_aversion}")
    
    def _create_lookup_dicts(self):
        """Create parameter lookup dictionaries"""
        
        # Product attributes
        self.product_cost = dict(zip(
            self.products_df['id'],
            self.products_df['unit_cost_vnd']
        ))
        self.product_weight = dict(zip(
            self.products_df['id'],
            self.products_df['weight_kg_per_unit']
        ))
        
        # Supplier attributes
        self.supplier_capacity = dict(zip(
            self.network['suppliers']['id'],
            self.network['suppliers']['capacity_kg_per_day']
        ))
        self.supplier_fixed_cost = dict(zip(
            self.network['suppliers']['id'],
            self.network['suppliers']['fixed_cost_vnd']
        ))
        
        # Supplier-product matrix
        self.sp_cost = {}
        self.sp_moq = {}
        self.sp_available = {}
        
        for _, row in self.supplier_product_df.iterrows():
            s, p = row['supplier_id'], row['product_id']
            self.sp_cost[(s, p)] = row['unit_cost_vnd']
            self.sp_moq[(s, p)] = row['moq_units']
            self.sp_available[(s, p)] = row['available']
    
    def build_model(self) -> Tuple[LpProblem, Dict]:
        """
        Build two-stage stochastic MILP in extensive form
        FIXED: Binary upper bounds, strengthened Stage 2
        """
        
        print("\nBuilding two-stage stochastic MILP...")
        
        model = LpProblem("Stochastic_Procurement_Weather", LpMinimize)
        
        # =================================================================
        # STAGE 1 VARIABLES (scenario-independent)
        # =================================================================
        print("  Creating Stage 1 variables...")
        
        # x[s,p]: Base procurement quantity
        x = LpVariable.dicts("stage1_procure",
                            ((s, p) for s in self.suppliers for p in self.products),
                            lowBound=0,
                            cat='Continuous')
        
        # y[s,p]: Supplier activation (binary)
        y = LpVariable.dicts("stage1_activate",
                            ((s, p) for s in self.suppliers for p in self.products),
                            cat='Binary')
        
        # =================================================================
        # STAGE 2 VARIABLES (scenario-dependent)
        # =================================================================
        print("  Creating Stage 2 variables (per scenario)...")
        
        # x_extra[k,s,p]: Emergency procurement in scenario k
        x_extra = LpVariable.dicts("stage2_emergency",
                                ((k, s, p) 
                                    for k in range(len(self.scenarios))
                                    for s in self.suppliers 
                                    for p in self.products),
                                lowBound=0,
                                cat='Continuous')
        
        # u[k,p]: Unmet demand in scenario k
        u = LpVariable.dicts("stage2_unmet",
                            ((k, p) 
                            for k in range(len(self.scenarios))
                            for p in self.products),
                            lowBound=0,
                            cat='Continuous')
        
        # =================================================================
        # OBJECTIVE FUNCTION
        # =================================================================
        print("  Formulating objective...")
        
        # Stage 1 cost (deterministic)
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
        
        # Stage 2 expected cost (over scenarios)
        stage2_costs = []
        
        for k, scenario in enumerate(self.scenarios):
            prob = scenario.probability
            
            # Emergency procurement cost (2x normal due to urgency)
            emergency_cost = lpSum([
                2.0 * self.sp_cost.get((s, p), self.product_cost[p]) * x_extra[k, s, p]
                for s in self.suppliers
                for p in self.products
                if self.sp_available.get((s, p), False)
            ])
            
            # Unmet demand penalty (weather-dependent spoilage)
            spoilage_mult = scenario.spoilage_multiplier
            
            penalty_multiplier = min(10.0, 5.0 * spoilage_mult)  # Max 10x instead of 20x
            penalty_cost = lpSum([
                penalty_multiplier * self.product_cost[p] * u[k, p]
                for p in self.products
            ])
            
            scenario_cost = emergency_cost + penalty_cost
            stage2_costs.append(prob * scenario_cost)
        
        # Expected Stage 2 cost
        stage2_expected_cost = lpSum(stage2_costs)
        
        # Total objective
        model += stage1_variable_cost + stage1_fixed_cost + stage2_expected_cost, "Total_Expected_Cost"
        
        # =================================================================
        # STAGE 1 CONSTRAINTS
        # =================================================================
        print("  Adding Stage 1 constraints...")
        
        # 1. Supplier capacity
        for s in self.suppliers:
            capacity = self.supplier_capacity[s]
            model += (
                lpSum([x[s, p] * self.product_weight[p] 
                    for p in self.products
                    if self.sp_available.get((s, p), False)]) <= capacity,
                f"S1_Capacity_{s}"
            )
        
        # 2. MOQ logic (FIXED: Both bounds)
        M = 100000  # Big-M
        for s in self.suppliers:
            for p in self.products:
                if self.sp_available.get((s, p), False):
                    moq = self.sp_moq.get((s, p), 0)
                    
                    # Lower bound: if used, must meet MOQ
                    model += (x[s, p] >= moq * y[s, p], f"S1_MOQ_lower_{s}_{p}")
                    
                    # CRITICAL FIX: Upper bound to force binary enforcement
                    model += (x[s, p] <= M * y[s, p], f"S1_MOQ_upper_{s}_{p}")
        
        # =================================================================
        # STAGE 2 CONSTRAINTS (per scenario) - STRENGTHENED
        # =================================================================
        print("  Adding Stage 2 constraints (per scenario)...")
        
        for k, scenario in enumerate(self.scenarios):
            
            # Weather impact factors
            capacity_factor = scenario.capacity_reduction_factor
            spoilage_mult = scenario.spoilage_multiplier
            severity = scenario.severity_level
            
            # SIMPLE APPROACH: Direct survival rates by severity
            survival_rates = {
                1: 0.99,  # Severity 1: minimal loss
                2: 0.97,  # Severity 2: 3% loss
                3: 0.92,  # Severity 3: 8% loss  
                4: 0.82,  # Severity 4: 18% loss
                5: 0.70   # Severity 5: 35% loss (not 50%)
            }
            survival_rate = survival_rates.get(severity, 0.90)
            
            # 1. Demand satisfaction with weather-induced losses
            for p in self.products:
                demand = self.total_demand.get(p, 0)
                
                if demand > 0:
                    # Stage 1 supply after weather losses
                    stage1_effective = lpSum([
                        x[s, p] * survival_rate
                        for s in self.suppliers
                        if self.sp_available.get((s, p), False)
                    ])
                    
                    # Emergency supply
                    emergency_supply = lpSum([
                        x_extra[k, s, p]
                        for s in self.suppliers
                        if self.sp_available.get((s, p), False)
                    ])
                    
                    # Demand satisfaction
                    model += (
                        stage1_effective + emergency_supply + u[k, p] >= demand,
                        f"S2_Demand_{k}_{p}"
                    )
            
            # 2. Weather-adjusted emergency capacity
            for s in self.suppliers:
                base_capacity = self.supplier_capacity[s]
                adjusted_capacity = base_capacity * capacity_factor
                
                # Emergency procurement limited by weather-adjusted capacity
                # Can only use 30% of adjusted capacity for emergency
                model += (
                    lpSum([x_extra[k, s, p] * self.product_weight[p]
                        for p in self.products
                        if self.sp_available.get((s, p), False)]) <= adjusted_capacity * 0.8,
                    f"S2_Emergency_Capacity_{k}_{s}"
                )
            
            # # 3. STRENGTHENED: Force Stage 2 activation under severe weather
            # if scenario.severity_level >= 4:  # Heavy rain or worse
            #     # Must have some emergency procurement or unmet demand
            #     total_emergency = lpSum([
            #         x_extra[k, s, p]
            #         for s in self.suppliers
            #         for p in self.products
            #         if self.sp_available.get((s, p), False)
            #     ])
                
            #     total_unmet = lpSum([u[k, p] for p in self.products])
                
            #     # At least 1% of total demand must trigger recourse
            #     min_recourse = sum(self.total_demand.values()) * 0.01
                
            #     model += (
            #         total_emergency + total_unmet >= min_recourse,
            #         f"S2_Force_Recourse_{k}"
            #     )
        
        var_counts = model.numVariables()
        const_counts = model.numConstraints()
        
        print(f"✓ Model built:")
        print(f"  - Variables: {var_counts}")
        print(f"  - Constraints: {const_counts}")
        print(f"  - Scenarios: {len(self.scenarios)}")
        
        return model, {'x': x, 'y': y, 'x_extra': x_extra, 'u': u}
    
    def solve(self, 
             time_limit: int = 600,
             gap_tolerance: float = 0.02) -> Tuple[str, Dict]:
        """
        Solve the stochastic model
        
        Args:
            time_limit: Max solve time (seconds)
            gap_tolerance: MIP gap tolerance (2% default for stochastic)
        """
        
        model, vars_dict = self.build_model()
        
        # Solver
        solver = PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=gap_tolerance,
            msg=1
        )
        
        print(f"\nSolving stochastic model (time limit: {time_limit}s, gap: {gap_tolerance*100}%)...")
        start_time = time.time()
        model.solve(solver)
        solve_time = time.time() - start_time
        
        status = LpStatus[model.status]
        print(f"\n✓ Status: {status}")
        print(f"✓ Solve time: {solve_time:.2f} seconds")
        
        if status in ['Optimal', 'Feasible']:
            obj_value = value(model.objective)
            print(f"✓ Objective value: {obj_value:,.0f} VND")
            
            solution = self._extract_solution(vars_dict)
            solution['objective_value'] = obj_value
            solution['solve_time'] = solve_time
            solution['status'] = status
            
            # Compute scenario-specific costs
            solution['scenario_costs'] = self._compute_scenario_costs(vars_dict)
            
            return status, solution
        else:
            print(f"⚠ Model status: {status}")
            return status, {}
    
    def _extract_solution(self, vars_dict: Dict) -> Dict:
        """Extract solution from variables"""
        
        x = vars_dict['x']
        y = vars_dict['y']
        x_extra = vars_dict['x_extra']
        u = vars_dict['u']
        
        # Stage 1 procurement
        stage1_procurement = []
        for s in self.suppliers:
            for p in self.products:
                qty = value(x[s, p])
                if qty and qty > 0.01:
                    stage1_procurement.append({
                        'supplier_id': s,
                        'product_id': p,
                        'quantity_units': round(qty, 2),
                        'cost_vnd': round(qty * self.sp_cost.get((s, p), self.product_cost[p]), 0)
                    })
        
        # Stage 2 recourse (per scenario)
        stage2_recourse = {}
        for k, scenario in enumerate(self.scenarios):
            emergency = []
            for s in self.suppliers:
                for p in self.products:
                    qty = value(x_extra[k, s, p])
                    if qty and qty > 0.01:
                        emergency.append({
                            'supplier_id': s,
                            'product_id': p,
                            'quantity_units': round(qty, 2)
                        })
            
            unmet = []
            for p in self.products:
                shortage = value(u[k, p])
                if shortage and shortage > 0.01:
                    unmet.append({
                        'product_id': p,
                        'unmet_quantity': round(shortage, 2)
                    })
            
            stage2_recourse[scenario.name] = {
                'scenario_id': scenario.scenario_id,
                'severity_level': scenario.severity_level,
                'probability': scenario.probability,
                'emergency_procurement': pd.DataFrame(emergency) if emergency else pd.DataFrame(),
                'unmet_demand': pd.DataFrame(unmet) if unmet else pd.DataFrame()
            }
        
        # Supplier usage
        supplier_usage = []
        for s in self.suppliers:
            products_used = [p for p in self.products if value(y[s, p]) and value(y[s, p]) > 0.5]
            if products_used:
                supplier_usage.append({
                    'supplier_id': s,
                    'products_supplied': products_used,
                    'num_products': len(products_used)
                })
        
        return {
            'stage1_procurement': pd.DataFrame(stage1_procurement),
            'stage2_recourse': stage2_recourse,
            'supplier_usage': pd.DataFrame(supplier_usage)
        }
    
    def _compute_scenario_costs(self, vars_dict: Dict) -> pd.DataFrame:
        """Compute cost breakdown per scenario"""
        x = vars_dict['x']
        y = vars_dict['y']
        x_extra = vars_dict['x_extra']
        u = vars_dict['u']
        
        scenario_costs = []
        
        # FIXED: Calculate Stage 1 fixed cost (same for all scenarios)
        stage1_fixed = sum([
            self.supplier_fixed_cost[s] * value(y[s, p])
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False) and value(y[s, p])
        ])
        
        for k, scenario in enumerate(self.scenarios):
            
            # Stage 1 variable cost (same for all scenarios)
            stage1_variable = sum([
                value(x[s, p]) * self.sp_cost.get((s, p), self.product_cost[p])
                for s in self.suppliers
                for p in self.products
                if self.sp_available.get((s, p), False) and value(x[s, p])
            ])
            
            # FIXED: Stage 1 total = variable + fixed
            stage1_total = stage1_variable + stage1_fixed
            
            # Emergency cost
            emergency_cost = sum([
                2.0 * value(x_extra[k, s, p]) * self.sp_cost.get((s, p), self.product_cost[p])
                for s in self.suppliers
                for p in self.products
                if self.sp_available.get((s, p), False) and value(x_extra[k, s, p])
            ]) if x_extra else 0
            
            # Penalty cost
            spoilage_mult = scenario.spoilage_multiplier
            penalty_cost = sum([
                10.0 * spoilage_mult * self.product_cost[p] * value(u[k, p])
                for p in self.products
                if value(u[k, p])
            ]) if u else 0
            
            # FIXED: Total cost includes fixed cost
            total_cost = stage1_total + emergency_cost + penalty_cost
            
            scenario_costs.append({
                'scenario_name': scenario.name,
                'severity_level': scenario.severity_level,
                'probability': scenario.probability,
                'stage1_cost': stage1_total,  # FIXED
                'emergency_cost': emergency_cost,
                'penalty_cost': penalty_cost,
                'total_cost': total_cost
            })
        
        return pd.DataFrame(scenario_costs)


if __name__ == "__main__":
    # Test
    import sys
    sys.path.insert(0, '../..')
    
    from data_generation.network_generator import DaNangNetworkGenerator
    from data_generation.product_generator import ProductCatalogGenerator
    from data_generation.demand_generator import DemandPatternGenerator
    from weather.manual_scenarios import ManualWeatherScenarios
    
    print("="*80)
    print("STOCHASTIC PROCUREMENT MODEL - TEST")
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
    
    # Load weather scenarios
    scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    
    # Optimize
    model = StochasticProcurementModel(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        weather_scenarios=scenarios,
        risk_aversion=0.0
    )
    
    status, solution = model.solve(time_limit=300)
    
    if status in ['Optimal', 'Feasible']:
        print("\n" + "="*80)
        print("STOCHASTIC SOLUTION")
        print("="*80)
        
        print("\nStage 1 Procurement:")
        print(solution['stage1_procurement'].head(10).to_string(index=False))
        
        print("\n\nScenario Costs:")
        print(solution['scenario_costs'].to_string(index=False))
        
        print("\n\nStage 2 Recourse (sample - worst scenario):")
        worst_scenario = solution['scenario_costs'].loc[solution['scenario_costs']['total_cost'].idxmax()]
        print(f"Scenario: {worst_scenario['scenario_name']}")
        recourse = solution['stage2_recourse'][worst_scenario['scenario_name']]
        if not recourse['emergency_procurement'].empty:
            print("\nEmergency Procurement:")
            print(recourse['emergency_procurement'].to_string(index=False))
        if not recourse['unmet_demand'].empty:
            print("\nUnmet Demand:")
            print(recourse['unmet_demand'].to_string(index=False))