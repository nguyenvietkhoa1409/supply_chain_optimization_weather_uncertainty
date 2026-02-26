"""
Integrated Stochastic Procurement + VRP
Two-stage stochastic model with complete supply chain decisions

Stage 1: Procurement (weather-independent)
Stage 2: Routing (weather-dependent, per scenario)

This provides the full extensive-form stochastic supply chain optimization
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
from weather_vrp import WeatherAwareVRP


class IntegratedStochasticModel:
    """
    Complete two-stage stochastic supply chain model
    
    Combines:
    - Stage 1: Procurement decisions (from StochasticProcurementModel)
    - Stage 2: Routing decisions (from WeatherAwareVRP, per scenario)
    
    Objective: min E[Procurement + Routing + Emergency + Unmet]
    """
    
    def __init__(self,
                 network: Dict,
                 products_df: pd.DataFrame,
                 supplier_product_df: pd.DataFrame,
                 demand_df: pd.DataFrame,
                 weather_scenarios: List,
                 vehicle_config: Dict = None,
                 risk_aversion: float = 0.0):
        """
        Args:
            All standard inputs for procurement + VRP
            risk_aversion: CVaR parameter λ ∈ [0,1]
        """
        
        self.network = network
        self.products_df = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df = demand_df
        self.scenarios = weather_scenarios
        self.vehicle_config = vehicle_config
        self.risk_aversion = risk_aversion
        
        print(f"Integrated Stochastic Model Initialized:")
        print(f"  - Weather scenarios: {len(self.scenarios)}")
        print(f"  - Risk aversion: {self.risk_aversion}")
    
    def solve_sequential(self, time_limit_procurement: int = 600,
                        time_limit_vrp: int = 300) -> Tuple[str, Dict]:
        """
        Solve using sequential approach (procurement → routing)
        
        This is computationally tractable and maintains modularity
        
        Steps:
        1. Solve stochastic procurement (Stage 1 + emergency/unmet Stage 2)
        2. For each scenario, solve VRP with procurement decisions fixed
        3. Combine costs and evaluate
        
        Returns:
            (status, combined_solution)
        """
        
        print("\n" + "="*80)
        print("INTEGRATED STOCHASTIC OPTIMIZATION - SEQUENTIAL APPROACH")
        print("="*80)
        
        # =================================================================
        # STEP 1: Solve Stochastic Procurement
        # =================================================================
        print("\n" + "-"*80)
        print("STEP 1: Solving Stochastic Procurement")
        print("-"*80)
        
        procurement_model = StochasticProcurementModel(
            network=self.network,
            products_df=self.products_df,
            supplier_product_df=self.supplier_product_df,
            demand_df=self.demand_df,
            weather_scenarios=self.scenarios,
            risk_aversion=self.risk_aversion
        )
        
        proc_status, proc_solution = procurement_model.solve(
            time_limit=time_limit_procurement
        )
        
        if proc_status not in ['Optimal', 'Feasible']:
            print(f"⚠ Procurement optimization failed: {proc_status}")
            return proc_status, {}
        
        print(f"\n✓ Procurement objective: {proc_solution['objective_value']:,.0f} VND")
        
        # =================================================================
        # STEP 2: Solve VRP for Each Scenario
        # =================================================================
        print("\n" + "-"*80)
        print("STEP 2: Solving VRP per Scenario")
        print("-"*80)
        
        routing_solutions = {}
        routing_costs = []
        
        for k, scenario in enumerate(self.scenarios):
            print(f"\n  Scenario {k}: {scenario.name} (p={scenario.probability:.2f})")
            
            # Create VRP with procurement decisions
            vrp = WeatherAwareVRP(
                network=self.network,
                products_df=self.products_df,
                demand_df=self.demand_df,
                procurement_solution=proc_solution['stage1_procurement'],
                weather_scenarios=self.scenarios,
                vehicle_config=self.vehicle_config
            )
            
            # Solve VRP for this scenario
            vrp_status, vrp_solution = vrp.solve(
                scenario_id=k,
                time_limit=time_limit_vrp
            )
            
            if vrp_status in ['Optimal', 'Feasible']:
                routing_cost = vrp_solution['objective_value']
                routing_solutions[scenario.name] = vrp_solution
                routing_costs.append({
                    'scenario': scenario.name,
                    'probability': scenario.probability,
                    'routing_cost': routing_cost
                })
                
                print(f"    ✓ Routing cost: {routing_cost:,.0f} VND")
            else:
                print(f"    ⚠ VRP failed for scenario {k}")
                routing_costs.append({
                    'scenario': scenario.name,
                    'probability': scenario.probability,
                    'routing_cost': np.inf
                })
        
        # =================================================================
        # STEP 3: Combine and Evaluate
        # =================================================================
        print("\n" + "-"*80)
        print("STEP 3: Computing Total Expected Cost")
        print("-"*80)
        
        routing_df = pd.DataFrame(routing_costs)
        expected_routing = (routing_df['routing_cost'] * routing_df['probability']).sum()
        
        # Total cost = Procurement + Expected Routing
        total_objective = proc_solution['objective_value'] + expected_routing
        
        print(f"\nProcurement cost:    {proc_solution['objective_value']:>15,.0f} VND")
        print(f"Expected routing:    {expected_routing:>15,.0f} VND")
        print(f"{'='*40}")
        print(f"Total expected cost: {total_objective:>15,.0f} VND")
        
        # Combine solution
        combined_solution = {
            'status': 'Optimal',
            'objective_value': total_objective,
            'procurement_cost': proc_solution['objective_value'],
            'expected_routing_cost': expected_routing,
            'procurement_solution': proc_solution,
            'routing_solutions': routing_solutions,
            'routing_costs_df': routing_df
        }
        
        return 'Optimal', combined_solution
    
    def generate_report(self, solution: Dict) -> str:
        """Generate comprehensive solution report"""
        
        report = []
        report.append("="*80)
        report.append("INTEGRATED STOCHASTIC SUPPLY CHAIN OPTIMIZATION REPORT")
        report.append("="*80)
        
        # Overall costs
        report.append("\n1. COST BREAKDOWN")
        report.append("-"*80)
        report.append(f"Procurement (Stage 1):    {solution['procurement_cost']:>15,.0f} VND")
        report.append(f"Expected Routing (Stage 2): {solution['expected_routing_cost']:>15,.0f} VND")
        report.append(f"{'='*40}")
        report.append(f"Total Expected Cost:       {solution['objective_value']:>15,.0f} VND")
        
        # Procurement breakdown
        report.append("\n2. PROCUREMENT DECISIONS")
        report.append("-"*80)
        proc_sol = solution['procurement_solution']
        
        report.append(f"Suppliers used: {len(proc_sol['supplier_usage'])}")
        report.append(f"Products procured: {len(proc_sol['stage1_procurement'])}")
        
        total_procurement_qty = proc_sol['stage1_procurement']['quantity_units'].sum()
        report.append(f"Total procurement: {total_procurement_qty:,.0f} units")
        
        # Routing breakdown
        report.append("\n3. ROUTING COSTS BY SCENARIO")
        report.append("-"*80)
        routing_df = solution['routing_costs_df']
        
        for _, row in routing_df.iterrows():
            report.append(f"{row['scenario']:25s}  p={row['probability']:.2f}  "
                         f"Cost: {row['routing_cost']:>12,.0f} VND")
        
        report.append(f"\nExpected routing cost: {solution['expected_routing_cost']:,.0f} VND")
        report.append(f"Routing cost std dev:  {routing_df['routing_cost'].std():,.0f} VND")
        
        # Sample routes
        report.append("\n4. SAMPLE ROUTING SOLUTION (Best Scenario)")
        report.append("-"*80)
        
        best_scenario = routing_df.loc[routing_df['routing_cost'].idxmin(), 'scenario']
        if best_scenario in solution['routing_solutions']:
            routes = solution['routing_solutions'][best_scenario]['routes']
            
            for route in routes[:2]:  # Show first 2 routes
                report.append(f"\nVehicle {route['vehicle_id']}:")
                for stop in route['stops']:
                    report.append(f"  → {stop['location']} (arrival: {stop['arrival_time']}h)")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


if __name__ == "__main__":
    
    # FIXED: Add proper paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'src'))
    
    from data_generation.network_generator import DaNangNetworkGenerator
    from data_generation.product_generator import ProductCatalogGenerator
    from data_generation.demand_generator import DemandPatternGenerator
    from weather.manual_scenarios import ManualWeatherScenarios
    
    print("="*80)
    print("INTEGRATED STOCHASTIC MODEL - TEST")
    print("="*80)
    
    # Load data
    network_gen = DaNangNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=6, n_dcs=2, n_stores=4)
    
    product_gen = ProductCatalogGenerator(seed=42)
    products = product_gen.generate_products(n_products=5)
    supplier_product = product_gen.generate_supplier_product_matrix(
        network['suppliers'], products
    )
    
    demand_gen = DemandPatternGenerator(seed=42)
    daily_demand = demand_gen.generate_demand_plan(
        network['stores'], products, planning_horizon_days=7
    )
    weekly_demand = demand_gen.aggregate_to_weekly(daily_demand)
    
    # Use subset of scenarios for faster testing
    all_scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    scenarios = [all_scenarios[i] for i in [0, 2, 4]]  # Normal, Moderate, Typhoon
    
    # Renormalize probabilities
    total_prob = sum(s.probability for s in scenarios)
    for s in scenarios:
        s.probability = s.probability / total_prob
    
    # Solve integrated model
    model = IntegratedStochasticModel(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        weather_scenarios=scenarios,
        vehicle_config={'num_vehicles': 2},
        risk_aversion=0.0
    )
    
    status, solution = model.solve_sequential(
        time_limit_procurement=300,
        time_limit_vrp=180
    )
    
    if status == 'Optimal':
        report = model.generate_report(solution)
        print("\n" + report)