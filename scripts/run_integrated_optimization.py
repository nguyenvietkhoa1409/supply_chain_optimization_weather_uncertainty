#!/usr/bin/env python3
"""
Complete Integrated Stochastic Supply Chain Optimization
Procurement + VRP with Weather Uncertainty

Pipeline:
1. Load supply chain data
2. Load weather scenarios
3. Solve stochastic procurement
4. Solve weather-aware VRP per scenario
5. Compute integrated metrics (VSS, EVPI)
6. Generate comprehensive report
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

from data_generation.network_generator import DaNangNetworkGenerator
from data_generation.product_generator import ProductCatalogGenerator
from data_generation.demand_generator import DemandPatternGenerator
from weather.manual_scenarios import ManualWeatherScenarios
from optimization.integrated_stochastic import IntegratedStochasticModel
from optimization.deterministic_baseline import DeterministicBaselineModel


def main():
    print("="*80)
    print("INTEGRATED STOCHASTIC SUPPLY CHAIN OPTIMIZATION")
    print("Weather-Aware Procurement + Routing")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Loading Supply Chain Data")
    print("-"*80)
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')
    
    suppliers = pd.read_csv(os.path.join(data_dir, 'suppliers.csv'))
    stores = pd.read_csv(os.path.join(data_dir, 'stores.csv'))
    dcs = pd.read_csv(os.path.join(data_dir, 'distribution_centers.csv'))
    all_locations = pd.read_csv(os.path.join(data_dir, 'network_topology.csv'))
    distance_matrix = pd.read_csv(os.path.join(data_dir, 'distance_matrix.csv'), index_col=0)
    
    network = {
        'suppliers': suppliers,
        'stores': stores,
        'dcs': dcs,
        'all_locations': all_locations,
        'distance_matrix': distance_matrix
    }
    
    products = pd.read_csv(os.path.join(data_dir, 'products.csv'))
    supplier_product = pd.read_csv(os.path.join(data_dir, 'supplier_product_matrix.csv'))
    weekly_demand = pd.read_csv(os.path.join(data_dir, 'weekly_demand.csv'))
    
    print(f"✓ Loaded {len(suppliers)} suppliers, {len(products)} products, {len(stores)} stores")
    
    # ========================================================================
    # STEP 2: Load Weather Scenarios
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Loading Weather Scenarios")
    print("-"*80)
    
    season = input("\nSelect season (1=Dry, 2=Monsoon, 3=Subset for testing): ").strip()
    
    if season == '1':
        scenarios = ManualWeatherScenarios.create_dry_season_scenarios()
        season_name = "Dry Season"
    elif season == '3':
        # Testing subset: 3 representative scenarios
        all_scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
        scenarios = [all_scenarios[i] for i in [0, 2, 4]]
        total_prob = sum(s.probability for s in scenarios)
        for s in scenarios:
            s.probability = s.probability / total_prob
        season_name = "Monsoon Subset (Testing)"
    else:
        scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
        season_name = "Monsoon Season"
    
    print(f"\n✓ Loaded {len(scenarios)} {season_name} scenarios")
    scenario_table = ManualWeatherScenarios.get_scenario_summary_table(scenarios)
    print("\n" + scenario_table.to_string(index=False))
    
    # ========================================================================
    # STEP 3: Solve Integrated Stochastic Model
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Solving Integrated Stochastic Model")
    print("-"*80)
    
    model = IntegratedStochasticModel(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        weather_scenarios=scenarios,
        vehicle_config={
            'num_vehicles': 3,
            'capacity_kg': 1000,
            'base_speed_kmh': 40,
            'cost_per_km': 5000,
            'cost_per_hour': 50000,
            'max_route_time_hours': 8
        },
        risk_aversion=0.0
    )
    
    status, solution = model.solve_sequential(
        time_limit_procurement=600,
        time_limit_vrp=300
    )
    
    if status != 'Optimal':
        print(f"\n⚠ Optimization failed: {status}")
        return
    
    # ========================================================================
    # STEP 4: Compare with Deterministic Baseline
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Computing Deterministic Baseline (for VSS)")
    print("-"*80)
    
    det_model = DeterministicBaselineModel(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand,
        weather_scenarios=scenarios
    )
    
    det_status, det_solution = det_model.solve(time_limit=300)
    
    if det_status == 'Optimal':
        det_objective = det_solution['deterministic_objective']
        
        # Compute VSS
        vss = det_objective - solution['objective_value']
        vss_pct = (vss / det_objective * 100)
        
        print(f"\nDeterministic baseline: {det_objective:,.0f} VND")
        print(f"Stochastic solution:    {solution['objective_value']:,.0f} VND")
        print(f"VSS (cost savings):     {vss:,.0f} VND ({vss_pct:.2f}%)")
    else:
        print(f"⚠ Deterministic baseline failed: {det_status}")
        vss = None
        vss_pct = None
    
    # ========================================================================
    # STEP 5: Generate Report
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 5: Generating Comprehensive Report")
    print("-"*80)
    
    report = model.generate_report(solution)
    
    # Add VSS section
    if vss is not None:
        report += "\n\n5. VALUE OF STOCHASTIC SOLUTION (VSS)"
        report += "\n" + "-"*80
        report += f"\nDeterministic (Expected Weather): {det_objective:>15,.0f} VND"
        report += f"\nStochastic (Hedging Strategy):    {solution['objective_value']:>15,.0f} VND"
        report += f"\n{'='*40}"
        report += f"\nVSS (Cost Savings):                {vss:>15,.0f} VND"
        report += f"\nVSS (%):                           {vss_pct:>15.2f}%"
        
        if vss_pct > 5:
            report += "\n\n  ✓ SIGNIFICANT value from weather-aware optimization!"
        elif vss_pct > 0:
            report += "\n\n  ✓ Moderate benefit from stochastic approach"
        else:
            report += "\n\n  ⚠ Limited benefit (check model assumptions)"
    
    print("\n" + report)
    
    # ========================================================================
    # STEP 6: Save Results
    # ========================================================================
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save report
    with open(os.path.join(results_dir, 'integrated_optimization_report.txt'), 'w') as f:
        f.write(report)
    
    # Save routing solutions
    routing_df = solution['routing_costs_df']
    routing_df.to_csv(os.path.join(results_dir, 'routing_costs_by_scenario.csv'), index=False)
    
    # Save procurement
    solution['procurement_solution']['stage1_procurement'].to_csv(
        os.path.join(results_dir, 'integrated_procurement.csv'), index=False
    )
    
    print(f"\n✓ Results saved to {results_dir}/")
    
    print("\n" + "="*80)
    print("INTEGRATED OPTIMIZATION COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()