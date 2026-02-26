#!/usr/bin/env python3
"""
Risk Aversion Comparison - Compare risk-neutral vs CVaR
Tests multiple λ values to show trade-off between cost and risk
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
from optimization.cvar_procurement import CVaRProcurementModel


def main():
    print("="*80)
    print("RISK AVERSION COMPARISON - CVaR vs Risk-Neutral")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
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
    
    scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    
    # Test different risk aversion levels
    lambda_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    results = []
    
    for lambda_val in lambda_values:
        print(f"\n{'='*80}")
        print(f"Testing λ = {lambda_val}")
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
        
        status, solution = model.solve(time_limit=600)
        
        if status in ['Optimal', 'Feasible']:
            scenario_costs = solution['scenario_costs']
            
            # Calculate metrics
            expected_cost = (scenario_costs['total_cost'] * scenario_costs['probability']).sum()
            worst_10_pct = scenario_costs.nlargest(int(len(scenario_costs) * 0.1) + 1, 'total_cost')
            cvar = (worst_10_pct['total_cost'] * worst_10_pct['probability']).sum() / worst_10_pct['probability'].sum()
            
            results.append({
                'lambda': lambda_val,
                'risk_profile': 'Risk-Neutral' if lambda_val == 0 else f'Risk-Averse (λ={lambda_val})',
                'objective': solution['objective_value'],
                'expected_cost': expected_cost,
                'cvar_90': cvar,
                'var_threshold': solution['var_threshold'],
                'worst_case': scenario_costs['total_cost'].max(),
                'best_case': scenario_costs['total_cost'].min(),
                'std_dev': scenario_costs['total_cost'].std(),
                'solve_time': solution['solve_time']
            })
            
            print(f"\n✓ Expected cost: {expected_cost:,.0f} VND")
            print(f"✓ CVaR (90%): {cvar:,.0f} VND")
            print(f"✓ Worst case: {results[-1]['worst_case']:,.0f} VND")
    
    # Generate report
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RISK AVERSION COMPARISON REPORT")
    print("="*80)
    print("\n" + df.to_string(index=False))
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    print("\n1. Expected Cost vs Risk Aversion:")
    print(f"   Risk-Neutral (λ=0):  {df.loc[df['lambda']==0, 'expected_cost'].values[0]:,.0f} VND")
    print(f"   Risk-Averse (λ=1):   {df.loc[df['lambda']==1, 'expected_cost'].values[0]:,.0f} VND")
    cost_increase = ((df.loc[df['lambda']==1, 'expected_cost'].values[0] / 
                     df.loc[df['lambda']==0, 'expected_cost'].values[0] - 1) * 100)
    print(f"   Cost increase: {cost_increase:.2f}%")
    
    print("\n2. Worst-Case Protection:")
    print(f"   Risk-Neutral worst: {df.loc[df['lambda']==0, 'worst_case'].values[0]:,.0f} VND")
    print(f"   Risk-Averse worst:  {df.loc[df['lambda']==1, 'worst_case'].values[0]:,.0f} VND")
    worst_reduction = ((1 - df.loc[df['lambda']==1, 'worst_case'].values[0] / 
                            df.loc[df['lambda']==0, 'worst_case'].values[0]) * 100)
    print(f"   Worst-case reduction: {worst_reduction:.2f}%")
    
    print("\n3. Trade-off Recommendation:")
    # Find sweet spot (λ where CVaR reduction / cost increase is maximized)
    df['cvar_reduction'] = (df['cvar_90'].iloc[0] - df['cvar_90']) / df['cvar_90'].iloc[0] * 100
    df['cost_increase'] = (df['expected_cost'] - df['expected_cost'].iloc[0]) / df['expected_cost'].iloc[0] * 100
    df['efficiency'] = df['cvar_reduction'] / (df['cost_increase'] + 0.01)  # Avoid div by 0
    
    best_lambda = df.loc[df['efficiency'].idxmax(), 'lambda']
    print(f"   Recommended λ: {best_lambda} (best CVaR reduction per cost increase)")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    df.to_csv(os.path.join(results_dir, 'risk_aversion_comparison.csv'), index=False)
    
    print(f"\n✓ Results saved to {results_dir}/risk_aversion_comparison.csv")
    print("\n" + "="*80)
    print("COMPARISON COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()