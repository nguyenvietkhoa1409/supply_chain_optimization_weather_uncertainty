#!/usr/bin/env python3
"""
Quick Test Script - Run from project root directory
Tests the procurement optimization with generated data
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generation.network_generator import DaNangNetworkGenerator
from data_generation.product_generator import ProductCatalogGenerator
from data_generation.demand_generator import DemandPatternGenerator
from optimization.procurement_base import ProcurementOptimizer

print("="*70)
print("PROCUREMENT OPTIMIZATION - QUICK TEST")
print("="*70)

# Check if data already exists
data_dir = os.path.join(os.path.dirname(__file__), 'data', 'synthetic')

if os.path.exists(os.path.join(data_dir, 'network_topology.csv')):
    print("\n✓ Using existing generated data from data/synthetic/")
    
    import pandas as pd
    
    # Load existing data
    network = {
        'suppliers': pd.read_csv(os.path.join(data_dir, 'suppliers.csv')),
        'dcs': pd.read_csv(os.path.join(data_dir, 'distribution_centers.csv')),
        'stores': pd.read_csv(os.path.join(data_dir, 'stores.csv')),
        'all_locations': pd.read_csv(os.path.join(data_dir, 'network_topology.csv')),
        'distance_matrix': pd.read_csv(os.path.join(data_dir, 'distance_matrix.csv'), index_col=0)
    }
    
    products = pd.read_csv(os.path.join(data_dir, 'products.csv'))
    supplier_product = pd.read_csv(os.path.join(data_dir, 'supplier_product_matrix.csv'))
    weekly_demand = pd.read_csv(os.path.join(data_dir, 'weekly_demand.csv'))
    
else:
    print("\n⚠ No existing data found. Generating new data...")
    
    # Generate test data
    print("\nGenerating test data...")
    network_gen = DaNangNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=5, n_dcs=2, n_stores=8)
    
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

# Run optimization
print("\n" + "-"*70)
print("Running Procurement Optimization...")
print("-"*70)

optimizer = ProcurementOptimizer(
    network=network,
    products_df=products,
    supplier_product_df=supplier_product,
    demand_df=weekly_demand
)

status, solution = optimizer.solve(time_limit=60)

if status == 'Optimal':
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\n✓ Total Cost: {solution['objective_value']:,.0f} VND")
    print(f"✓ Solve Time: {solution['solve_time']:.2f} seconds")
    
    print("\n\nProcurement Plan (Top 10):")
    print(solution['procurement'].head(10).to_string(index=False))
    
    print(f"\n\nTotal Procurement Records: {len(solution['procurement'])}")
    
    print("\n\nSupplier Usage:")
    print(solution['supplier_usage'].to_string(index=False))
    
    if not solution['unmet_demand'].empty:
        print("\n\n⚠ Unmet Demand:")
        print(solution['unmet_demand'].to_string(index=False))
    else:
        print("\n\n✓ All demand satisfied - no shortages!")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
else:
    print(f"\n⚠ Optimization failed with status: {status}")