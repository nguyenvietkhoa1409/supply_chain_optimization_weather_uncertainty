#!/usr/bin/env python3
"""
Master Data Generation Script
Generates complete synthetic Da Nang supply chain dataset
UPDATED: Uses 9 suppliers (7 specialized + 2 general)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation.network_generator import DaNangNetworkGenerator
from data_generation.product_generator import ProductCatalogGenerator
from data_generation.demand_generator import DemandPatternGenerator
from weather.manual_scenarios import ManualWeatherScenarios

import json


def main():
    print("="*80)
    print("DA NANG SUPPLY CHAIN - COMPLETE DATA GENERATION")
    print("="*80)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')
    
    # Configuration
    config = {
        'seed': 42,
        'network': {
            'n_suppliers': 9,      # UPDATED: 9 suppliers (7 specialized + 2 general)
            'n_dcs': 2,
            'n_stores': 6
        },
        'products': {
            'n_products': 10
        },
        'demand': {
            'planning_horizon_days': 30
        }
    }
    
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    print("\nImprovements applied:")
    print("  ✓ 9 suppliers (7 specialized + 2 general wholesale)")
    print("  ✓ General supplier can supply ALL products")
    print("  ✓ Supplier capacity increased 2-3x")
    print("  ✓ Demand reduced by 60% to match capacity")
    print("  ✓ Lower MOQs (5-20 units)")
    
    # Step 1: Generate Network
    print("\n" + "-"*80)
    print("STEP 1: Generating Supply Chain Network")
    print("-"*80)
    
    network_gen = DaNangNetworkGenerator(seed=config['seed'])
    network = network_gen.generate_network(**config['network'])
    network_gen.save_network(network, output_dir)
    
    total_capacity = network['suppliers']['capacity_kg_per_day'].sum()
    print(f"\nTotal supplier capacity: {total_capacity:,.0f} kg/day")
    
    # Step 2: Generate Products
    print("\n" + "-"*80)
    print("STEP 2: Generating Product Catalog")
    print("-"*80)
    
    product_gen = ProductCatalogGenerator(seed=config['seed'])
    products = product_gen.generate_products(**config['products'])
    supplier_product = product_gen.generate_supplier_product_matrix(
        network['suppliers'],
        products
    )
    product_gen.save_catalog(products, supplier_product, output_dir)
    
    print(f"\nSupplier-product combinations: {len(supplier_product)}")
    print(f"Avg suppliers per product: {len(supplier_product) / len(products):.1f}")
    
    # Step 3: Generate Demand
    print("\n" + "-"*80)
    print("STEP 3: Generating Demand Patterns")
    print("-"*80)
    
    demand_gen = DemandPatternGenerator(seed=config['seed'])
    daily_demand = demand_gen.generate_demand_plan(
        stores_df=network['stores'],
        products_df=products,
        **config['demand']
    )
    demand_gen.save_demand_plan(daily_demand, output_dir)
    
    # Calculate total demand in kg
    import pandas as pd
    merged = daily_demand.merge(products, left_on='product_id', right_on='id')
    total_demand_kg = (merged['demand_units'] * merged['weight_kg_per_unit']).sum()
    daily_demand_kg = total_demand_kg / config['demand']['planning_horizon_days']
    
    print(f"\nTotal demand: {total_demand_kg:,.0f} kg over 30 days")
    print(f"Daily demand: {daily_demand_kg:,.0f} kg/day")
    print(f"Capacity/Demand ratio: {total_capacity / daily_demand_kg:.2f}x")
    
    if total_capacity / daily_demand_kg < 2.0:
        print("  ⚠ WARNING: Capacity might still be tight!")
    else:
        print("  ✓ Sufficient capacity buffer!")
    
    # Step 4: Generate Weather Scenarios
    print("\n" + "-"*80)
    print("STEP 4: Generating Weather Scenarios")
    print("-"*80)
    
    scenario_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'weather', 'scenarios')
    os.makedirs(scenario_dir, exist_ok=True)
    
    dry_scenarios = ManualWeatherScenarios.create_dry_season_scenarios()
    ManualWeatherScenarios.save_scenarios(
        dry_scenarios,
        'dry_season_scenarios.json',
        scenario_dir
    )
    
    monsoon_scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    ManualWeatherScenarios.save_scenarios(
        monsoon_scenarios,
        'monsoon_season_scenarios.json',
        scenario_dir
    )
    
    all_scenarios = dry_scenarios + monsoon_scenarios
    ManualWeatherScenarios.save_scenarios(
        all_scenarios,
        'all_scenarios.json',
        scenario_dir
    )
    
    # Summary Report
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE - SUMMARY")
    print("="*80)
    
    print(f"\n📍 Network:")
    print(f"   - {len(network['suppliers'])} suppliers (INCLUDES GENERAL)")
    print(f"   - Total capacity: {total_capacity:,.0f} kg/day")
    print(f"   - {len(network['stores'])} retail stores")
    
    print(f"\n📦 Products:")
    print(f"   - {len(products)} fresh food products")
    print(f"   - {len(supplier_product)} supplier-product combinations")
    print(f"   - Categories: {products['category'].value_counts().to_dict()}")
    
    print(f"\n📊 Demand:")
    print(f"   - {len(daily_demand)} daily demand records")
    print(f"   - Planning horizon: {config['demand']['planning_horizon_days']} days")
    print(f"   - Total demand: {daily_demand['demand_units'].sum():,.0f} units")
    print(f"   - Daily: {daily_demand_kg:,.0f} kg/day")
    print(f"   - Capacity ratio: {total_capacity / daily_demand_kg:.2f}x")
    
    print(f"\n🌦️  Weather Scenarios:")
    print(f"   - {len(dry_scenarios)} dry season scenarios")
    print(f"   - {len(monsoon_scenarios)} monsoon season scenarios")
    print(f"   - {len(all_scenarios)} total scenarios")
    
    print(f"\n✓ All data saved to: {output_dir}")
    print(f"✓ Weather scenarios saved to: {scenario_dir}")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Review generated data")
    print("  2. Run: python test_optimization.py")
    print("  3. Expected: <5% unmet demand, cost ~800M-1.5B VND")
    print("="*80)


if __name__ == "__main__":
    main()