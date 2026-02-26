"""
Simplified VRP for debugging
Removes complex constraints to identify infeasibility source
"""

import pandas as pd
import numpy as np
from pulp import *
from typing import Dict, List


def build_simple_vrp(network, products_df, demand_df, num_vehicles=2):
    """
    Ultra-simple VRP - just visit all stores once
    No time windows, no freshness, minimal constraints
    """
    
    print("\n" + "="*80)
    print("SIMPLIFIED VRP DEBUG")
    print("="*80)
    
    stores = network['stores']['id'].tolist()
    depot = network['dcs']['id'].iloc[0]  # Use actual DC
    
    # Get distance matrix
    dist_matrix = network['distance_matrix']
    
    print(f"\nStores: {stores}")
    print(f"Depot: {depot}")
    print(f"Vehicles: {num_vehicles}")
    
    # Calculate total demand in kg
    total_demand_kg = 0
    for _, prod in products_df.iterrows():
        prod_id = prod['id']
        prod_weight = prod['weight_kg_per_unit']
        
        prod_demand = demand_df[demand_df['product_id'] == prod_id]['demand_units'].sum()
        total_demand_kg += prod_demand * prod_weight
    
    print(f"\nTotal demand: {total_demand_kg:.2f} kg")
    print(f"Vehicle capacity: 1000 kg × {num_vehicles} = {1000 * num_vehicles} kg")
    print(f"Capacity ratio: {(1000 * num_vehicles) / total_demand_kg:.2f}x")
    
    if total_demand_kg > 1000 * num_vehicles:
        print("\n⚠️ WARNING: Total demand exceeds vehicle capacity!")
        print("   VRP will be infeasible unless we allow multiple trips or unmet demand")
    
    # Build simple model
    model = LpProblem("Simple_VRP", LpMinimize)
    
    locations = [depot] + stores
    vehicles = list(range(num_vehicles))
    
    # Decision variables: x[i,j,v] = 1 if vehicle v goes from i to j
    x = LpVariable.dicts("route",
                        ((i, j, v) for i in locations for j in locations 
                         for v in vehicles if i != j),
                        cat='Binary')
    
    # Objective: minimize total distance
    total_distance = lpSum([
        dist_matrix.loc[i, j] * x[i, j, v]
        for i in locations
        for j in locations
        for v in vehicles
        if i != j and i in dist_matrix.index and j in dist_matrix.columns
    ])
    
    model += total_distance, "Total_Distance"
    
    # Constraint 1: Each store visited exactly once
    for store in stores:
        model += (
            lpSum([x[i, store, v] for i in locations for v in vehicles 
                  if i != store and (i, store, v) in x]) == 1,
            f"Visit_{store}"
        )
    
    # Constraint 2: Flow conservation
    for v in vehicles:
        for loc in locations:
            inflow = lpSum([x[i, loc, v] for i in locations 
                           if i != loc and (i, loc, v) in x])
            outflow = lpSum([x[loc, j, v] for j in locations 
                            if j != loc and (loc, j, v) in x])
            
            model += (inflow == outflow, f"Flow_{loc}_{v}")
    
    # Constraint 3: Each vehicle starts from depot at most once
    for v in vehicles:
        model += (
            lpSum([x[depot, j, v] for j in stores if (depot, j, v) in x]) <= 1,
            f"Start_{v}"
        )
    
    print(f"\n✓ Simple model built:")
    print(f"  - Variables: {model.numVariables()}")
    print(f"  - Constraints: {model.numConstraints()}")
    
    # Solve
    print("\nSolving simple VRP...")
    solver = PULP_CBC_CMD(msg=1, timeLimit=60)
    model.solve(solver)
    
    status = LpStatus[model.status]
    print(f"\n✓ Status: {status}")
    
    if status == 'Optimal':
        print(f"✓ Objective: {value(model.objective):.2f} km")
        
        # Extract routes
        print("\nRoutes:")
        for v in vehicles:
            route = [depot]
            current = depot
            
            while True:
                next_loc = None
                for j in locations:
                    if j != current and (current, j, v) in x:
                        if value(x[current, j, v]) and value(x[current, j, v]) > 0.5:
                            next_loc = j
                            break
                
                if next_loc is None or next_loc == depot:
                    break
                
                route.append(next_loc)
                current = next_loc
            
            if len(route) > 1:
                print(f"  Vehicle {v}: {' → '.join(route)}")
    
    return status, model


if __name__ == "__main__":
    import sys
    import os
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from data_generation.network_generator import DaNangNetworkGenerator
    from data_generation.product_generator import ProductCatalogGenerator
    from data_generation.demand_generator import DemandPatternGenerator
    
    # Generate data
    network_gen = DaNangNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=6, n_dcs=2, n_stores=4)
    
    product_gen = ProductCatalogGenerator(seed=42)
    products = product_gen.generate_products(n_products=5)
    
    demand_gen = DemandPatternGenerator(seed=42)
    daily_demand = demand_gen.generate_demand_plan(
        network['stores'], products, planning_horizon_days=7
    )
    weekly_demand = demand_gen.aggregate_to_weekly(daily_demand)
    
    # Test simple VRP
    status, model = build_simple_vrp(network, products, weekly_demand, num_vehicles=2)
    
    if status != 'Optimal':
        print("\n⚠️ Even simple VRP is infeasible!")
        print("Trying with more vehicles...")
        
        status, model = build_simple_vrp(network, products, weekly_demand, num_vehicles=4)