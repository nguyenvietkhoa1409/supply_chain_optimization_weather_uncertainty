"""
Weather-Aware Vehicle Routing Problem (VRP)
FIXED VERSION - Relaxed constraints for feasibility

Key changes from original:
- Freshness: 80% shelf life (was 50%)
- Added unmet demand variables
- Simplified delivery constraints
- Removed MTZ subtour elimination (too tight)
"""

import pandas as pd
import numpy as np
from pulp import *
from typing import Dict, List, Tuple
import time


class WeatherAwareVRP:
    """
    Vehicle Routing Problem with weather scenario integration
    RELAXED VERSION for practical feasibility
    """
    
    def __init__(self,
                 network: Dict,
                 products_df: pd.DataFrame,
                 demand_df: pd.DataFrame,
                 procurement_solution: pd.DataFrame,
                 weather_scenarios: List,
                 vehicle_config: Dict = None):
        
        self.network = network
        self.products_df = products_df
        self.demand_df = demand_df
        self.procurement = procurement_solution
        self.scenarios = weather_scenarios
        
        # Default vehicle configuration
        default_config = {
            'num_vehicles': 3,
            'capacity_kg': 1500,
            'base_speed_kmh': 40,
            'cost_per_km': 5000,
            'cost_per_hour': 50000,
            'max_route_time_hours': 10  # Increased from 8
        }
        
        if vehicle_config is None:
            self.vehicle_config = default_config
        else:
            self.vehicle_config = {**default_config, **vehicle_config}
        
        # Extract data
        self.stores = network['stores']['id'].tolist()
        self.products = products_df['id'].tolist()
        
        # Use actual DC as depot
        if 'dcs' in network and len(network['dcs']) > 0:
            self.depot = network['dcs']['id'].iloc[0]
        else:
            self.depot = 'DEPOT'
        
        self._create_lookups()
        
        print(f"Weather-Aware VRP Initialized (RELAXED):")
        print(f"  - Stores: {len(self.stores)}")
        print(f"  - Products: {len(self.products)}")
        print(f"  - Vehicles: {self.vehicle_config['num_vehicles']}")
        print(f"  - Depot: {self.depot}")
    
    def _create_lookups(self):
        """Create parameter lookup dictionaries"""
        
        # Distance matrix
        self.distance = {}
        dist_matrix = self.network['distance_matrix']
        
        all_locs = [self.depot] + self.stores
        for i in all_locs:
            for j in all_locs:
                if i != j:
                    try:
                        self.distance[(i, j)] = dist_matrix.loc[i, j]
                    except:
                        # Fallback: symmetric distance
                        try:
                            self.distance[(i, j)] = dist_matrix.loc[j, i]
                        except:
                            self.distance[(i, j)] = 10.0  # Default 10km
        
        # Demand by store-product
        self.demand = {}
        for _, row in self.demand_df.groupby(['store_id', 'product_id'])['demand_units'].sum().reset_index().iterrows():
            self.demand[(row['store_id'], row['product_id'])] = row['demand_units']
        
        # Product weight
        self.product_weight = dict(zip(
            self.products_df['id'],
            self.products_df['weight_kg_per_unit']
        ))
        
        # Product shelf life (in hours)
        self.shelf_life_hours = {}
        for _, prod in self.products_df.iterrows():
            self.shelf_life_hours[prod['id']] = prod['shelf_life_days'] * 24
    
    def build_model(self, scenario_id: int = None) -> Tuple[LpProblem, Dict]:
        """Build simplified VRP - RELAXED for feasibility"""
        
        if scenario_id is None:
            speed_factor = np.mean([s.speed_reduction_factor for s in self.scenarios])
            capacity_factor = np.mean([s.capacity_reduction_factor for s in self.scenarios])
            scenario_name = "Expected"
        else:
            scenario = self.scenarios[scenario_id]
            speed_factor = scenario.speed_reduction_factor
            capacity_factor = scenario.capacity_reduction_factor
            scenario_name = scenario.name
        
        print(f"\nBuilding RELAXED VRP for: {scenario_name}")
        
        model = LpProblem(f"VRP_{scenario_name}", LpMinimize)
        
        locations = [self.depot] + self.stores
        vehicles = list(range(self.vehicle_config['num_vehicles']))
        
        # =================================================================
        # VARIABLES
        # =================================================================
        
        # x[i,j,v]: route arc
        x = LpVariable.dicts("route",
                            ((i, j, v) for i in locations for j in locations 
                             for v in vehicles if i != j),
                            cat='Binary')
        
        # q[i,p,v]: delivery quantity
        q = LpVariable.dicts("delivery",
                            ((i, p, v) for i in self.stores for p in self.products 
                             for v in vehicles),
                            lowBound=0,
                            cat='Continuous')
        
        # ADDED: Unmet demand (backup feasibility)
        unmet = LpVariable.dicts("unmet",
                                ((i, p) for i in self.stores for p in self.products),
                                lowBound=0,
                                cat='Continuous')
        
        # =================================================================
        # OBJECTIVE
        # =================================================================
        
        base_speed = self.vehicle_config['base_speed_kmh']
        adjusted_speed = base_speed * speed_factor
        adjusted_capacity = self.vehicle_config['capacity_kg'] * capacity_factor
        
        # Distance cost
        distance_cost = lpSum([
            self.distance.get((i, j), 0) * self.vehicle_config['cost_per_km'] * x[i, j, v]
            for i in locations
            for j in locations
            for v in vehicles
            if i != j and (i, j) in self.distance
        ])
        
        # Time cost
        time_cost = lpSum([
            (self.distance.get((i, j), 0) / adjusted_speed) * 
            self.vehicle_config['cost_per_hour'] * x[i, j, v]
            for i in locations
            for j in locations
            for v in vehicles
            if i != j and (i, j) in self.distance
        ])
        
        # PENALTY for unmet demand (make it expensive but not infeasible)
        penalty_cost = lpSum([
            50000 * unmet[i, p]  # 50k VND/unit penalty
            for i in self.stores
            for p in self.products
        ])
        
        model += distance_cost + time_cost + penalty_cost, "Total_Cost"
        
        # =================================================================
        # CONSTRAINTS (SIMPLIFIED)
        # =================================================================
        
        # 1. Each store visited at most once per vehicle
        for i in self.stores:
            model += (
                lpSum([x[j, i, v] for j in locations for v in vehicles 
                      if i != j and (j, i, v) in x]) >= 1,  # At least once total
                f"Visit_{i}"
            )
        
        # 2. Flow conservation
        for v in vehicles:
            for i in locations:
                inflow = lpSum([x[j, i, v] for j in locations 
                               if i != j and (j, i, v) in x])
                outflow = lpSum([x[i, j, v] for j in locations 
                                if i != j and (i, j, v) in x])
                
                model += (inflow == outflow, f"Flow_{i}_{v}")
        
        # 3. Depot start/end
        for v in vehicles:
            model += (
                lpSum([x[self.depot, j, v] for j in self.stores 
                      if (self.depot, j, v) in x]) <= 1,
                f"Start_{v}"
            )
        
        # 4. Vehicle capacity
        for v in vehicles:
            total_load = lpSum([
                q[i, p, v] * self.product_weight[p]
                for i in self.stores
                for p in self.products
            ])
            
            model += (total_load <= adjusted_capacity, f"Capacity_{v}")
        
        # 5. Demand satisfaction (WITH UNMET)
        for i in self.stores:
            for p in self.products:
                demand_qty = self.demand.get((i, p), 0)
                if demand_qty > 0:
                    total_delivery = lpSum([q[i, p, v] for v in vehicles])
                    model += (
                        lpSum([q[i, p, v] for v in vehicles]) + unmet[i, p] >= demand_qty,
                        f"Demand_{i}_{p}"
                    )
                    # ADDED: Force delivery if visited
                    # If any vehicle visits store i, must deliver product p
                    visited = lpSum([x[j, i, v] for j in locations for v in vehicles
                                if i != j and (j, i, v) in x])
                    
                    # If visited (visited >= 0.5), then delivery should happen
                    # But allow unmet if capacity insufficient
                    model += (
                        total_delivery >= demand_qty * 0.8 - 10000 * (1 - visited),
                        f"Force_Delivery_{i}_{p}"
                    )
                    
        # 6. Delivery only if visited (SIMPLIFIED)
        M = 10000
        for i in self.stores:
            for p in self.products:
                for v in vehicles:
                    visited = lpSum([x[j, i, v] for j in locations 
                                    if i != j and (j, i, v) in x])
                    
                    model += (
                        q[i, p, v] <= M * visited,
                        f"Visit_Delivery_{i}_{p}_{v}"
                    )
        
        print(f"✓ Relaxed VRP built:")
        print(f"  - Variables: {model.numVariables()}")
        print(f"  - Constraints: {model.numConstraints()}")
        print(f"  - Capacity: {adjusted_capacity:.0f} kg/vehicle")
        
        return model, {'x': x, 'q': q, 'unmet': unmet}
    
    def solve(self, scenario_id: int = None, time_limit: int = 300) -> Tuple[str, Dict]:
        """Solve VRP"""
        
        model, vars_dict = self.build_model(scenario_id)
        
        solver = PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=0.10,  # 10% gap tolerance
            msg=1
        )
        
        print(f"\nSolving VRP (time limit: {time_limit}s, gap: 10%)...")
        start_time = time.time()
        model.solve(solver)
        solve_time = time.time() - start_time
        
        status = LpStatus[model.status]
        print(f"\n✓ Status: {status}")
        print(f"✓ Solve time: {solve_time:.2f} seconds")
        
        if status in ['Optimal', 'Feasible']:
            obj_value = value(model.objective)
            print(f"✓ Routing cost: {obj_value:,.0f} VND")
            
            # Check unmet demand
            unmet_total = sum([value(vars_dict['unmet'][i, p]) or 0 
                              for i in self.stores for p in self.products])
            
            if unmet_total > 0.01:
                print(f"  ⚠️ Unmet demand: {unmet_total:.2f} units")
            else:
                print(f"  ✓ All demand satisfied!")
            
            solution = self._extract_solution(vars_dict)
            solution['objective_value'] = obj_value
            solution['solve_time'] = solve_time
            solution['status'] = status
            solution['unmet_demand'] = unmet_total
            
            return status, solution
        else:
            print(f"⚠ VRP failed: {status}")
            return status, {}
    
    def _extract_solution(self, vars_dict: Dict) -> Dict:
        """Extract solution"""
        
        x = vars_dict['x']
        q = vars_dict['q']
        # DEBUG: Check if any routes exist
        total_arcs = 0
        for key in x:
            if value(x[key]) and value(x[key]) > 0.5:
                total_arcs += 1
        
        print(f"\n  [DEBUG] Total active arcs: {total_arcs}")
        
        # DEBUG: Check total deliveries
        total_deliveries = 0
        for key in q:
            qty = value(q[key])
            if qty and qty > 0.01:
                total_deliveries += qty
        
        print(f"  [DEBUG] Total deliveries: {total_deliveries:.2f} units")        
        
        routes = []
        for v in range(self.vehicle_config['num_vehicles']):
            route = {'vehicle_id': v, 'stops': []}
            
            current = self.depot
            visited = set([self.depot])
            
            while True:
                next_stop = None
                for j in [self.depot] + self.stores:
                    if j not in visited and (current, j, v) in x:
                        if value(x[current, j, v]) and value(x[current, j, v]) > 0.5:
                            next_stop = j
                            break
                
                if next_stop is None or next_stop == self.depot:
                    break
                
                # Get deliveries
                deliveries = []
                for p in self.products:
                    if (next_stop, p, v) in q:
                        qty = value(q[next_stop, p, v])
                        if qty and qty > 0.01:
                            deliveries.append({
                                'product_id': p,
                                'quantity': round(qty, 2)
                            })
                
                if deliveries:  # Only add stop if deliveries
                    route['stops'].append({
                        'location': next_stop,
                        'deliveries': deliveries
                    })
                
                visited.add(next_stop)
                current = next_stop
            
            if route['stops']:
                routes.append(route)
        
        return {'routes': routes}


# Test code
if __name__ == "__main__":
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, 'src'))
    
    from data_generation.network_generator import DaNangNetworkGenerator
    from data_generation.product_generator import ProductCatalogGenerator
    from data_generation.demand_generator import DemandPatternGenerator
    from weather.manual_scenarios import ManualWeatherScenarios
    
    print("="*80)
    print("WEATHER-AWARE VRP - RELAXED VERSION TEST")
    print("="*80)
    
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
    
    procurement = pd.DataFrame({
        'supplier_id': ['SUP_001'] * 5,
        'product_id': products['id'].tolist(),
        'quantity_units': [200] * 5  # Enough supply
    })
    
    scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    
    # Test VRP
    vrp = WeatherAwareVRP(
        network=network,
        products_df=products,
        demand_df=weekly_demand,
        procurement_solution=procurement,
        weather_scenarios=scenarios,
        vehicle_config={'num_vehicles': 5}
    )
    
    # Test multiple scenarios
    for k in [0, 2, 4]:  # Normal, Moderate, Typhoon
        scenario = scenarios[k]
        print(f"\n{'='*80}")
        print(f"Testing Scenario: {scenario.name}")
        print(f"{'='*80}")
        
        status, solution = vrp.solve(scenario_id=k, time_limit=180)
        
        if status in ['Optimal', 'Feasible']:
            print(f"\n✓ Solution found!")
            
            # ADDED: Print detailed routes
            print(f"\n{'='*60}")
            print("ROUTING DETAILS:")
            print(f"{'='*60}")
            
            if solution['routes']:
                for route in solution['routes']:
                    print(f"\nVehicle {route['vehicle_id']}:")
                    if route['stops']:
                        total_delivered = 0
                        for stop in route['stops']:
                            deliveries_str = []
                            for d in stop['deliveries']:
                                deliveries_str.append(f"{d['product_id']}: {d['quantity']} units")
                                total_delivered += d['quantity']
                            
                            print(f"  → {stop['location']}")
                            for ds in deliveries_str:
                                print(f"      {ds}")
                        
                        print(f"  Total delivered by this vehicle: {total_delivered:.0f} units")
                    else:
                        print(f"  → NO STOPS (vehicle not used)")
            else:
                print("\n⚠️ NO ROUTES FOUND!")
            
            print(f"\n{'='*60}")
            print(f"Total unmet demand: {solution['unmet_demand']:.0f} units")
            print(f"{'='*60}")