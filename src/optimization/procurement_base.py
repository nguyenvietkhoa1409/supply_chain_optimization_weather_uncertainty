"""
Procurement Optimization - Deterministic Base Model
MILP formulation for fresh food procurement with supplier selection and capacity constraints
"""

import pandas as pd
import numpy as np
from pulp import *
from typing import Dict, List, Tuple
import time


class ProcurementOptimizer:
    """
    Deterministic procurement optimization model
    
    Decision variables:
    - x[s,p]: procurement quantity from supplier s for product p
    - y[s,p]: binary, 1 if supplier s is used for product p (for fixed cost)
    
    Objective:
    Minimize total cost = variable cost + fixed ordering cost + penalty for unmet demand
    
    Constraints:
    - Demand satisfaction: sum over suppliers >= demand
    - Supplier capacity: sum of product weights <= supplier capacity  
    - MOQ logic: if y[s,p]=1, then x[s,p] >= MOQ
    - Non-negativity
    """
    
    def __init__(self, 
                 network: Dict,
                 products_df: pd.DataFrame,
                 supplier_product_df: pd.DataFrame,
                 demand_df: pd.DataFrame):
        """
        Args:
            network: Dict with 'suppliers', 'stores', 'distance_matrix'
            products_df: Product catalog
            supplier_product_df: Supplier-product availability matrix
            demand_df: Demand forecast (aggregated to weekly/period level)
        """
        self.network = network
        self.products_df = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df = demand_df
        
        # Extract key data
        self.suppliers = network['suppliers']['id'].tolist()
        self.products = products_df['id'].tolist()
        
        # Create lookup dicts for fast access
        self._create_lookup_dicts()
        
        # Aggregate demand by product (sum across stores for procurement)
        self.total_demand = demand_df.groupby('product_id')['demand_units'].sum().to_dict()
    
    def _create_lookup_dicts(self):
        """Create fast lookup dictionaries"""
        
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
        
        # Supplier-product matrix (cost, MOQ, availability)
        self.sp_cost = {}
        self.sp_moq = {}
        self.sp_available = {}
        
        for _, row in self.supplier_product_df.iterrows():
            s, p = row['supplier_id'], row['product_id']
            self.sp_cost[(s, p)] = row['unit_cost_vnd']
            self.sp_moq[(s, p)] = row['moq_units']
            self.sp_available[(s, p)] = row['available']
    
    def build_model(self) -> LpProblem:
        """
        Build deterministic procurement MILP
        
        Returns:
            PuLP LpProblem instance
        """
        print("Building procurement optimization model...")
        
        # Create problem
        model = LpProblem("Procurement_Optimization", LpMinimize)
        
        # Decision variables
        print("  Creating decision variables...")
        
        # x[s,p]: procurement quantity
        x = LpVariable.dicts("procure",
                            ((s, p) for s in self.suppliers for p in self.products),
                            lowBound=0,
                            cat='Continuous')
        
        # y[s,p]: binary activation variable for fixed cost
        y = LpVariable.dicts("use_supplier",
                            ((s, p) for s in self.suppliers for p in self.products),
                            cat='Binary')
        
        # u[p]: unmet demand (penalty variable)
        u = LpVariable.dicts("unmet_demand",
                            self.products,
                            lowBound=0,
                            cat='Continuous')
        
        # Objective function
        print("  Formulating objective...")
        
        variable_cost = lpSum([
            self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False)
        ])
        
        fixed_cost = lpSum([
            self.supplier_fixed_cost[s] * y[s, p]
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False)
        ])
        
        # Penalty for unmet demand (5x unit price to strongly discourage)
        penalty_cost = lpSum([
            5.0 * self.product_cost[p] * u[p]
            for p in self.products
        ])
        
        model += variable_cost + fixed_cost + penalty_cost, "Total_Cost"
        
        # Constraints
        print("  Adding constraints...")
        
        # 1. Demand satisfaction
        for p in self.products:
            demand = self.total_demand.get(p, 0)
            if demand > 0:
                model += (
                    lpSum([x[s, p] for s in self.suppliers
                          if self.sp_available.get((s, p), False)]) + u[p] >= demand,
                    f"Demand_{p}"
                )
        
        # 2. Supplier capacity (weight-based)
        for s in self.suppliers:
            capacity = self.supplier_capacity[s]
            model += (
                lpSum([x[s, p] * self.product_weight[p] 
                      for p in self.products
                      if self.sp_available.get((s, p), False)]) <= capacity,
                f"Capacity_{s}"
            )
        
        # 3. MOQ logic: if y[s,p]=1, then x[s,p] >= MOQ
        M = 100000  # Big-M for logical constraints
        for s in self.suppliers:
            for p in self.products:
                if self.sp_available.get((s, p), False):
                    moq = self.sp_moq.get((s, p), 0)
                    
                    # x[s,p] >= MOQ * y[s,p]
                    model += (x[s, p] >= moq * y[s, p], f"MOQ_lower_{s}_{p}")
                    
                    # x[s,p] <= M * y[s,p] (if not used, force to 0)
                    model += (x[s, p] <= M * y[s, p], f"MOQ_upper_{s}_{p}")
        
        print(f"✓ Model built: {model.numVariables()} variables, {model.numConstraints()} constraints")
        
        return model, {'x': x, 'y': y, 'u': u}
    
    def solve(self, 
             time_limit: int = 300,
             gap_tolerance: float = 0.01,
             solver_name: str = 'CBC') -> Tuple[str, Dict]:
        """
        Solve the optimization model
        
        Args:
            time_limit: Maximum solve time in seconds
            gap_tolerance: MIP optimality gap tolerance (default 1%)
            solver_name: 'CBC' (default, open-source) or 'GUROBI' (if available)
            
        Returns:
            (status, solution_dict)
        """
        # Build model
        model, vars_dict = self.build_model()
        
        # Select solver
        if solver_name.upper() == 'CBC':
            solver = PULP_CBC_CMD(
                timeLimit=time_limit,
                gapRel=gap_tolerance,
                msg=1
            )
        elif solver_name.upper() == 'GUROBI':
            try:
                solver = GUROBI_CMD(
                    timeLimit=time_limit,
                    mip=gap_tolerance,
                    msg=1
                )
            except:
                print("Gurobi not available, falling back to CBC")
                solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap_tolerance, msg=1)
        else:
            solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap_tolerance, msg=1)
        
        # Solve
        print(f"\nSolving with {solver_name}...")
        start_time = time.time()
        model.solve(solver)
        solve_time = time.time() - start_time
        
        # Extract status
        status = LpStatus[model.status]
        print(f"\n✓ Status: {status}")
        print(f"✓ Solve time: {solve_time:.2f} seconds")
        
        if status == 'Optimal':
            obj_value = value(model.objective)
            print(f"✓ Objective value: {obj_value:,.0f} VND")
            
            # Extract solution
            solution = self._extract_solution(vars_dict)
            solution['objective_value'] = obj_value
            solution['solve_time'] = solve_time
            solution['status'] = status
            
            return status, solution
        else:
            print(f"⚠ Model did not solve to optimality")
            return status, {}
    
    def _extract_solution(self, vars_dict: Dict) -> Dict:
        """Extract solution from PuLP variables"""
        
        x = vars_dict['x']
        y = vars_dict['y']
        u = vars_dict['u']
        
        # Procurement decisions
        procurement = []
        for s in self.suppliers:
            for p in self.products:
                qty = value(x[s, p])
                if qty is not None and qty > 0.01:  # Filter out numerical zeros
                    procurement.append({
                        'supplier_id': s,
                        'product_id': p,
                        'quantity_units': round(qty, 2),
                        'cost_vnd': round(qty * self.sp_cost.get((s, p), self.product_cost[p]), 0)
                    })
        
        # Supplier usage
        supplier_usage = []
        for s in self.suppliers:
            products_used = [p for p in self.products if value(y[s, p]) and value(y[s, p]) > 0.5]
            if products_used:
                supplier_usage.append({
                    'supplier_id': s,
                    'products_supplied': products_used,
                    'num_products': len(products_used),
                    'fixed_cost_vnd': self.supplier_fixed_cost[s]
                })
        
        # Unmet demand
        unmet = []
        for p in self.products:
            shortage = value(u[p])
            if shortage is not None and shortage > 0.01:
                unmet.append({
                    'product_id': p,
                    'unmet_quantity': round(shortage, 2),
                    'penalty_cost_vnd': round(shortage * 5.0 * self.product_cost[p], 0)
                })
        
        return {
            'procurement': pd.DataFrame(procurement),
            'supplier_usage': pd.DataFrame(supplier_usage),
            'unmet_demand': pd.DataFrame(unmet) if unmet else pd.DataFrame()
        }


if __name__ == "__main__":
    # Test with generated data
    import sys
    sys.path.append('..')
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from data_generation.network_generator import DaNangNetworkGenerator
    from data_generation.product_generator import ProductCatalogGenerator
    from data_generation.demand_generator import DemandPatternGenerator
    
    print("="*70)
    print("PROCUREMENT OPTIMIZATION TEST")
    print("="*70)
    
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
    
    # Optimize
    print("\nInitializing optimizer...")
    optimizer = ProcurementOptimizer(
        network=network,
        products_df=products,
        supplier_product_df=supplier_product,
        demand_df=weekly_demand
    )
    
    status, solution = optimizer.solve(time_limit=60)
    
    if status == 'Optimal':
        print("\n" + "="*70)
        print("SOLUTION SUMMARY")
        print("="*70)
        
        print("\nProcurement Plan:")
        print(solution['procurement'].to_string(index=False))
        
        print("\n\nSupplier Usage:")
        print(solution['supplier_usage'].to_string(index=False))
        
        if not solution['unmet_demand'].empty:
            print("\n\nUnmet Demand:")
            print(solution['unmet_demand'].to_string(index=False))
        else:
            print("\n\n✓ All demand satisfied!")
