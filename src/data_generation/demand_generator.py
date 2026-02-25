"""
Demand Generator - Creates realistic demand patterns for stores and products
UPDATED: Reduced baseline demand by 60% to match supplier capacity
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import json


class DemandPatternGenerator:
    """
    Generates realistic demand patterns for fresh food retail
    
    UPDATED: Baseline demand reduced to 60% of original levels
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def generate_demand_plan(self,
                            stores_df: pd.DataFrame,
                            products_df: pd.DataFrame,
                            planning_horizon_days: int = 30) -> pd.DataFrame:
        """Generate daily demand forecast for each store-product combination"""
        
        demand_data = []
        
        start_date = pd.Timestamp('2024-10-01')
        dates = pd.date_range(start=start_date, periods=planning_horizon_days, freq='D')
        
        for date in dates:
            day_of_week = date.dayofweek
            dow_multiplier = self._get_dow_multiplier(day_of_week)
            
            for _, store in stores_df.iterrows():
                store_factor = store['demand_factor']
                
                for _, product in products_df.iterrows():
                    # REDUCED base demand
                    base_demand = self._get_base_demand(
                        product['category'],
                        store_factor
                    )
                    
                    seasonal_mult = self._get_seasonal_multiplier(
                        product['category'],
                        date.month
                    )
                    
                    expected_demand = base_demand * store_factor * dow_multiplier * seasonal_mult
                    
                    # Stochastic variation
                    cv = 0.40
                    shape = 1 / (cv ** 2)
                    scale = expected_demand / shape
                    
                    actual_demand = np.random.gamma(shape, scale)
                    actual_demand = max(0, round(actual_demand))
                    
                    demand_data.append({
                        'date': date.date(),
                        'store_id': store['id'],
                        'product_id': product['id'],
                        'product_category': product['category'],
                        'demand_units': actual_demand,
                        'expected_demand': round(expected_demand, 2)
                    })
        
        return pd.DataFrame(demand_data)
    
    def _get_base_demand(self, category: str, store_factor: float) -> float:
        """
        EXTREME REDUCTION: 40% of already-reduced levels
        
        Previous (60% reduction):
        - Seafood: 18-36
        - Vegetable: 30-60
        - Meat: 24-48
        - Fruit: 24-42
        
        NEW (76% total reduction from original):
        - Seafood: 10-20
        - Vegetable: 15-30
        - Meat: 12-24
        - Fruit: 12-20
        """
        base_demands = {
            'seafood': np.random.uniform(10, 20),      # Was 18-36
            'vegetable': np.random.uniform(15, 30),    # Was 30-60
            'meat': np.random.uniform(12, 24),         # Was 24-48
            'fruit': np.random.uniform(12, 20)         # Was 24-42
        }
        
        return base_demands.get(category, 15)
    
    def _get_dow_multiplier(self, day_of_week: int) -> float:
        """Day-of-week demand multiplier"""
        dow_multipliers = {
            0: 0.95,  # Monday
            1: 0.93,  # Tuesday
            2: 0.94,  # Wednesday
            3: 0.97,  # Thursday
            4: 1.10,  # Friday
            5: 1.25,  # Saturday
            6: 1.20   # Sunday
        }
        return dow_multipliers.get(day_of_week, 1.0)
    
    def _get_seasonal_multiplier(self, category: str, month: int) -> float:
        """Seasonal demand variation by product category"""
        
        patterns = {
            'seafood': {
                1: 0.9, 2: 0.9, 3: 0.95, 4: 1.0,
                5: 1.15, 6: 1.20, 7: 1.20, 8: 1.15,
                9: 1.0, 10: 0.95, 11: 0.95, 12: 1.0
            },
            'vegetable': {
                1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
                5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0,
                9: 1.0, 10: 1.0, 11: 1.0, 12: 1.05
            },
            'meat': {
                1: 1.10, 2: 1.05, 3: 1.0, 4: 1.0,
                5: 0.95, 6: 0.95, 7: 0.95, 8: 0.95,
                9: 1.0, 10: 1.0, 11: 1.05, 12: 1.15
            },
            'fruit': {
                1: 0.9, 2: 0.9, 3: 0.95, 4: 1.05,
                5: 1.15, 6: 1.20, 7: 1.20, 8: 1.15,
                9: 1.05, 10: 1.0, 11: 0.95, 12: 0.95
            }
        }
        
        category_pattern = patterns.get(category, {m: 1.0 for m in range(1, 13)})
        return category_pattern.get(month, 1.0)
    
    def aggregate_to_weekly(self, daily_demand_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily demand to weekly for procurement planning"""
        weekly = daily_demand_df.copy()
        weekly['week'] = pd.to_datetime(weekly['date']).dt.isocalendar().week
        
        aggregated = weekly.groupby(['week', 'store_id', 'product_id']).agg({
            'demand_units': 'sum',
            'expected_demand': 'sum',
            'product_category': 'first'
        }).reset_index()
        
        return aggregated
    
    def compute_demand_stats(self, demand_df: pd.DataFrame) -> dict:
        """Compute summary statistics for demand patterns"""
        
        stats = {
            'total_demand_units': int(demand_df['demand_units'].sum()),
            'planning_horizon_days': demand_df['date'].nunique(),
            'avg_daily_demand_per_store_product': demand_df.groupby(['store_id', 'product_id'])['demand_units'].mean().mean(),
            'demand_by_category': demand_df.groupby('product_category')['demand_units'].sum().to_dict(),
            'coefficient_of_variation': {
                'overall': float(demand_df['demand_units'].std() / demand_df['demand_units'].mean())
            },
            'zero_demand_pct': float((demand_df['demand_units'] == 0).sum() / len(demand_df) * 100)
        }
        
        return stats
    
    def save_demand_plan(self, 
                        daily_demand_df: pd.DataFrame,
                        output_dir: str = '../data/synthetic'):
        """Save demand plan to CSV"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        daily_demand_df.to_csv(f'{output_dir}/daily_demand.csv', index=False)
        
        weekly_demand = self.aggregate_to_weekly(daily_demand_df)
        weekly_demand.to_csv(f'{output_dir}/weekly_demand.csv', index=False)
        
        stats = self.compute_demand_stats(daily_demand_df)
        with open(f'{output_dir}/demand_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Demand plan saved to {output_dir}/")


if __name__ == "__main__":
    from network_generator import DaNangNetworkGenerator
    from product_generator import ProductCatalogGenerator
    
    network_gen = DaNangNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=6, n_dcs=2, n_stores=8)
    
    product_gen = ProductCatalogGenerator(seed=42)
    products = product_gen.generate_products(n_products=10)
    
    demand_gen = DemandPatternGenerator(seed=42)
    daily_demand = demand_gen.generate_demand_plan(
        stores_df=network['stores'],
        products_df=products,
        planning_horizon_days=30
    )
    
    print("\n" + "="*60)
    print("DEMAND PLAN SUMMARY (REDUCED 60%)")
    print("="*60)
    print(f"\nTotal records: {len(daily_demand)}")
    print(f"Total demand: {daily_demand['demand_units'].sum():,.0f} units")
    print(f"Daily average: {daily_demand['demand_units'].sum()/30:,.0f} units/day")
    
    # Calculate in kg
    merged = daily_demand.merge(products, left_on='product_id', right_on='id')
    total_kg = (merged['demand_units'] * merged['weight_kg_per_unit']).sum()
    print(f"\nTotal: {total_kg:,.0f} kg over 30 days")
    print(f"Daily: {total_kg/30:,.0f} kg/day")
    
    demand_gen.save_demand_plan(daily_demand, '../data/synthetic')