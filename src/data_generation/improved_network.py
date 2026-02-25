"""
Improved Network Generator - Higher capacities to meet demand
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from network_generator import DaNangNetworkGenerator


class ImprovedNetworkGenerator(DaNangNetworkGenerator):
    """Enhanced version with higher supplier capacities"""
    
    def generate_suppliers(self, n_suppliers: int = 5) -> pd.DataFrame:
        """Generate suppliers with HIGHER capacities"""
        
        suppliers = []
        
        # Increased capacity ranges (2-3x higher)
        archetypes = [
            {
                'name': 'Tho Quang Seafood',
                'type': 'seafood',
                'base_lat': 16.0833,
                'base_lon': 108.2167,
                'capacity_range': (2000, 3000),  # Was 800-1200
                'fixed_cost_range': (300000, 500000)
            },
            {
                'name': 'Hoa Vang Vegetables',
                'type': 'vegetables',
                'base_lat': 16.0167,
                'base_lon': 108.1167,
                'capacity_range': (1500, 2500),  # Was 600-1000
                'fixed_cost_range': (200000, 400000)
            },
            {
                'name': 'Hoa Khanh Meat',
                'type': 'meat',
                'base_lat': 16.0375,
                'base_lon': 108.1528,
                'capacity_range': (1200, 2000),  # Was 500-800
                'fixed_cost_range': (250000, 450000)
            },
            {
                'name': 'Lien Chieu Farm',
                'type': 'vegetables',
                'base_lat': 16.0750,
                'base_lon': 108.1500,
                'capacity_range': (1000, 1800),  # Was 400-700
                'fixed_cost_range': (150000, 350000)
            },
            {
                'name': 'Nam O Fishing Port',
                'type': 'seafood',
                'base_lat': 16.1167,
                'base_lon': 108.1667,
                'capacity_range': (1800, 2500),  # Was 700-1000
                'fixed_cost_range': (280000, 480000)
            }
        ]
        
        for i in range(min(n_suppliers, len(archetypes))):
            arch = archetypes[i]
            
            lat = arch['base_lat'] + np.random.normal(0, 0.01)
            lon = arch['base_lon'] + np.random.normal(0, 0.01)
            
            capacity = np.random.uniform(*arch['capacity_range'])
            fixed_cost = np.random.uniform(*arch['fixed_cost_range'])
            
            suppliers.append({
                'id': f'SUP_{i+1:03d}',
                'name': f"{arch['name']} {i+1}",
                'type': 'supplier',
                'subtype': arch['type'],
                'latitude': lat,
                'longitude': lon,
                'capacity_kg_per_day': round(capacity, 2),
                'fixed_cost_vnd': round(fixed_cost, 0)
            })
        
        return pd.DataFrame(suppliers)


if __name__ == "__main__":
    # Test improved generator
    gen = ImprovedNetworkGenerator(seed=42)
    network = gen.generate_network(n_suppliers=5, n_dcs=2, n_stores=8)
    
    print("\n" + "="*70)
    print("IMPROVED NETWORK - HIGHER CAPACITIES")
    print("="*70)
    
    print("\nSuppliers:")
    print(network['suppliers'][['id', 'name', 'subtype', 'capacity_kg_per_day']].to_string(index=False))
    
    total_capacity = network['suppliers']['capacity_kg_per_day'].sum()
    print(f"\nTotal Supplier Capacity: {total_capacity:,.0f} kg/day")
    
    # Save
    gen.save_network(network, '../../../data/synthetic')
    print("\n✓ Saved improved network!")
