"""
Network Generator - Creates realistic Da Nang supply chain topology
Generates suppliers, distribution centers, and retail stores with geographic coordinates
INCLUDES: 1 general supplier that can supply all products
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import json


@dataclass
class Location:
    """Represents a location in the supply chain network"""
    id: str
    name: str
    type: str  # 'supplier', 'dc', 'store'
    latitude: float
    longitude: float
    capacity: float = None
    fixed_cost: float = 0.0


class DaNangNetworkGenerator:
    """
    Generates a realistic supply chain network for Da Nang
    
    Geographic context:
    - Da Nang center: ~16.0544°N, 108.2022°E
    - Suppliers: Rural areas (farms, fishing ports) + 1 general wholesale market
    - Distribution centers: Industrial zones
    - Stores: Urban retail clusters
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
        # Da Nang center coordinates
        self.center_lat = 16.0544
        self.center_lon = 108.2022
        
    def generate_suppliers(self, n_suppliers: int = 9) -> pd.DataFrame:
        """
        Generate supplier locations.
        UPDATED: 9 suppliers (7 specialized + 2 general/wholesale)
        for improved network coverage and reduced concentration constraint pressure.
        """
        suppliers = []

        # 7 Specialized + 2 General suppliers
        archetypes = [
            # --- Original 5 specialized ---
            {
                'name': 'Tho Quang Seafood',
                'type': 'seafood',
                'base_lat': 16.0833,
                'base_lon': 108.2167,
                'capacity_range': (2000, 3000),
                'fixed_cost_range': (300000, 500000)
            },
            {
                'name': 'Hoa Vang Vegetables',
                'type': 'vegetables',
                'base_lat': 16.0167,
                'base_lon': 108.1167,
                'capacity_range': (1500, 2500),
                'fixed_cost_range': (200000, 400000)
            },
            {
                'name': 'Hoa Khanh Meat',
                'type': 'meat',
                'base_lat': 16.0375,
                'base_lon': 108.1528,
                'capacity_range': (1200, 2000),
                'fixed_cost_range': (250000, 450000)
            },
            {
                'name': 'Lien Chieu Farm',
                'type': 'vegetables',
                'base_lat': 16.0750,
                'base_lon': 108.1500,
                'capacity_range': (1000, 1800),
                'fixed_cost_range': (150000, 350000)
            },
            {
                'name': 'Nam O Fishing Port',
                'type': 'seafood',
                'base_lat': 16.1167,
                'base_lon': 108.1667,
                'capacity_range': (1800, 2500),
                'fixed_cost_range': (280000, 480000)
            },
            # --- NEW: 3 additional suppliers ---
            {
                # Second seafood: My Khe beach fishing port (east coast, different zone)
                'name': 'My Khe Fishing Cooperative',
                'type': 'seafood',
                'base_lat': 16.0600,
                'base_lon': 108.2450,
                'capacity_range': (1200, 1800),
                'fixed_cost_range': (200000, 380000)
            },
            {
                # Second meat: Cam Le industrial slaughterhouse district
                'name': 'Cam Le Livestock',
                'type': 'meat',
                'base_lat': 16.0230,
                'base_lon': 108.1980,
                'capacity_range': (900, 1500),
                'fixed_cost_range': (220000, 400000)
            },
            {
                # Second general wholesale: Ngu Hanh Son area (south Da Nang)
                'name': 'Ngu Hanh Son Wholesale',
                'type': 'general',
                'base_lat': 15.9980,
                'base_lon': 108.2490,
                'capacity_range': (700, 1100),
                'fixed_cost_range': (1200000, 1600000)
            },
        ]

        # Generate all specialized/general suppliers up to n_suppliers
        for i in range(min(len(archetypes), n_suppliers)):
            arch = archetypes[i]

            lat = arch['base_lat'] + np.random.normal(0, 0.01)
            lon = arch['base_lon'] + np.random.normal(0, 0.01)

            capacity   = np.random.uniform(*arch['capacity_range'])
            fixed_cost = np.random.uniform(*arch['fixed_cost_range'])

            suppliers.append({
                'id':                  f'SUP_{i+1:03d}',
                'name':                f"{arch['name']} {i+1}",
                'type':                'supplier',
                'subtype':             arch['type'],
                'latitude':            lat,
                'longitude':           lon,
                'capacity_kg_per_day': round(capacity, 2),
                'fixed_cost_vnd':      round(fixed_cost, 0)
            })

        # Legacy: if caller explicitly passes n_suppliers >= 6 but < len(archetypes),
        # still add the original general wholesale market (SUP_006) for backward compat.
        if n_suppliers == 6 and len(suppliers) < 6:
            suppliers.append({
                'id': 'SUP_006',
                'name': 'Da Nang Wholesale Market',
                'type': 'supplier',
                'subtype': 'general',
                'latitude': self.center_lat,
                'longitude': self.center_lon,
                'capacity_kg_per_day': 900.0,
                'fixed_cost_vnd': 1500000.0
            })

        return pd.DataFrame(suppliers)

    
    def generate_distribution_centers(self, n_dcs: int = 2) -> pd.DataFrame:
        """Generate distribution center locations"""
        dcs = []
        
        dc_locations = [
            {
                'name': 'Hoa Khanh DC',
                'lat': 16.0319,
                'lon': 108.1497,
                'capacity': 5000,
                'fixed_cost': 1500000
            },
            {
                'name': 'Lien Chieu DC',
                'lat': 16.0806,
                'lon': 108.1453,
                'capacity': 3000,
                'fixed_cost': 1000000
            }
        ]
        
        for i in range(min(n_dcs, len(dc_locations))):
            loc = dc_locations[i]
            
            dcs.append({
                'id': f'DC_{i+1:03d}',
                'name': loc['name'],
                'type': 'distribution_center',
                'latitude': loc['lat'],
                'longitude': loc['lon'],
                'capacity_kg_per_day': loc['capacity'],
                'fixed_cost_vnd': loc['fixed_cost']
            })
        
        return pd.DataFrame(dcs)
    
    def generate_stores(self, n_stores: int = 8) -> pd.DataFrame:
        """Generate retail store locations"""
        stores = []
        
        clusters = [
            {'name': 'Hai Chau Downtown', 'lat': 16.0678, 'lon': 108.2208, 'demand_factor': 1.2},
            {'name': 'Son Tra Beach', 'lat': 16.0833, 'lon': 108.2500, 'demand_factor': 1.0},
            {'name': 'Ngu Hanh Son', 'lat': 16.0019, 'lon': 108.2517, 'demand_factor': 0.9},
            {'name': 'Cam Le Residential', 'lat': 16.0247, 'lon': 108.1956, 'demand_factor': 0.8},
            {'name': 'Thanh Khe Urban', 'lat': 16.0614, 'lon': 108.1878, 'demand_factor': 1.1},
            {'name': 'Lien Chieu Suburb', 'lat': 16.0944, 'lon': 108.1556, 'demand_factor': 0.7},
            {'name': 'Hoa Vang Rural', 'lat': 16.0500, 'lon': 108.0833, 'demand_factor': 0.6},
            {'name': 'Son Tra Hills', 'lat': 16.1000, 'lon': 108.2667, 'demand_factor': 0.85}
        ]
        
        for i in range(min(n_stores, len(clusters))):
            cluster = clusters[i]
            
            lat = cluster['lat'] + np.random.normal(0, 0.005)
            lon = cluster['lon'] + np.random.normal(0, 0.005)
            
            stores.append({
                'id': f'STORE_{i+1:03d}',
                'name': f"{cluster['name']} Store {i+1}",
                'type': 'store',
                'latitude': lat,
                'longitude': lon,
                'demand_factor': cluster['demand_factor']
            })
        
        return pd.DataFrame(stores)
    
    def compute_distance_matrix(self, locations_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Haversine distance matrix between all locations"""
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            
            a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            DETOUR_FACTOR = 1.5
            return R * c * DETOUR_FACTOR
        
        n = len(locations_df)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i, j] = haversine(
                        locations_df.iloc[i]['latitude'],
                        locations_df.iloc[i]['longitude'],
                        locations_df.iloc[j]['latitude'],
                        locations_df.iloc[j]['longitude']
                    )
        
        distance_df = pd.DataFrame(
            distance_matrix,
            index=locations_df['id'].values,
            columns=locations_df['id'].values
        )
        
        return distance_df
    
    def generate_network(self, 
                        n_suppliers: int = 6,  # UPDATED: default 6 (was 5)
                        n_dcs: int = 2, 
                        n_stores: int = 8) -> dict:
        """Generate complete supply chain network"""
        print(f"Generating Da Nang supply chain network...")
        print(f"  - {n_suppliers} suppliers (including 1 general)")
        print(f"  - {n_dcs} distribution centers")
        print(f"  - {n_stores} retail stores")
        
        suppliers = self.generate_suppliers(n_suppliers)
        dcs = self.generate_distribution_centers(n_dcs)
        stores = self.generate_stores(n_stores)
        
        all_locations = pd.concat([suppliers, dcs, stores], ignore_index=True)
        distance_matrix = self.compute_distance_matrix(all_locations)
        
        print(f"✓ Generated {len(all_locations)} locations")
        print(f"✓ Computed {len(distance_matrix)}x{len(distance_matrix)} distance matrix")
        
        metadata = {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'seed': self.seed,
            'center_coordinates': {
                'latitude': self.center_lat,
                'longitude': self.center_lon
            },
            'counts': {
                'suppliers': len(suppliers),
                'distribution_centers': len(dcs),
                'stores': len(stores),
                'total_locations': len(all_locations)
            },
            'distance_range_km': {
                'min': distance_matrix[distance_matrix > 0].min().min(),
                'max': distance_matrix.max().max(),
                'mean': distance_matrix[distance_matrix > 0].mean().mean()
            }
        }
        
        return {
            'suppliers': suppliers,
            'dcs': dcs,
            'stores': stores,
            'all_locations': all_locations,
            'distance_matrix': distance_matrix,
            'metadata': metadata
        }
    
    def save_network(self, network: dict, output_dir: str = '../data/synthetic'):
        """Save network data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        network['suppliers'].to_csv(f'{output_dir}/suppliers.csv', index=False)
        network['dcs'].to_csv(f'{output_dir}/distribution_centers.csv', index=False)
        network['stores'].to_csv(f'{output_dir}/stores.csv', index=False)
        network['all_locations'].to_csv(f'{output_dir}/network_topology.csv', index=False)
        network['distance_matrix'].to_csv(f'{output_dir}/distance_matrix.csv')
        
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(network['metadata'], f, indent=2)
        
        print(f"\n✓ Network saved to {output_dir}/")


if __name__ == "__main__":
    generator = DaNangNetworkGenerator(seed=42)
    network = generator.generate_network(n_suppliers=6, n_dcs=2, n_stores=8)
    
    print("\n" + "="*60)
    print("NETWORK SUMMARY")
    print("="*60)
    print(f"\nSuppliers ({len(network['suppliers'])}):")
    print(network['suppliers'][['id', 'name', 'subtype', 'capacity_kg_per_day']].to_string(index=False))
    
    print(f"\nTotal Capacity: {network['suppliers']['capacity_kg_per_day'].sum():,.0f} kg/day")
    
    generator.save_network(network, '../data/synthetic')