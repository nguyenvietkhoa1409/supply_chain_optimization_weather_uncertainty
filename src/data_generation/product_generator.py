"""
Product Generator - Creates fresh food product catalog for Da Nang
UPDATED: Ensures EVERY product has at least 2 suppliers (using general supplier)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import json


@dataclass
class Product:
    """Represents a fresh food product"""
    id: str
    name: str
    category: str
    shelf_life_days: float
    unit_cost_vnd: float
    unit_price_vnd: float
    weight_kg_per_unit: float
    temperature_sensitivity: str


class ProductCatalogGenerator:
    """Generates realistic fresh food product catalog for Da Nang market"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_products(self, n_products: int = 10) -> pd.DataFrame:
        """Generate product catalog"""
        
        product_templates = [
            # Seafood
            {
                'name': 'Fresh Fish (Cá)',
                'category': 'seafood',
                'shelf_life_range': (1, 2),
                'cost_range': (80000, 120000),
                'margin': 0.35,
                'weight_per_unit': 0.5,
                'temp_sensitivity': 'high'
            },
            {
                'name': 'Fresh Shrimp (Tôm)',
                'category': 'seafood',
                'shelf_life_range': (1, 2),
                'cost_range': (150000, 200000),
                'margin': 0.40,
                'weight_per_unit': 0.3,
                'temp_sensitivity': 'high'
            },
            {
                'name': 'Squid (Mực)',
                'category': 'seafood',
                'shelf_life_range': (1, 3),
                'cost_range': (100000, 140000),
                'margin': 0.35,
                'weight_per_unit': 0.4,
                'temp_sensitivity': 'high'
            },
            # Vegetables
            {
                'name': 'Leafy Greens (Rau xanh)',
                'category': 'vegetable',
                'shelf_life_range': (3, 5),
                'cost_range': (15000, 25000),
                'margin': 0.30,
                'weight_per_unit': 0.2,
                'temp_sensitivity': 'medium'
            },
            {
                'name': 'Tomatoes (Cà chua)',
                'category': 'vegetable',
                'shelf_life_range': (4, 7),
                'cost_range': (20000, 30000),
                'margin': 0.25,
                'weight_per_unit': 0.3,
                'temp_sensitivity': 'medium'
            },
            {
                'name': 'Cabbage (Bắp cải)',
                'category': 'vegetable',
                'shelf_life_range': (5, 8),
                'cost_range': (12000, 20000),
                'margin': 0.28,
                'weight_per_unit': 0.5,
                'temp_sensitivity': 'low'
            },
            # Meat
            {
                'name': 'Chicken (Gà)',
                'category': 'meat',
                'shelf_life_range': (2, 3),
                'cost_range': (60000, 80000),
                'margin': 0.30,
                'weight_per_unit': 0.6,
                'temp_sensitivity': 'high'
            },
            {
                'name': 'Pork (Thịt heo)',
                'category': 'meat',
                'shelf_life_range': (2, 4),
                'cost_range': (70000, 90000),
                'margin': 0.32,
                'weight_per_unit': 0.5,
                'temp_sensitivity': 'high'
            },
            {
                'name': 'Beef (Thịt bò)',
                'category': 'meat',
                'shelf_life_range': (2, 4),
                'cost_range': (180000, 220000),
                'margin': 0.35,
                'weight_per_unit': 0.4,
                'temp_sensitivity': 'high'
            },
            # Fruits
            {
                'name': 'Dragon Fruit (Thanh long)',
                'category': 'fruit',
                'shelf_life_range': (5, 10),
                'cost_range': (25000, 35000),
                'margin': 0.30,
                'weight_per_unit': 0.4,
                'temp_sensitivity': 'low'
            }
        ]
        
        products = []
        
        for i in range(min(n_products, len(product_templates))):
            template = product_templates[i]
            
            shelf_life = np.random.uniform(*template['shelf_life_range'])
            cost = np.random.uniform(*template['cost_range'])
            price = cost * (1 + template['margin'])
            
            products.append({
                'id': f'PROD_{i+1:03d}',
                'name': template['name'],
                'category': template['category'],
                'shelf_life_days': round(shelf_life, 1),
                'unit_cost_vnd': round(cost, 0),
                'unit_price_vnd': round(price, 0),
                'margin_pct': round(template['margin'] * 100, 1),
                'weight_kg_per_unit': template['weight_per_unit'],
                'temperature_sensitivity': template['temp_sensitivity']
            })
        
        return pd.DataFrame(products)
    
    def generate_supplier_product_matrix(self, 
                                        suppliers_df: pd.DataFrame,
                                        products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate supplier-product availability matrix
        
        CRITICAL FIX: Ensures EVERY product has at least 2 suppliers
        Strategy: General supplier (subtype='general') can supply ALL products
        """
        
        matrix_data = []
        product_coverage = {p: [] for p in products_df['id'].tolist()}
        
        # Pass 1: Specialized suppliers with their primary products
        for _, supplier in suppliers_df.iterrows():
            supplier_type = supplier.get('subtype', 'general')
            supplier_id = supplier['id']
            
            # Skip general supplier in this pass
            if supplier_type == 'general':
                continue
            
            for _, product in products_df.iterrows():
                product_category = product['category']
                product_id = product['id']
                
                can_supply = False
                
                # Primary matches (100% probability)
                if supplier_type == 'seafood' and product_category == 'seafood':
                    can_supply = True
                elif supplier_type == 'vegetables' and product_category in ['vegetable', 'fruit']:
                    can_supply = True
                elif supplier_type == 'meat' and product_category == 'meat':
                    can_supply = True
                
                # Cross-category matching (70% probability for flexibility)
                elif supplier_type == 'meat' and product_category in ['seafood', 'fruit']:
                    if np.random.random() < 0.7:
                        can_supply = True
                elif supplier_type == 'seafood' and product_category in ['meat', 'fruit']:
                    if np.random.random() < 0.7:
                        can_supply = True
                elif supplier_type == 'vegetables' and product_category in ['meat', 'seafood']:
                    if np.random.random() < 0.6:
                        can_supply = True
                
                if can_supply:
                    base_cost = product['unit_cost_vnd']
                    supplier_cost = base_cost * np.random.uniform(0.85, 1.15)
                    moq = np.random.randint(5, 20)  # LOW MOQs
                    
                    matrix_data.append({
                        'supplier_id': supplier_id,
                        'product_id': product_id,
                        'unit_cost_vnd': round(supplier_cost, 0),
                        'moq_units': moq,
                        'lead_time_days': round(np.random.uniform(0.5, 2.0), 1),
                        'available': True
                    })
                    
                    product_coverage[product_id].append(supplier_id)
        
        # Pass 2: General supplier supplies EVERYTHING
        general_suppliers = suppliers_df[suppliers_df['subtype'] == 'general']
        
        if len(general_suppliers) > 0:
            general_id = general_suppliers.iloc[0]['id']
            print(f"\n✓ Adding general supplier {general_id} for ALL products")
            
            for _, product in products_df.iterrows():
                base_cost = product['unit_cost_vnd']
                # General supplier: 10-15% markup (convenience premium)
                supplier_cost = base_cost * np.random.uniform(1.10, 1.15)
                moq = np.random.randint(10, 30)
                
                matrix_data.append({
                    'supplier_id': general_id,
                    'product_id': product['id'],
                    'unit_cost_vnd': round(supplier_cost, 0),
                    'moq_units': moq,
                    'lead_time_days': round(np.random.uniform(1.0, 2.5), 1),
                    'available': True
                })
                
                product_coverage[product['id']].append(general_id)
        
        df = pd.DataFrame(matrix_data)
        
        # Validation: Check coverage
        print("\nSupplier-Product Coverage Validation:")
        all_covered = True
        for _, prod in products_df.iterrows():
            suppliers = df[df['product_id'] == prod['id']]
            n_suppliers = len(suppliers)
            
            if n_suppliers >= 2:
                print(f"  ✓ {prod['id']} ({prod['name']}): {n_suppliers} suppliers")
            elif n_suppliers == 1:
                print(f"  ⚠ {prod['id']} ({prod['name']}): only 1 supplier")
                all_covered = False
            else:
                print(f"  ❌ {prod['id']} ({prod['name']}): NO suppliers!")
                all_covered = False
        
        if all_covered:
            print("\n✓ ALL products have at least 2 suppliers!")
        else:
            print("\n⚠ WARNING: Some products have insufficient coverage!")
        
        return df
    
    def compute_product_stats(self, products_df: pd.DataFrame) -> dict:
        """Compute aggregate statistics for the product catalog"""
        stats = {
            'total_products': len(products_df),
            'by_category': products_df['category'].value_counts().to_dict(),
            'shelf_life': {
                'min_days': products_df['shelf_life_days'].min(),
                'max_days': products_df['shelf_life_days'].max(),
                'mean_days': products_df['shelf_life_days'].mean()
            },
            'pricing': {
                'avg_cost_vnd': products_df['unit_cost_vnd'].mean(),
                'avg_price_vnd': products_df['unit_price_vnd'].mean(),
                'avg_margin_pct': products_df['margin_pct'].mean()
            },
            'temperature_sensitivity': products_df['temperature_sensitivity'].value_counts().to_dict()
        }
        return stats
    
    def save_catalog(self, 
                    products_df: pd.DataFrame,
                    supplier_product_df: pd.DataFrame,
                    output_dir: str = '../data/synthetic'):
        """Save product catalog to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        products_df.to_csv(f'{output_dir}/products.csv', index=False)
        supplier_product_df.to_csv(f'{output_dir}/supplier_product_matrix.csv', index=False)
        
        stats = self.compute_product_stats(products_df)
        with open(f'{output_dir}/product_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ Product catalog saved to {output_dir}/")


if __name__ == "__main__":
    from network_generator import DaNangNetworkGenerator
    
    network_gen = DaNangNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=6, n_dcs=2, n_stores=8)
    
    product_gen = ProductCatalogGenerator(seed=42)
    products = product_gen.generate_products(n_products=10)
    
    supplier_product = product_gen.generate_supplier_product_matrix(
        network['suppliers'],
        products
    )
    
    print("\n" + "="*60)
    print("PRODUCT CATALOG")
    print("="*60)
    print(products.to_string(index=False))
    
    print(f"\n\nSupplier-Product Matrix: {len(supplier_product)} combinations")
    print(f"Avg suppliers per product: {len(supplier_product) / len(products):.1f}")
    
    product_gen.save_catalog(products, supplier_product, '../data/synthetic')