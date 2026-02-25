"""
Improved Product Generator - Better supplier-product coverage
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from product_generator import ProductCatalogGenerator


class ImprovedProductGenerator(ProductCatalogGenerator):
    """Enhanced version with more flexible supplier-product matching"""
    
    def generate_supplier_product_matrix(self, 
                                    suppliers_df: pd.DataFrame,
                                    products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate COMPREHENSIVE supplier-product availability
        
        Strategy: Ensure EVERY product has AT LEAST 2 suppliers
        """
        matrix_data = []
        
        # Track coverage per product
        product_coverage = {p: [] for p in products_df['id'].tolist()}
        
        for _, supplier in suppliers_df.iterrows():
            supplier_type = supplier.get('subtype', 'general')
            supplier_id = supplier['id']
            
            for _, product in products_df.iterrows():
                product_category = product['category']
                product_id = product['id']
                
                # Determine if supplier can provide this product
                can_supply = False
                
                # PRIMARY MATCHES (high probability)
                if supplier_type == 'seafood' and product_category == 'seafood':
                    can_supply = True
                elif supplier_type == 'vegetables' and product_category in ['vegetable', 'fruit']:
                    can_supply = True
                elif supplier_type == 'meat' and product_category == 'meat':
                    can_supply = True
                
                # SECONDARY MATCHES (moderate probability for flexibility)
                elif supplier_type == 'meat' and product_category == 'seafood':
                    if np.random.random() < 0.5:  # 50% chance
                        can_supply = True
                elif supplier_type == 'seafood' and product_category == 'meat':
                    if np.random.random() < 0.3:  # 30% chance
                        can_supply = True
                elif supplier_type == 'vegetables' and product_category == 'meat':
                    if np.random.random() < 0.2:  # 20% chance (fresh market vendors)
                        can_supply = True
                
                if can_supply:
                    # Add cost variation per supplier (±15%)
                    base_cost = product['unit_cost_vnd']
                    supplier_cost = base_cost * np.random.uniform(0.85, 1.15)
                    
                    # LOWER MOQs (5-25 instead of 5-30)
                    moq = np.random.randint(5, 25)
                    
                    # Lead time
                    lead_time = np.random.uniform(0.5, 2.0)
                    
                    matrix_data.append({
                        'supplier_id': supplier_id,
                        'product_id': product_id,
                        'unit_cost_vnd': round(supplier_cost, 0),
                        'moq_units': moq,
                        'lead_time_days': round(lead_time, 1),
                        'available': True
                    })
                    
                    product_coverage[product_id].append(supplier_id)
        
        # CRITICAL: Ensure every product has at least 2 suppliers
        # If not, add emergency suppliers
        all_supplier_ids = suppliers_df['id'].tolist()
        
        for product_id, suppliers_list in product_coverage.items():
            if len(suppliers_list) < 2:
                print(f"  ⚠ WARNING: {product_id} only has {len(suppliers_list)} supplier(s)")
                
                # Find product details
                product = products_df[products_df['id'] == product_id].iloc[0]
                
                # Add 2 random suppliers with slightly higher cost
                available_suppliers = [s for s in all_supplier_ids if s not in suppliers_list]
                np.random.shuffle(available_suppliers)
                
                for emergency_supplier in available_suppliers[:2]:
                    base_cost = product['unit_cost_vnd']
                    # Emergency suppliers have 20% higher cost
                    supplier_cost = base_cost * np.random.uniform(1.15, 1.25)
                    moq = np.random.randint(5, 20)
                    
                    matrix_data.append({
                        'supplier_id': emergency_supplier,
                        'product_id': product_id,
                        'unit_cost_vnd': round(supplier_cost, 0),
                        'moq_units': moq,
                        'lead_time_days': round(np.random.uniform(0.5, 2.0), 1),
                        'available': True
                    })
                    
                    product_coverage[product_id].append(emergency_supplier)
                    
                    print(f"    → Added emergency supplier {emergency_supplier}")
        
        return pd.DataFrame(matrix_data)


if __name__ == "__main__":
    # Test
    from improved_network import ImprovedNetworkGenerator
    
    network_gen = ImprovedNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=5, n_dcs=2, n_stores=8)
    
    product_gen = ImprovedProductGenerator(seed=42)
    products = product_gen.generate_products(n_products=10)
    supplier_product = product_gen.generate_supplier_product_matrix(
        network['suppliers'],
        products
    )
    
    print("\n" + "="*70)
    print("IMPROVED SUPPLIER-PRODUCT MATRIX")
    print("="*70)
    
    print(f"\nTotal combinations: {len(supplier_product)}")
    print(f"Average suppliers per product: {len(supplier_product) / len(products):.1f}")
    
    # Check coverage
    for _, prod in products.iterrows():
        suppliers_for_prod = supplier_product[supplier_product['product_id'] == prod['id']]
        print(f"\n{prod['name']} ({prod['id']}): {len(suppliers_for_prod)} suppliers")
        if len(suppliers_for_prod) == 0:
            print("  ⚠ WARNING: No suppliers!")
    
    # Save
    product_gen.save_catalog(products, supplier_product, '../../../data/synthetic')
    print("\n✓ Saved improved product catalog!")
