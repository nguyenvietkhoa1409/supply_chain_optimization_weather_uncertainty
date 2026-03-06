"""
Product Generator - Creates fresh food product catalog for Da Nang
UPDATED v2: Adds volume_m3_per_unit for dual-capacity (weight + cubic) VRP constraint
            Consistent with Patel et al. dual-capacity formulation (Wk, Vk constraints)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import json


@dataclass
class Product:
    id: str
    name: str
    category: str
    shelf_life_days: float
    unit_cost_vnd: float
    unit_price_vnd: float
    weight_kg_per_unit: float
    volume_m3_per_unit: float           # NEW: for cubic-load constraint
    temperature_sensitivity: str
    requires_refrigeration: bool        # NEW: flag for xe lạnh routing preference


class ProductCatalogGenerator:
    """Generates realistic fresh food product catalog for Da Nang market"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_products(self, n_products: int = 10) -> pd.DataFrame:
        """
        Generate product catalog.

        NEW in v2:
        - volume_m3_per_unit  added (derived from weight + packing density)
        - requires_refrigeration flag  (seafood & meat → True)
        """

        product_templates = [
            # ── Seafood ──────────────────────────────────────────────
            {
                "name":              "Fresh Fish (Cá)",
                "category":          "seafood",
                "shelf_life_range":  (1, 2),
                "cost_range":        (80_000, 120_000),
                "margin":            0.35,
                "weight_per_unit":   0.5,
                "volume_m3":         0.003,    # ice-box packing: 0.006 m³/kg
                "temp_sensitivity":  "high",
                "requires_refrig":   True,
            },
            {
                "name":              "Fresh Shrimp (Tôm)",
                "category":          "seafood",
                "shelf_life_range":  (1, 2),
                "cost_range":        (150_000, 200_000),
                "margin":            0.40,
                "weight_per_unit":   0.3,
                "volume_m3":         0.002,    # dense packing
                "temp_sensitivity":  "high",
                "requires_refrig":   True,
            },
            {
                "name":              "Squid (Mực)",
                "category":          "seafood",
                "shelf_life_range":  (1, 3),
                "cost_range":        (100_000, 140_000),
                "margin":            0.35,
                "weight_per_unit":   0.4,
                "volume_m3":         0.003,
                "temp_sensitivity":  "high",
                "requires_refrig":   True,
            },
            # ── Vegetables ───────────────────────────────────────────
            {
                "name":              "Leafy Greens (Rau xanh)",
                "category":          "vegetable",
                "shelf_life_range":  (3, 5),
                "cost_range":        (15_000, 25_000),
                "margin":            0.30,
                "weight_per_unit":   0.2,
                "volume_m3":         0.003,    # bulky: 0.015 m³/kg
                "temp_sensitivity":  "medium",
                "requires_refrig":   False,
            },
            {
                "name":              "Tomatoes (Cà chua)",
                "category":          "vegetable",
                "shelf_life_range":  (4, 7),
                "cost_range":        (20_000, 30_000),
                "margin":            0.25,
                "weight_per_unit":   0.3,
                "volume_m3":         0.003,
                "temp_sensitivity":  "medium",
                "requires_refrig":   False,
            },
            {
                "name":              "Cabbage (Bắp cải)",
                "category":          "vegetable",
                "shelf_life_range":  (5, 8),
                "cost_range":        (12_000, 20_000),
                "margin":            0.28,
                "weight_per_unit":   0.5,
                "volume_m3":         0.006,    # bulky: 0.012 m³/kg
                "temp_sensitivity":  "low",
                "requires_refrig":   False,
            },
            # ── Meat ─────────────────────────────────────────────────
            {
                "name":              "Chicken (Gà)",
                "category":          "meat",
                "shelf_life_range":  (2, 3),
                "cost_range":        (60_000, 80_000),
                "margin":            0.30,
                "weight_per_unit":   0.6,
                "volume_m3":         0.005,    # vacuum-packed: 0.008 m³/kg
                "temp_sensitivity":  "high",
                "requires_refrig":   True,
            },
            {
                "name":              "Pork (Thịt heo)",
                "category":          "meat",
                "shelf_life_range":  (2, 4),
                "cost_range":        (70_000, 90_000),
                "margin":            0.32,
                "weight_per_unit":   0.5,
                "volume_m3":         0.004,
                "temp_sensitivity":  "high",
                "requires_refrig":   True,
            },
            {
                "name":              "Beef (Thịt bò)",
                "category":          "meat",
                "shelf_life_range":  (2, 4),
                "cost_range":        (180_000, 220_000),
                "margin":            0.35,
                "weight_per_unit":   0.4,
                "volume_m3":         0.003,
                "temp_sensitivity":  "high",
                "requires_refrig":   True,
            },
            # ── Fruit ────────────────────────────────────────────────
            {
                "name":              "Dragon Fruit (Thanh long)",
                "category":          "fruit",
                "shelf_life_range":  (5, 10),
                "cost_range":        (25_000, 35_000),
                "margin":            0.30,
                "weight_per_unit":   0.4,
                "volume_m3":         0.004,    # medium density
                "temp_sensitivity":  "low",
                "requires_refrig":   False,
            },
        ]

        products = []
        for i in range(min(n_products, len(product_templates))):
            t = product_templates[i]
            shelf_life = np.random.uniform(*t["shelf_life_range"])
            cost       = np.random.uniform(*t["cost_range"])
            price      = cost * (1 + t["margin"])

            products.append({
                "id":                   f"PROD_{i+1:03d}",
                "name":                 t["name"],
                "category":             t["category"],
                "shelf_life_days":      round(shelf_life, 1),
                "unit_cost_vnd":        round(cost, 0),
                "unit_price_vnd":       round(price, 0),
                "margin_pct":           round(t["margin"] * 100, 1),
                "weight_kg_per_unit":   t["weight_per_unit"],
                "volume_m3_per_unit":   t["volume_m3"],      # NEW
                "temperature_sensitivity": t["temp_sensitivity"],
                "requires_refrigeration":  t["requires_refrig"],   # NEW
            })

        df = pd.DataFrame(products)
        print(f"✓ Products generated:")
        print(f"  Requires refrigeration: {df['requires_refrigeration'].sum()} / {len(df)}")
        print(f"  Volume range: {df['volume_m3_per_unit'].min():.3f} – "
              f"{df['volume_m3_per_unit'].max():.3f} m³/unit")
        return df

    def generate_supplier_product_matrix(self,
                                         suppliers_df: pd.DataFrame,
                                         products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate supplier-product availability matrix.
        UNCHANGED logic; kept for API compatibility.
        """
        matrix_data = []
        product_coverage = {p: [] for p in products_df["id"].tolist()}

        # Pass 1: Specialized suppliers
        for _, supplier in suppliers_df.iterrows():
            supplier_type = supplier.get("subtype", "general")
            supplier_id   = supplier["id"]
            if supplier_type == "general":
                continue

            for _, product in products_df.iterrows():
                product_category = product["category"]
                product_id       = product["id"]
                can_supply       = False

                if supplier_type == "seafood" and product_category == "seafood":
                    can_supply = True
                elif supplier_type == "vegetables" and product_category in ["vegetable", "fruit"]:
                    can_supply = True
                elif supplier_type == "meat" and product_category == "meat":
                    can_supply = True
                elif supplier_type == "meat" and product_category in ["seafood", "fruit"]:
                    if np.random.random() < 0.7:
                        can_supply = True
                elif supplier_type == "seafood" and product_category in ["meat", "fruit"]:
                    if np.random.random() < 0.7:
                        can_supply = True
                elif supplier_type == "vegetables" and product_category in ["meat", "seafood"]:
                    if np.random.random() < 0.6:
                        can_supply = True

                if can_supply:
                    base_cost     = product["unit_cost_vnd"]
                    supplier_cost = base_cost * np.random.uniform(0.85, 1.15)
                    moq           = np.random.randint(5, 20)
                    matrix_data.append({
                        "supplier_id":    supplier_id,
                        "product_id":     product_id,
                        "unit_cost_vnd":  round(supplier_cost, 0),
                        "moq_units":      moq,
                        "lead_time_days": round(np.random.uniform(0.5, 2.0), 1),
                        "available":      True,
                    })
                    product_coverage[product_id].append(supplier_id)

        # Pass 2: General supplier covers ALL products
        general_suppliers = suppliers_df[suppliers_df["subtype"] == "general"]
        if len(general_suppliers) > 0:
            general_id = general_suppliers.iloc[0]["id"]
            print(f"\n✓ General supplier {general_id} covers ALL products (10-15% premium)")
            for _, product in products_df.iterrows():
                base_cost     = product["unit_cost_vnd"]
                supplier_cost = base_cost * np.random.uniform(1.10, 1.15)
                moq           = np.random.randint(10, 30)
                matrix_data.append({
                    "supplier_id":    general_id,
                    "product_id":     product["id"],
                    "unit_cost_vnd":  round(supplier_cost, 0),
                    "moq_units":      moq,
                    "lead_time_days": round(np.random.uniform(1.0, 2.5), 1),
                    "available":      True,
                })
                product_coverage[product["id"]].append(general_id)

        df = pd.DataFrame(matrix_data)

        print("\nSupplier-Product Coverage Validation:")
        all_covered = True
        for _, prod in products_df.iterrows():
            n_sup = len(df[df["product_id"] == prod["id"]])
            sym   = "✓" if n_sup >= 2 else ("⚠" if n_sup == 1 else "❌")
            if n_sup < 2:
                all_covered = False
            print(f"  {sym} {prod['id']} ({prod['name']}): {n_sup} suppliers")

        print("\n✓ ALL products have ≥2 suppliers!" if all_covered else
              "\n⚠ WARNING: Some products have insufficient coverage!")
        return df

    def compute_product_stats(self, products_df: pd.DataFrame) -> dict:
        stats = {
            "total_products":  len(products_df),
            "by_category":     products_df["category"].value_counts().to_dict(),
            "shelf_life":      {
                "min_days":  products_df["shelf_life_days"].min(),
                "max_days":  products_df["shelf_life_days"].max(),
                "mean_days": products_df["shelf_life_days"].mean(),
            },
            "pricing": {
                "avg_cost_vnd":  products_df["unit_cost_vnd"].mean(),
                "avg_price_vnd": products_df["unit_price_vnd"].mean(),
                "avg_margin_pct": products_df["margin_pct"].mean(),
            },
            "requires_refrigeration": int(products_df["requires_refrigeration"].sum()),
        }
        return stats

    def save_catalog(self,
                     products_df: pd.DataFrame,
                     supplier_product_df: pd.DataFrame,
                     output_dir: str = "../data/synthetic"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        products_df.to_csv(f"{output_dir}/products.csv", index=False)
        supplier_product_df.to_csv(f"{output_dir}/supplier_product_matrix.csv", index=False)
        stats = self.compute_product_stats(products_df)
        with open(f"{output_dir}/product_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Product catalog saved to {output_dir}/")