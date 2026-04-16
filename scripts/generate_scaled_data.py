#!/usr/bin/env python3
"""
Generate Scaled-Down Data for Tractable Stochastic PDP Optimization
====================================================================
Target scale: K=4, V=4, S=4, R=5, P=6

Estimated MIP size after scaling:
  N = 1 DC + 4 suppliers + 5 stores = 10 PDP nodes
  Arc binary vars ≈ 4 × 4 × 50 ≈ 800   (vs 3,222 at full scale → −75%)
  Total constraints ≈ 3,000             (vs 7,083 → −57%)
  Expected solve time: 30–180 s         (vs 1,800+ s)

Demand calibrated so that:
  L1/L2 (4 vehicles): ~60% fleet utilization  → comfortable operations
  L4    (2 vehicles): ~85% utilization        → meaningful strain
  L5    (0 vehicles): full unmet              → severe penalty scenario

Product coverage: Exactly 3 suppliers per product (satisfies concentration_max=0.40).

Run this script ONCE to regenerate all data files in data/synthetic/.
Then run scripts/test_scaled_feasibility.py to verify.
"""

import os
import sys
import json
import math
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DISTRIBUTION CENTER  (keep 1 DC — Hoa Khanh)
# ─────────────────────────────────────────────────────────────────────────────
DC_DATA = [
    {
        "id": "DC_001",
        "name": "Hoa Khanh Distribution Center",
        "type": "dc",
        "latitude": 16.0700,
        "longitude": 108.1750,
        "capacity_kg_per_day": 5000.0,
        "fixed_cost_vnd": 0.0,
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. SUPPLIERS  (S=4: seafood, vegetable, meat, general)
# ─────────────────────────────────────────────────────────────────────────────
# Coverage rule: every product must appear in ≥ 3 suppliers (for concentration_max=0.40).
# SUP_004 (general) covers ALL 6 products as emergency backup.
SUPPLIER_DATA = [
    {
        "id": "SUP_001", "name": "Tho Quang Seafood", "type": "supplier", "subtype": "seafood",
        "latitude": 16.0883, "longitude": 108.2153,
        "capacity_kg_per_day": 700.0, "fixed_cost_vnd": 420_000.0,
        "time_window_open": 4, "time_window_close": 8, "service_time_min": 30,
    },
    {
        "id": "SUP_002", "name": "Hoa Vang Vegetables", "type": "supplier", "subtype": "vegetables",
        "latitude": 16.0144, "longitude": 108.1144,
        "capacity_kg_per_day": 600.0, "fixed_cost_vnd": 300_000.0,
        "time_window_open": 5, "time_window_close": 9, "service_time_min": 25,
    },
    {
        "id": "SUP_003", "name": "Hoa Khanh Meat Market", "type": "supplier", "subtype": "meat",
        "latitude": 16.0533, "longitude": 108.1605,
        "capacity_kg_per_day": 850.0, "fixed_cost_vnd": 350_000.0,
        "time_window_open": 4, "time_window_close": 9, "service_time_min": 35,
    },
    {
        "id": "SUP_004", "name": "Da Nang Wholesale Market", "type": "supplier", "subtype": "general",
        "latitude": 16.0544, "longitude": 108.2022,
        "capacity_kg_per_day": 2500.0, "fixed_cost_vnd": 1_500_000.0,
        "time_window_open": 4, "time_window_close": 11, "service_time_min": 45,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. STORES  (R=5: 5 zones across Da Nang)
# ─────────────────────────────────────────────────────────────────────────────
STORE_DATA = [
    {
        "id": "STORE_001", "name": "Hai Chau Downtown", "type": "store",
        "latitude": 16.0678, "longitude": 108.2208, "demand_factor": 1.2,
        "time_window_open": 6, "time_window_close": 11, "service_time_min": 15,
    },
    {
        "id": "STORE_002", "name": "Son Tra Beach", "type": "store",
        "latitude": 16.0833, "longitude": 108.2500, "demand_factor": 1.0,
        "time_window_open": 6, "time_window_close": 11, "service_time_min": 15,
    },
    {
        "id": "STORE_003", "name": "Ngu Hanh Son", "type": "store",
        "latitude": 16.0019, "longitude": 108.2517, "demand_factor": 0.9,
        "time_window_open": 6, "time_window_close": 11, "service_time_min": 15,
    },
    {
        "id": "STORE_004", "name": "Thanh Khe Urban", "type": "store",
        "latitude": 16.0614, "longitude": 108.1878, "demand_factor": 1.1,
        "time_window_open": 6, "time_window_close": 11, "service_time_min": 15,
    },
    {
        "id": "STORE_005", "name": "Lien Chieu Suburb", "type": "store",
        "latitude": 16.0944, "longitude": 108.1556, "demand_factor": 0.8,
        "time_window_open": 6, "time_window_close": 11, "service_time_min": 15,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. PRODUCTS  (P=6: 2 seafood [refrig], 2 vegetable, 2 meat [no refrig])
# ─────────────────────────────────────────────────────────────────────────────
# Only SEAFOOD requires refrigeration → ref_truck (1500kg) covers all ref. demand:
#   ref demand = (Fish + Shrimp) × 5.0 = ~530 kg < 1500 kg  ✓
PRODUCT_DATA = [
    {"id": "PROD_001", "name": "Fresh Fish (Cá)",          "category": "seafood",    "shelf_life_days": 1.4,
     "unit_cost_vnd": 118_000.0, "unit_price_vnd": 159_000.0, "margin_pct": 35.0,
     "weight_kg_per_unit": 0.50, "volume_m3_per_unit": 0.003, "temperature_sensitivity": "high",  "requires_refrigeration": True},
    {"id": "PROD_002", "name": "Fresh Shrimp (Tôm)",       "category": "seafood",    "shelf_life_days": 1.7,
     "unit_cost_vnd": 180_000.0, "unit_price_vnd": 252_000.0, "margin_pct": 40.0,
     "weight_kg_per_unit": 0.30, "volume_m3_per_unit": 0.002, "temperature_sensitivity": "high",  "requires_refrigeration": True},
    {"id": "PROD_003", "name": "Leafy Greens (Rau xanh)",  "category": "vegetable",  "shelf_life_days": 3.1,
     "unit_cost_vnd": 24_000.0,  "unit_price_vnd": 31_000.0,  "margin_pct": 30.0,
     "weight_kg_per_unit": 0.20, "volume_m3_per_unit": 0.003, "temperature_sensitivity": "medium", "requires_refrigeration": False},
    {"id": "PROD_004", "name": "Tomatoes (Cà chua)",       "category": "vegetable",  "shelf_life_days": 5.8,
     "unit_cost_vnd": 27_000.0,  "unit_price_vnd": 34_000.0,  "margin_pct": 25.0,
     "weight_kg_per_unit": 0.30, "volume_m3_per_unit": 0.003, "temperature_sensitivity": "medium", "requires_refrigeration": False},
    {"id": "PROD_005", "name": "Chicken (Gà)",             "category": "meat",       "shelf_life_days": 2.8,
     "unit_cost_vnd": 65_000.0,  "unit_price_vnd": 85_000.0,  "margin_pct": 30.0,
     "weight_kg_per_unit": 0.60, "volume_m3_per_unit": 0.005, "temperature_sensitivity": "medium", "requires_refrigeration": False},
    {"id": "PROD_006", "name": "Pork (Thịt heo)",          "category": "meat",       "shelf_life_days": 2.4,
     "unit_cost_vnd": 74_000.0,  "unit_price_vnd": 97_000.0,  "margin_pct": 32.0,
     "weight_kg_per_unit": 0.50, "volume_m3_per_unit": 0.004, "temperature_sensitivity": "medium", "requires_refrigeration": False},
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. SUPPLIER-PRODUCT MATRIX
# ─────────────────────────────────────────────────────────────────────────────
# Coverage: each product has EXACTLY 3 suppliers → satisfies concentration_max=0.40
#   (1 / 0.40 = 2.5 → need ≥ 3 suppliers; with 3: 3 × 0.40 = 1.20 × demand ≥ demand ✓)
#
# PROD_001 Fish:    SUP_001 (seafood), SUP_003 (meat/cross), SUP_004 (general)
# PROD_002 Shrimp:  SUP_001 (seafood), SUP_002 (coastal farm), SUP_004 (general)
# PROD_003 Greens:  SUP_002 (vegetable), SUP_003 (cross-sell), SUP_004 (general)
# PROD_004 Tomato:  SUP_002 (vegetable), SUP_003 (cross-sell), SUP_004 (general)
# PROD_005 Chicken: SUP_001 (poultry), SUP_003 (meat), SUP_004 (general)
# PROD_006 Pork:    SUP_001 (cross), SUP_003 (meat), SUP_004 (general)

SP_ASSIGNMENTS = [
    # SUP_001 — Tho Quang Seafood (handles seafood + some poultry/pork from port area)
    ("SUP_001", "PROD_001", 0.88),  # Fish  — home product
    ("SUP_001", "PROD_002", 0.90),  # Shrimp — home product
    ("SUP_001", "PROD_005", 0.92),  # Chicken — poultry section
    ("SUP_001", "PROD_006", 0.93),  # Pork — cross sell
    # SUP_002 — Hoa Vang Vegetables
    ("SUP_002", "PROD_002", 0.95),  # Shrimp — coastal farm adjacent
    ("SUP_002", "PROD_003", 0.87),  # Greens — home product
    ("SUP_002", "PROD_004", 0.89),  # Tomatoes — home product
    # SUP_003 — Hoa Khanh Meat Market
    ("SUP_003", "PROD_001", 0.91),  # Fish  — wet market adjacent
    ("SUP_003", "PROD_003", 0.95),  # Greens — morning market cross-sell
    ("SUP_003", "PROD_004", 0.94),  # Tomatoes — morning market
    ("SUP_003", "PROD_005", 0.88),  # Chicken — home product
    ("SUP_003", "PROD_006", 0.86),  # Pork — home product
    # SUP_004 — Da Nang Wholesale Market (general: all 6 products, premium price)
    ("SUP_004", "PROD_001", 1.15),  # Fish
    ("SUP_004", "PROD_002", 1.18),  # Shrimp
    ("SUP_004", "PROD_003", 1.12),  # Greens
    ("SUP_004", "PROD_004", 1.10),  # Tomatoes
    ("SUP_004", "PROD_005", 1.13),  # Chicken
    ("SUP_004", "PROD_006", 1.14),  # Pork
]

# ─────────────────────────────────────────────────────────────────────────────
# 6. BASE DEMAND PER STORE (units/day at demand_factor=1.0)
#    Calibrated to target ~2,500 kg total demand (across all stores)
#    → ~60% of 4-vehicle fleet capacity (4,300 kg non-ref + 1,500 ref)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DEMAND = {
    "PROD_001": 143,   # Fish:    143 × 0.5 kg × Σfactor(5.0) = 357.5 kg ref
    "PROD_002": 114,   # Shrimp:  114 × 0.3 kg × 5.0 = 171.0 kg ref   (total ref: 528.5 kg < 1500 ✓)
    "PROD_003": 380,   # Greens:  380 × 0.2 kg × 5.0 = 380.0 kg
    "PROD_004": 310,   # Tomato:  310 × 0.3 kg × 5.0 = 465.0 kg
    "PROD_005": 200,   # Chicken: 200 × 0.6 kg × 5.0 = 600.0 kg
    "PROD_006": 140,   # Pork:    140 × 0.5 kg × 5.0 = 350.0 kg
    # Total: (357.5 + 171.0 + 380.0 + 465.0 + 600.0 + 350.0) = 2,323.5 kg ≈ target
}

DATE_RANGE = pd.date_range("2024-10-01", periods=7, freq="D")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def build_distance_matrix(all_nodes):
    ids = [n["id"] for n in all_nodes]
    mat = pd.DataFrame(index=ids, columns=ids, dtype=float)
    for i, n1 in enumerate(all_nodes):
        for j, n2 in enumerate(all_nodes):
            if i == j:
                mat.at[n1["id"], n2["id"]] = 0.0
            else:
                d = haversine_km(n1["latitude"], n1["longitude"], n2["latitude"], n2["longitude"])
                mat.at[n1["id"], n2["id"]] = round(d, 3)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate():
    print("=" * 60)
    print("Generating SCALED data (K=4, V=4, S=4, R=5, P=6)")
    print("=" * 60)

    # ── suppliers.csv ─────────────────────────────────────────
    df_sup = pd.DataFrame(SUPPLIER_DATA)
    df_sup.to_csv(os.path.join(DATA_DIR, "suppliers.csv"), index=False)
    print(f"  ✓ suppliers.csv       : {len(df_sup)} rows")

    # ── stores.csv ────────────────────────────────────────────
    df_sto = pd.DataFrame(STORE_DATA)
    df_sto.to_csv(os.path.join(DATA_DIR, "stores.csv"), index=False)
    print(f"  ✓ stores.csv          : {len(df_sto)} rows")

    # ── distribution_centers.csv ──────────────────────────────
    df_dc = pd.DataFrame(DC_DATA)
    df_dc.to_csv(os.path.join(DATA_DIR, "distribution_centers.csv"), index=False)
    print(f"  ✓ distribution_centers.csv : {len(df_dc)} rows")

    # ── products.csv ──────────────────────────────────────────
    df_prod = pd.DataFrame(PRODUCT_DATA)
    df_prod.to_csv(os.path.join(DATA_DIR, "products.csv"), index=False)
    refrig_count = df_prod["requires_refrigeration"].sum()
    print(f"  ✓ products.csv        : {len(df_prod)} rows ({refrig_count} refrigerated)")

    # ── supplier_product_matrix.csv ───────────────────────────
    prod_costs = {r["id"]: r["unit_cost_vnd"] for r in PRODUCT_DATA}
    prod_weights = {r["id"]: r["weight_kg_per_unit"] for r in PRODUCT_DATA}
    sp_rows = []
    for (sup_id, prod_id, price_ratio) in SP_ASSIGNMENTS:
        base_cost = prod_costs[prod_id]
        moq = max(10, int(50 / prod_weights[prod_id]))  # ~50 kg minimum
        sp_rows.append({
            "supplier_id": sup_id,
            "product_id":  prod_id,
            "unit_cost_vnd": round(base_cost * price_ratio, 0),
            "moq_units":  moq,
            "lead_time_days": 1,
            "available": True,
        })
    df_sp = pd.DataFrame(sp_rows)
    df_sp.to_csv(os.path.join(DATA_DIR, "supplier_product_matrix.csv"), index=False)

    # Coverage check
    cov = df_sp.groupby("product_id")["supplier_id"].count()
    min_cov = cov.min()
    print(f"  ✓ supplier_product_matrix.csv : {len(df_sp)} rows | "
          f"min coverage={min_cov} (need ≥3)")
    assert min_cov >= 3, f"Coverage violation: product {cov.idxmin()} has only {min_cov} suppliers!"

    # ── daily_demand.csv ───────────────────────────────────────
    demand_rows = []
    for date in DATE_RANGE:
        # Day-of-week multiplier: Mon/Tue bigger, Sun smaller
        dow = date.day_of_week  # 0=Mon, 6=Sun
        dow_factor = {0: 1.05, 1: 1.02, 2: 1.00, 3: 1.00, 4: 1.03, 5: 1.10, 6: 0.90}[dow]
        for store in STORE_DATA:
            s_id   = store["id"]
            s_fac  = store["demand_factor"]
            for prod in PRODUCT_DATA:
                p_id  = prod["id"]
                base  = BASE_DEMAND[p_id]
                # Gaussian noise ±8%, floored at 50% of base
                noise = np.random.normal(1.0, 0.08)
                noise = max(0.50, noise)
                units = max(5, round(base * s_fac * dow_factor * noise, 1))
                demand_rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id":   s_id,
                    "product_id": p_id,
                    "demand_units": units,
                })
    df_dem = pd.DataFrame(demand_rows)
    df_dem.to_csv(os.path.join(DATA_DIR, "daily_demand.csv"), index=False)

    # Summary for 2024-10-01 (the date used in optimization)
    oct1 = df_dem[df_dem["date"] == "2024-10-01"]
    prod_info = {r["id"]: r for r in PRODUCT_DATA}
    total_kg = sum(
        r["demand_units"] * prod_info[r["product_id"]]["weight_kg_per_unit"]
        for _, r in oct1.iterrows()
    )
    ref_kg = sum(
        r["demand_units"] * prod_info[r["product_id"]]["weight_kg_per_unit"]
        for _, r in oct1.iterrows()
        if prod_info[r["product_id"]]["requires_refrigeration"]
    )
    print(f"  ✓ daily_demand.csv    : {len(df_dem)} rows | "
          f"2024-10-01 total={total_kg:.0f} kg (ref={ref_kg:.0f} kg, non-ref={total_kg-ref_kg:.0f} kg)")

    # ── distance_matrix.csv ───────────────────────────────────
    all_nodes = DC_DATA + SUPPLIER_DATA + STORE_DATA
    dist_mat  = build_distance_matrix(all_nodes)
    dist_mat.to_csv(os.path.join(DATA_DIR, "distance_matrix.csv"))
    print(f"  ✓ distance_matrix.csv : {len(all_nodes)}×{len(all_nodes)} matrix")

    # ── network_topology.csv ──────────────────────────────────
    topo_rows = []
    for n in DC_DATA:
        topo_rows.append({**n, "node_type": "dc"})
    for n in SUPPLIER_DATA:
        topo_rows.append({**{k: v for k, v in n.items()
                             if k not in ("fixed_cost_vnd", "time_window_open",
                                          "time_window_close", "service_time_min")},
                          "node_type": "supplier"})
    for n in STORE_DATA:
        topo_rows.append({**n, "node_type": "store"})
    df_topo = pd.DataFrame(topo_rows)
    df_topo.to_csv(os.path.join(DATA_DIR, "network_topology.csv"), index=False)
    print(f"  ✓ network_topology.csv: {len(df_topo)} nodes")

    # ── metadata.json ─────────────────────────────────────────
    meta = {
        "generated_by": "generate_scaled_data.py",
        "scale": {"K": 4, "V": 4, "S": 4, "R": 5, "P": 6},
        "n_suppliers": len(SUPPLIER_DATA),
        "n_stores":    len(STORE_DATA),
        "n_products":  len(PRODUCT_DATA),
        "n_dcs":       len(DC_DATA),
        "demand_date": "2024-10-01",
        "total_demand_kg_oct1": round(total_kg, 1),
        "ref_demand_kg_oct1":   round(ref_kg, 1),
        "concentration_max":    0.40,
        "min_supplier_coverage": int(min_cov),
    }
    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ metadata.json")

    print("\n✅ All data files generated successfully.\n")
    print("Next step: python scripts/test_scaled_feasibility.py")
    return df_sup, df_sto, df_prod, df_sp, df_dem, dist_mat


if __name__ == "__main__":
    generate()
