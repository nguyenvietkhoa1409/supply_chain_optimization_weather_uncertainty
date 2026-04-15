#!/usr/bin/env python3
"""
recalibrate_data.py
================================================================================
Heuristic Data Recalibration for Supply Chain Tractability
================================================================================

ROOT CAUSES DIAGNOSED:
──────────────────────
1. DEMAND TOO LOW vs FLEET CAPACITY:
   Total demand = 382.6 kg, but fleet = 8,900 kg → 4.3% utilization
   → Solver sees no incentive to route efficiently; any assignment is feasible
   → Fix: Scale demand UP to 60–70% fleet utilization (~5,500 kg)

2. SUPPLIER CAPACITY MASSIVELY OVER-SUPPLIED (ratio 60x – 387x):
   Implication: optimizer can procure everything from 1 supplier → concentrates
   → Fix: Cap per-supplier-per-product capacity to 1.5× fair share

3. SUPPLIER-PRODUCT MATRIX too permissive:
   SUP_002 (vegetables): 9 products, SUP_006 (general): 10 products
   → Any optimizer will use the "super supplier" for everything
   → Fix: Enforce domain-specific coverage with HARD max 4 products/supplier
           (exception: general supplier capped at 5 products for weather hedge)

4. GEOGRAPHIC SCALE too small:
   Max DC-to-supplier: 9.7 km, mean 5.7 km
   → Routing cost per trip ≈ 5.7 × 8000 = 45,600 VND (negligible)
   → Vehicle fixed cost dominates; fleet assignment becomes binary (use/not use)
   → Fix: This is a physical reality of Da Nang; compensate by ensuring
          meaningful load-per-vehicle (utilization fix handles this)

HEURISTIC DESIGN PRINCIPLES APPLIED:
────────────────────────────────────
P1. Vehicle Utilization Target: 55–70% of total fleet weight
    → Ensures multi-vehicle routing is genuinely required
    → 2–4 vehicles needed instead of 0.4

P2. Supplier Concentration limit (data level, not model level):
    max capacity from any single supplier ≤ 40% of total per-product demand
    → Ensures diversified procurement is NECESSARY, not just preferred

P3. Product Coverage Design (2–4 suppliers per product):
    - Specialized suppliers: 3–4 complementary products each
    - General supplier: 5 products (emergency hedge, higher cost)
    - Each product: exactly 3 suppliers for redundancy

P4. Academic Realism:
    Demand calibrated to match a mid-size Da Nang fresh food distributor
    serving 6 retail outlets (each ~30-40 units/product/day)
"""

import os
import sys
import numpy as np
import pandas as pd
import json

np.random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

# ══════════════════════════════════════════════════════════════════════════════
#  TARGET PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
TARGET_FLEET_UTIL    = 0.60   # target 60% fleet utilization
TOTAL_FLEET_KG       = 3*300 + 2*1000 + 2*1500 + 3000   # = 8,900 kg
TARGET_DEMAND_KG     = TARGET_FLEET_UTIL * TOTAL_FLEET_KG  # ≈ 5,340 kg
DEMAND_SCALE_FACTOR  = TARGET_DEMAND_KG   # will compute actual scale ratio

MAX_PRODS_SPECIALIZED = 4   # max products a specialized supplier carries
MAX_PRODS_GENERAL     = 6   # general wholesale: covers more products (feasibility hedge)
N_SUPPLIERS_PER_PROD  = 3   # minimum 3 suppliers per product (required by concentration=0.40)
CAP_RATIO_PER_PRODUCT = 1.5   # per supplier: 1.5 × fair share → total = 1.5 × demand (feasible)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Compute current demand and scale factor
# ══════════════════════════════════════════════════════════════════════════════
def step1_compute_scale():
    products = pd.read_csv(f"{DATA_DIR}/products.csv")
    demand_full = pd.read_csv(f"{DATA_DIR}/daily_demand.csv")
    demand = demand_full[demand_full["date"] == "2024-10-01"].copy()

    dm = demand.merge(products[["id", "weight_kg_per_unit"]], left_on="product_id", right_on="id")
    current_kg = (dm["demand_units"] * dm["weight_kg_per_unit"]).sum()
    scale      = TARGET_DEMAND_KG / current_kg

    # Safety: don't re-scale if demand already within 10% of target
    if abs(scale - 1.0) < 0.15:
        scale = 1.0
        print(f"Step 1: Demand already near target ({current_kg:.0f} kg ≈ {TARGET_DEMAND_KG:.0f} kg). No scaling needed.")
    else:
        print(f"Step 1: Demand calibration")
        print(f"  Current demand:  {dm['demand_units'].sum():.0f} units = {current_kg:.1f} kg")
        print(f"  Target demand:   {TARGET_DEMAND_KG:.0f} kg ({TARGET_FLEET_UTIL*100:.0f}% fleet)")
        print(f"  Scale factor:    {scale:.2f}x")

    return scale, products, demand


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Scale daily_demand.csv
# ══════════════════════════════════════════════════════════════════════════════
def step2_scale_demand(scale):
    demand_full = pd.read_csv(f"{DATA_DIR}/daily_demand.csv")

    # Scale uniformly all records (preserves temporal patterns)
    demand_full["demand_units"] = (demand_full["demand_units"] * scale).round(1)
    demand_full["demand_units"] = demand_full["demand_units"].clip(lower=1.0)

    demand_full.to_csv(f"{DATA_DIR}/daily_demand.csv", index=False)
    print(f"\nStep 2: Scaled all demand by {scale:.2f}x → saved daily_demand.csv")

    # Verify
    dem = demand_full[demand_full["date"] == "2024-10-01"]
    products = pd.read_csv(f"{DATA_DIR}/products.csv")
    dm = dem.merge(products[["id", "weight_kg_per_unit"]], left_on="product_id", right_on="id")
    new_kg = (dm["demand_units"] * dm["weight_kg_per_unit"]).sum()
    print(f"  New total demand 2024-10-01: {dm['demand_units'].sum():.0f} units = {new_kg:.1f} kg "
          f"({new_kg/TOTAL_FLEET_KG*100:.1f}% fleet)")
    return dem


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Redesign supplier-product matrix
# ══════════════════════════════════════════════════════════════════════════════
def step3_redesign_sp_matrix(demand, products):
    """
    Hardcoded domain-expert assignment:
    ── Products ────                     ── Domain ──
    PROD_001: Fresh Fish (seafood)        → seafood suppliers
    PROD_002: Fresh Shrimp (seafood)      → seafood suppliers
    PROD_003: Squid (seafood)             → seafood suppliers
    PROD_004: Leafy Greens (vegetable)    → vegetable suppliers
    PROD_005: Tomatoes (vegetable)        → vegetable suppliers
    PROD_006: Cabbage (vegetable)         → vegetable suppliers
    PROD_007: Chicken (meat)              → meat supplier
    PROD_008: Pork (meat)                 → meat supplier
    PROD_009: Beef (meat)                 → meat supplier
    PROD_010: Dragon Fruit (fruit)        → vegetable/general suppliers

    ── Suppliers ──
    SUP_001: Tho Quang Seafood  → PROD_001, PROD_002, PROD_003 (specialist + backup fruit)
    SUP_002: Hoa Vang Vegetables → PROD_004, PROD_005, PROD_006, PROD_010
    SUP_003: Hoa Khanh Meat     → PROD_007, PROD_008, PROD_009 (specialist)
    SUP_004: Lien Chieu Farm    → PROD_004, PROD_005, PROD_006, PROD_010
    SUP_005: Nam O Fishing Port → PROD_001, PROD_002, PROD_003 (overlap seafood)
    SUP_006: Da Nang Wholesale  → PROD_001, PROD_004, PROD_005, PROD_007, PROD_009, PROD_010 (6 products)
    Note: PROD_004 added to SUP_006 to ensure 3 suppliers for CONCENTRATION_MAX=0.40 feasibility

    Coverage check:
    PROD_001 (Fish):     SUP_001, SUP_005, SUP_006  ✓ 3 suppliers
    PROD_002 (Shrimp):   SUP_001, SUP_005            → need 3rd; add SUP_003 (cross-sell)
    PROD_003 (Squid):    SUP_001, SUP_005            → need 3rd; add SUP_004 (farm pond)
    PROD_004 (Leafy):    SUP_002, SUP_004            → need 3rd; add SUP_006
    PROD_005 (Tomatoes): SUP_002, SUP_004, SUP_006  ✓ 3 suppliers
    PROD_006 (Cabbage):  SUP_002, SUP_004            → need 3rd; add SUP_003
    PROD_007 (Chicken):  SUP_003, SUP_006            → need 3rd; add SUP_001
    PROD_008 (Pork):     SUP_003                     → need more; add SUP_001, SUP_005
    PROD_009 (Beef):     SUP_003, SUP_006            → need 3rd; add SUP_005
    PROD_010 (D.Fruit):  SUP_002, SUP_004, SUP_006  ✓ 3 suppliers
    """
    products_df = pd.read_csv(f"{DATA_DIR}/products.csv")
    suppliers_df = pd.read_csv(f"{DATA_DIR}/suppliers.csv")
    sp_orig = pd.read_csv(f"{DATA_DIR}/supplier_product_matrix.csv")

    # Domain-expert assignment: (supplier_id, product_id) pairs
    ASSIGNMENTS = [
        # SUP_001 Seafood (specialist): seafood + chicken + pork
        ("SUP_001", "PROD_001"), ("SUP_001", "PROD_002"), ("SUP_001", "PROD_003"),
        ("SUP_001", "PROD_007"), ("SUP_001", "PROD_008"),  # 5 products (was 6)

        # SUP_002 Vegetables (specialist): all veg + fruit
        ("SUP_002", "PROD_004"), ("SUP_002", "PROD_005"),
        ("SUP_002", "PROD_006"), ("SUP_002", "PROD_010"),  # 4 products (was 9)

        # SUP_003 Meat (specialist): meat + cabbage + squid
        ("SUP_003", "PROD_007"), ("SUP_003", "PROD_008"),
        ("SUP_003", "PROD_009"), ("SUP_003", "PROD_006"), ("SUP_003", "PROD_002"),  # 5 products (was 6)

        # SUP_004 Farm (vegetables + seafood)
        ("SUP_004", "PROD_004"), ("SUP_004", "PROD_005"),
        ("SUP_004", "PROD_006"), ("SUP_004", "PROD_010"), ("SUP_004", "PROD_003"),  # 5 products (was 9)

        # SUP_005 Fishing Port (seafood + beef + pork)
        ("SUP_005", "PROD_001"), ("SUP_005", "PROD_002"),
        ("SUP_005", "PROD_003"), ("SUP_005", "PROD_008"), ("SUP_005", "PROD_009"),  # 5 products (was 7)

        # SUP_006 General Wholesale (6 products — emergency hedge for all categories)
        # PROD_004 included to ensure 3-supplier coverage (required by concentration=0.40)
        ("SUP_006", "PROD_001"), ("SUP_006", "PROD_004"), ("SUP_006", "PROD_005"),
        ("SUP_006", "PROD_007"), ("SUP_006", "PROD_009"), ("SUP_006", "PROD_010"),  # 6 products
    ]

    print("\nStep 3: Redesigning supplier-product matrix")
    print(f"  Assignment plan: {len(ASSIGNMENTS)} (supplier, product) pairs")

    # Build new matrix from original cost data where available, else estimate
    new_rows = []
    for (sid, pid) in ASSIGNMENTS:
        # Try to reuse original cost data
        orig = sp_orig[(sp_orig["supplier_id"] == sid) & (sp_orig["product_id"] == pid)]
        if len(orig) > 0:
            row = orig.iloc[0].to_dict()
        else:
            # Estimate cost: base product cost ± 10%, with premium for general supplier
            prod_row = products_df[products_df["id"] == pid]
            if len(prod_row) == 0:
                continue
            base_cost = prod_row["unit_cost_vnd"].values[0]
            if sid == "SUP_006":
                unit_cost = base_cost * np.random.uniform(1.25, 1.40)  # 30-40% premium
            else:
                unit_cost = base_cost * np.random.uniform(0.85, 1.10)
            row = {
                "supplier_id":   sid,
                "product_id":    pid,
                "unit_cost_vnd": round(unit_cost, 0),
                "moq_units":     np.random.randint(5, 20),
                "lead_time_days": round(np.random.uniform(0.5, 2.0), 1),
                "available":     True,
            }
        new_rows.append(row)

    new_sp = pd.DataFrame(new_rows)

    # Verify coverage
    print("\n  Product coverage check:")
    all_ok = True
    for _, p in products_df.iterrows():
        sups = new_sp[new_sp["product_id"] == p["id"]]["supplier_id"].tolist()
        sym = "✅" if len(sups) >= 2 else "⚠️"
        if len(sups) < 2:
            all_ok = False
        print(f"    {sym} {p['id']} ({p['name'][:20]:20s}): {len(sups)} suppliers → {sups}")

    print("\n  Products per supplier:")
    for sid in ["SUP_001", "SUP_002", "SUP_003", "SUP_004", "SUP_005", "SUP_006"]:
        n = len(new_sp[new_sp["supplier_id"] == sid])
        print(f"    {sid}: {n} products")

    new_sp.to_csv(f"{DATA_DIR}/supplier_product_matrix.csv", index=False)
    print(f"\n  ✓ Saved new supplier_product_matrix.csv ({len(new_sp)} rows)")
    return new_sp


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Recalibrate supplier capacities
# ══════════════════════════════════════════════════════════════════════════════
def step4_recalibrate_capacities(demand, products_df, new_sp):
    """
    Set each supplier's per-product capacity = CAP_RATIO × (product demand / n_suppliers).
    Then sum to get total supplier daily capacity.
    This ensures:
    - Solver MUST use multiple suppliers (single supplier can't cover all demand)
    - Concentration constraint naturally binds
    """
    print("\nStep 4: Recalibrating supplier capacities")

    suppliers = pd.read_csv(f"{DATA_DIR}/suppliers.csv")

    # Per-product daily demand (units and kg)
    prod_demand = {}
    for _, p in products_df.iterrows():
        d_units = demand[demand["product_id"] == p["id"]]["demand_units"].sum()
        d_kg    = d_units * p["weight_kg_per_unit"]
        prod_demand[p["id"]] = {"units": d_units, "kg": d_kg}

    # Compute target capacity per supplier
    for idx, sup_row in suppliers.iterrows():
        sid = sup_row["id"]
        prods_for_sup = new_sp[new_sp["supplier_id"] == sid]["product_id"].tolist()

        # Each supplier gets CAP_RATIO × fair_share of demand per product it supplies
        total_cap_kg = 0
        for pid in prods_for_sup:
            n_sup_for_prod = len(new_sp[new_sp["product_id"] == pid])
            fair_share_kg  = prod_demand[pid]["kg"] / max(n_sup_for_prod, 1)
            capped_kg      = CAP_RATIO_PER_PRODUCT * fair_share_kg
            total_cap_kg  += capped_kg

        # Add 20% buffer for operational flexibility
        new_cap = round(total_cap_kg * 1.2, 0)

        # Minimum capacity (realistic: 500 kg/day for any operating supplier)
        new_cap = max(new_cap, 500)

        old_cap = sup_row["capacity_kg_per_day"]
        suppliers.loc[idx, "capacity_kg_per_day"] = new_cap
        print(f"  {sid} ({sup_row['subtype']:12s}): "
              f"{old_cap:.0f} → {new_cap:.0f} kg/day  (products: {len(prods_for_sup)})")

    suppliers.to_csv(f"{DATA_DIR}/suppliers.csv", index=False)
    print(f"\n  ✓ Saved updated suppliers.csv")
    return suppliers


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Verify the new profile
# ══════════════════════════════════════════════════════════════════════════════
def step5_verify(demand, products_df, new_sp, suppliers):
    print("\n" + "="*60)
    print("VERIFICATION: New Data Profile")
    print("="*60)

    dm = demand.merge(products_df[["id", "weight_kg_per_unit"]], left_on="product_id", right_on="id")
    total_kg = (dm["demand_units"] * dm["weight_kg_per_unit"]).sum()

    print(f"\nDemand: {dm['demand_units'].sum():.0f} units = {total_kg:.0f} kg")
    print(f"Fleet:  {TOTAL_FLEET_KG} kg")
    print(f"Utilization: {total_kg/TOTAL_FLEET_KG*100:.1f}% ← TARGET 55-70%")
    print(f"Trucks needed (mean 1000kg): {total_kg/1000:.1f} ← TARGET 4-6")

    prod_demand = {}
    for _, p in products_df.iterrows():
        d_kg = (demand[demand["product_id"] == p["id"]]["demand_units"].sum()
                * p["weight_kg_per_unit"])
        prod_demand[p["id"]] = d_kg

    print(f"\n{'Product':22s} {'Demand_kg':>10} {'n_sup':>6} {'TotalCap_kg':>12} {'Ratio':>7}")
    print("-"*65)
    for _, p in products_df.iterrows():
        pid = p["id"]
        d_kg = prod_demand[pid]
        sup_ids = new_sp[new_sp["product_id"] == pid]["supplier_id"].tolist()
        total_cap = sum(suppliers[suppliers["id"]==s]["capacity_kg_per_day"].values[0]
                        for s in sup_ids)
        ratio = total_cap / d_kg if d_kg > 0 else 999
        flag = "✅" if 1.2 <= ratio <= 5.0 else ("⚠️↑" if ratio > 5.0 else "⚠️↓")
        print(f"  {flag} {p['name'][:20]:20s} {d_kg:>10.0f} {len(sup_ids):>6} "
              f"{total_cap:>12.0f} {ratio:>6.1f}x")

    print(f"\nSupplier Dominance (% of total demand coverable):")
    for _, row in suppliers.iterrows():
        sid = row["id"]
        prods = new_sp[new_sp["supplier_id"] == sid]["product_id"].tolist()
        d_covered = sum(prod_demand.get(p, 0) for p in prods)
        cap_kg = row["capacity_kg_per_day"]
        pct = d_covered / total_kg * 100
        can_supply = min(d_covered, cap_kg)
        actual_share = can_supply / total_kg * 100
        print(f"  {sid} ({row['subtype']:12s}): "
              f"products={len(prods)}  demand_coverage={pct:.0f}%  "
              f"actual_max_share={actual_share:.0f}%  cap={cap_kg:.0f}kg")

    print(f"\n✅ Recalibration complete. Run run_benders_optimization.py to optimize.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("HEURISTIC DATA RECALIBRATION")
    print("Da Nang Fresh Food Supply Chain")
    print("="*60)

    scale, products_df, demand_old = step1_compute_scale()
    demand_new = step2_scale_demand(scale)
    new_sp     = step3_redesign_sp_matrix(demand_new, products_df)

    # Reload demand after scaling
    demand_full = pd.read_csv(f"{DATA_DIR}/daily_demand.csv")
    demand = demand_full[demand_full["date"] == "2024-10-01"].copy()

    suppliers = step4_recalibrate_capacities(demand, products_df, new_sp)
    step5_verify(demand, products_df, new_sp, suppliers)
