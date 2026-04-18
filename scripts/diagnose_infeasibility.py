#!/usr/bin/env python3
"""
diagnose_infeasibility.py
=========================
Chẩn đoán toàn diện các nguyên nhân có thể gây INFEASIBLE trong
TwoPhaseExtensiveFormOptimizer.

Các test case được kiểm tra:
  [T1]  Capacity gap  — tổng fleet capacity có đủ để pickup 70% demand không?
  [T2]  MandatoryPickup conflict — constraint pickup_total >= committed_accessible
        có xung đột với vehicle capacity không?
  [T3]  Concentration + baseline — S1Conc + S1Base có mâu thuẫn nhau không?
  [T4]  dVisit feasibility — "every store must be visited" có khả thi với fleet không?
  [T5]  PROD_008 coverage — Pork chỉ do SUP_001/003/005/006 supply,
        không có subtype khớp với scenarios Level 4-5 không?
  [T6]  MTZ time window — route time có vượt quá bigM = 24h không?
  [T7]  Scenario-level fleet check — mỗi scenario có ít nhất 1 xe operative không?
  [T8]  Emergency cap too tight — em_cap có đủ bù cho inaccessible portion không?
  [T9]  Concentration vs MOQ — moq * conc_max có nhỏ hơn moq không (impossible)?
  [T10] Single-supplier products — product nào chỉ có 1 supplier duy nhất?
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["GRB_LICENSE_FILE"] = r"D:\gurobi.lic"

import pandas as pd
import numpy as np

from data_generation.fleet_config import VEHICLE_TYPES, expand_fleet, to_optimizer_fleet, get_effective_capacity
from weather.manual_scenarios import ManualWeatherScenarios
from weather.scenario_adapter import get_data_driven_scenarios

# ── Constants từ optimizer ──────────────────────────────────────────────────
_MIN_CAP_KG   = 10.0
BASELINE_RATIO = 0.70
CONC_MAX       = 0.40
EMERGENCY_RATIO = 0.40
BIG_M_TIME     = 24.0
T_DEPART_DC    = 4.0

PASS = "✅ PASS"
WARN = "⚠️  WARN"
FAIL = "❌ FAIL"

# ───────────────────────────────────────────────────────────────────────────────
def separator(title=""):
    print()
    print("=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)

def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    suppliers       = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
    stores          = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    dcs             = pd.read_csv(os.path.join(data_dir, "distribution_centers.csv"))
    distance_matrix = pd.read_csv(os.path.join(data_dir, "distance_matrix.csv"), index_col=0)
    products        = pd.read_csv(os.path.join(data_dir, "products.csv"))
    sp_matrix       = pd.read_csv(os.path.join(data_dir, "supplier_product_matrix.csv"))
    demand_full     = pd.read_csv(os.path.join(data_dir, "daily_demand.csv"))
    daily_demand    = demand_full[demand_full["date"] == "2024-10-01"].copy()

    if "requires_refrigeration" not in products.columns:
        products["requires_refrigeration"] = (products["temperature_sensitivity"] == "high")

    return suppliers, stores, dcs, distance_matrix, products, sp_matrix, daily_demand


# ───────────────────────────────────────────────────────────────────────────────
def t1_capacity_gap(scenarios, fleet_instances, sp_matrix, products_df, daily_demand, suppliers):
    """T1: Tổng fleet capacity (weighted by scenario prob) đủ pickup 70% demand?"""
    separator("[T1] Fleet Capacity vs Baseline Demand")
    
    prod_weight = dict(zip(products_df["id"], products_df["weight_kg_per_unit"]))
    total_demand_units = daily_demand.groupby("product_id")["demand_units"].sum().to_dict()
    
    # Tổng demand theo kg
    total_demand_kg = sum(
        total_demand_units.get(p, 0) * prod_weight.get(p, 1.0)
        for p in products_df["id"]
    )
    required_pickup_kg = BASELINE_RATIO * total_demand_kg
    
    print(f"  Total demand (kg)           : {total_demand_kg:,.1f}")
    print(f"  Required pickup @ {BASELINE_RATIO*100:.0f}% baseline : {required_pickup_kg:,.1f} kg")
    
    issues = []
    for k, sc in enumerate(scenarios):
        sev = sc.severity_level
        ops = [v for v, vh in enumerate(fleet_instances)
               if vh["capacity_kg"] * vh["weather_capacity_factor"].get(sev, 1.0) >= _MIN_CAP_KG]
        
        total_fleet_cap = sum(
            fleet_instances[v]["capacity_kg"] * fleet_instances[v]["weather_capacity_factor"].get(sev, 1.0)
            for v in ops
        )
        
        # Accessible supply capacity (supplier side)
        sup_subtype = dict(zip(suppliers["id"], suppliers["subtype"]))
        acc_sups = [s for s in suppliers["id"] if sc.get_supplier_accessible(sup_subtype.get(s, "general")) == 1]
        
        acc_supply_kg = sum(
            row["capacity_kg_per_day"]
            for _, row in suppliers.iterrows()
            if row["id"] in acc_sups
        )
        
        status = PASS if total_fleet_cap >= required_pickup_kg else FAIL
        if total_fleet_cap < required_pickup_kg:
            issues.append(f"Scenario {sc.name} (k={k}): cap={total_fleet_cap:.0f} < needed={required_pickup_kg:.0f}")
        
        print(f"  k={k} [{sc.name:<30}] severity={sev}  "
              f"ops={len(ops)}/{len(fleet_instances)}  "
              f"fleet_cap={total_fleet_cap:,.0f} kg  "
              f"acc_supply={acc_supply_kg:,.0f} kg  {status}")
    
    if issues:
        print("\n  ISSUES FOUND:")
        for iss in issues:
            print(f"    {FAIL} {iss}")
    else:
        print(f"\n  {PASS} All scenarios have sufficient fleet capacity for baseline.")


# ───────────────────────────────────────────────────────────────────────────────
def t2_mandatory_pickup_conflict(scenarios, fleet_instances, sp_matrix, products_df, 
                                  daily_demand, suppliers):
    """T2: MandatoryPickup — pickup_total >= committed_accessible
    Problem: x[s,p] is NOT gated by accessibility. If solver orders from accessible
    supplier with x[s,p], then pickup_total MUST equal that → but vehicle capacity
    might be insufficient for ALL products simultaneously.
    """
    separator("[T2] MandatoryPickup Constraint Feasibility")
    
    prod_weight   = dict(zip(products_df["id"], products_df["weight_kg_per_unit"]))
    total_demand  = daily_demand.groupby("product_id")["demand_units"].sum().to_dict()
    sup_subtype   = dict(zip(suppliers["id"], suppliers["subtype"]))
    sp_avail      = {(r["supplier_id"], r["product_id"]): r["available"] 
                     for _, r in sp_matrix.iterrows()}
    sup_cap       = dict(zip(suppliers["id"], suppliers["capacity_kg_per_day"]))
    
    print("  The MandatoryPickup constraint requires:")
    print("    pickup_total[k,p] >= committed_accessible[k,p] = Σ_{s∈acc} x[s,p]")
    print("  This means ALL accessible committed qty MUST be physically picked up.")
    print("  But vehicles also serve ALL products simultaneously → joint capacity binding.")
    print()
    
    issues = []
    for k, sc in enumerate(scenarios):
        sev = sc.severity_level
        ops = [v for v, vh in enumerate(fleet_instances)
               if vh["capacity_kg"] * vh["weather_capacity_factor"].get(sev, 1.0) >= _MIN_CAP_KG]
        acc_sups = [s for s in suppliers["id"] 
                    if sc.get_supplier_accessible(sup_subtype.get(s, "general")) == 1]
        
        if not ops or not acc_sups:
            print(f"  k={k} [{sc.name:<30}]: NO ops or NO accessible sups → inventory=0 (OK, skip)")
            continue
        
        total_fleet_cap = sum(
            fleet_instances[v]["capacity_kg"] * fleet_instances[v]["weather_capacity_factor"].get(sev, 1.0)
            for v in ops
        )
        
        # Min committed from accessible sups (assuming baseline is fully allocated to accessible)
        # Worst case: solver assigns ALL of baseline to accessible suppliers
        min_committed_kg = 0.0
        for p in products_df["id"]:
            d = total_demand.get(p, 0)
            if d <= 0:
                continue
            acc_avail = [s for s in acc_sups if sp_avail.get((s, p), False)]
            if not acc_avail:
                continue
            # Baseline requires 0.70 * d units ordered; if all accessible, all must be picked up
            # Upper bound: supplier capacity constraint also limits
            per_sup_cap_kg = [min(CONC_MAX * d * prod_weight[p],
                                  sup_cap.get(s, 99999)) for s in acc_avail]
            feasible_max_units = sum(c / prod_weight[p] for c in per_sup_cap_kg)
            committed_units = min(BASELINE_RATIO * d, feasible_max_units)
            min_committed_kg += committed_units * prod_weight[p]
        
        status = PASS if total_fleet_cap >= min_committed_kg * 0.95 else FAIL
        if status == FAIL:
            issues.append(f"k={k} [{sc.name}]: fleet_cap={total_fleet_cap:.0f} < "
                          f"min_committed_kg={min_committed_kg:.0f}")
        
        print(f"  k={k} [{sc.name:<30}]  fleet_cap={total_fleet_cap:,.0f}  "
              f"est_committed_kg={min_committed_kg:,.0f}  {status}")
    
    print()
    if issues:
        print(f"  {FAIL} CRITICAL: MandatoryPickup may be infeasible in some scenarios!")
        for iss in issues:
            print(f"    → {iss}")
        print()
        print("  ROOT CAUSE ANALYSIS:")
        print("  MandatoryPickup[k,p]: pickup_total >= committed_accessible is a HARD constraint.")
        print("  If the solver cannot find a single-vehicle routing that picks up ALL of x[s,p]")
        print("  from accessible suppliers (e.g. due to joint capacity across all products),")
        print("  the model becomes infeasible or forces solver to branch excessively.")
        print("  RECOMMENDATION: Relax to pickup_total >= ALPHA * committed_accessible (alpha~0.85)")
    else:
        print(f"  {PASS} MandatoryPickup appears feasible capacity-wise.")


# ───────────────────────────────────────────────────────────────────────────────
def t3_concentration_vs_baseline(products_df, daily_demand, suppliers, sp_matrix):
    """T3: Kiểm tra S1Conc (x[s,p] <= 0.4*d) xung đột S1Base (Σx >= 0.7*d)?"""
    separator("[T3] Concentration (40%) vs Baseline (70%) Constraint Conflict")
    
    total_demand = daily_demand.groupby("product_id")["demand_units"].sum().to_dict()
    sp_avail     = {(r["supplier_id"], r["product_id"]): r["available"] 
                    for _, r in sp_matrix.iterrows()}
    sup_cap      = dict(zip(suppliers["id"], suppliers["capacity_kg_per_day"]))
    sp_moq       = {(r["supplier_id"], r["product_id"]): r["moq_units"]
                    for _, r in sp_matrix.iterrows()}
    prod_weight  = dict(zip(products_df["id"], products_df["weight_kg_per_unit"]))
    
    print(f"  Concentration max : {CONC_MAX*100:.0f}% of demand per (supplier, product)")
    print(f"  Baseline min      : {BASELINE_RATIO*100:.0f}% of demand total per product")
    print(f"  Minimum suppliers needed per product: ceil(baseline/conc) = "
          f"ceil({BASELINE_RATIO}/{CONC_MAX}) = {int(np.ceil(BASELINE_RATIO/CONC_MAX))}")
    print()
    
    issues = []
    for p in products_df["id"]:
        d = total_demand.get(p, 0)
        if d <= 0:
            continue
        
        avail_sups = [s for s in suppliers["id"] if sp_avail.get((s, p), False)]
        n_avail = len(avail_sups)
        
        # Max feasible per supplier considering capacity
        max_per_sup = []
        for s in avail_sups:
            conc_limit_units = CONC_MAX * d
            cap_limit_units  = sup_cap.get(s, 99999) / prod_weight.get(p, 1.0)
            moq              = sp_moq.get((s, p), 0)
            effective_max    = min(conc_limit_units, cap_limit_units)
            max_per_sup.append(effective_max if effective_max >= moq else 0.0)
        
        max_total = sum(max_per_sup)
        required  = BASELINE_RATIO * d
        
        name = products_df.loc[products_df["id"] == p, "name"].values[0]
        
        if max_total < required:
            status = FAIL
            issues.append(f"{p} ({name}): max_feasible={max_total:.1f} < required={required:.1f} units")
        elif n_avail < int(np.ceil(BASELINE_RATIO / CONC_MAX)):
            status = WARN
        else:
            status = PASS
        
        print(f"  {p} ({name:<30}) d={d:,.0f}  suppliers={n_avail}  "
              f"max_total={max_total:,.1f}  required={required:,.1f}  {status}")
    
    print()
    if issues:
        print(f"  {FAIL} CRITICAL: Some products CANNOT satisfy baseline with concentration constraint!")
        for iss in issues:
            print(f"    → {iss}")
    else:
        print(f"  {PASS} All products can mathematically satisfy baseline + concentration constraints.")


# ───────────────────────────────────────────────────────────────────────────────
def t4_store_visit_feasibility(scenarios, fleet_instances, stores, dcs, distance_matrix):
    """T4: dVisit[k,r] >= 1 — mỗi store PHẢI được visit ít nhất 1 xe.
    Cần ít nhất len(stores) arc activations. Check route time feasibility."""
    separator("[T4] Mandatory Store Visit Feasibility (dVisit >= 1)")
    
    print(f"  All {len(stores)} stores MUST be visited in every scenario (hard constraint).")
    print(f"  This requires fleet to physically route to each store even if unmet=0.")
    print()
    
    T_DIST_DEP = 10.0  # 10:00 AM departure
    STORE_SVC  = 0.25  # 15 min service per store
    DC = dcs["id"].iloc[0]
    
    issues = []
    for k, sc in enumerate(scenarios):
        sev = sc.severity_level
        ops = [v for v, vh in enumerate(fleet_instances)
               if vh["capacity_kg"] * vh["weather_capacity_factor"].get(sev, 1.0) >= _MIN_CAP_KG]
        
        if not ops:
            print(f"  k={k} [{sc.name:<30}]: NO vehicles → dVisit INFEASIBLE (all stores unmet)")
            issues.append(f"k={k}: No vehicles but dVisit >= 1 enforced → INFEASIBLE")
            continue
        
        road_factor = sc.speed_reduction_factor
        # Use slowest available vehicle
        speeds = [max(fleet_instances[v]["base_speed_kmh"] / road_factor, 1.0) for v in ops]
        min_speed = min(speeds)
        max_speed = max(speeds)
        
        # Estimate worst-case time to visit all stores: sequential round-trip
        # Approximate distances DC→store (use distance matrix)
        store_ids = stores["id"].tolist()
        total_dist_all_stores = 0.0
        for r in store_ids:
            try:
                d_out = float(distance_matrix.loc[DC, r])
            except:
                d_out = 15.0  # fallback km
            try:
                d_back = float(distance_matrix.loc[r, DC])
            except:
                d_back = d_out
            total_dist_all_stores += d_out + d_back
        
        # If 1 vehicle visits all stores sequentially
        time_sequential = total_dist_all_stores / min_speed + len(store_ids) * STORE_SVC
        # Time window available: 24h - T_DIST_DEP = 14h
        time_available = BIG_M_TIME - T_DIST_DEP
        
        status = PASS if time_sequential <= time_available * len(ops) else WARN
        if time_sequential > time_available:
            issues.append(f"k={k}: sequential_time={time_sequential:.1f}h > window={time_available:.1f}h")
        
        print(f"  k={k} [{sc.name:<30}]  ops={len(ops)}  "
              f"speed={min_speed:.1f}-{max_speed:.1f}km/h  "
              f"sequential_time={time_sequential:.1f}h  window={time_available:.1f}h  {status}")
    
    print()
    if issues:
        print(f"  {WARN} Some scenarios have very tight routing time windows.")
        for iss in issues:
            print(f"    → {iss}")
        print()
        print("  NOTE: With multiple vehicles, parallel routing reduces time.")
        print("  BUT: if ops=0 and dVisit>=1 is enforced, model is DEFINITELY INFEASIBLE.")


# ───────────────────────────────────────────────────────────────────────────────
def t5_product_supplier_coverage(scenarios, suppliers, sp_matrix, products_df, daily_demand):
    """T5: Mỗi sản phẩm có đủ accessible supplier trong mọi scenario không?
    Đặc biệt PROD_008 (Pork) — chỉ có SUP_001(seafood), SUP_003(meat), SUP_005(seafood), SUP_006(general)
    """
    separator("[T5] Product-Supplier Coverage Across Scenarios")
    
    sup_subtype  = dict(zip(suppliers["id"], suppliers["subtype"]))
    sp_avail     = {(r["supplier_id"], r["product_id"]): r["available"] 
                    for _, r in sp_matrix.iterrows()}
    total_demand = daily_demand.groupby("product_id")["demand_units"].sum().to_dict()
    prod_weight  = dict(zip(products_df["id"], products_df["weight_kg_per_unit"]))
    
    issues = []
    for p in products_df["id"]:
        d = total_demand.get(p, 0)
        name = products_df.loc[products_df["id"] == p, "name"].values[0]
        all_sups = [s for s in suppliers["id"] if sp_avail.get((s, p), False)]
        
        print(f"\n  {p} ({name}): {len(all_sups)} suppliers = {all_sups}")
        
        for k, sc in enumerate(scenarios):
            acc_sups = [s for s in all_sups 
                        if sc.get_supplier_accessible(sup_subtype.get(s, "general")) == 1]
            
            # Max feasible procurement from accessible sups
            sup_cap_kg = dict(zip(suppliers["id"], suppliers["capacity_kg_per_day"]))
            max_acc_units = sum(
                min(CONC_MAX * d, sup_cap_kg.get(s, 99999) / prod_weight.get(p, 1.0))
                for s in acc_sups
            )
            
            required = BASELINE_RATIO * d
            em_cap = EMERGENCY_RATIO * d * (1 if sc.emergency_feasible else 0)
            can_meet = max_acc_units + em_cap >= required
            
            if not acc_sups:
                flag = FAIL
                issues.append(f"  {p}: k={k} [{sc.name}] NO accessible suppliers!")
            elif max_acc_units < required * 0.5:
                flag = WARN
            else:
                flag = PASS
            
            print(f"    k={k} [{sc.name:<28}] "
                  f"acc={len(acc_sups)}/{len(all_sups)}  "
                  f"max_acc={max_acc_units:,.0f}  "
                  f"em_cap={em_cap:,.0f}  "
                  f"required={required:,.0f}  {flag}")
    
    print()
    if issues:
        print(f"  {FAIL} CRITICAL: Some products have NO accessible suppliers in certain scenarios!")
        for iss in issues:
            print(f"    → {iss}")
        print()
        print("  ROOT CAUSE: With baseline_ratio=0.70, if a product's accessible supply")
        print("  is 0 and emergency_cap is too small, S1Base constraint forces INFEASIBLE.")


# ───────────────────────────────────────────────────────────────────────────────
def t6_mtz_time_window(scenarios, fleet_instances, suppliers, dcs, distance_matrix):
    """T6: MTZ time constraints — travel time supplier → supplier có hợp lý không?"""
    separator("[T6] MTZ Time Window Analysis (Phase 2A)")
    
    DC = dcs["id"].iloc[0]
    T_PROC_AVAIL = BIG_M_TIME - T_DEPART_DC  # 20h window
    
    print(f"  Phase 2A departure: {T_DEPART_DC:.0f}:00  |  BigM = {BIG_M_TIME:.0f}h  "
          f"→  window = {T_PROC_AVAIL:.0f}h")
    print()
    
    issues = []
    for k, sc in enumerate(scenarios):
        sev = sc.severity_level
        sup_subtype = dict(zip(suppliers["id"], suppliers["subtype"]))
        acc_sups = [s for s in suppliers["id"] 
                    if sc.get_supplier_accessible(sup_subtype.get(s, "general")) == 1]
        ops = [v for v, vh in enumerate(fleet_instances)
               if vh["capacity_kg"] * vh["weather_capacity_factor"].get(sev, 1.0) >= _MIN_CAP_KG]
        
        if not ops or not acc_sups:
            continue
        
        # Slowest vehicle speed
        min_speed = min(
            max(fleet_instances[v]["base_speed_kmh"] / sc.speed_reduction_factor, 1.0) 
            for v in ops
        )
        
        # Worst-case: visit ALL accessible suppliers
        total_dist = 0.0
        prev = DC
        for s in acc_sups:
            try:
                d = float(distance_matrix.loc[prev, s])
            except:
                d = 15.0
            total_dist += d
            prev = s
        # Return to DC
        try:
            total_dist += float(distance_matrix.loc[prev, DC])
        except:
            total_dist += 15.0
        
        SUP_SVC = 0.5  # 30min per supplier
        total_time = total_dist / min_speed + len(acc_sups) * SUP_SVC
        
        status = PASS if total_time <= T_PROC_AVAIL else WARN
        if total_time > T_PROC_AVAIL:
            issues.append(f"k={k} [{sc.name}]: time={total_time:.1f}h > window={T_PROC_AVAIL:.1f}h")
        
        print(f"  k={k} [{sc.name:<30}]  acc_sups={len(acc_sups)}  "
              f"min_speed={min_speed:.1f}km/h  "
              f"total_dist={total_dist:.1f}km  "
              f"time={total_time:.2f}h  {status}")
    
    if issues:
        print(f"\n  {WARN} Time window tight in some scenarios.")
        for iss in issues:
            print(f"    → {iss}")


# ───────────────────────────────────────────────────────────────────────────────
def t7_scenario_fleet_availability(scenarios, fleet_instances):
    """T7: Mỗi scenario có ít nhất 1 xe không? Level 5 đặc biệt nguy hiểm."""
    separator("[T7] Fleet Availability Per Scenario")
    
    issues = []
    for k, sc in enumerate(scenarios):
        sev = sc.severity_level
        ops = []
        breakdown = {}
        for v, vh in enumerate(fleet_instances):
            eff_cap = vh["capacity_kg"] * vh["weather_capacity_factor"].get(sev, 1.0)
            if eff_cap >= _MIN_CAP_KG:
                ops.append(v)
            breakdown[vh["type_id"]] = breakdown.get(vh["type_id"], 0) + (1 if eff_cap >= _MIN_CAP_KG else 0)
        
        if not ops:
            status = FAIL
            issues.append(f"k={k} [{sc.name}]: ZERO operable vehicles!")
        elif len(ops) < 2:
            status = WARN
        else:
            status = PASS
        
        type_str = ", ".join(f"{t}×{c}" for t, c in breakdown.items() if c > 0)
        print(f"  k={k} [{sc.name:<30}]  severity={sev}  "
              f"ops={len(ops)}/{len(fleet_instances)}  [{type_str}]  {status}")
    
    print()
    if issues:
        print(f"  {FAIL} CRITICAL: Some scenarios have NO operable vehicles!")
        for iss in issues:
            print(f"    → {iss}")
        print()
        print("  When ops=[], inventory[k,p]=0 is enforced (ok) BUT")
        print("  dVisit[k,r] >= 1 is STILL enforced (⚠️ INFEASIBLE if no vehicles!)")
        print("  → Check if the `if not ops: continue` branch in Phase 2B is correct.")


# ───────────────────────────────────────────────────────────────────────────────
def t8_emergency_cap_adequacy(scenarios, suppliers, sp_matrix, products_df, daily_demand):
    """T8: Emergency cap đủ bù cho inaccessible portion không?"""
    separator("[T8] Emergency Cap Adequacy for Inaccessible Suppliers")
    
    sup_subtype  = dict(zip(suppliers["id"], suppliers["subtype"]))
    sp_avail     = {(r["supplier_id"], r["product_id"]): r["available"] 
                    for _, r in sp_matrix.iterrows()}
    total_demand = daily_demand.groupby("product_id")["demand_units"].sum().to_dict()
    prod_weight  = dict(zip(products_df["id"], products_df["weight_kg_per_unit"]))
    sup_cap      = dict(zip(suppliers["id"], suppliers["capacity_kg_per_day"]))
    
    print("  Emergency cap formula in optimizer:")
    print("    if all suppliers accessible: em_cap = 0.05 * d")
    print("    if some inaccessible:        em_cap = min(0.40*d, inacc_fraction * d * 1.2)")
    print()
    
    issues = []
    for k, sc in enumerate(scenarios):
        if sc.severity_level < 4:
            continue  # Only check severe scenarios
        
        print(f"  Scenario k={k} [{sc.name}], severity={sc.severity_level}:")
        for p in products_df["id"]:
            d = total_demand.get(p, 0)
            if d <= 0:
                continue
            name = products_df.loc[products_df["id"] == p, "name"].values[0]
            
            all_sups = [s for s in suppliers["id"] if sp_avail.get((s, p), False)]
            acc_sups  = [s for s in all_sups if sc.get_supplier_accessible(sup_subtype.get(s, "general")) == 1]
            inacc_sups = [s for s in all_sups if s not in acc_sups]
            
            # Max from accessible
            max_acc = sum(
                min(CONC_MAX * d, sup_cap.get(s, 99999) / prod_weight.get(p, 1.0))
                for s in acc_sups
            )
            
            # Emergency cap (as in optimizer code)
            if not sc.emergency_feasible:
                em_cap = 0.0
            elif not inacc_sups:
                em_cap = 0.05 * d
            else:
                inacc_fraction = len(inacc_sups) / max(len(all_sups), 1)
                em_cap = min(EMERGENCY_RATIO * d, inacc_fraction * d * 1.2)
            
            shortfall = max(0.0, BASELINE_RATIO * d - max_acc)
            gap = shortfall - em_cap
            status = PASS if gap <= 0 else FAIL
            
            if gap > 0:
                issues.append(f"  k={k} p={p}: shortfall={shortfall:.0f} > em_cap={em_cap:.0f} → gap={gap:.0f}")
            
            print(f"    {p} ({name:<25})  "
                  f"acc={len(acc_sups)}/{len(all_sups)}  "
                  f"max_acc={max_acc:,.0f}  "
                  f"shortfall={shortfall:,.0f}  "
                  f"em_cap={em_cap:,.0f}  "
                  f"gap={''+str(round(gap,0)) if gap>0 else '0'}  {status}")
    
    print()
    if issues:
        print(f"  {FAIL} EMERGENCY CAP INSUFFICIENT for some product-scenario pairs!")
        for iss in issues:
            print(f"    → {iss}")
        print()
        print("  ROOT CAUSE: S1Base says Σx[s,p] >= 0.70*d (including inaccessible suppliers).")
        print("  In Phase 2A, accessible pickup + emergency MUST cover what's needed for Phase 2B.")
        print("  If accessible supply + em_cap < demand, unmet is forced → heavy penalty but FEASIBLE.")
        print("  However, if S1Base forces x[s,p] orders that CANNOT be picked up AND")
        print("  MandatoryPickup forces physical pickup → INFEASIBLE.")
    else:
        print(f"  {PASS} Emergency cap appears adequate for accessible supply gaps.")


# ───────────────────────────────────────────────────────────────────────────────
def t9_moq_concentration_conflict(products_df, daily_demand, sp_matrix, suppliers):
    """T9: MOQ * conc_max constraint confict — moq > conc_max * d?"""
    separator("[T9] MOQ vs Concentration Limit Conflict")
    
    total_demand = daily_demand.groupby("product_id")["demand_units"].sum().to_dict()
    sp_avail     = {(r["supplier_id"], r["product_id"]): r["available"] 
                    for _, r in sp_matrix.iterrows()}
    sp_moq       = {(r["supplier_id"], r["product_id"]): r["moq_units"]
                    for _, r in sp_matrix.iterrows()}
    
    print("  S1MOQlo: x[s,p] >= MOQ * y[s,p]")
    print("  S1Conc:  x[s,p] <= 0.40 * total_demand[p]")
    print("  Conflict if: MOQ > 0.40 * total_demand[p]  (impossible to satisfy both)")
    print()
    
    issues = []
    for _, row in sp_matrix.iterrows():
        s, p = row["supplier_id"], row["product_id"]
        if not row["available"]:
            continue
        d = total_demand.get(p, 0)
        moq = row["moq_units"]
        conc_limit = CONC_MAX * d
        name = products_df.loc[products_df["id"] == p, "name"].values[0]
        
        if moq > conc_limit:
            status = FAIL
            issues.append(f"{s} × {p} ({name}): MOQ={moq:.0f} > conc_limit={conc_limit:.0f}")
        elif moq > conc_limit * 0.8:
            status = WARN
        else:
            status = PASS
        
        if status != PASS:
            print(f"  {s} × {p:<12} ({name:<25})  "
                  f"d={d:,.0f}  MOQ={moq:,.0f}  conc_limit={conc_limit:,.0f}  {status}")
    
    print()
    if issues:
        print(f"  {FAIL} CRITICAL: MOQ > CONC_LIMIT → y[s,p]=1 is IMPOSSIBLE!")
        for iss in issues:
            print(f"    → {iss}")
        print()
        print("  This means: if y[s,p]=1, then x[s,p]>=MOQ AND x[s,p]<=conc_limit")
        print("  Since MOQ > conc_limit, the only feasible solution is y[s,p]=0, x[s,p]=0.")
        print("  This effectively removes that supplier-product pair from consideration,")
        print("  potentially making S1Base infeasible for products with few suppliers.")
    else:
        print(f"  {PASS} No MOQ vs Concentration conflicts detected.")


# ───────────────────────────────────────────────────────────────────────────────
def t10_single_supplier_products(products_df, sp_matrix, scenarios, suppliers):
    """T10: Sản phẩm nào chỉ có 1 supplier → concentration constraint bị tight."""
    separator("[T10] Single-Supplier Products (High Risk)")
    
    sp_avail    = {(r["supplier_id"], r["product_id"]): r["available"] 
                   for _, r in sp_matrix.iterrows()}
    sup_subtype = dict(zip(suppliers["id"], suppliers["subtype"]))
    
    issues = []
    for p in products_df["id"]:
        name = products_df.loc[products_df["id"] == p, "name"].values[0]
        all_sups = [s for s in suppliers["id"] if sp_avail.get((s, p), False)]
        
        if len(all_sups) <= 2:
            min_sups_needed = int(np.ceil(BASELINE_RATIO / CONC_MAX))
            status = FAIL if len(all_sups) < min_sups_needed else WARN
            issues.append(p)
            
            acc_by_scenario = {}
            for k, sc in enumerate(scenarios):
                acc = [s for s in all_sups 
                       if sc.get_supplier_accessible(sup_subtype.get(s, "general")) == 1]
                acc_by_scenario[sc.name] = acc
            
            worst_acc = min(len(v) for v in acc_by_scenario.values())
            scenario_breakdown = " | ".join(
                f"k={k}:{len(acc_by_scenario[sc.name])}" 
                for k, sc in enumerate(scenarios)
            )
            print(f"  {status} {p} ({name}): {len(all_sups)} total suppliers, "
                  f"worst_accessible={worst_acc}")
            print(f"      Suppliers: {all_sups}")
            print(f"      Per-scenario accessible: {scenario_breakdown}")
    
    print()
    if issues:
        print(f"  {WARN} Products with very few suppliers risk infeasibility under Conc constraint:")
        for p in issues:
            name = products_df.loc[products_df["id"] == p, "name"].values[0]
            print(f"    → {p} ({name})")


# ───────────────────────────────────────────────────────────────────────────────
def t11_inaccessible_supplier_baseline_conflict(scenarios, suppliers, sp_matrix, products_df, daily_demand):
    """T11 (CRITICAL): Stage 1 baseline constraint Σ x[s,p] >= 0.70*d summed over ALL suppliers.
    Under severe weather, inaccessible suppliers contribute to x[s,p] but cannot be picked up.
    MandatoryPickup then requires: Σ_{s∈acc} qty_pickup >= Σ_{s∈acc} x[s,p]
    But S1Base only guarantees Σ_all x[s,p] >= 0.70*d, NOT Σ_{s∈acc} x[s,p] >= 0.70*d.
    
    The solver may legally set most x[s,p] on INACCESSIBLE suppliers.
    Then: committed_accessible is small → MandatoryPickup is easy
         BUT then: dist_sum <= inventory = small → cannot serve demand → heavy penalty
    OR: solver is forced to commit accessible values to meet MandatoryPickup
        + global baseline → overcommitting accessible beyond capacity → INFEASIBLE
    """
    separator("[T11] CRITICAL: S1Base Inaccessible Supplier Interaction")
    
    sup_subtype  = dict(zip(suppliers["id"], suppliers["subtype"]))
    sp_avail     = {(r["supplier_id"], r["product_id"]): r["available"] 
                    for _, r in sp_matrix.iterrows()}
    total_demand = daily_demand.groupby("product_id")["demand_units"].sum().to_dict()
    prod_weight  = dict(zip(products_df["id"], products_df["weight_kg_per_unit"]))
    sup_cap      = dict(zip(suppliers["id"], suppliers["capacity_kg_per_day"]))
    
    print("  S1Base: Σ_{all s} x[s,p] >= 0.70 * d  (Stage 1, all suppliers)")
    print("  MandatoryPickup: Σ_v qty_pickup[k,s,p,v] >= Σ_{s∈acc} x[s,p]")
    print()
    print("  KEY ISSUE: S1Base is GLOBAL (across all suppliers, accessible or not).")
    print("  Solver can satisfy S1Base by loading inaccessible suppliers with high x values.")
    print("  But MandatoryPickup then forces pickup of whatever is on ACCESSIBLE suppliers.")
    print("  Combined, this creates complex interactions that may over-constrain the model.")
    print()
    
    for k, sc in enumerate(scenarios):
        if sc.severity_level < 4:
            continue
        
        print(f"  Scenario k={k} [{sc.name}] (severity={sc.severity_level}):")
        for p in products_df["id"]:
            d = total_demand.get(p, 0)
            if d <= 0:
                continue
            name = products_df.loc[products_df["id"] == p, "name"].values[0]
            all_sups = [s for s in suppliers["id"] if sp_avail.get((s, p), False)]
            acc_sups  = [s for s in all_sups 
                         if sc.get_supplier_accessible(sup_subtype.get(s, "general")) == 1]
            inacc_sups = [s for s in all_sups if s not in acc_sups]
            
            max_acc_units = sum(
                min(CONC_MAX * d, sup_cap.get(s, 99999) / prod_weight.get(p, 1.0))
                for s in acc_sups
            )
            max_inacc_units = sum(
                min(CONC_MAX * d, sup_cap.get(s, 99999) / prod_weight.get(p, 1.0))
                for s in inacc_sups
            )
            
            # Can baseline be satisfied WITHOUT relying on inaccessible?
            can_satisfy_acc_only = max_acc_units >= BASELINE_RATIO * d
            
            print(f"    {p} ({name:<25})  "
                  f"acc={len(acc_sups)}/{len(all_sups)}  "
                  f"max_acc={max_acc_units:,.0f}  max_inacc={max_inacc_units:,.0f}  "
                  f"req={BASELINE_RATIO*d:,.0f}  "
                  f"{'[acc sufficient]' if can_satisfy_acc_only else f'[NEEDS inacc → MandatoryPickup CANNOT enforce pickup!]'}")


# ───────────────────────────────────────────────────────────────────────────────
def print_summary_and_recommendations():
    separator("DIAGNOSIS SUMMARY & RECOMMENDATIONS")
    print("""
  Based on the analysis above, the most likely root causes of infeasibility are:

  [HIGH RISK] T2 + T11 — MandatoryPickup + S1Base Interaction
  ─────────────────────────────────────────────────────────────
  MandatoryPickup (line 544-548 in optimizer):
    pickup_total >= committed_accessible  (HARD constraint)
  
  This forces the solver to physically pick up ALL quantity ordered from
  accessible suppliers. Combined with S1Base (global baseline across all
  suppliers), this can create deadlock:
  
    Option A: Solver sets x[s,p] on INACCESSIBLE sups → committed_accessible
              is small → pickup is easy, but inventory is 0 → demand unmet
    Option B: Solver sets x[s,p] on ACCESSIBLE sups → committed_accessible
              is large → pickup must be large → may exceed joint vehicle capacity
  
  RECOMMENDATION: Change MandatoryPickup from HARD to SOFT:
    pickup_total + pickup_slack >= committed_accessible
    Or: Remove MandatoryPickup entirely (waste_vars already handle this)

  [HIGH RISK] T3 — dVisit >= 1 with zero-vehicle scenarios
  ──────────────────────────────────────────────────────────
  If any scenario has ops=[], the Phase 2B code does `if not ops: continue`.
  BUT dVisit[k,r] >= 1 was added outside this check, meaning constraints
  exist for arcs that have NO variables defined → INFEASIBLE.
  
  Check: Is `dVisit` constraint inside or outside the `if not ops:` block?

  [MEDIUM RISK] T8 — Emergency cap for severe scenarios
  ──────────────────────────────────────────────────────
  The 5% emergency cap for "fully accessible" scenarios may still leave
  coverage gaps when vehicle capacity limits effective pickup below baseline.

  [QUICK WINS — Model Relaxation for Diagnostics]
  ────────────────────────────────────────────────
  1. Remove MandatoryPickup constraint temporarily → see if model becomes feasible
  2. Set baseline_ratio = 0.50 (from 0.70) → looser requirement
  3. Set concentration_max = 0.60 (from 0.40) → more flexibility per supplier
  4. Add a small IIS test with Gurobi (model.computeIIS()) to find exact conflict
""")


# ───────────────────────────────────────────────────────────────────────────────
def run_gurobi_iis_test(scenarios, fleet_instances, suppliers, stores, dcs,
                         distance_matrix, products_df, sp_matrix, daily_demand):
    """Run a minimal IIS (Irreducible Infeasible Subsystem) test on a single scenario."""
    separator("[BONUS] Gurobi IIS Test on Single Worst Scenario (Level 4)")
    
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        print("  gurobipy not available, skipping IIS test.")
        return
    
    # Pick the Level-4 scenario (most complex with partial accessibility)
    target_sc = next((sc for sc in scenarios if sc.severity_level == 4), None)
    if target_sc is None:
        print("  No Level-4 scenario found, skipping.")
        return
    
    print(f"  Running IIS on scenario: [{target_sc.name}]...")
    print("  Building mini model (single scenario)...")
    
    from optimization.two_phase_optimizer import TwoPhaseExtensiveFormOptimizer
    import tempfile, os
    
    opt = TwoPhaseExtensiveFormOptimizer(
        network={"suppliers": suppliers, "stores": stores, "dcs": dcs,
                 "distance_matrix": distance_matrix,
                 "all_locations": pd.concat([suppliers, stores, dcs])},
        products_df=products_df,
        supplier_product_df=sp_matrix,
        demand_df=daily_demand,
        weather_scenarios=[target_sc],
        fleet_instances=fleet_instances,
        baseline_ratio=BASELINE_RATIO,
        concentration_max=CONC_MAX,
    )
    
    model, _ = opt.build_model()
    lp_path = os.path.join(tempfile.gettempdir(), "iis_test_single_scenario.lp")
    model.writeLP(lp_path)
    print(f"  LP written: {lp_path}")
    
    try:
        env = gp.Env()
        env.setParam("OutputFlag", 0)
        grb = gp.read(lp_path, env)
        grb.optimize()
        
        if grb.status == GRB.INFEASIBLE:
            print(f"  {FAIL} Single-scenario model is INFEASIBLE! Computing IIS...")
            grb.computeIIS()
            
            iis_constraints = [(c.ConstrName, c.IISConstr) for c in grb.getConstrs() if c.IISConstr]
            iis_bounds  = [(v.VarName, v.IISLB, v.IISUB) for v in grb.getVars() 
                           if v.IISLB or v.IISUB]
            
            print(f"\n  IIS contains {len(iis_constraints)} constraints and {len(iis_bounds)} bounds:")
            print("  Constraints in IIS:")
            for name, _ in iis_constraints[:30]:  # show top 30
                print(f"    {name}")
            if len(iis_constraints) > 30:
                print(f"    ... and {len(iis_constraints)-30} more")
            
            if iis_bounds:
                print("  Variable bounds in IIS:")
                for name, lb, ub in iis_bounds[:20]:
                    print(f"    {name}  lb={lb}  ub={ub}")
        elif grb.status == GRB.OPTIMAL:
            print(f"  {PASS} Single-scenario (Level 4) is FEASIBLE. "
                  f"ObjVal={grb.ObjVal:,.0f}")
            print("  Infeasibility may be in multi-scenario coupling (non-anticipativity).")
        else:
            print(f"  Solver status: {grb.status}")
    except Exception as e:
        print(f"  IIS test failed: {e}")


# ───────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  INFEASIBILITY DIAGNOSTIC SCRIPT")
    print("  Two-Phase Stochastic MILP — Fresh Food Supply Chain")
    print("=" * 70)
    
    # Load data
    suppliers, stores, dcs, distance_matrix, products_df, sp_matrix, daily_demand = load_data()
    
    # Fleet
    fleet_vehicles  = expand_fleet(VEHICLE_TYPES)
    fleet_instances = to_optimizer_fleet(fleet_vehicles)
    
    # Scenarios (monsoon — same as main run)
    try:
        scenarios = get_data_driven_scenarios(season="monsoon", target_count=5, merge_duplicates=True)
        season_name = "Monsoon (data-driven)"
    except Exception:
        scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
        season_name = "Monsoon (manual)"
    
    # Normalize
    total_p = sum(s.probability for s in scenarios)
    if abs(total_p - 1.0) > 0.01:
        for s in scenarios:
            s.probability /= total_p
    
    print(f"\n  Fleet: {len(fleet_vehicles)} vehicles ({len(fleet_instances)} optimizer instances)")
    print(f"  Scenarios: {len(scenarios)} ({season_name})")
    print(f"  Suppliers: {len(suppliers)} | Products: {len(products_df)} | Stores: {len(stores)}")
    print()
    for sc in scenarios:
        print(f"    [{sc.name:<30}] sev={sc.severity_level}  p={sc.probability:.2f}  "
              f"acc={list(sc.supplier_accessibility.values())}")
    
    # Run all tests
    t1_capacity_gap(scenarios, fleet_instances, sp_matrix, products_df, daily_demand, suppliers)
    t2_mandatory_pickup_conflict(scenarios, fleet_instances, sp_matrix, products_df, daily_demand, suppliers)
    t3_concentration_vs_baseline(products_df, daily_demand, suppliers, sp_matrix)
    t4_store_visit_feasibility(scenarios, fleet_instances, stores, dcs, distance_matrix)
    t5_product_supplier_coverage(scenarios, suppliers, sp_matrix, products_df, daily_demand)
    t6_mtz_time_window(scenarios, fleet_instances, suppliers, dcs, distance_matrix)
    t7_scenario_fleet_availability(scenarios, fleet_instances)
    t8_emergency_cap_adequacy(scenarios, suppliers, sp_matrix, products_df, daily_demand)
    t9_moq_concentration_conflict(products_df, daily_demand, sp_matrix, suppliers)
    t10_single_supplier_products(products_df, sp_matrix, scenarios, suppliers)
    t11_inaccessible_supplier_baseline_conflict(scenarios, suppliers, sp_matrix, products_df, daily_demand)
    
    print_summary_and_recommendations()
    
    # Optional IIS test (runs fast on single scenario)
    print("\nRun Gurobi IIS test on single Level-4 scenario? (y/n): ", end="")
    try:
        ans = input().strip().lower()
    except:
        ans = "n"
    
    if ans == "y":
        run_gurobi_iis_test(scenarios, fleet_instances, suppliers, stores, dcs,
                             distance_matrix, products_df, sp_matrix, daily_demand)
    
    print("\n" + "=" * 70)
    print("  DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
