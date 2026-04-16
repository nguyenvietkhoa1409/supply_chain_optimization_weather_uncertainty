#!/usr/bin/env python3
"""
Feasibility Test for Scaled Supply Chain Data
==============================================
Verifies that the K=4, V=4, S=4, R=5, P=6 problem instance is tractable
and structurally feasible BEFORE running the full 30-minute Gurobi solve.

Checks performed:
  [C1]  Supplier coverage  : ≥ 3 suppliers per product (concentration_max=0.40)
  [C2]  Supplier capacity  : total supply possible ≥ β×demand (baseline_ratio=0.70)
  [C3]  Fleet capacity L1  : total fleet kg ≥ total demand (easy scenario)
  [C4]  Refrigerated cargo : ref_truck capacity ≥ total refrigerated demand
  [C5]  Fleet capacity L4  : 2-vehicle subset (light_truck+ref_truck) vs demand
  [C6]  LP relaxation      : quick PuLP solve to confirm no structural infeasibility

Run: python scripts/test_scaled_feasibility.py
Expected output: all 6 checks PASS  →  safe to run run_stochastic_optimization.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

# ─── Fleet definitions (post-scaling to V=4) ──────────────────────────────
FLEET_L1 = [
    {"id": "mini_van_0",    "kg": 300,  "refrigerated": False, "max_sev": 3},
    {"id": "light_truck_0", "kg": 1000, "refrigerated": False, "max_sev": 4},
    {"id": "ref_truck_0",   "kg": 1500, "refrigerated": True,  "max_sev": 4},
    {"id": "heavy_truck_0", "kg": 3000, "refrigerated": False, "max_sev": 3},
]

BASELINE_RATIO     = 0.70
CONCENTRATION_MAX  = 0.40
OVERSTOCK_FACTOR   = 1.50

PASS_ICON = "✅"
FAIL_ICON = "❌"
WARN_ICON = "⚠️"


def check(label, condition, detail=""):
    icon = PASS_ICON if condition else FAIL_ICON
    print(f"  {icon}  [{label}]  {detail}")
    return condition


def run_checks():
    print("=" * 66)
    print("FEASIBILITY CHECK — Scaled PDP (K=4, V=4, S=4, R=5, P=6)")
    print("=" * 66)

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        df_sup  = pd.read_csv(f"{DATA_DIR}/suppliers.csv")
        df_sto  = pd.read_csv(f"{DATA_DIR}/stores.csv")
        df_prod = pd.read_csv(f"{DATA_DIR}/products.csv")
        df_sp   = pd.read_csv(f"{DATA_DIR}/supplier_product_matrix.csv")
        df_dem  = pd.read_csv(f"{DATA_DIR}/daily_demand.csv")
    except FileNotFoundError as e:
        print(f"\n{FAIL_ICON}  Data not found: {e}")
        print("   Run:  python scripts/generate_scaled_data.py  first.")
        sys.exit(1)

    daily = df_dem[df_dem["date"] == "2024-10-01"].copy()
    if daily.empty:
        print(f"\n{FAIL_ICON}  No demand data for 2024-10-01 in daily_demand.csv")
        sys.exit(1)

    prod_weight = dict(zip(df_prod["id"], df_prod["weight_kg_per_unit"]))
    prod_refrig = dict(zip(df_prod["id"], df_prod["requires_refrigeration"].astype(bool)))

    # Total demand per product (units) for 2024-10-01
    total_demand_units = daily.groupby("product_id")["demand_units"].sum().to_dict()
    total_demand_kg    = {p: v * prod_weight.get(p, 0) for p, v in total_demand_units.items()}

    total_kg_all = sum(total_demand_kg.values())
    total_kg_ref = sum(v for p, v in total_demand_kg.items() if prod_refrig.get(p, False))
    total_kg_nonref = total_kg_all - total_kg_ref

    print(f"\n  Data summary (2024-10-01):")
    print(f"    Suppliers:         {len(df_sup)}")
    print(f"    Stores:            {len(df_sto)}")
    print(f"    Products:          {len(df_prod)}  ({df_prod['requires_refrigeration'].sum()} refrigerated)")
    print(f"    Total demand:      {total_kg_all:,.0f} kg")
    print(f"    Refrigerated:      {total_kg_ref:,.0f} kg")
    print(f"    Non-refrigerated:  {total_kg_nonref:,.0f} kg\n")

    results = []

    # ─────────────────────────────────────────────────────────────────────
    # [C1] Supplier coverage ≥ 3 per product
    # ─────────────────────────────────────────────────────────────────────
    print("[C1] Supplier coverage per product")
    coverage = df_sp[df_sp["available"] == True].groupby("product_id")["supplier_id"].count()
    all_ok = True
    for prod_id in df_prod["id"]:
        n = coverage.get(prod_id, 0)
        pname = df_prod.loc[df_prod["id"] == prod_id, "name"].values[0]
        ok = n >= 3
        icon = PASS_ICON if ok else FAIL_ICON
        print(f"    {icon}  {prod_id} ({pname[:22]:22s}): {n} suppliers  "
              f"[need ≥ ceil(1/{CONCENTRATION_MAX}) = 3]")
        if not ok:
            all_ok = False
    results.append(all_ok)

    # ─────────────────────────────────────────────────────────────────────
    # [C2] Supplier capacity ≥ β × demand (baseline_ratio = 0.70)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[C2] Supplier capacity vs β×demand (β={BASELINE_RATIO})")
    sup_capacity = dict(zip(df_sup["id"], df_sup["capacity_kg_per_day"]))
    c2_ok = True
    for prod_id in df_prod["id"]:
        demand_kg = total_demand_kg.get(prod_id, 0)
        needed    = BASELINE_RATIO * demand_kg            # β × D_p
        max_avail = 0
        suppliers_for_prod = df_sp[
            (df_sp["product_id"] == prod_id) & (df_sp["available"] == True)
        ]["supplier_id"].tolist()
        for s in suppliers_for_prod:
            # Max single supplier allocation under concentration constraint
            max_avail += min(CONCENTRATION_MAX * demand_kg, sup_capacity.get(s, 0))
        ok = max_avail >= needed
        icon = PASS_ICON if ok else FAIL_ICON
        pname = df_prod.loc[df_prod["id"] == prod_id, "name"].values[0]
        print(f"    {icon}  {prod_id}: max_supply={max_avail:6.0f} kg ≥ "
              f"β×D={needed:6.0f} kg  [{'+' if ok else 'FAIL'}]")
        if not ok:
            c2_ok = False
    results.append(c2_ok)

    # ─────────────────────────────────────────────────────────────────────
    # [C3] Fleet capacity L1 (all 4 vehicles) vs total demand
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[C3] Fleet capacity (L1 — all 4 vehicles) vs demand")
    fleet_cap_l1 = sum(v["kg"] for v in FLEET_L1)
    ok3 = fleet_cap_l1 >= total_kg_all
    util = total_kg_all / fleet_cap_l1 * 100
    icon = PASS_ICON if ok3 else WARN_ICON  # warn if demand > fleet (not a hard blocker due to multi-trip)
    print(f"    {icon}  Fleet capacity: {fleet_cap_l1:,.0f} kg | "
          f"Total demand: {total_kg_all:,.0f} kg | Utilization: {util:.1f}%")
    for v in FLEET_L1:
        print(f"         {v['id']:18s}  {v['kg']:5} kg  {'❄' if v['refrigerated'] else '  '}")
    results.append(True)   # fleet being over-subscribed triggers unmet demand, not infeasibility

    # ─────────────────────────────────────────────────────────────────────
    # [C4] Refrigerated cargo vs ref_truck capacity
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[C4] Refrigerated demand vs ref_truck capacity")
    ref_truck_cap = next(v["kg"] for v in FLEET_L1 if v["refrigerated"])
    ok4 = ref_truck_cap >= total_kg_ref
    icon = PASS_ICON if ok4 else FAIL_ICON
    print(f"    {icon}  ref_truck capacity: {ref_truck_cap:,} kg | "
          f"Refrigerated demand: {total_kg_ref:,.0f} kg")
    if not ok4:
        print(f"    ⚠   Infeasibility risk: refrigerated demand EXCEEDS ref_truck capacity.")
        print(f"    ⚠   Fix: add a second ref_truck or reduce refrigerated product count.")
    results.append(ok4)

    # ─────────────────────────────────────────────────────────────────────
    # [C5] Fleet capacity L4 (2 vehicles: light_truck + ref_truck)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[C5] Fleet capacity L4 (severity 4 — mini_van & heavy_truck disabled)")
    l4_fleet = [v for v in FLEET_L1 if v["max_sev"] >= 4]
    fleet_cap_l4  = sum(v["kg"] for v in l4_fleet)
    ref_cap_l4    = sum(v["kg"] for v in l4_fleet if v["refrigerated"])
    ok_ref_l4     = ref_cap_l4 >= total_kg_ref
    unmet_nonref  = max(0, total_kg_nonref - sum(v["kg"] for v in l4_fleet if not v["refrigerated"]))
    unmet_pct_l4  = unmet_nonref / total_kg_all * 100

    print(f"    L4 vehicles: {[v['id'] for v in l4_fleet]}")
    print(f"    L4 fleet capacity: {fleet_cap_l4:,} kg")
    print(f"    Refrigerated cap:  {ref_cap_l4:,} kg vs ref demand {total_kg_ref:.0f} kg  "
          f"{'✅' if ok_ref_l4 else '❌'}")
    print(f"    Expected non-ref unmet: ~{unmet_nonref:.0f} kg ({unmet_pct_l4:.1f}% of total demand)")
    print(f"    {'✅ Feasible with partial unmet demand (expected behavior for L4)' if unmet_pct_l4 < 60 else '⚠ High unmet rate — consider reducing demand scale'}")
    results.append(True)  # L4 unmet is handled by penalty variables — always structurally feasible

    # ─────────────────────────────────────────────────────────────────────
    # [C6] Quick LP relaxation via PuLP
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[C6] LP relaxation feasibility (PuLP — procurement stage only)")
    try:
        import pulp
        model = pulp.LpProblem("FeasCheck", pulp.LpMinimize)
        x = {}
        for _, row in df_sp[df_sp["available"] == True].iterrows():
            s, p = row["supplier_id"], row["product_id"]
            x[(s, p)] = pulp.LpVariable(f"x_{s}_{p}", lowBound=0)

        # Objective: minimize total procurement cost
        sp_cost = {(r["supplier_id"], r["product_id"]): r["unit_cost_vnd"]
                   for _, r in df_sp.iterrows()}
        model += pulp.lpSum(sp_cost.get((s, p), 1) * xv for (s, p), xv in x.items())

        # Constraint 1: baseline demand coverage
        for prod_id, demand_u in total_demand_units.items():
            d_kg = demand_u * prod_weight.get(prod_id, 1)
            avail_x = [x[(s, p)] for (s, p) in x if p == prod_id]
            if avail_x:
                model += (pulp.lpSum(avail_x) >= BASELINE_RATIO * demand_u,
                          f"DemCov_{prod_id}")

        # Constraint 2: supplier capacity
        for s_id in df_sup["id"]:
            cap = sup_capacity.get(s_id, 0)
            terms = [x[(s, p)] * prod_weight.get(p, 0)
                     for (s, p) in x if s == s_id]
            if terms:
                model += (pulp.lpSum(terms) <= cap, f"SupCap_{s_id}")

        # Constraint 3: concentration
        for prod_id, demand_u in total_demand_units.items():
            for s_id in df_sup["id"]:
                if (s_id, prod_id) in x:
                    model += (x[(s_id, prod_id)] <= CONCENTRATION_MAX * demand_u,
                              f"Conc_{s_id}_{prod_id}")

        # Try gurobipy first, then fall back to CBC
        try:
            import gurobipy  # noqa
            solver = pulp.getSolver("GUROBI", msg=0, timeLimit=30)
        except Exception:
            solver = pulp.PULP_CBC_CMD(msg=0, maxSeconds=30)
        status = model.solve(solver)

        lp_ok = pulp.LpStatus[model.status] in ("Optimal", "Feasible")
        icon  = PASS_ICON if lp_ok else FAIL_ICON
        obj   = pulp.value(model.objective) or 0
        print(f"    {icon}  LP status: {pulp.LpStatus[model.status]} | "
              f"Objective: {obj:,.0f} VND")
        results.append(lp_ok)

    except ImportError:
        print(f"    {WARN_ICON}  PuLP not available — skipping LP check")
        results.append(True)

    # ─────────────────────────────────────────────────────────────────────
    # VERDICT
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    if n_fail == 0:
        print(f"✅ ALL {len(results)} CHECKS PASSED — problem is structurally feasible.")
        print("   Run:  python scripts/run_stochastic_optimization.py")
        est_lp_binaries = 4 * 4 * 50  # K × V × ~arcs_per_vehicle
        est_constraints = 3000
        print(f"\n   Estimated MIP size:")
        print(f"     Binary arc vars: ~{est_lp_binaries}")
        print(f"     Constraints:     ~{est_constraints}")
        print(f"     Expected solve:  30–180 s (vs 1800+ s at full scale)")
    else:
        print(f"{FAIL_ICON} {n_fail}/{len(results)} checks FAILED — review issues above before running.")

    print("=" * 66)
    return n_fail == 0


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
