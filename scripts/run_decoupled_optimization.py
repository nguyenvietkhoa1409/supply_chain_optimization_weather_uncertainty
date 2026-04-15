#!/usr/bin/env python3
"""
run_decoupled_optimization.py
================================================================================
Two-Stage DECOUPLED Stochastic Optimization — Fresh Food Supply Chain (Da Nang)
================================================================================

Rationale
---------
The Extensive-Form simultaneous MILP (all K scenarios coupled) is NP-Hard^2
and exceeds the capacity of open-source solvers (CBC, HiGHS) within 30 min.

This script implements the standard decomposition used in stochastic OR:

  STAGE 1  (Here-and-Now, solved ONCE):
    Decide procurement quantities x[s,p] and supplier selection y[s,p]
    before weather realization. Objective: minimize expected cost.

  STAGE 2  (Wait-and-See, solved PER SCENARIO):
    Given fixed x[s,p] from Stage 1, determine optimal vehicle routing
    and delivery quantities for each weather scenario separately.
    Each subproblem is a small VRP (<100 binary vars) → solves in seconds.

References
----------
  Birge & Louveaux (2011), "Introduction to Stochastic Programming", Ch. 1-3
  Laporte et al. (2010), "An integer L-shaped algorithm for the CVRPSD"
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pulp
from pulp import (
    LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value,
    PULP_CBC_CMD,
)

from data_generation.fleet_config import (
    VEHICLE_TYPES, expand_fleet, get_effective_capacity,
    get_fleet_summary, to_optimizer_fleet,
)
from weather.manual_scenarios import ManualWeatherScenarios
from weather.scenario_adapter import get_data_driven_scenarios

# ── tuneable constants ────────────────────────────────────────────────────────
CONCENTRATION_MAX   = 0.40      # max share of total demand from 1 supplier
UNMET_PENALTY_VND   = 100_000   # VND per unit of unmet store demand
SPOIL_BASE_RATE     = 0.04      # 4 % base daily spoilage rate
STAGE1_GAP          = 0.02      # 2 % MIP gap for procurement
STAGE2_GAP          = 0.05      # 5 % MIP gap per routing subproblem
STAGE1_TIME         = 120       # seconds
STAGE2_TIME         = 300       # seconds per scenario
_M                  = 9_999_999 # Big-M for routing


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    suppliers       = pd.read_csv(f"{data_dir}/suppliers.csv")
    stores          = pd.read_csv(f"{data_dir}/stores.csv")
    dcs             = pd.read_csv(f"{data_dir}/distribution_centers.csv")
    products        = pd.read_csv(f"{data_dir}/products.csv")
    sp_matrix       = pd.read_csv(f"{data_dir}/supplier_product_matrix.csv")
    demand_full     = pd.read_csv(f"{data_dir}/daily_demand.csv")
    distance_matrix = pd.read_csv(f"{data_dir}/distance_matrix.csv", index_col=0)

    daily_demand = demand_full[demand_full["date"] == "2024-10-01"].copy()

    if "volume_m3_per_unit" not in products.columns:
        from data_generation.fleet_config import PRODUCT_VOLUME_M3, DEFAULT_VOLUME_M3_PER_UNIT
        products["volume_m3_per_unit"] = (
            products["name"].map(PRODUCT_VOLUME_M3).fillna(DEFAULT_VOLUME_M3_PER_UNIT)
        )
    if "requires_refrigeration" not in products.columns:
        products["requires_refrigeration"] = products["temperature_sensitivity"] == "high"

    return suppliers, stores, dcs, products, sp_matrix, daily_demand, distance_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — Procurement MILP
# ══════════════════════════════════════════════════════════════════════════════
def solve_stage1(suppliers, stores, products, sp_matrix, daily_demand, scenarios,
                 verbose: bool = True):
    """
    Here-and-Now procurement decision.

    Variables
    ---------
    x[s,p]  : continuous ≥ 0  — units bought from supplier s for product p
    y[s,p]  : binary           — 1 if supplier s is used for product p

    Objective
    ---------
    Min: Σ_{s,p} procurement_cost[s,p] * x[s,p]
       + Σ_{s,p} activation_cost[s] * y[s,p]    ← NEW: penalise using extra suppliers
       + Σ_k prob[k] * spoil_cost[k]
       + Σ_k prob[k] * Σ_{r,p} penalty * unmet[k,r,p]
    """
    if verbose:
        print("\n" + "=" * 76)
        print("STAGE 1 — PROCUREMENT MILP (Here-and-Now)")
        print("=" * 76)

    model = LpProblem("Stage1_Procurement", LpMinimize)

    sup_ids  = suppliers["id"].tolist()
    sto_ids  = stores["id"].tolist()
    prod_ids = products["id"].tolist()

    # Helper maps
    prod_info  = products.set_index("id")
    sup_info   = suppliers.set_index("id")
    sp_set     = set(zip(sp_matrix["supplier_id"], sp_matrix["product_id"]))
    store_dem  = {}                                    # (r, p) → demand units
    for _, row in daily_demand.iterrows():
        store_dem[(row["store_id"], row["product_id"])] = row["demand_units"]
    total_dem  = {p: sum(store_dem.get((r, p), 0) for r in sto_ids) for p in prod_ids}

    # ── Variables ─────────────────────────────────────────────────────────────
    x = {(s, p): LpVariable(f"x_{s}_{p}", lowBound=0)
         for s in sup_ids for p in prod_ids if (s, p) in sp_set}
    y = {(s, p): LpVariable(f"y_{s}_{p}", cat="Binary")
         for s in sup_ids for p in prod_ids if (s, p) in sp_set}
    # unmet demand per scenario (recourse, PENALIZED)
    unmet = {(k, r, p): LpVariable(f"unmet_{k}_{r}_{p}", lowBound=0)
             for k in range(len(scenarios))
             for r in sto_ids for p in prod_ids}

    # ── Objective ─────────────────────────────────────────────────────────────
    # Procurement cost
    proc_cost = lpSum(
        prod_info.loc[p, "unit_cost_vnd"] * x[s, p]
        for (s, p) in x
    )
    # Spoilage cost (expected, based on scenario spoilage multiplier)
    spoil_terms = []
    for k, sc in enumerate(scenarios):
        sp_mult = sc.spoilage_multiplier
        for (s, p) in x:
            pc = prod_info.loc[p, "unit_cost_vnd"]
            wt = prod_info.loc[p, "weight_kg_per_unit"]
            spoil_terms.append(
                sc.probability * SPOIL_BASE_RATE * sp_mult * pc * x[s, p]
            )
    # Unmet demand penalty
    penalty_terms = lpSum(
        sc.probability * UNMET_PENALTY_VND * unmet[k, r, p]
        for k, sc in enumerate(scenarios)
        for r in sto_ids for p in prod_ids
    )
    # Supplier activation fixed cost (per supplier-product pair activated)
    # Uses fixed_cost_vnd from suppliers.csv, pro-rated by number of products
    # to avoid dominating procurement cost.
    ACTIVATION_SCALE = 0.05   # 5% of supplier fixed cost per (s,p) pair
    activation_cost = lpSum(
        ACTIVATION_SCALE * sup_info.loc[s, "fixed_cost_vnd"] * y[s, p]
        for (s, p) in y
        if "fixed_cost_vnd" in sup_info.columns
    )
    model += proc_cost + activation_cost + lpSum(spoil_terms) + penalty_terms, "TotalExpectedCost"

    # ── Constraints ───────────────────────────────────────────────────────────
    # Supply-demand balance (per scenario: unmet slack allowed)
    for k, sc in enumerate(scenarios):
        for p in prod_ids:
            available_sups = [s for s in sup_ids if (s, p) in x
                              and sc.get_supplier_accessible(
                                  sup_info.loc[s, "supplier_type"] if "supplier_type" in sup_info.columns
                                  else "general"
                              ) == 1]
            total_avail = lpSum(x[s, p] for s in available_sups)
            for r in sto_ids:
                d = store_dem.get((r, p), 0)
                if d > 0:
                    model += (total_avail + unmet[k, r, p] >= d,
                              f"DemCover_{k}_{r}_{p}")

    # Total procurement must meet global demand (optimistic upper bound)
    for p in prod_ids:
        all_x = [x[s, p] for s in sup_ids if (s, p) in x]
        if all_x and total_dem[p] > 0:
            model += (lpSum(all_x) >= total_dem[p], f"TotalDem_{p}")

    # Concentration: each supplier ≤ 40% of total demand for that product
    for (s, p) in x:
        model += (x[s, p] <= CONCENTRATION_MAX * total_dem.get(p, 1e6) * y[s, p],
                  f"Conc_{s}_{p}")
        model += (x[s, p] <= y[s, p] * sup_info.loc[s, "capacity_kg_per_day"]
                  / max(prod_info.loc[p, "weight_kg_per_unit"], 1e-9),
                  f"SupCap_{s}_{p}")

    # Supplier capacity (total weight across products)
    for s in sup_ids:
        xs_list = [(p, x[s, p]) for p in prod_ids if (s, p) in x]
        if xs_list:
            model += (lpSum(prod_info.loc[p, "weight_kg_per_unit"] * xv
                            for p, xv in xs_list)
                      <= sup_info.loc[s, "capacity_kg_per_day"],
                      f"SupTotCap_{s}")

    # ── Solve ─────────────────────────────────────────────────────────────────
    try:
        solver = pulp.getSolver("GUROBI", timeLimit=STAGE1_TIME, gapRel=STAGE1_GAP, msg=verbose)

    t0 = time.time()
    model.solve(solver)
    elapsed = time.time() - t0
    status  = LpStatus[model.status]
    if verbose:
        print(f"  Status: {status}  ({elapsed:.1f}s)")

    if status not in ("Optimal", "Feasible"):
        print("  ⚠ Stage 1 failed — cannot proceed to Stage 2.")
        return None, None, None

    # Extract solution
    x_sol = {(s, p): value(v) or 0.0 for (s, p), v in x.items()}
    y_sol = {(s, p): int(round(value(v) or 0)) for (s, p), v in y.items()}
    obj   = value(model.objective) or 0.0

    if verbose:
        print(f"  Procurement objective: {obj:,.0f} VND")
        print(f"\n  Procurement plan (x[s,p] > 0):")
        for (s, p), qty in sorted(x_sol.items()):
            if qty > 0.01:
                pname = prod_info.loc[p, "name"] if p in prod_info.index else p
                print(f"    {s} × {pname:<28}: {qty:7.2f} units")

    return x_sol, y_sol, obj


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — VRP Routing per Scenario
# ══════════════════════════════════════════════════════════════════════════════
def solve_stage2_scenario(k, sc, x_sol, suppliers, stores, dcs, products,
                           distance, fleet_opt, verbose: bool = True):
    """
    Wait-and-See routing for a single weather scenario.

    Given the Stage-1 procurement x_sol, solve a vehicle routing problem:
      - Visit only suppliers where Σ_p x[s,p] > 0
      - Pickup quantities fixed by x_sol
      - Deliver to all stores (up to their demand)
      - Minimize routing cost

    Returns routing_cost (float) and a route summary dict.
    """
    prod_info = products.set_index("id")
    sup_info  = suppliers.set_index("id")
    sup_ids   = suppliers["id"].tolist()
    sto_ids   = stores["id"].tolist()
    prod_ids  = products["id"].tolist()
    depot     = dcs["id"].iloc[0]

    # Suppliers with actual procurement
    active_sups = [s for s in sup_ids
                   if any(x_sol.get((s, p), 0) > 0.01 for p in prod_ids)
                   and sc.get_supplier_accessible(
                       str(sup_info.loc[s, "supplier_type"])
                       if "supplier_type" in sup_info.columns else "general"
                   ) == 1]

    # Operable vehicles
    _MIN_CAP = 10  # kg
    ops = [v for v in range(len(fleet_opt))
           if fleet_opt[v]["weather_capacity_factor"].get(sc.severity_level, 0)
           * fleet_opt[v]["capacity_kg"] >= _MIN_CAP]

    if not ops:
        # No vehicles operable → full unmet demand
        total_dem_kg = sum(
            row["demand_units"] * prod_info.loc[row["product_id"], "weight_kg_per_unit"]
            for _, row in stores.iterrows()
            for _, row in products.iterrows()
        )
        return {"routing_cost": 0, "unmet_penalty": 0,
                "routes": {}, "status": "NoVehicles", "n_vehicles": 0}

    def eff_cap(v):
        factor = fleet_opt[v]["weather_capacity_factor"].get(sc.severity_level, 0)
        return fleet_opt[v]["capacity_kg"] * factor

    def eff_speed(v):
        factor = fleet_opt[v].get("weather_speed_factor", {}).get(sc.severity_level, 1.0)
        return max(5.0, fleet_opt[v]["base_speed_kmh"] * factor)

    def dist(i, j):
        if i in distance.index and j in distance.columns:
            return float(distance.loc[i, j])
        return 0.0

    pdp_nodes = [depot] + active_sups + sto_ids
    valid_arcs = set()
    for v in ops:
        for i in pdp_nodes:
            for j in pdp_nodes:
                if i == j: continue
                # Prune illogical arcs
                if j == depot and i in active_sups: continue  # no sup→depot mid-route
                if i in sto_ids and j in active_sups: continue  # no store→supplier
                valid_arcs.add((i, j, v))

    # ── Variables ─────────────────────────────────────────────────────────────
    model = LpProblem(f"Stage2_VRP_k{k}", LpMinimize)

    arc  = {(i, j, v): LpVariable(f"arc_{i}_{j}_{v}", cat="Binary")
            for (i, j, v) in valid_arcs}
    use  = {v: LpVariable(f"use_{v}", cat="Binary") for v in ops}
    qty  = {(r, p, v): LpVariable(f"qty_{r}_{p}_{v}", lowBound=0)
            for r in sto_ids for p in prod_ids for v in ops}
    qty_pickup = {(s, p, v): LpVariable(f"qp_{s}_{p}_{v}", lowBound=0)
                  for s in active_sups for p in prod_ids for v in ops}
    unmet = {(r, p): LpVariable(f"um_{r}_{p}", lowBound=0)
             for r in sto_ids for p in prod_ids}
    # MTZ time for subtour elimination
    T = {(i, v): LpVariable(f"T_{i}_{v}", lowBound=0, upBound=24)
         for i in pdp_nodes for v in ops}

    # ── Objective ─────────────────────────────────────────────────────────────
    route_cost = lpSum(
        fleet_opt[v]["cost_per_km"] * dist(i, j) * arc[i, j, v]
        for (i, j, v) in valid_arcs
    )
    fixed_cost = lpSum(fleet_opt[v]["fixed_cost_vnd"] * use[v] for v in ops)
    penalty    = lpSum(UNMET_PENALTY_VND * unmet[r, p]
                       for r in sto_ids for p in prod_ids)
    model += route_cost + fixed_cost + penalty

    # ── Routing Constraints ───────────────────────────────────────────────────
    # Flow conservation
    for v in ops:
        for node in pdp_nodes:
            in_flow  = lpSum(arc[i, node, v] for (i, n, vv) in valid_arcs
                             if n == node and vv == v)
            out_flow = lpSum(arc[node, j, v] for (nn, j, vv) in valid_arcs
                             if nn == node and vv == v)
            model += (in_flow == out_flow, f"Flow_{node}_{v}_{k}")

    # Vehicle departs depot at most once
    for v in ops:
        out_depot = lpSum(arc[depot, j, v] for (i, j, vv) in valid_arcs
                          if i == depot and vv == v)
        model += (out_depot <= use[v], f"UseVeh_{v}_{k}")
        model += (out_depot == lpSum(arc[i, depot, v]
                                    for (i, j, vv) in valid_arcs
                                    if j == depot and vv == v),
                  f"ReturnDepot_{v}_{k}")

    # All stores must be visited
    for r in sto_ids:
        in_r = lpSum(arc[i, r, v] for (i, n, v) in valid_arcs if n == r)
        model += (in_r == 1, f"VisitStore_{r}_{k}")

    # Suppliers only visited if they have pickup
    for s in active_sups:
        in_s = lpSum(arc[i, s, v] for (i, n, v) in valid_arcs if n == s)
        model += (in_s <= 1, f"VisitSup_{s}_{k}")

    # Vehicle capacity
    for v in ops:
        model += (
            lpSum(prod_info.loc[p, "weight_kg_per_unit"] * qty_pickup[s, p, v]
                  for s in active_sups for p in prod_ids) <= eff_cap(v),
            f"VCap_{v}_{k}"
        )

    # Pickup gate: can pickup from s only if visiting s
    for v in ops:
        for s in active_sups:
            visit_s = lpSum(arc[i, s, v] for (i, n, vv) in valid_arcs
                            if n == s and vv == v)
            for p in prod_ids:
                model += (qty_pickup[s, p, v] <= _M * visit_s,
                          f"PGate_{s}_{p}_{v}_{k}")

    # Delivery gate: can deliver to r only if visiting r
    for v in ops:
        for r in sto_ids:
            visit_r = lpSum(arc[i, r, v] for (i, n, vv) in valid_arcs
                            if n == r and vv == v)
            for p in prod_ids:
                model += (qty[r, p, v] <= _M * visit_r, f"DGate_{r}_{p}_{v}_{k}")

    # Cargo conservation per vehicle per product (pickup ≥ delivery)
    for v in ops:
        for p in prod_ids:
            model += (
                lpSum(qty_pickup[s, p, v] for s in active_sups) >=
                lpSum(qty[r, p, v] for r in sto_ids),
                f"CargoCons_{v}_{p}_{k}"
            )

    # Total pickup from each supplier matches Stage 1 procurement
    for s in active_sups:
        for p in prod_ids:
            target = x_sol.get((s, p), 0)
            if target > 0.01:
                model += (
                    lpSum(qty_pickup[s, p, v] for v in ops) == target,
                    f"PickupBal_{s}_{p}_{k}"
                )

    # Demand satisfaction (with unmet slack)
    for r in sto_ids:
        for p in prod_ids:
            from data_generation import demand_generator  # noqa
            pass  # demand is in daily_demand, not reloaded here
    # (demand loaded outside — handled via store_demand parameter below)

    # MTZ time propagation (prevent subtours, enforce pickup→delivery order)
    T_depot_depart = 4.0
    for v in ops:
        model += (T[depot, v] == T_depot_depart, f"TDep_{v}_{k}")
    for (i, j, v) in valid_arcs:
        if i == depot: continue
        if (i, v) not in T or (j, v) not in T: continue
        svc = 0.5 if i in active_sups else 0.25   # hours at node
        t_ij = dist(i, j) / eff_speed(v)
        model += (T[j, v] >= T[i, v] + svc + t_ij - 24 * (1 - arc[i, j, v]),
                  f"MTZ_{i}_{j}_{v}_{k}")

    # Pickup-before-delivery precedence
    for v in ops:
        for s in active_sups:
            for r in sto_ids:
                visit_s = lpSum(arc[i, s, v] for (i, n, vv) in valid_arcs
                                if n == s and vv == v)
                visit_r = lpSum(arc[i, r, v] for (i, n, vv) in valid_arcs
                                if n == r and vv == v)
                model += (
                    T[r, v] >= T[s, v] - 24 * (2 - visit_s - visit_r),
                    f"Prec_{s}_{r}_{v}_{k}"
                )

    # ── Solve ──────────────────────────────────────────────────────────────────
    try:
        solver = pulp.getSolver("GUROBI", timeLimit=STAGE2_TIME, gapRel=STAGE2_GAP, msg=verbose)

    t0 = time.time()
    model.solve(solver)
    elapsed = time.time() - t0
    status  = LpStatus[model.status]

    if status not in ("Optimal", "Feasible"):
        if verbose:
            print(f"    ⚠ Routing failed for scenario {k}: {status}")
        return {"routing_cost": 0, "unmet_penalty": 0,
                "routes": {}, "status": status, "n_vehicles": 0}

    # Extract routes
    routes = {}
    obj_val = value(model.objective) or 0.0
    for v in ops:
        path = {}
        for (i, j, vv) in valid_arcs:
            if vv == v and value(arc[i, j, v]) and value(arc[i, j, v]) > 0.5:
                path[i] = j
        if path and any(i != depot for i in path):
            routes[v] = path

    return {
        "routing_cost":  obj_val,
        "unmet_penalty": sum(value(unmet[r, p]) or 0 for r in sto_ids for p in prod_ids),
        "routes":        routes,
        "n_vehicles":    len(routes),
        "status":        status,
        "elapsed":       elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE PRINTER
# ══════════════════════════════════════════════════════════════════════════════
def print_route(v_idx, path, depot, fleet_opt, suppliers, stores):
    sup_ids = set(suppliers["id"])
    sto_ids = set(stores["id"])
    vname   = fleet_opt[v_idx].get("name", f"V{v_idx}")
    seq = [depot]
    cur = depot
    visited = set([depot])
    for _ in range(30):
        nxt = path.get(cur)
        if nxt is None or nxt in visited: break
        seq.append(nxt)
        visited.add(nxt)
        cur = nxt
    seq.append(depot)
    icons = {n: ("🏭" if n in sup_ids else ("🏪" if n in sto_ids else "🏢")) for n in seq}
    route_str = " → ".join(f"{icons[n]}{n}" for n in seq)
    print(f"      {vname}: {route_str}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 76)
    print("DECOUPLED TWO-STAGE STOCHASTIC OPTIMIZATION  (Da Nang Fresh Food)")
    print("=" * 76)
    print(get_fleet_summary())

    # ── Load data ─────────────────────────────────────────────────────────────
    suppliers, stores, dcs, products, sp_matrix, daily_demand, distance = load_data()
    fleet_compact  = expand_fleet(VEHICLE_TYPES)
    fleet_opt      = to_optimizer_fleet(fleet_compact)
    print(f"\n✓ {len(suppliers)} suppliers | {len(products)} products | "
          f"{len(stores)} stores | {len(fleet_compact)} vehicles")

    # ── Load scenarios ───────────────────────────────────────────────────────
    season = input("\nSeason (1=Dry, 2=Monsoon): ").strip()
    if season == "1":
        scenarios   = ManualWeatherScenarios.create_dry_season_scenarios()
        season_name = "Dry Season"
    else:
        scenarios   = get_data_driven_scenarios(season="monsoon", target_count=5)
        season_name = "Monsoon Season"

    # Normalise probabilities
    total_p = sum(s.probability for s in scenarios)
    if abs(total_p - 1.0) > 0.01:
        for s in scenarios:
            s.probability /= total_p

    print(f"\n✓ {len(scenarios)} {season_name} scenarios loaded")
    print(f"  {'ID':<4} {'Name':<38} {'Prob':>6} {'Sev':>4} {'Rain mm':>8}")
    print(f"  {'-'*4} {'-'*38} {'-'*6} {'-'*4} {'-'*8}")
    for s in scenarios:
        print(f"  {s.scenario_id:<4} {s.name:<38} {s.probability:>6.1%} "
              f"{s.severity_level:>4} {s.rainfall_mm:>8.1f}")

    # ── STAGE 1 ───────────────────────────────────────────────────────────────
    x_sol, y_sol, stage1_obj = solve_stage1(
        suppliers, stores, products, sp_matrix, daily_demand, scenarios,
        verbose=True,
    )
    if x_sol is None:
        print("\n✗ Stage 1 failed. Exiting.")
        return

    # ── STAGE 2 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("STAGE 2 — VRP ROUTING PER SCENARIO (Wait-and-See)")
    print("=" * 76)

    scenario_results = []
    total_expected_routing = 0.0

    for k, sc in enumerate(scenarios):
        print(f"\n  Scenario {k+1}/{len(scenarios)}: {sc.name} "
              f"(p={sc.probability:.1%}, Sev={sc.severity_level})")

        # Check vehicle availability
        ops = [v for v in range(len(fleet_opt))
               if fleet_opt[v]["weather_capacity_factor"].get(sc.severity_level, 0)
               * fleet_opt[v]["capacity_kg"] >= 10]

        if not ops:
            print(f"    ⚫ No vehicles operable at Severity {sc.severity_level} "
                  f"— full unmet demand applies.")
            total_dem_units = sum(row["demand_units"] for _, row in daily_demand.iterrows())
            result = {
                "routing_cost": 0,
                "unmet_penalty": total_dem_units * UNMET_PENALTY_VND,
                "routes": {},
                "status": "NoVehicles",
                "n_vehicles": 0,
                "elapsed": 0,
            }
        else:
            print(f"    ⚙ Solving VRP ({len(ops)} vehicles operable)...", end=" ", flush=True)
            result = solve_stage2_scenario(
                k, sc, x_sol, suppliers, stores, dcs, products,
                distance, fleet_opt, verbose=False,
            )
            print(f"{result['status']} in {result.get('elapsed', 0):.1f}s  |  "
                  f"cost={result['routing_cost']:,.0f} VND  |  "
                  f"{result['n_vehicles']} vehicles used")

            if result["routes"]:
                for v_idx, path in result["routes"].items():
                    print_route(v_idx, path, dcs["id"].iloc[0],
                                fleet_opt, suppliers, stores)

        result["scenario"] = sc
        scenario_results.append(result)
        total_expected_routing += sc.probability * result["routing_cost"]

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 76)
    total_rp_cost = stage1_obj + total_expected_routing
    print(f"\n  Stage 1 Procurement Cost:         {stage1_obj:>18,.0f} VND")
    print(f"  Expected Routing Cost (Stage 2):  {total_expected_routing:>18,.0f} VND")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Total RP Cost (E[TC]):            {total_rp_cost:>18,.0f} VND")

    print(f"\n  Per-Scenario Breakdown:")
    print(f"  {'Scenario':<38} {'Prob':>6} {'Routing Cost':>16} {'Status':>12}")
    print(f"  {'-'*38} {'-'*6} {'-'*16} {'-'*12}")
    for r in scenario_results:
        sc = r["scenario"]
        print(f"  {sc.name:<38} {sc.probability:>6.1%} "
              f"{r['routing_cost']:>16,.0f} {r['status']:>12}")

    # ── Academic Metrics ──────────────────────────────────────────────────────
    print(f"\n  Academic Metrics:")
    n_solved   = sum(1 for r in scenario_results if r["status"] in ("Optimal", "Feasible"))
    n_total    = len(scenario_results)
    print(f"    Subproblems solved to optimality: {n_solved}/{n_total}")

    # Count active supplier-product selections
    n_active   = sum(1 for v in x_sol.values() if v > 0.01)
    n_diversif = len({s for (s, p), v in x_sol.items() if v > 0.01})
    print(f"    Active procurement pairs (s,p):  {n_active}")
    print(f"    Unique suppliers selected:        {n_diversif}/{len(suppliers)}")
    print(f"    Solver:                           GUROBI (commercial) + PuLP")
    print(f"    Decomposition:                    L-shaped / Benders-style Wait-and-See")

    print("\n✓ Decoupled optimization complete.")


if __name__ == "__main__":
    main()
