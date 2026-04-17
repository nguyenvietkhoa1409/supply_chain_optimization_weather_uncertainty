"""
procurement_vrp.py — Phase 2A: Procurement VRP
===============================================
Route: DC → Accessible Suppliers → DC  (pickup tour)

Faithful to Patel et al. (2024) Procurement Model:
  - Vehicles depart DC, visit accessible supplier nodes
  - Pick up committed quantities (x[s,p] from Stage 1)
  - Return to DC — building physical inventory for Phase 2B

Key linking constraint:
  inventory[p] = Σ_{s accessible, v} qty_pickup[s,p,v]

Inaccessible supplier commitments → spoilage cost (opportunity cost)
No vehicles operable → inventory = 0, cost = 0 (always feasible)
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pulp import (
    LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value,
    PULP_CBC_CMD,
)

try:
    import pulp as _pulp
    _HAS_GUROBI = True
except Exception:
    _HAS_GUROBI = False

_MIN_CAP_KG  = 10.0      # minimum effective capacity to be considered operable
_BIG_M_QTY   = 50_000    # Big-M for pickup gate
_BIG_M_TIME  = 24.0      # time horizon (hours)
_T_DEPART_DC = 4.0       # vehicles leave DC at 4 AM


class ProcurementVRP:
    """
    Phase 2A VRP: DC → Suppliers → DC (pickup tour).

    Given Stage-1 procurement decisions x_sol = {(s, p): quantity},
    vehicles physically collect the committed goods from accessible suppliers.

    Usage
    -----
    vrp = ProcurementVRP(network, products_df, fleet, x_sol, scenario)
    result = vrp.solve()
    # result["inventory"]  → {product_id: qty_collected}
    # result["cost"]       → float (VND)
    # result["routes"]     → list of route dicts
    # result["spoilage"]   → {product_id: qty_lost} (inaccessible suppliers)
    """

    _DC_SEVERITY_LIMITS = {"hoakhanh": 3, "lienchieu": 5}

    def __init__(
        self,
        network:      Dict,
        products_df:  pd.DataFrame,
        fleet:        List[Dict],   # optimizer-format from to_optimizer_fleet()
        x_sol:        Dict,         # {(supplier_id, product_id): quantity}
        scenario,                   # WeatherScenario
        time_limit:   int   = 120,
        gap:          float = 0.05,
        verbose:      bool  = False,
    ):
        self.network     = network
        self.products_df = products_df
        self.fleet       = fleet
        self.x_sol       = x_sol
        self.sc          = scenario
        self.time_limit  = time_limit
        self.gap         = gap
        self.verbose     = verbose

        self.suppliers = network["suppliers"]["id"].tolist()
        self.products  = products_df["id"].tolist()

        self._build_lookups()

    # ------------------------------------------------------------------
    def _build_lookups(self):
        sup_df = self.network["suppliers"]
        self.sup_subtype = {
            r["id"]: r.get("subtype", "general")
            for _, r in sup_df.iterrows()
        }
        self.sup_svc_h = (
            dict(zip(sup_df["id"], sup_df["service_time_min"] / 60.0))
            if "service_time_min" in sup_df.columns
            else {s: 0.5 for s in self.suppliers}
        )
        self.prod_weight = dict(zip(
            self.products_df["id"],
            self.products_df["weight_kg_per_unit"],
        ))
        self.prod_refrig = dict(zip(
            self.products_df["id"],
            self.products_df["requires_refrigeration"].astype(bool),
        ))

        self.depot = self._select_depot()
        nodes = [self.depot] + self.suppliers
        dm = self.network["distance_matrix"]
        self.dist: Dict = {}
        for i in nodes:
            for j in nodes:
                if i != j:
                    try:    self.dist[(i, j)] = float(dm.loc[i, j])
                    except Exception:
                        try: self.dist[(i, j)] = float(dm.loc[j, i])
                        except Exception: self.dist[(i, j)] = 10.0

    def _select_depot(self) -> str:
        sev = self.sc.severity_level
        for _, dc in self.network["dcs"].iterrows():
            key = dc["name"].lower().replace(" ", "").replace("_", "")
            max_sev = next(
                (v for k, v in self._DC_SEVERITY_LIMITS.items() if k in key), 5
            )
            if sev <= max_sev:
                return dc["id"]
        return self.network["dcs"]["id"].iloc[-1]

    def _accessible_suppliers(self) -> List[str]:
        """Suppliers accessible under this scenario with non-zero procurement."""
        return [
            s for s in self.suppliers
            if self.sc.get_supplier_accessible(self.sup_subtype.get(s, "general")) == 1
            and any(self.x_sol.get((s, p), 0) > 0.01 for p in self.products)
        ]

    def _eff_cap(self, v: int) -> float:
        veh = self.fleet[v]
        return veh["capacity_kg"] * veh["weather_capacity_factor"].get(
            self.sc.severity_level, 1.0
        )

    def _eff_speed(self, v: int) -> float:
        veh  = self.fleet[v]
        road = 1.0 / max(self.sc.speed_reduction_factor, 0.1)
        vtf  = veh["weather_speed_factor"].get(self.sc.severity_level, 1.0)
        return max(veh["base_speed_kmh"] * road * vtf, 1.0)

    # ------------------------------------------------------------------
    def _compute_spoilage(self) -> Dict[str, float]:
        """Goods committed from inaccessible suppliers → spoilage."""
        spoilage: Dict[str, float] = {}
        for p in self.products:
            qty = sum(
                self.x_sol.get((s, p), 0)
                for s in self.suppliers
                if self.sc.get_supplier_accessible(
                    self.sup_subtype.get(s, "general")
                ) == 0
            )
            if qty > 0.01:
                spoilage[p] = qty
        return spoilage

    # ------------------------------------------------------------------
    def solve(self) -> Dict:
        """
        Solve Phase 2A and return:
          status        : str
          inventory     : {product_id: qty}   ← input for Phase 2B
          pickup_detail : {(sup_id, prod_id): qty}
          routes        : list of dicts
          cost          : float (VND)
          spoilage      : {product_id: qty}
        """
        spoilage   = self._compute_spoilage()
        accessible = self._accessible_suppliers()
        zero_inv   = {p: 0.0 for p in self.products}

        # Edge case: nothing accessible
        if not accessible:
            return dict(
                status="NoAccessibleSuppliers",
                inventory=zero_inv, pickup_detail={},
                routes=[], cost=0.0, spoilage=spoilage,
            )

        # Operable vehicles
        ops = [v for v in range(len(self.fleet)) if self._eff_cap(v) >= _MIN_CAP_KG]
        if not ops:
            return dict(
                status="NoVehicles",
                inventory=zero_inv, pickup_detail={},
                routes=[], cost=0.0, spoilage=spoilage,
            )

        depot = self.depot
        nodes = [depot] + accessible
        model = LpProblem(f"ProcVRP_{self.sc.name.replace(' ', '_')}", LpMinimize)

        # ── Variables ────────────────────────────────────────────────
        arc = {
            (i, j, v): LpVariable(f"parc_{i}_{j}_{v}", cat="Binary")
            for i in nodes for j in nodes if i != j
            for v in ops
        }
        use_v = {v: LpVariable(f"puse_{v}", cat="Binary") for v in ops}

        # Pickup quantities (only for (s,p) with actual commitment)
        qty_pickup = {
            (s, p, v): LpVariable(f"ppick_{s}_{p}_{v}", lowBound=0)
            for s in accessible
            for p in self.products
            if self.x_sol.get((s, p), 0) > 0.01
            for v in ops
        }

        T = {
            (i, v): LpVariable(f"pT_{i}_{v}", lowBound=0, upBound=_BIG_M_TIME)
            for i in nodes for v in ops
        }

        # ── Objective ────────────────────────────────────────────────
        km_cost = lpSum(
            (self.fleet[v]["cost_per_km"]
             + self.fleet[v]["cost_per_hour"] / self._eff_speed(v))
            * self.dist.get((i, j), 0)
            * arc[i, j, v]
            for (i, j, v) in arc
        )
        fix_cost = lpSum(
            self.fleet[v]["fixed_cost_vnd"] * use_v[v]
            for v in ops
        )
        model += km_cost + fix_cost

        # ── Constraints ──────────────────────────────────────────────
        # Flow conservation at every node for every vehicle
        for v in ops:
            for i in nodes:
                in_f  = lpSum(arc[j, i, v] for j in nodes if j != i
                              if (j, i, v) in arc)
                out_f = lpSum(arc[i, j, v] for j in nodes if j != i
                              if (i, j, v) in arc)
                model += (in_f == out_f, f"pFlow_{i}_{v}")

        # Vehicle departs depot at most once
        for v in ops:
            out_depot = lpSum(
                arc[depot, s, v]
                for s in accessible
                if (depot, s, v) in arc
            )
            model += (out_depot <= use_v[v], f"pDepart_{v}")

        # Vehicle capacity (weight-based)
        for v in ops:
            model += (
                lpSum(
                    self.prod_weight[p] * qty_pickup[s, p, v]
                    for (s, p, vv) in qty_pickup if vv == v
                ) <= self._eff_cap(v),
                f"pCap_{v}"
            )

        # Pickup gate: can only pick up from s if vehicle visits s
        for (s, p, v) in qty_pickup:
            visit_s = lpSum(
                arc[i, s, v] for i in nodes if i != s if (i, s, v) in arc
            )
            model += (qty_pickup[s, p, v] <= _BIG_M_QTY * visit_s,
                      f"pGate_{s}_{p}_{v}")

        # Total pickup ≤ Stage-1 commitment (per supplier-product)
        for s in accessible:
            for p in self.products:
                committed = self.x_sol.get((s, p), 0)
                if committed > 0.01:
                    pv_vars = [
                        qty_pickup[s, p, v]
                        for v in ops if (s, p, v) in qty_pickup
                    ]
                    if pv_vars:
                        model += (lpSum(pv_vars) <= committed,
                                  f"pCommit_{s}_{p}")

        # MTZ time propagation (subtour elimination)
        for v in ops:
            model += (T[depot, v] == _T_DEPART_DC, f"pTdep_{v}")
        for (i, j, v) in arc:
            if i == depot:
                continue
            if (i, v) not in T or (j, v) not in T:
                continue
            svc_i = self.sup_svc_h.get(i, 0.5)
            t_ij  = self.dist.get((i, j), 0) / self._eff_speed(v)
            model += (
                T[j, v] >= T[i, v] + svc_i + t_ij
                          - _BIG_M_TIME * (1 - arc[i, j, v]),
                f"pMTZ_{i}_{j}_{v}"
            )

        # ── Solve ────────────────────────────────────────────────────
        try:
            import pulp
            solver = pulp.getSolver(
                "GUROBI", msg=self.verbose,
                timeLimit=self.time_limit, gapRel=self.gap,
            )
        except Exception:
            solver = PULP_CBC_CMD(
                msg=self.verbose, timeLimit=self.time_limit, gapRel=self.gap,
            )

        t0 = time.time()
        model.solve(solver)
        status = LpStatus[model.status]

        # ── Fallback if VRP fails (full pickup assumed) ──────────────
        if status not in ("Optimal", "Feasible"):
            pickup_detail = {
                (s, p): self.x_sol.get((s, p), 0)
                for s in accessible for p in self.products
                if self.x_sol.get((s, p), 0) > 0.01
            }
            inventory = {p: 0.0 for p in self.products}
            for (s, p), q in pickup_detail.items():
                inventory[p] = inventory.get(p, 0.0) + q
            return dict(
                status=f"Fallback({status})",
                inventory=inventory,
                pickup_detail=pickup_detail,
                routes=[], cost=0.0, spoilage=spoilage,
            )

        # ── Extract solution ─────────────────────────────────────────
        vrp_obj = value(model.objective) or 0.0

        pickup_detail: Dict = {}
        for (s, p, v), var in qty_pickup.items():
            q = value(var) or 0.0
            if q > 0.01:
                pickup_detail[(s, p)] = pickup_detail.get((s, p), 0.0) + q

        inventory = {p: 0.0 for p in self.products}
        for (s, p), q in pickup_detail.items():
            inventory[p] = inventory.get(p, 0.0) + q

        # Reconstruct routes
        routes = []
        for v in ops:
            if (value(use_v[v]) or 0) < 0.5:
                continue
            path, cur, seen = [depot], depot, {depot}
            for _ in range(len(accessible) + 2):
                nxt = next(
                    (j for j in accessible
                     if j not in seen
                     and (value(arc.get((cur, j, v))) or 0) > 0.5),
                    None,
                )
                if nxt is None:
                    break
                path.append(nxt)
                seen.add(nxt)
                cur = nxt
            if len(path) > 1:
                path.append(depot)
                pickups_on_route = {
                    s: {
                        p: round(pickup_detail.get((s, p), 0), 2)
                        for p in self.products
                        if pickup_detail.get((s, p), 0) > 0.01
                    }
                    for s in path[1:-1]
                }
                routes.append(dict(
                    vehicle_id   = self.fleet[v]["vehicle_id"],
                    vehicle_type = self.fleet[v]["type_id"],
                    refrigerated = self.fleet[v]["refrigerated"],
                    route        = path,
                    pickups      = pickups_on_route,
                    elapsed_s    = round(time.time() - t0, 1),
                ))

        return dict(
            status        = status,
            inventory     = inventory,
            pickup_detail = pickup_detail,
            routes        = routes,
            cost          = vrp_obj,
            spoilage      = spoilage,
        )
