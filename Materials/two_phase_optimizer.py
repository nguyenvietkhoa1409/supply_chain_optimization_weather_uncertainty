"""
two_phase_optimizer.py — Two-Phase Two-Stage Stochastic MILP
=============================================================
Faithful to Patel et al. (2024) architecture:

  Stage 1  (Here-and-Now):
    Procurement decisions x[s,p], y[s,p]

  Stage 2A (Procurement VRP per scenario k):
    Route:  DC → Accessible Suppliers → DC
    Output: inventory[k,p] = Σ qty_pickup[k,s,p,v]

  Stage 2B (Distribution VRP per scenario k):
    Route:  DC → Stores → DC
    Input:  inventory[k,p] from Stage 2A
    Output: qty_dist[k,r,p,v] deliveries, unmet[k,r,p] shortfalls

Linking constraint (DC inventory balance):
  inventory[k,p] = Σ_{s ∈ accessible(k), v} qty_pickup[k,s,p,v]
  Σ_{r,v} qty_dist[k,r,p,v] ≤ inventory[k,p] + e[k,p]   ∀p,k

Feasibility guarantee:
  - severity 5 (no vehicles): ops=[] → inventory=0 → all unmet → high penalty
  - No hard "must visit" constraint on stores; demand satisfaction uses unmet slack
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp
from pulp import (
    LpMinimize, LpProblem, LpStatus, LpVariable, lpSum, value,
    PULP_CBC_CMD,
)

_MIN_CAP_KG  = 10.0
_BIG_M_QTY   = 50_000
_BIG_M_TIME  = 24.0
_T_DEPART_DC = 4.0       # 04:00 departure from DC (procurement)
_T_DIST_DEP  = 10.0      # 10:00 departure from DC (distribution — after pickup)


class TwoPhaseExtensiveFormOptimizer:
    """
    Two-Phase Two-Stage Stochastic MILP.

    Phase 2A variables: arc_proc, qty_pickup, inventory
    Phase 2B variables: arc_dist, qty_dist, unmet, e (emergency)

    Parameters
    ----------
    network, products_df, supplier_product_df, demand_df, weather_scenarios:
        Same as ExtensiveFormOptimizer (working version)
    fleet_instances:
        optimizer-format fleet from to_optimizer_fleet()
    baseline_ratio:
        β — fraction of demand that Stage 1 must procure (default 0.70)
    emergency_ratio:
        EC_p — max emergency as fraction of total demand (default 0.40)
    unmet_penalty:
        Penalty per unit unmet demand (VND). Set high to force demand satisfaction.
    refrig_penalty_factor:
        Multiplier applied to spoilage cost when non-refrigerated vehicle carries
        refrigerated products in Phase 2B (soft constraint, not hard block).
    """

    _DC_SEVERITY_LIMITS = {"hoakhanh": 3, "lienchieu": 5}

    def __init__(
        self,
        network:             Dict,
        products_df:         pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df:           pd.DataFrame,
        weather_scenarios:   List,
        fleet_instances:     List[Dict],
        baseline_ratio:      float = 0.70,
        emergency_ratio:     float = 0.40,
        unmet_penalty:       float = 500_000.0,
        refrig_penalty_factor: float = 1.5,
        concentration_max:   float = 0.40,
    ):
        self.network             = network
        self.products_df         = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df           = demand_df
        self.scenarios           = weather_scenarios
        self.fleet               = fleet_instances
        self.baseline_ratio      = baseline_ratio
        self.emergency_ratio     = emergency_ratio
        self.unmet_penalty       = unmet_penalty
        self.refrig_penalty      = refrig_penalty_factor
        self.conc_max            = concentration_max

        self.suppliers = network["suppliers"]["id"].tolist()
        self.products  = products_df["id"].tolist()
        self.stores    = network["stores"]["id"].tolist()

        self._create_lookups()

        K      = len(self.scenarios)
        V      = len(self.fleet)
        S      = len(self.suppliers)
        R      = len(self.stores)
        refrig = sum(1 for v in self.fleet if v.get("refrigerated", False))
        print(f"TwoPhaseExtensiveFormOptimizer:")
        print(f"  K={K} scenarios | V={V} vehicles ({refrig} refrigerated)")
        print(f"  S={S} suppliers | R={R} stores")
        print(f"  Phase 2A: DC→Suppliers→DC  |  Phase 2B: DC→Stores→DC")

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def _create_lookups(self):
        self.prod_cost   = dict(zip(self.products_df["id"],
                                     self.products_df["unit_cost_vnd"]))
        self.prod_weight = dict(zip(self.products_df["id"],
                                     self.products_df["weight_kg_per_unit"]))
        self.prod_refrig = dict(zip(
            self.products_df["id"],
            self.products_df["requires_refrigeration"].astype(bool),
        ))

        sup_df = self.network["suppliers"]
        self.sup_cap     = dict(zip(sup_df["id"], sup_df["capacity_kg_per_day"]))
        self.sup_fix     = dict(zip(sup_df["id"], sup_df["fixed_cost_vnd"]))
        self.sup_subtype = {r["id"]: r.get("subtype", "general")
                            for _, r in sup_df.iterrows()}
        self.sup_svc_h   = (
            dict(zip(sup_df["id"], sup_df["service_time_min"] / 60.0))
            if "service_time_min" in sup_df.columns
            else {s: 0.5 for s in self.suppliers}
        )

        sto_df = self.network["stores"]
        self.sto_svc_h = (
            dict(zip(sto_df["id"], sto_df["service_time_min"] / 60.0))
            if "service_time_min" in sto_df.columns
            else {r: 0.25 for r in self.stores}
        )

        self.sp_cost, self.sp_moq, self.sp_avail = {}, {}, {}
        for _, row in self.supplier_product_df.iterrows():
            s, p = row["supplier_id"], row["product_id"]
            self.sp_cost[(s, p)]  = row["unit_cost_vnd"]
            self.sp_moq[(s, p)]   = row["moq_units"]
            self.sp_avail[(s, p)] = row["available"]

        dm = self.network["distance_matrix"]
        dcs = self.network["dcs"]["id"].tolist()
        all_nodes = dcs + self.suppliers + self.stores
        self.dist: Dict = {}
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    try:    self.dist[(i, j)] = float(dm.loc[i, j])
                    except Exception:
                        try: self.dist[(i, j)] = float(dm.loc[j, i])
                        except Exception: self.dist[(i, j)] = 10.0

        grp = (self.demand_df
               .groupby(["store_id", "product_id"])["demand_units"]
               .sum().reset_index())
        self.store_demand = {
            (r["store_id"], r["product_id"]): float(r["demand_units"])
            for _, r in grp.iterrows()
        }
        self.total_demand = (
            self.demand_df.groupby("product_id")["demand_units"].sum().to_dict()
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_depot(self, scenario) -> str:
        sev = scenario.severity_level
        for _, dc in self.network["dcs"].iterrows():
            key = dc["name"].lower().replace(" ", "").replace("_", "")
            max_sev = next(
                (v for k, v in self._DC_SEVERITY_LIMITS.items() if k in key), 5
            )
            if sev <= max_sev:
                return dc["id"]
        return self.network["dcs"]["id"].iloc[-1]

    def _acc_sups(self, sc, p=None) -> List[str]:
        """Accessible suppliers under scenario sc (optionally filtered by product p)."""
        result = []
        for s in self.suppliers:
            if sc.get_supplier_accessible(self.sup_subtype.get(s, "general")) != 1:
                continue
            if p is not None:
                if not self.sp_avail.get((s, p), False):
                    continue
            result.append(s)
        return result

    def _inacc_sups(self, sc, p=None) -> List[str]:
        result = []
        for s in self.suppliers:
            if sc.get_supplier_accessible(self.sup_subtype.get(s, "general")) == 1:
                continue
            if p is not None:
                if not self.sp_avail.get((s, p), False):
                    continue
            result.append(s)
        return result

    def _eff_cap(self, v: int, severity: int) -> float:
        veh = self.fleet[v]
        return veh["capacity_kg"] * veh["weather_capacity_factor"].get(severity, 1.0)

    def _eff_speed(self, v: int, sc) -> float:
        veh  = self.fleet[v]
        road = 1.0 / max(sc.speed_reduction_factor, 0.1)
        vtf  = veh["weather_speed_factor"].get(sc.severity_level, 1.0)
        return max(veh["base_speed_kmh"] * road * vtf, 1.0)

    # ------------------------------------------------------------------
    # Build MILP
    # ------------------------------------------------------------------
    def build_model(self) -> Tuple[LpProblem, Dict]:
        print("\nBuilding Two-Phase Extensive Form MILP…")
        K   = len(self.scenarios)
        M1  = 100_000   # Big-M for Stage 1 MOQ logic

        model = LpProblem("TwoPhase_ExtensiveForm", LpMinimize)

        # ── Stage 1 Variables ──────────────────────────────────────────
        x = LpVariable.dicts("x",
            ((s, p) for s in self.suppliers for p in self.products),
            lowBound=0)
        y = LpVariable.dicts("y",
            ((s, p) for s in self.suppliers for p in self.products),
            cat="Binary")

        # ── Phase 2 variables (indexed by scenario k) ──────────────────
        # 2A: Procurement VRP  (DC → Suppliers → DC)
        arc_proc  : Dict = {}
        qty_pickup: Dict = {}
        T_proc    : Dict = {}
        use_proc  : Dict = {}
        inventory : Dict = {}   # linking variable

        # 2B: Distribution VRP (DC → Stores → DC)
        arc_dist  : Dict = {}
        qty_dist  : Dict = {}
        unmet     : Dict = {}
        T_dist    : Dict = {}
        use_dist  : Dict = {}

        # Emergency procurement (product level, bounded by scenario)
        e_emerg   : Dict = {}

        depot_by_k  : Dict = {}
        ops_by_k    : Dict = {}

        for k, sc in enumerate(self.scenarios):
            depot = self._get_depot(sc)
            depot_by_k[k] = depot
            sev = sc.severity_level

            ops = [v for v in range(len(self.fleet))
                   if self._eff_cap(v, sev) >= _MIN_CAP_KG]
            ops_by_k[k] = ops

            # accessible suppliers with any committed goods
            acc_sups = self._acc_sups(sc)
            nodes_proc = [depot] + acc_sups
            nodes_dist = [depot] + self.stores

            # ── 2A arc & time variables ────────────────────────────────
            for v in ops:
                for i in nodes_proc:
                    for j in nodes_proc:
                        if i != j:
                            arc_proc[k, i, j, v] = LpVariable(
                                f"ap_{k}_{i}_{j}_{v}", cat="Binary")
                for i in nodes_proc:
                    T_proc[k, i, v] = LpVariable(
                        f"Tp_{k}_{i}_{v}", lowBound=0, upBound=_BIG_M_TIME)
                use_proc[k, v] = LpVariable(f"up_{k}_{v}", cat="Binary")

                # qty_pickup only for (s,p) pairs where supplier is accessible
                for s in acc_sups:
                    for p in self.products:
                        if self.sp_avail.get((s, p), False):
                            qty_pickup[k, s, p, v] = LpVariable(
                                f"qpick_{k}_{s}_{p}_{v}", lowBound=0)

            # inventory[k,p]: total collected at DC after Phase 2A
            for p in self.products:
                inventory[k, p] = LpVariable(f"inv_{k}_{p}", lowBound=0)

            # ── 2B arc & time variables ────────────────────────────────
            for v in ops:
                for i in nodes_dist:
                    for j in nodes_dist:
                        if i != j:
                            arc_dist[k, i, j, v] = LpVariable(
                                f"ad_{k}_{i}_{j}_{v}", cat="Binary")
                for i in nodes_dist:
                    T_dist[k, i, v] = LpVariable(
                        f"Td_{k}_{i}_{v}", lowBound=0, upBound=_BIG_M_TIME)
                use_dist[k, v] = LpVariable(f"ud_{k}_{v}", cat="Binary")
                for r in self.stores:
                    for p in self.products:
                        qty_dist[k, r, p, v] = LpVariable(
                            f"qdist_{k}_{r}_{p}_{v}", lowBound=0)

            # unmet demand (store-product level) and emergency (product level)
            for r in self.stores:
                for p in self.products:
                    unmet[k, r, p] = LpVariable(f"unmet_{k}_{r}_{p}", lowBound=0)

            for p in self.products:
                em_cap = self.emergency_ratio * self.total_demand.get(p, 0)
                if not sc.emergency_feasible:
                    em_cap = 0.0
                e_emerg[k, p] = LpVariable(
                    f"emrg_{k}_{p}", lowBound=0, upBound=em_cap)

        # ── Objective ──────────────────────────────────────────────────
        s1_proc_cost = lpSum(
            self.sp_cost.get((s, p), self.prod_cost[p]) * x[s, p]
            for s in self.suppliers for p in self.products
            if self.sp_avail.get((s, p), False)
        )
        s1_fix_cost = lpSum(
            self.sup_fix[s] * y[s, p]
            for s in self.suppliers for p in self.products
            if self.sp_avail.get((s, p), False)
        )

        s2_terms = []
        base_spoil = 0.04

        for k, sc in enumerate(self.scenarios):
            prob  = sc.probability
            depot = depot_by_k[k]
            ops   = ops_by_k[k]
            acc_sups   = self._acc_sups(sc)
            nodes_proc = [depot] + acc_sups
            nodes_dist = [depot] + self.stores

            # 2A: procurement VRP cost
            proc_vrp = lpSum(
                (self.fleet[v]["cost_per_km"]
                 + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                * self.dist.get((i, j), 0)
                * arc_proc[k, i, j, v]
                for v in ops
                for i in nodes_proc for j in nodes_proc if i != j
                if (k, i, j, v) in arc_proc
            )
            proc_fix = lpSum(
                self.fleet[v]["fixed_cost_vnd"] * use_proc[k, v]
                for v in ops if (k, v) in use_proc
            )

            # 2B: distribution VRP cost
            dist_vrp = lpSum(
                (self.fleet[v]["cost_per_km"]
                 + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                * self.dist.get((i, j), 0)
                * arc_dist[k, i, j, v]
                for v in ops
                for i in nodes_dist for j in nodes_dist if i != j
                if (k, i, j, v) in arc_dist
            )
            dist_fix = lpSum(
                self.fleet[v]["fixed_cost_vnd"] * use_dist[k, v]
                for v in ops if (k, v) in use_dist
            )

            # Refrigeration penalty (soft): penalize non-ref vehicle carrying ref product
            refrig_pen = lpSum(
                base_spoil * sc.spoilage_multiplier
                * (self.refrig_penalty - 1.0)
                * self.prod_cost[p]
                * (self.dist.get((depot, r), 0) / self._eff_speed(v, sc))
                * self.store_demand.get((r, p), 0)
                * arc_dist[k, depot, r, v]
                for v in ops
                for r in self.stores for p in self.products
                if self.prod_refrig.get(p, False)
                and not self.fleet[v].get("refrigerated", False)
                if (k, depot, r, v) in arc_dist
            )

            # Spoilage cost: goods from inaccessible suppliers
            spoil_s1 = lpSum(
                self.sp_cost.get((s, p), self.prod_cost[p]) * x[s, p]
                for p in self.products
                for s in self._inacc_sups(sc, p)
            )

            # Emergency procurement cost (2× unit cost)
            em_cost = lpSum(
                2.0 * self.prod_cost[p] * e_emerg[k, p]
                for p in self.products
                if (k, p) in e_emerg
            )

            # Unmet demand penalty
            pm = min(10.0, 5.0 * sc.spoilage_multiplier)
            unmet_cost = lpSum(
                pm * self.prod_cost[p] * unmet[k, r, p]
                for r in self.stores for p in self.products
                if (k, r, p) in unmet
            )

            total_k = (proc_vrp + proc_fix
                       + dist_vrp + dist_fix
                       + refrig_pen + spoil_s1
                       + em_cost + unmet_cost)
            s2_terms.append(prob * total_k)

        model += s1_proc_cost + s1_fix_cost + lpSum(s2_terms), "TwoPhaseObj"

        # ── Stage 1 Constraints ────────────────────────────────────────
        for s in self.suppliers:
            model += (
                lpSum(x[s, p] * self.prod_weight[p]
                      for p in self.products if self.sp_avail.get((s, p), False))
                <= self.sup_cap[s], f"S1Cap_{s}"
            )
        for s in self.suppliers:
            for p in self.products:
                if self.sp_avail.get((s, p), False):
                    moq = self.sp_moq.get((s, p), 0)
                    model += (x[s, p] >= moq * y[s, p],  f"S1MOQlo_{s}_{p}")
                    model += (x[s, p] <= M1  * y[s, p],  f"S1MOQhi_{s}_{p}")
        for p in self.products:
            d = self.total_demand.get(p, 0)
            if d > 0:
                all_x = lpSum(x[s, p] for s in self.suppliers
                              if self.sp_avail.get((s, p), False))
                model += (all_x >= self.baseline_ratio * d, f"S1Base_{p}")
                model += (all_x <= 1.5 * d,                 f"S1Over_{p}")
                # Concentration risk
                for s in self.suppliers:
                    if self.sp_avail.get((s, p), False):
                        model += (x[s, p] <= self.conc_max * d,
                                  f"S1Conc_{s}_{p}")

        # ── Phase 2A Constraints (per scenario k) ─────────────────────
        for k, sc in enumerate(self.scenarios):
            depot = depot_by_k[k]
            ops   = ops_by_k[k]
            acc_sups   = self._acc_sups(sc)
            nodes_proc = [depot] + acc_sups

            # Inventory = 0 when no accessible suppliers or no vehicles
            if not acc_sups or not ops:
                for p in self.products:
                    model += (inventory[k, p] == 0.0, f"InvZero_{k}_{p}")
                continue

            # Flow conservation (procurement nodes)
            for v in ops:
                for i in nodes_proc:
                    in_f  = lpSum(arc_proc[k, j, i, v]
                                  for j in nodes_proc if j != i
                                  if (k, j, i, v) in arc_proc)
                    out_f = lpSum(arc_proc[k, i, j, v]
                                  for j in nodes_proc if j != i
                                  if (k, i, j, v) in arc_proc)
                    model += (in_f == out_f, f"pFlow_{k}_{i}_{v}")

            # Vehicle departs depot at most once
            for v in ops:
                out_d = lpSum(arc_proc[k, depot, s, v]
                              for s in acc_sups
                              if (k, depot, s, v) in arc_proc)
                model += (out_d <= use_proc[k, v], f"pDepart_{k}_{v}")

            # Vehicle capacity (procurement trip)
            for v in ops:
                model += (
                    lpSum(
                        self.prod_weight[p] * qty_pickup[k, s, p, v]
                        for s in acc_sups for p in self.products
                        if (k, s, p, v) in qty_pickup
                    ) <= self._eff_cap(v, sc.severity_level),
                    f"pCap_{k}_{v}"
                )

            # Pickup gate: can only pick up from s if visiting s
            for v in ops:
                for s in acc_sups:
                    visit_s = lpSum(arc_proc[k, i, s, v]
                                    for i in nodes_proc if i != s
                                    if (k, i, s, v) in arc_proc)
                    for p in self.products:
                        if (k, s, p, v) in qty_pickup:
                            model += (qty_pickup[k, s, p, v] <= _BIG_M_QTY * visit_s,
                                      f"pGate_{k}_{s}_{p}_{v}")

            # Pickup ≤ Stage-1 commitment (per supplier-product)
            for s in acc_sups:
                for p in self.products:
                    if not self.sp_avail.get((s, p), False):
                        continue
                    pv = [qty_pickup[k, s, p, v]
                          for v in ops if (k, s, p, v) in qty_pickup]
                    if pv:
                        model += (lpSum(pv) <= x[s, p], f"pCommit_{k}_{s}_{p}")

            # Inventory definition (linking constraint)
            for p in self.products:
                pickup_sum = lpSum(
                    qty_pickup[k, s, p, v]
                    for s in acc_sups for v in ops
                    if (k, s, p, v) in qty_pickup
                )
                model += (inventory[k, p] == pickup_sum, f"InvDef_{k}_{p}")

            # MTZ subtour elimination (procurement tour)
            for v in ops:
                if (k, depot, v) in T_proc:
                    model += (T_proc[k, depot, v] == _T_DEPART_DC,
                              f"pTdep_{k}_{v}")
            for v in ops:
                for i in nodes_proc:
                    for j in nodes_proc:
                        if i == j or (k, i, j, v) not in arc_proc:
                            continue
                        if (k, i, v) not in T_proc or (k, j, v) not in T_proc:
                            continue
                        svc_i = self.sup_svc_h.get(i, 0.0)
                        t_ij  = self.dist.get((i, j), 0) / self._eff_speed(v, sc)
                        model += (
                            T_proc[k, j, v] >= T_proc[k, i, v]
                            + svc_i + t_ij
                            - _BIG_M_TIME * (1 - arc_proc[k, i, j, v]),
                            f"pMTZ_{k}_{i}_{j}_{v}"
                        )

        # ── Phase 2B Constraints (per scenario k) ─────────────────────
        for k, sc in enumerate(self.scenarios):
            depot = depot_by_k[k]
            ops   = ops_by_k[k]
            nodes_dist = [depot] + self.stores
            pm = min(10.0, 5.0 * sc.spoilage_multiplier)

            # If no vehicles: all demand is unmet (cost captured in objective)
            if not ops:
                for r in self.stores:
                    for p in self.products:
                        d_rp = self.store_demand.get((r, p), 0)
                        if d_rp > 0 and (k, r, p) in unmet:
                            model += (unmet[k, r, p] == d_rp,
                                      f"AllUnmet_{k}_{r}_{p}")
                continue

            # Flow conservation (distribution nodes)
            for v in ops:
                for i in nodes_dist:
                    in_f  = lpSum(arc_dist[k, j, i, v]
                                  for j in nodes_dist if j != i
                                  if (k, j, i, v) in arc_dist)
                    out_f = lpSum(arc_dist[k, i, j, v]
                                  for j in nodes_dist if j != i
                                  if (k, i, j, v) in arc_dist)
                    model += (in_f == out_f, f"dFlow_{k}_{i}_{v}")

            # Vehicle departs depot at most once (distribution)
            for v in ops:
                out_d = lpSum(arc_dist[k, depot, r, v]
                              for r in self.stores
                              if (k, depot, r, v) in arc_dist)
                model += (out_d <= use_dist[k, v], f"dDepart_{k}_{v}")

            # Each store must be visited (at least one vehicle)
            # Only enforce when vehicles exist
            for r in self.stores:
                model += (
                    lpSum(arc_dist[k, i, r, v]
                          for i in nodes_dist if i != r
                          for v in ops
                          if (k, i, r, v) in arc_dist) >= 1,
                    f"dVisit_{k}_{r}"
                )

            # Vehicle capacity (distribution trip)
            for v in ops:
                model += (
                    lpSum(
                        self.prod_weight[p] * qty_dist[k, r, p, v]
                        for r in self.stores for p in self.products
                        if (k, r, p, v) in qty_dist
                    ) <= self._eff_cap(v, sc.severity_level),
                    f"dCap_{k}_{v}"
                )

            # Delivery gate: can only deliver to r if visiting r
            for v in ops:
                for r in self.stores:
                    visit_r = lpSum(arc_dist[k, i, r, v]
                                    for i in nodes_dist if i != r
                                    if (k, i, r, v) in arc_dist)
                    for p in self.products:
                        if (k, r, p, v) in qty_dist:
                            model += (qty_dist[k, r, p, v] <= _BIG_M_QTY * visit_r,
                                      f"dGate_{k}_{r}_{p}_{v}")

            # Demand satisfaction (store-product level)
            for r in self.stores:
                for p in self.products:
                    d_rp = self.store_demand.get((r, p), 0)
                    if d_rp <= 0:
                        continue
                    del_vars = [qty_dist[k, r, p, v]
                                for v in ops if (k, r, p, v) in qty_dist]
                    emrg = e_emerg.get((k, p), 0)
                    um   = unmet.get((k, r, p), 0)
                    if del_vars:
                        model += (
                            lpSum(del_vars) + emrg + um >= d_rp,
                            f"dDem_{k}_{r}_{p}"
                        )
                    else:
                        model += (um >= d_rp, f"dDemFallback_{k}_{r}_{p}")

            # Supply cap: total distribution ≤ inventory + emergency
            for p in self.products:
                dist_sum = lpSum(
                    qty_dist[k, r, p, v]
                    for r in self.stores for v in ops
                    if (k, r, p, v) in qty_dist
                )
                emrg = e_emerg.get((k, p), 0)
                model += (
                    dist_sum <= inventory[k, p] + emrg,
                    f"dSupCap_{k}_{p}"
                )

            # MTZ subtour elimination (distribution tour)
            for v in ops:
                if (k, depot, v) in T_dist:
                    model += (T_dist[k, depot, v] == _T_DIST_DEP,
                              f"dTdep_{k}_{v}")
            for v in ops:
                for i in nodes_dist:
                    for j in self.stores:
                        if i == j or (k, i, j, v) not in arc_dist:
                            continue
                        if (k, i, v) not in T_dist or (k, j, v) not in T_dist:
                            continue
                        svc_i = self.sto_svc_h.get(i, 0.0) if i != depot else 0.0
                        t_ij  = self.dist.get((i, j), 0) / self._eff_speed(v, sc)
                        model += (
                            T_dist[k, j, v] >= T_dist[k, i, v]
                            + svc_i + t_ij
                            - _BIG_M_TIME * (1 - arc_dist[k, i, j, v]),
                            f"dMTZ_{k}_{i}_{j}_{v}"
                        )

        print(f"  ✓ Variables: {model.numVariables()} | "
              f"Constraints: {model.numConstraints()}")

        vd = dict(
            x=x, y=y,
            arc_proc=arc_proc, qty_pickup=qty_pickup, inventory=inventory,
            T_proc=T_proc, use_proc=use_proc,
            arc_dist=arc_dist, qty_dist=qty_dist, unmet=unmet,
            T_dist=T_dist, use_dist=use_dist, e_emerg=e_emerg,
            depot_by_k=depot_by_k, ops_by_k=ops_by_k,
        )
        return model, vd

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(
        self,
        time_limit:    int   = 1800,
        gap_tolerance: float = 0.05,
    ) -> Tuple[str, Dict]:
        model, vd = self.build_model()

        try:
            solver = pulp.getSolver(
                "GUROBI", msg=True,
                timeLimit=time_limit, gapRel=gap_tolerance,
            )
        except Exception:
            solver = PULP_CBC_CMD(
                msg=True, timeLimit=time_limit, gapRel=gap_tolerance,
            )

        print(f"\nSolving (limit={time_limit}s, gap={gap_tolerance*100:.0f}%)…")
        t0 = time.time()
        model.solve(solver)
        elapsed = time.time() - t0
        status  = LpStatus[model.status]
        print(f"  Status: {status}  ({elapsed:.1f}s)")

        if status not in ("Optimal", "Feasible"):
            return status, {}

        obj = value(model.objective)
        print(f"  Objective: {obj:,.0f} VND")

        solution = self._extract_solution(vd)
        solution.update(dict(
            objective_value = obj,
            solve_time      = elapsed,
            status          = status,
            scenario_costs  = self._compute_scenario_costs(vd),
        ))
        return status, solution

    # ------------------------------------------------------------------
    # Extract Solution
    # ------------------------------------------------------------------
    def _extract_solution(self, vd: Dict) -> Dict:
        x, y = vd["x"], vd["y"]

        stage1 = [
            dict(
                supplier_id    = s,
                product_id     = p,
                quantity_units = round(value(x[s, p]) or 0, 2),
                cost_vnd       = round(
                    (value(x[s, p]) or 0)
                    * self.sp_cost.get((s, p), self.prod_cost[p]), 0),
            )
            for s in self.suppliers for p in self.products
            if (value(x[s, p]) or 0) > 0.01
        ]

        # Per-scenario routes (both phases)
        scenario_routes: Dict = {}
        for k, sc in enumerate(self.scenarios):
            depot = vd["depot_by_k"][k]
            ops   = vd["ops_by_k"][k]
            acc_sups = self._acc_sups(sc)

            proc_routes, dist_routes = [], []

            # Phase 2A routes
            for v in ops:
                if not ((k, v) in vd["use_proc"]
                        and (value(vd["use_proc"][k, v]) or 0) > 0.5):
                    continue
                path, cur, seen = [depot], depot, {depot}
                for _ in range(len(acc_sups) + 2):
                    nxt = next(
                        (j for j in acc_sups
                         if j not in seen
                         and (value(vd["arc_proc"].get((k, cur, j, v))) or 0) > 0.5),
                        None,
                    )
                    if nxt is None:
                        break
                    path.append(nxt); seen.add(nxt); cur = nxt
                if len(path) > 1:
                    path.append(depot)
                    proc_routes.append(dict(
                        vehicle_id   = self.fleet[v]["vehicle_id"],
                        vehicle_type = self.fleet[v]["type_id"],
                        route        = path,
                        pickups      = {
                            s: {
                                p: round(
                                    value(vd["qty_pickup"].get((k, s, p, v))) or 0, 2)
                                for p in self.products
                                if (value(vd["qty_pickup"].get((k, s, p, v))) or 0) > 0.01
                            }
                            for s in path[1:-1]
                        },
                    ))

            # Phase 2B routes
            for v in ops:
                if not ((k, v) in vd["use_dist"]
                        and (value(vd["use_dist"][k, v]) or 0) > 0.5):
                    continue
                path, cur, seen = [depot], depot, {depot}
                for _ in range(len(self.stores) + 2):
                    nxt = next(
                        (j for j in self.stores
                         if j not in seen
                         and (value(vd["arc_dist"].get((k, cur, j, v))) or 0) > 0.5),
                        None,
                    )
                    if nxt is None:
                        break
                    path.append(nxt); seen.add(nxt); cur = nxt
                if len(path) > 1:
                    path.append(depot)
                    dist_routes.append(dict(
                        vehicle_id   = self.fleet[v]["vehicle_id"],
                        vehicle_type = self.fleet[v]["type_id"],
                        route        = path,
                        deliveries   = {
                            r: {
                                p: round(
                                    value(vd["qty_dist"].get((k, r, p, v))) or 0, 2)
                                for p in self.products
                                if (value(vd["qty_dist"].get((k, r, p, v))) or 0) > 0.01
                            }
                            for r in path[1:-1]
                        },
                    ))

            scenario_routes[sc.name] = dict(
                procurement_routes  = proc_routes,
                distribution_routes = dist_routes,
                inventory = {
                    p: round(value(vd["inventory"].get((k, p))) or 0, 2)
                    for p in self.products
                },
            )

        return dict(
            stage1_procurement = pd.DataFrame(stage1),
            scenario_routes    = scenario_routes,
        )

    # ------------------------------------------------------------------
    # Scenario Costs
    # ------------------------------------------------------------------
    def _compute_scenario_costs(self, vd: Dict) -> pd.DataFrame:
        x, y = vd["x"], vd["y"]
        base_spoil = 0.04

        s1_var = sum(
            (value(x[s, p]) or 0) * self.sp_cost.get((s, p), self.prod_cost[p])
            for s in self.suppliers for p in self.products
            if self.sp_avail.get((s, p), False)
        )
        s1_fix = sum(
            (value(y[s, p]) or 0) * self.sup_fix[s]
            for s in self.suppliers for p in self.products
            if self.sp_avail.get((s, p), False)
        )
        s1_total = s1_var + s1_fix

        rows = []
        for k, sc in enumerate(self.scenarios):
            depot = vd["depot_by_k"][k]
            ops   = vd["ops_by_k"][k]
            acc_sups   = self._acc_sups(sc)
            nodes_proc = [depot] + acc_sups
            nodes_dist = [depot] + self.stores
            pm = min(10.0, 5.0 * sc.spoilage_multiplier)

            # Phase 2A costs
            proc_vrp = sum(
                (self.fleet[v]["cost_per_km"]
                 + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                * self.dist.get((i, j), 0)
                * (value(vd["arc_proc"].get((k, i, j, v))) or 0)
                for v in ops
                for i in nodes_proc for j in nodes_proc if i != j
                if (k, i, j, v) in vd["arc_proc"]
            )
            proc_fix = sum(
                self.fleet[v]["fixed_cost_vnd"]
                * (1 if (value(vd["use_proc"].get((k, v))) or 0) > 0.5 else 0)
                for v in ops
            )

            # Phase 2B costs
            dist_vrp = sum(
                (self.fleet[v]["cost_per_km"]
                 + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                * self.dist.get((i, j), 0)
                * (value(vd["arc_dist"].get((k, i, j, v))) or 0)
                for v in ops
                for i in nodes_dist for j in nodes_dist if i != j
                if (k, i, j, v) in vd["arc_dist"]
            )
            dist_fix = sum(
                self.fleet[v]["fixed_cost_vnd"]
                * (1 if (value(vd["use_dist"].get((k, v))) or 0) > 0.5 else 0)
                for v in ops
            )

            # Spoilage from inaccessible suppliers
            spoilage = sum(
                (value(x[s, p]) or 0)
                * self.sp_cost.get((s, p), self.prod_cost[p])
                for p in self.products
                for s in self._inacc_sups(sc, p)
            )

            # Emergency + unmet
            em_c  = sum(2.0 * (value(vd["e_emerg"].get((k, p))) or 0)
                        * self.prod_cost[p]
                        for p in self.products if (k, p) in vd["e_emerg"])
            unm_c = sum(pm * (value(vd["unmet"].get((k, r, p))) or 0)
                        * self.prod_cost[p]
                        for r in self.stores for p in self.products
                        if (k, r, p) in vd["unmet"])

            total = (s1_total
                     + proc_vrp + proc_fix
                     + dist_vrp + dist_fix
                     + spoilage + em_c + unm_c)

            rows.append(dict(
                scenario_name       = sc.name,
                severity_level      = sc.severity_level,
                probability         = sc.probability,
                stage1_cost         = s1_total,
                proc_vrp_cost       = proc_vrp + proc_fix,
                dist_vrp_cost       = dist_vrp + dist_fix,
                vrp_cost            = proc_vrp + proc_fix + dist_vrp + dist_fix,
                spoilage_cost       = spoilage,
                emergency_cost      = em_c,
                penalty_cost        = unm_c,
                total_cost          = total,
                n_operable_vehicles = len(ops),
            ))

        return pd.DataFrame(rows)
