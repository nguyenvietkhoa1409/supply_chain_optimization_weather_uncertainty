"""
two_phase_optimizer_v2.py  —  Two-Phase Two-Stage Stochastic MILP
==================================================================
CLEAN REWRITE: Emergency procurement removed entirely.

Design rationale
----------------
The original model allowed emergency procurement (e_emerg) as a DC-level
variable that flowed directly into the Phase-2B supply cap.  This gave the
solver a cheap substitute for Phase-2A routing:

    Original supply cap:  dist_sum ≤ inventory + e_emerg

Because e_emerg had no routing cost, the solver always preferred it over
dispatching Phase-2A vehicles.  Every attempted fix (waste penalty, mandatory
pickup, S1 baseline anchoring) either left the loophole open or created
cross-scenario infeasibility.

This version removes e_emerg completely.  The only demand-satisfaction levers
are now:
  1. Physical inventory built by Phase-2A pickup.
  2. Unmet-demand slack (heavily penalised in objective).

With no backdoor, the solver is genuinely forced to run Phase-2A whenever
vehicles and accessible suppliers are available.

Spoilage from inaccessible suppliers is still modelled as an opportunity cost
in the objective so that Stage-1 diversification remains meaningful.

Architecture
------------
Stage 1  (here-and-now):
    x[s,p]  — procurement quantity from supplier s for product p
    y[s,p]  — binary activation (for fixed ordering cost)

Stage 2A (per scenario k, procurement VRP):
    Route DC → accessible suppliers → DC
    qty_pickup[k,s,p,v]  — quantity picked up
    inventory[k,p]        = Σ qty_pickup   (linking variable)

Stage 2B (per scenario k, distribution VRP):
    Route DC → stores → DC
    qty_dist[k,r,p,v]     — quantity delivered to store r
    unmet[k,r,p]          — unmet demand slack (penalised)

Supply cap (Phase 2B):
    Σ_v qty_dist[k,r,p,v]  ≤  inventory[k,p]   ← NO emergency term
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

from data_generation.spoilage_model import compute_spoilage_rate

# ── Module-level constants ─────────────────────────────────────────────────────
_MIN_CAP_KG  = 10.0      # minimum effective vehicle capacity to be considered operable
_BIG_M_QTY   = 50_000    # Big-M for gate constraints on quantities
_BIG_M_TIME  = 24.0      # planning horizon in hours
_T_DEPART_DC = 4.0       # Phase-2A departure time (04:00)
_T_DIST_DEP  = 10.0      # Phase-2B departure time (10:00, after pickup returns)

# Penalty multiplier for unmet demand.
# Must be high enough that satisfying demand through routing is always cheaper.
# Set to 10× unit cost — well above any plausible procurement + routing cost.
_UNMET_PENALTY_MULT = 10.0

# Waste penalty multiplier for Stage-1 goods ordered but not picked up.
# Must exceed the savings from skipping Phase-2A.
# Analysis: skipping Phase-2A saves proc_fixed (~150k-1.2M VND/vehicle).
# Waste penalty = 3.5 × unit_cost × qty; at typical quantities this is ~50-200M.
# 3.5× is sufficient to make pickup always cheaper than abandoning goods.
_WASTE_MULT = 3.5


class TwoPhaseExtensiveFormOptimizer:
    """
    Two-Phase Two-Stage Stochastic MILP (no emergency procurement).

    Parameters
    ----------
    network              : dict with keys suppliers, dcs, stores, distance_matrix
    products_df          : DataFrame — id, unit_cost_vnd, weight_kg_per_unit,
                           requires_refrigeration
    supplier_product_df  : DataFrame — supplier_id, product_id, unit_cost_vnd,
                           moq_units, available
    demand_df            : DataFrame — store_id, product_id, demand_units
    weather_scenarios    : list of WeatherScenario objects
    fleet_instances      : list of vehicle dicts (optimizer format from
                           to_optimizer_fleet())
    baseline_ratio       : β — minimum fraction of demand procured at Stage 1
    concentration_max    : max fraction of demand from any single supplier
    refrig_penalty_factor: soft spoilage multiplier for non-refrigerated vehicles
                           carrying refrigerated products
    """

    _DC_SEVERITY_LIMITS = {"hoakhanh": 3, "lienchieu": 5}

    def __init__(
        self,
        network:              Dict,
        products_df:          pd.DataFrame,
        supplier_product_df:  pd.DataFrame,
        demand_df:            pd.DataFrame,
        weather_scenarios:    List,
        fleet_instances:      List[Dict],
        baseline_ratio:       float = 0.70,
        concentration_max:    float = 0.40,
        refrig_penalty_factor: float = 1.5,
    ):
        self.network    = network
        self.products_df = products_df
        self.sp_df      = supplier_product_df
        self.demand_df  = demand_df
        self.scenarios  = weather_scenarios
        self.fleet      = fleet_instances
        self.beta       = baseline_ratio
        self.conc_max   = concentration_max
        self.refrig_pen = refrig_penalty_factor

        self.suppliers = network["suppliers"]["id"].tolist()
        self.products  = products_df["id"].tolist()
        self.stores    = network["stores"]["id"].tolist()

        self._build_lookups()

        K = len(self.scenarios)
        V = len(self.fleet)
        print(f"TwoPhaseOptimizerV2 (no emergency procurement):")
        print(f"  K={K} scenarios | V={V} vehicles | "
              f"S={len(self.suppliers)} suppliers | R={len(self.stores)} stores")

    # ── Lookups ───────────────────────────────────────────────────────────────
    def _build_lookups(self):
        self.prod_cost   = dict(zip(self.products_df["id"],
                                    self.products_df["unit_cost_vnd"]))
        self.prod_weight = dict(zip(self.products_df["id"],
                                    self.products_df["weight_kg_per_unit"]))
        self.prod_refrig = {
            r["id"]: bool(r["requires_refrigeration"])
            for _, r in self.products_df.iterrows()
        }

        # Tiered penalty with goodwill loss factor
        self.prod_category = {
            r["id"]: r.get("category", "vegetable")
            for _, r in self.products_df.iterrows()
        }
        penalty_base = {"seafood": 4.5, "meat": 4.0, "vegetable": 2.5, "fruit": 2.0}
        goodwill_loss = 0.20
        self.prod_penalty_mult = {}
        for p in self.products:
            cat = self.prod_category.get(p, "vegetable")
            self.prod_penalty_mult[p] = penalty_base.get(cat, 3.0) + goodwill_loss
            if self.prod_refrig.get(p, False):
                self.prod_penalty_mult[p] = max(self.prod_penalty_mult[p], 5.0)

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
        for _, row in self.sp_df.iterrows():
            s, p = row["supplier_id"], row["product_id"]
            self.sp_cost[(s, p)]  = row["unit_cost_vnd"]
            self.sp_moq[(s, p)]   = row["moq_units"]
            self.sp_avail[(s, p)] = row["available"]

        dm        = self.network["distance_matrix"]
        dcs       = self.network["dcs"]["id"].tolist()
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

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_depot(self, sc) -> str:
        sev = sc.severity_level
        for _, dc in self.network["dcs"].iterrows():
            key = dc["name"].lower().replace(" ", "").replace("_", "")
            max_sev = next(
                (v for k, v in self._DC_SEVERITY_LIMITS.items() if k in key), 5
            )
            if sev <= max_sev:
                return dc["id"]
        return self.network["dcs"]["id"].iloc[-1]

    def _acc_sups(self, sc, p=None) -> List[str]:
        return [
            s for s in self.suppliers
            if sc.get_supplier_accessible(self.sup_subtype.get(s, "general")) == 1
            and (p is None or self.sp_avail.get((s, p), False))
        ]

    def _inacc_sups(self, sc, p=None) -> List[str]:
        return [
            s for s in self.suppliers
            if sc.get_supplier_accessible(self.sup_subtype.get(s, "general")) != 1
            and (p is None or self.sp_avail.get((s, p), False))
        ]

    def _eff_cap(self, v: int, severity: int) -> float:
        veh = self.fleet[v]
        return veh["capacity_kg"] * veh["weather_capacity_factor"].get(severity, 1.0)

    def _eff_speed(self, v: int, sc) -> float:
        veh  = self.fleet[v]
        road = 1.0 / max(sc.speed_reduction_factor, 0.1)
        vtf  = veh["weather_speed_factor"].get(sc.severity_level, 1.0)
        return max(veh["base_speed_kmh"] * road * vtf, 1.0)

    # ── Model construction ────────────────────────────────────────────────────
    def build_model(
        self,
        fixed_stage1: Optional[pd.DataFrame] = None,
    ) -> Tuple[LpProblem, Dict]:

        print("\nBuilding Two-Phase MILP (v2 — no emergency procurement)…")
        M1 = 100_000

        model = LpProblem("TwoPhase_v2", LpMinimize)

        # ── Stage 1 variables ──────────────────────────────────────────────
        if fixed_stage1 is not None:
            # EEV mode: fix Stage-1 procurement at EV solution
            fixed_qty = {}
            for _, row in fixed_stage1.iterrows():
                fixed_qty[(row["supplier_id"], row["product_id"])] = \
                    row["quantity_units"]
            x, y = {}, {}
            for s in self.suppliers:
                for p in self.products:
                    val  = fixed_qty.get((s, p), 0.0)
                    yval = 1 if val > 0 else 0
                    x[s, p] = LpVariable(f"x_{s}_{p}", lowBound=val, upBound=val)
                    y[s, p] = LpVariable(f"y_{s}_{p}", lowBound=yval, upBound=yval)
            print("  [EEV mode] Stage-1 fixed from deterministic EV solution.")
        else:
            x = LpVariable.dicts("x",
                ((s, p) for s in self.suppliers for p in self.products),
                lowBound=0)
            y = LpVariable.dicts("y",
                ((s, p) for s in self.suppliers for p in self.products),
                cat="Binary")

        # ── Per-scenario variable dictionaries ────────────────────────────
        arc_proc  : Dict = {}
        qty_pickup: Dict = {}
        T_proc    : Dict = {}
        use_proc  : Dict = {}
        inventory : Dict = {}
        waste_vars: Dict = {}

        arc_dist  : Dict = {}
        qty_dist  : Dict = {}
        unmet     : Dict = {}
        T_dist    : Dict = {}
        use_dist  : Dict = {}

        depot_by_k: Dict = {}
        ops_by_k  : Dict = {}

        for k, sc in enumerate(self.scenarios):
            depot     = self._get_depot(sc)
            depot_by_k[k] = depot
            ops       = [v for v in range(len(self.fleet))
                         if self._eff_cap(v, sc.severity_level) >= _MIN_CAP_KG]
            ops_by_k[k] = ops

            acc_sups   = self._acc_sups(sc)
            nodes_proc = [depot] + acc_sups
            nodes_dist = [depot] + self.stores

            # Phase-2A arcs, times, pickup quantities
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
                for s in acc_sups:
                    for p in self.products:
                        if self.sp_avail.get((s, p), False):
                            qty_pickup[k, s, p, v] = LpVariable(
                                f"qpick_{k}_{s}_{p}_{v}", lowBound=0)

            # Inventory (Phase-2A output = Phase-2B input)
            for p in self.products:
                inventory[k, p] = LpVariable(f"inv_{k}_{p}", lowBound=0)

            # Waste variables (ordered but not picked up)
            for s in acc_sups:
                for p in self.products:
                    if self.sp_avail.get((s, p), False):
                        waste_vars[k, s, p] = LpVariable(
                            f"waste_{k}_{s}_{p}", lowBound=0)

            # Phase-2B arcs, times, delivery quantities, unmet slack
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

            for r in self.stores:
                for p in self.products:
                    unmet[k, r, p] = LpVariable(f"unmet_{k}_{r}_{p}", lowBound=0)

        # ── Objective ─────────────────────────────────────────────────────
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

        s2_terms   = []
        base_spoil = 0.04
        
        # Minimum baseline fleet cost (assumes 4 vehicles are owned and their daily costs are sunk)
        n_owned = min(4, len(self.fleet))
        min_fleet_cost = sum(self.fleet[i].get("fixed_cost_vnd", 600_000) for i in range(n_owned))

        for k, sc in enumerate(self.scenarios):
            prob       = sc.probability
            depot      = depot_by_k[k]
            ops        = ops_by_k[k]
            acc_sups   = self._acc_sups(sc)
            nodes_proc = [depot] + acc_sups
            nodes_dist = [depot] + self.stores

            # Phase-2A routing cost
            proc_vrp = lpSum(
                ((self.fleet[v]["cost_per_km"]
                  + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                 * self.dist.get((i, j), 0)
                 + (self.fleet[v].get("loading_cost_per_stop", 25_000) if j != depot else 0))
                * arc_proc[k, i, j, v]
                for v in ops
                for i in nodes_proc for j in nodes_proc if i != j
                if (k, i, j, v) in arc_proc
            )
            proc_fix = lpSum(
                (0 if v < n_owned else self.fleet[v]["fixed_cost_vnd"]) * use_proc[k, v]
                for v in ops if (k, v) in use_proc
            )

            # Phase-2B routing cost
            dist_vrp = lpSum(
                ((self.fleet[v]["cost_per_km"]
                  + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                 * self.dist.get((i, j), 0)
                 + (self.fleet[v].get("loading_cost_per_stop", 25_000) if j != depot else 0))
                * arc_dist[k, i, j, v]
                for v in ops
                for i in nodes_dist for j in nodes_dist if i != j
                if (k, i, j, v) in arc_dist
            )
            dist_fix = lpSum(
                (0 if v < n_owned else self.fleet[v]["fixed_cost_vnd"]) * use_dist[k, v]
                for v in ops if (k, v) in use_dist
            )

            # Arrhenius-based refrigeration penalty (non-refrig vehicle carrying cold product)
            refrig_pen = lpSum(
                compute_spoilage_rate(
                    category=self.prod_category.get(p, "vegetable"),
                    transport_time_h=self.dist.get((depot, r), 0) / max(0.1, self._eff_speed(v, sc)),
                    is_refrigerated=False,
                    ambient_temp_c=getattr(sc, "temperature_celsius", 28.0)
                )
                * self.prod_cost[p]
                * (self.store_demand.get((r, p), 0) * getattr(sc, 'demand_reduction_factor', 1.0))
                * arc_dist[k, depot, r, v]
                for v in ops
                for r in self.stores for p in self.products
                if self.prod_refrig.get(p, False)
                and not self.fleet[v].get("refrigerated", False)
                if (k, depot, r, v) in arc_dist
            )

            # Spoilage opportunity cost from inaccessible suppliers (Assumes 24h wait)
            spoil_s1 = lpSum(
                compute_spoilage_rate(
                    category=self.prod_category.get(p, "vegetable"),
                    transport_time_h=24.0,
                    is_refrigerated=False,
                    ambient_temp_c=getattr(sc, "temperature_celsius", 28.0)
                )
                * getattr(sc, "spoilage_multiplier", 1.0)
                * self.sp_cost.get((s, p), self.prod_cost[p]) * x[s, p]
                for p in self.products
                for s in self._inacc_sups(sc, p)
            )

            # Unmet demand penalty  (NO emergency term)
            unmet_cost = lpSum(
                self.prod_penalty_mult[p] * self.prod_cost[p] * unmet[k, r, p]
                for r in self.stores for p in self.products
                if (k, r, p) in unmet
            )

            # Waste penalty (Stage-1 goods ordered but not physically picked up)
            waste_cost = lpSum(
                _WASTE_MULT * self.sp_cost.get((s, p), self.prod_cost[p])
                * waste_vars[k, s, p]
                for s in acc_sups for p in self.products
                if self.sp_avail.get((s, p), False)
                and (k, s, p) in waste_vars
            )

            total_k = (proc_vrp + proc_fix + dist_vrp + dist_fix
                       + refrig_pen + spoil_s1 + unmet_cost + waste_cost)
            s2_terms.append(prob * total_k)

        model += s1_proc_cost + s1_fix_cost + min_fleet_cost + lpSum(s2_terms), "Obj"

        # ── Stage-1 constraints ───────────────────────────────────────────

        # Supplier weight capacity
        if fixed_stage1 is None:
            for s in self.suppliers:
                model += (
                    lpSum(x[s, p] * self.prod_weight[p]
                          for p in self.products if self.sp_avail.get((s, p), False))
                    <= self.sup_cap[s], f"S1Cap_{s}"
                )

            # MOQ activation
            for s in self.suppliers:
                for p in self.products:
                    if self.sp_avail.get((s, p), False):
                        moq = self.sp_moq.get((s, p), 0)
                        model += (x[s, p] >= moq * y[s, p], f"S1MOQlo_{s}_{p}")
                        model += (x[s, p] <= M1  * y[s, p], f"S1MOQhi_{s}_{p}")

            # Per-product baseline, overstock, concentration
            for p in self.products:
                d = self.total_demand.get(p, 0)
                if d <= 0:
                    continue

                all_x = lpSum(x[s, p] for s in self.suppliers
                              if self.sp_avail.get((s, p), False))

                # S1AccBase: baseline anchored to accessible suppliers in sev=1
                # (prevents solver dumping all orders on inaccessible suppliers)
                best_sc   = self.scenarios[0]
                if best_sc.severity_level <= 2:
                    acc_best  = self._acc_sups(best_sc, p)
                    if acc_best:
                        acc_x_best = lpSum(
                            x[s, p] for s in acc_best
                            if self.sp_avail.get((s, p), False)
                        )
                        model += (acc_x_best >= self.beta * d, f"S1AccBase_{p}")

                # S1ScenarioBase: per-scenario floor scaled by capacity_reduction_factor
                # Floor is capped so it never exceeds concentration limits (feasibility).
                for k_s, sc_s in enumerate(self.scenarios):
                    acc_s = self._acc_sups(sc_s, p)
                    ops_s = ops_by_k.get(k_s, [])
                    if not acc_s or not ops_s:
                        continue
                    n_acc     = len([s for s in acc_s if self.sp_avail.get((s, p), False)])
                    max_acc   = self.conc_max * d * n_acc          # physical ceiling
                    raw_floor = self.beta * d * sc_s.capacity_reduction_factor
                    floor     = min(raw_floor, max_acc * 0.95)     # 5% safety margin
                    if floor > 1e-3:
                        acc_x_s = lpSum(
                            x[s, p] for s in acc_s
                            if self.sp_avail.get((s, p), False)
                        )
                        model += (acc_x_s >= floor, f"S1ScenBase_{k_s}_{p}")

                # Overstock prevention
                model += (all_x <= 1.5 * d, f"S1Over_{p}")

                # Concentration risk
                for s in self.suppliers:
                    if self.sp_avail.get((s, p), False):
                        model += (x[s, p] <= self.conc_max * d, f"S1Conc_{s}_{p}")

        # ── Phase-2A constraints ──────────────────────────────────────────
        for k, sc in enumerate(self.scenarios):
            depot      = depot_by_k[k]
            ops        = ops_by_k[k]
            acc_sups   = self._acc_sups(sc)
            nodes_proc = [depot] + acc_sups

            # No vehicles or no accessible suppliers → inventory = 0
            if not acc_sups or not ops:
                for p in self.products:
                    model += (inventory[k, p] == 0.0, f"InvZero_{k}_{p}")
                # Any accessible commitment becomes waste
                for s in acc_sups:
                    for p in self.products:
                        if self.sp_avail.get((s, p), False) \
                                and (k, s, p) in waste_vars:
                            model += (waste_vars[k, s, p] >= x[s, p],
                                      f"WasteLB_nv_{k}_{s}_{p}")
                continue

            # Flow conservation
            for v in ops:
                for i in nodes_proc:
                    in_f  = lpSum(arc_proc[k, j, i, v] for j in nodes_proc
                                  if j != i and (k, j, i, v) in arc_proc)
                    out_f = lpSum(arc_proc[k, i, j, v] for j in nodes_proc
                                  if j != i and (k, i, j, v) in arc_proc)
                    model += (in_f == out_f, f"pFlow_{k}_{i}_{v}")

            # Each vehicle departs depot at most once
            for v in ops:
                out_d = lpSum(arc_proc[k, depot, s, v] for s in acc_sups
                              if (k, depot, s, v) in arc_proc)
                model += (out_d <= use_proc[k, v], f"pDepart_{k}_{v}")

            # Vehicle weight capacity
            for v in ops:
                model += (
                    lpSum(self.prod_weight[p] * qty_pickup[k, s, p, v]
                          for s in acc_sups for p in self.products
                          if (k, s, p, v) in qty_pickup)
                    <= self._eff_cap(v, sc.severity_level), f"pCap_{k}_{v}"
                )

            # Pickup gate: only pick up from s if vehicle visits s
            for v in ops:
                for s in acc_sups:
                    visit_s = lpSum(arc_proc[k, i, s, v] for i in nodes_proc
                                    if i != s and (k, i, s, v) in arc_proc)
                    for p in self.products:
                        if (k, s, p, v) in qty_pickup:
                            model += (qty_pickup[k, s, p, v] <= _BIG_M_QTY * visit_s,
                                      f"pGate_{k}_{s}_{p}_{v}")

            # Pickup cannot exceed Stage-1 commitment
            for s in acc_sups:
                for p in self.products:
                    if not self.sp_avail.get((s, p), False):
                        continue
                    pv = [qty_pickup[k, s, p, v] for v in ops
                          if (k, s, p, v) in qty_pickup]
                    if pv:
                        model += (lpSum(pv) <= x[s, p], f"pCommit_{k}_{s}_{p}")

            # Waste lower bound: waste ≥ ordered − picked-up
            for s in acc_sups:
                for p in self.products:
                    if not self.sp_avail.get((s, p), False):
                        continue
                    pickup_sp = lpSum(qty_pickup[k, s, p, v] for v in ops
                                      if (k, s, p, v) in qty_pickup)
                    if (k, s, p) in waste_vars:
                        model += (waste_vars[k, s, p] >= x[s, p] - pickup_sp,
                                  f"WasteLB_{k}_{s}_{p}")

            # Inventory linking: inventory = sum of all physical pickups
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
                    for j in acc_sups:  # ─ FIX: Must only be suppliers, not depot!
                        if i == j or (k, i, j, v) not in arc_proc:
                            continue
                        if (k, i, v) not in T_proc or (k, j, v) not in T_proc:
                            continue
                        svc_i = self.sup_svc_h.get(i, 0.0)
                        t_ij  = self.dist.get((i, j), 0) / self._eff_speed(v, sc)
                        model += (
                            T_proc[k, j, v] >= T_proc[k, i, v] + svc_i + t_ij
                            - _BIG_M_TIME * (1 - arc_proc[k, i, j, v]),
                            f"pMTZ_{k}_{i}_{j}_{v}"
                        )

        # ── Phase-2B constraints ──────────────────────────────────────────
        for k, sc in enumerate(self.scenarios):
            depot      = depot_by_k[k]
            ops        = ops_by_k[k]
            nodes_dist = [depot] + self.stores

            # No vehicles: all demand is unmet
            if not ops:
                for r in self.stores:
                    for p in self.products:
                        d_rp = self.store_demand.get((r, p), 0) * getattr(sc, 'demand_reduction_factor', 1.0)
                        if d_rp > 0 and (k, r, p) in unmet:
                            model += (unmet[k, r, p] == d_rp,
                                      f"AllUnmet_{k}_{r}_{p}")
                continue

            # Flow conservation
            for v in ops:
                for i in nodes_dist:
                    in_f  = lpSum(arc_dist[k, j, i, v] for j in nodes_dist
                                  if j != i and (k, j, i, v) in arc_dist)
                    out_f = lpSum(arc_dist[k, i, j, v] for j in nodes_dist
                                  if j != i and (k, i, j, v) in arc_dist)
                    model += (in_f == out_f, f"dFlow_{k}_{i}_{v}")

            # Each vehicle departs depot at most once
            for v in ops:
                out_d = lpSum(arc_dist[k, depot, r, v] for r in self.stores
                              if (k, depot, r, v) in arc_dist)
                model += (out_d <= use_dist[k, v], f"dDepart_{k}_{v}")

            # Every store must be visited (unmet slack absorbs missing inventory)
            for r in self.stores:
                model += (
                    lpSum(arc_dist[k, i, r, v]
                          for i in nodes_dist if i != r
                          for v in ops if (k, i, r, v) in arc_dist) >= 1,
                    f"dVisit_{k}_{r}"
                )

            # Vehicle weight capacity
            for v in ops:
                model += (
                    lpSum(self.prod_weight[p] * qty_dist[k, r, p, v]
                          for r in self.stores for p in self.products
                          if (k, r, p, v) in qty_dist)
                    <= self._eff_cap(v, sc.severity_level), f"dCap_{k}_{v}"
                )

            # Delivery gate: only deliver to r if vehicle visits r
            for v in ops:
                for r in self.stores:
                    visit_r = lpSum(arc_dist[k, i, r, v] for i in nodes_dist
                                    if i != r and (k, i, r, v) in arc_dist)
                    for p in self.products:
                        if (k, r, p, v) in qty_dist:
                            model += (qty_dist[k, r, p, v] <= _BIG_M_QTY * visit_r,
                                      f"dGate_{k}_{r}_{p}_{v}")

            # Demand satisfaction: delivery + unmet ≥ store demand
            for r in self.stores:
                for p in self.products:
                    d_rp = self.store_demand.get((r, p), 0) * getattr(sc, 'demand_reduction_factor', 1.0)
                    if d_rp <= 0:
                        continue
                    del_vars = [qty_dist[k, r, p, v] for v in ops
                                if (k, r, p, v) in qty_dist]
                    um = unmet.get((k, r, p), 0)
                    if del_vars:
                        model += (lpSum(del_vars) + um >= d_rp,
                                  f"dDem_{k}_{r}_{p}")
                    else:
                        model += (um >= d_rp, f"dDemFB_{k}_{r}_{p}")

            # Supply cap: total deliveries ≤ physical inventory only
            # NO emergency term — this is the key constraint that forces Phase-2A.
            for p in self.products:
                dist_sum = lpSum(
                    qty_dist[k, r, p, v]
                    for r in self.stores for v in ops
                    if (k, r, p, v) in qty_dist
                )
                model += (dist_sum <= inventory[k, p], f"dSupCap_{k}_{p}")

            # MTZ subtour elimination (distribution tour)
            for v in ops:
                if (k, depot, v) in T_dist:
                    model += (T_dist[k, depot, v] == _T_DIST_DEP, f"dTdep_{k}_{v}")
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
                            T_dist[k, j, v] >= T_dist[k, i, v] + svc_i + t_ij
                            - _BIG_M_TIME * (1 - arc_dist[k, i, j, v]),
                            f"dMTZ_{k}_{i}_{j}_{v}"
                        )

        print(f"  ✓ Variables:   {model.numVariables()}")
        print(f"  ✓ Constraints: {model.numConstraints()}")

        vd = dict(
            x=x, y=y,
            arc_proc=arc_proc, qty_pickup=qty_pickup, inventory=inventory,
            T_proc=T_proc,     use_proc=use_proc,
            arc_dist=arc_dist, qty_dist=qty_dist,     unmet=unmet,
            T_dist=T_dist,     use_dist=use_dist,
            waste_vars=waste_vars,
            depot_by_k=depot_by_k, ops_by_k=ops_by_k,
        )
        return model, vd

    # ── Solver ────────────────────────────────────────────────────────────────
    def solve(
        self,
        time_limit:    int   = 1800,
        gap_tolerance: float = 0.05,
        fixed_stage1:  Optional[pd.DataFrame] = None,
    ) -> Tuple[str, Dict]:

        model, vd = self.build_model(fixed_stage1=fixed_stage1)

        import tempfile, os
        lp_path = os.path.join(tempfile.gettempdir(), "two_phase_v2.lp")
        model.writeLP(lp_path)

        try:
            import gurobipy as gp
            from gurobipy import GRB

            env = gp.Env()
            env.setParam("OutputFlag", 1)
            env.setParam("TimeLimit",  time_limit)
            env.setParam("MIPGap",     gap_tolerance)
            env.setParam("LogFile",    "")
            grb = gp.read(lp_path, env)
            print(f"  Gurobi {'.'.join(str(v) for v in gp.gurobi.version())} — solving…")

            t0 = time.time()
            grb.optimize()
            elapsed = time.time() - t0

            FEASIBLE = {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED,
                        GRB.SUBOPTIMAL, 13}
            if grb.status in FEASIBLE and grb.SolCount > 0:
                obj_val = grb.ObjVal
                gap_pct = (abs(obj_val - grb.ObjBound)
                           / max(abs(obj_val), 1e-10) * 100)
                status  = "Optimal" if grb.status == GRB.OPTIMAL else "Feasible"
                print(f"  Status: {status}  obj={obj_val:,.0f}  "
                      f"gap={gap_pct:.2f}%  ({elapsed:.1f}s)")
                var_map = {v.VarName: v for v in grb.getVars()}
                for lv in model.variables():
                    gv = var_map.get(lv.name)
                    lv.varValue = gv.X if gv is not None else 0.0
            else:
                status = ("Infeasible" if grb.status == GRB.INFEASIBLE
                          else "Not Solved")
                print(f"  Status: {status} (code={grb.status}, "
                      f"sols={grb.SolCount})")
                return status, {}

        except ImportError:
            print("  gurobipy not available — using CBC")
            solver = PULP_CBC_CMD(msg=True, timeLimit=time_limit,
                                  gapRel=gap_tolerance)
            t0 = time.time()
            model.solve(solver)
            elapsed = time.time() - t0
            status  = LpStatus[model.status]
            obj_val = value(model.objective)
            if status == "Not Solved" and obj_val is not None and obj_val > 0:
                status = "Feasible"
            print(f"  Status: {status}  ({elapsed:.1f}s)")
            if status not in ("Optimal", "Feasible"):
                return status, {}

        solution = self._extract(vd)
        solution.update(dict(
            objective_value = obj_val,
            solve_time      = elapsed,
            status          = status,
            scenario_costs  = self._scenario_costs(vd),
        ))
        return status, solution

    # ── Solution extraction ───────────────────────────────────────────────────
    def _extract(self, vd: Dict) -> Dict:
        x = vd["x"]

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

        scenario_routes: Dict = {}
        for k, sc in enumerate(self.scenarios):
            depot    = vd["depot_by_k"][k]
            ops      = vd["ops_by_k"][k]
            acc_sups = self._acc_sups(sc)

            proc_routes, dist_routes = [], []

            for v in ops:
                if (value(vd["use_proc"].get((k, v))) or 0) < 0.5:
                    continue
                path, cur, seen = [depot], depot, {depot}
                for _ in range(len(acc_sups) + 2):
                    nxt = next(
                        (j for j in acc_sups
                         if j not in seen
                         and (value(vd["arc_proc"].get((k, cur, j, v))) or 0) > 0.5),
                        None)
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
                            s: {p: round(value(vd["qty_pickup"][(k, s, p, v)]) or 0, 2)
                                for p in self.products
                                if (k, s, p, v) in vd["qty_pickup"]
                                and (value(vd["qty_pickup"][(k, s, p, v)]) or 0) > 0.01}
                            for s in path[1:-1]
                        },
                    ))

            for v in ops:
                if (value(vd["use_dist"].get((k, v))) or 0) < 0.5:
                    continue
                path, cur, seen = [depot], depot, {depot}
                for _ in range(len(self.stores) + 2):
                    nxt = next(
                        (j for j in self.stores
                         if j not in seen
                         and (value(vd["arc_dist"].get((k, cur, j, v))) or 0) > 0.5),
                        None)
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
                            r: {p: round(value(vd["qty_dist"][(k, r, p, v)]) or 0, 2)
                                for p in self.products
                                if (k, r, p, v) in vd["qty_dist"]
                                and (value(vd["qty_dist"][(k, r, p, v)]) or 0) > 0.01}
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

    def _scenario_costs(self, vd: Dict) -> pd.DataFrame:
        x = vd["x"]
        s1_var = sum(
            (value(x[s, p]) or 0) * self.sp_cost.get((s, p), self.prod_cost[p])
            for s in self.suppliers for p in self.products
            if self.sp_avail.get((s, p), False)
        )
        s1_fix = sum(
            (value(vd["y"][s, p]) or 0) * self.sup_fix[s]
            for s in self.suppliers for p in self.products
            if self.sp_avail.get((s, p), False)
        )
        s1_total = s1_var + s1_fix

        rows = []
        for k, sc in enumerate(self.scenarios):
            depot      = vd["depot_by_k"][k]
            ops        = vd["ops_by_k"][k]
            acc_sups   = self._acc_sups(sc)
            nodes_proc = [depot] + acc_sups
            nodes_dist = [depot] + self.stores

            proc_vrp = sum(
                ((self.fleet[v]["cost_per_km"]
                  + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                 * self.dist.get((i, j), 0)
                 + (self.fleet[v].get("loading_cost_per_stop", 25_000) if j != depot else 0))
                * (value(vd["arc_proc"].get((k, i, j, v))) or 0)
                for v in ops
                for i in nodes_proc for j in nodes_proc if i != j
                if (k, i, j, v) in vd["arc_proc"]
            )
            proc_fix = sum(
                (0 if v < min(4, len(self.fleet)) else self.fleet[v]["fixed_cost_vnd"])
                * (1 if (value(vd["use_proc"].get((k, v))) or 0) > 0.5 else 0)
                for v in ops
            )
            dist_vrp = sum(
                ((self.fleet[v]["cost_per_km"]
                  + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                 * self.dist.get((i, j), 0)
                 + (self.fleet[v].get("loading_cost_per_stop", 25_000) if j != depot else 0))
                * (value(vd["arc_dist"].get((k, i, j, v))) or 0)
                for v in ops
                for i in nodes_dist for j in nodes_dist if i != j
                if (k, i, j, v) in vd["arc_dist"]
            )
            dist_fix = sum(
                (0 if v < min(4, len(self.fleet)) else self.fleet[v]["fixed_cost_vnd"])
                * (1 if (value(vd["use_dist"].get((k, v))) or 0) > 0.5 else 0)
                for v in ops
            )
            spoilage = sum(
                compute_spoilage_rate(
                    category=self.prod_category.get(p, "vegetable"),
                    transport_time_h=24.0,
                    is_refrigerated=False,
                    ambient_temp_c=getattr(sc, "temperature_celsius", 28.0)
                ) * getattr(sc, "spoilage_multiplier", 1.0)
                * (value(x[s, p]) or 0)
                * self.sp_cost.get((s, p), self.prod_cost[p])
                for p in self.products
                for s in self._inacc_sups(sc, p)
            )
            # Thêm refrigeration soft penalty vào mục spoilage cost return report
            refrig_pen_cost = sum(
                compute_spoilage_rate(
                    category=self.prod_category.get(p, "vegetable"),
                    transport_time_h=self.dist.get((depot, r), 0) / max(0.1, self._eff_speed(v, sc)),
                    is_refrigerated=False,
                    ambient_temp_c=getattr(sc, "temperature_celsius", 28.0)
                ) * self.prod_cost[p]
                  * (self.store_demand.get((r, p), 0) * getattr(sc, 'demand_reduction_factor', 1.0))
                  * (value(vd["arc_dist"][(k, depot, r, v)]) or 0)
                for v in ops
                for r in self.stores for p in self.products
                if self.prod_refrig.get(p, False) and not self.fleet[v].get("refrigerated", False)
                if (k, depot, r, v) in vd["arc_dist"]
            )
            spoilage += refrig_pen_cost
            unm_c = sum(
                self.prod_penalty_mult[p] * (value(vd["unmet"].get((k, r, p))) or 0) * self.prod_cost[p]
                for r in self.stores for p in self.products
                if (k, r, p) in vd["unmet"]
            )
            total = (s1_total + proc_vrp + proc_fix + dist_vrp + dist_fix
                     + spoilage + unm_c)

            rows.append(dict(
                scenario_name       = sc.name,
                severity_level      = sc.severity_level,
                probability         = sc.probability,
                stage1_cost         = s1_total,
                proc_vrp_cost       = proc_vrp + proc_fix,
                dist_vrp_cost       = dist_vrp + dist_fix,
                vrp_cost            = proc_vrp + proc_fix + dist_vrp + dist_fix,
                spoilage_cost       = spoilage,
                emergency_cost      = 0.0,     # removed from model
                penalty_cost        = unm_c,
                total_cost          = total,
                n_operable_vehicles = len(ops),
            ))
        return pd.DataFrame(rows)