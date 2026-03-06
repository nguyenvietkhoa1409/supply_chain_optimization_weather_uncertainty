"""
Extensive Form Optimizer – True Two-Stage Stochastic MILP
UPDATED: Heterogeneous Fleet Support + BUG-Q1 Fix

Changes from previous version
──────────────────────────────
[FLEET-1]  Heterogeneous fleet (4 types, 7 instances) replaces 3 identical vehicles.
    Inspired by Patel et al. (2024) F_k·X_k + L_ijk·d_ij·O_k cost structure where
    each vehicle type k has distinct fixed cost F_k and transport cost L_ijk.

[FLEET-2]  Per-vehicle fixed deployment cost in Stage 2 objective.
    Mirrors "F_k·X_k" from Patel et al. procurement model.
    A vehicle incurs its fixed_cost_vnd whenever it departs the depot.

[FLEET-3]  Vehicles with effective_capacity < 50 kg under a given scenario
    are automatically excluded from routing in that scenario (large trucks in typhoon).

[FLEET-4]  Refrigerated vehicles reduce spoilage for carried goods (65% reduction).
    Models cold-chain benefit: maintains ~4°C, cutting Arrhenius degradation.

[FLEET-5]  Per-vehicle effective speed = base_speed / road_slowdown × type_weather_factor.
    Mini trucks retain more urban speed in rain; large trucks lose more.

[BUG-Q1]  Fixed sp_transit: switched from qty-based to arc-based computation.
    Root cause: qty[k,r,p,v] had no lower bound → solver set qty=0 to avoid
    sp_transit cost → sp_transit=0 → ref_truck spoilage_reduction benefit invisible
    → n_refrigerated_active=0 in all scenarios.
    Previous attempted fix (lower bound on qty) caused infeasibility when fleet
    capacity < total procurement.
    Correct fix: sp_transit = base_spoil × mult × (1-reduction) × Σ_p product_cost[p]
                 × store_demand[r,p] × (dist[depot→r] / speed[v]) × arc[k,depot,r,v]
    arc[k,depot,r,v] is binary and always ≥ 0 — no new constraints, no infeasibility.
    Solver now sees ref_truck spoilage saving on every store-visit arc it commits to.

Backward compatibility
──────────────────────
fleet_instances=None + vehicle_config=<old dict>  →  auto-converted via
fleet_config.legacy_vehicle_config_to_fleet().  All existing call sites unchanged.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pulp import (
    PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus,
    LpVariable, lpSum, value,
)

try:
    from fleet_config import legacy_vehicle_config_to_fleet
except ImportError:
    try:
        from src.fleet_config import legacy_vehicle_config_to_fleet
    except ImportError:
        legacy_vehicle_config_to_fleet = None

_MIN_OPERABLE_CAP_KG = 50.0


class ExtensiveFormOptimizer:
    """True two-stage stochastic extensive-form optimizer with heterogeneous fleet."""

    _DC_MAX_SEVERITY = {"hoakhanh": 3, "lienchieu": 5}

    def __init__(
        self,
        network:             Dict,
        products_df:         pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df:           pd.DataFrame,
        weather_scenarios:   List,
        vehicle_config:      Optional[Dict] = None,
        fleet_instances:     Optional[List] = None,
        risk_aversion:       float = 0.0,
        cvar_alpha:          float = 0.95,
        baseline_ratio:      float = 0.70,
        emergency_ratio:     float = 0.40,
    ):
        self.network             = network
        self.products_df         = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df           = demand_df
        self.scenarios           = weather_scenarios
        self.risk_aversion       = risk_aversion
        self.cvar_alpha          = cvar_alpha
        self.baseline_ratio      = baseline_ratio
        self.emergency_ratio     = emergency_ratio

        # ── Fleet resolution [FLEET-1] ─────────────────────────────────────
        if fleet_instances is not None:
            self.fleet = fleet_instances
        elif vehicle_config is not None and legacy_vehicle_config_to_fleet is not None:
            self.fleet = legacy_vehicle_config_to_fleet(vehicle_config)
        else:
            self.fleet = [
                {
                    "vehicle_id": f"std_{v}", "type_id": "standard",
                    "name": f"Standard #{v+1}",
                    "capacity_kg": 1000, "fixed_cost_vnd": 0,
                    "cost_per_km": 5000, "cost_per_hour": 50000,
                    "base_speed_kmh": 40, "refrigerated": False,
                    "spoilage_reduction": 0.0,
                    "weather_capacity_factor": {s: 1.0 for s in range(1, 6)},
                    "weather_speed_factor":    {s: 1.0 for s in range(1, 6)},
                }
                for v in range(3)
            ]

        self.suppliers = network["suppliers"]["id"].tolist()
        self.products  = products_df["id"].tolist()
        self.stores    = network["stores"]["id"].tolist()

        self._create_lookups()

        K = len(self.scenarios)
        V = len(self.fleet)
        R = len(self.stores)
        refrig = sum(1 for v in self.fleet if v["refrigerated"])

        print("ExtensiveFormOptimizer (Heterogeneous Fleet):")
        print(f"  K={K} scenarios, V={V} vehicles ({refrig} refrigerated), R={R} stores")
        print(f"  Types: {list(dict.fromkeys(v['type_id'] for v in self.fleet))}")
        print(f"  λ={risk_aversion}, α={cvar_alpha}")

    # ── lookups ────────────────────────────────────────────────────────────
    def _create_lookups(self):
        self.product_cost   = dict(zip(self.products_df["id"], self.products_df["unit_cost_vnd"]))
        self.product_weight = dict(zip(self.products_df["id"], self.products_df["weight_kg_per_unit"]))
        self.supplier_capacity   = dict(zip(self.network["suppliers"]["id"],
                                            self.network["suppliers"]["capacity_kg_per_day"]))
        self.supplier_fixed_cost = dict(zip(self.network["suppliers"]["id"],
                                            self.network["suppliers"]["fixed_cost_vnd"]))
        self.supplier_subtype = {
            r["id"]: r.get("subtype", "general")
            for _, r in self.network["suppliers"].iterrows()
        }
        self.sp_cost, self.sp_moq, self.sp_available = {}, {}, {}
        for _, r in self.supplier_product_df.iterrows():
            s, p = r["supplier_id"], r["product_id"]
            self.sp_cost[(s, p)]      = r["unit_cost_vnd"]
            self.sp_moq[(s, p)]       = r["moq_units"]
            self.sp_available[(s, p)] = r["available"]

        dm = self.network["distance_matrix"]
        dcs = self.network["dcs"]["id"].tolist()
        self.all_nodes = dcs + self.stores
        self.distance  = {}
        for i in self.all_nodes:
            for j in self.all_nodes:
                if i != j:
                    try:    self.distance[(i, j)] = float(dm.loc[i, j])
                    except Exception:
                        try: self.distance[(i, j)] = float(dm.loc[j, i])
                        except Exception: self.distance[(i, j)] = 10.0

        grp = (self.demand_df.groupby(["store_id", "product_id"])["demand_units"]
               .sum().reset_index())
        self.store_demand = {
            (r["store_id"], r["product_id"]): float(r["demand_units"])
            for _, r in grp.iterrows()
        }
        self.total_demand = (
            self.demand_df.groupby("product_id")["demand_units"].sum().to_dict()
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _get_depot(self, scenario) -> str:
        sev = scenario.severity_level
        for _, dc in self.network["dcs"].iterrows():
            key     = dc["name"].lower().replace(" ", "").replace("_", "")
            max_sev = next((v for k, v in self._DC_MAX_SEVERITY.items() if k in key), 5)
            if sev <= max_sev:
                return dc["id"]
        return self.network["dcs"]["id"].iloc[-1]

    def _accessible_suppliers(self, sc, p):
        return [s for s in self.suppliers
                if self.sp_available.get((s, p), False)
                and sc.get_supplier_accessible(self.supplier_subtype.get(s, "general")) == 1]

    def _inaccessible_suppliers(self, sc, p):
        return [s for s in self.suppliers
                if self.sp_available.get((s, p), False)
                and sc.get_supplier_accessible(self.supplier_subtype.get(s, "general")) == 0]

    def _eff_cap(self, v_idx: int, severity: int) -> float:
        veh = self.fleet[v_idx]
        return veh["capacity_kg"] * veh["weather_capacity_factor"].get(severity, 1.0)

    def _eff_speed(self, v_idx: int, sc) -> float:
        veh  = self.fleet[v_idx]
        road = 1.0 / max(sc.speed_reduction_factor, 0.1)
        vtf  = veh["weather_speed_factor"].get(sc.severity_level, 1.0)
        return max(veh["base_speed_kmh"] * road * vtf, 1.0)

    # ── build model ────────────────────────────────────────────────────────
    def build_model(self) -> Tuple[LpProblem, Dict]:
        print("\nBuilding extensive-form MILP (heterogeneous fleet)…")
        K     = len(self.scenarios)
        M_big = 24.0
        M_qty = 100_000
        M_s1  = 100_000

        model = LpProblem("ExtensiveForm_Fleet", LpMinimize)

        # Stage 1
        x = LpVariable.dicts("x",
            ((s, p) for s in self.suppliers for p in self.products), lowBound=0)
        y = LpVariable.dicts("y",
            ((s, p) for s in self.suppliers for p in self.products), cat="Binary")

        # Stage 2 recourse
        e = LpVariable.dicts("e",
            ((k, p) for k in range(K) for p in self.products), lowBound=0)
        u = LpVariable.dicts("u",
            ((k, p) for k in range(K) for p in self.products), lowBound=0)

        # VRP variables (only for operable vehicles per scenario)
        arc, qty, T_mtz       = {}, {}, {}
        depot_by_k            = {}
        operable_by_k: Dict   = {}

        for k, sc in enumerate(self.scenarios):
            depot = self._get_depot(sc)
            depot_by_k[k] = depot
            nodes = [depot] + self.stores
            V = len(self.fleet)
            ops = [v for v in range(V)
                   if self._eff_cap(v, sc.severity_level) >= _MIN_OPERABLE_CAP_KG]
            operable_by_k[k] = ops

            for v in ops:
                for i in nodes:
                    for j in nodes:
                        if i != j:
                            arc[k, i, j, v] = LpVariable(
                                f"arc_{k}_{i}_{j}_{v}", cat="Binary")
                for r in self.stores:
                    for p in self.products:
                        qty[k, r, p, v] = LpVariable(f"qty_{k}_{r}_{p}_{v}", lowBound=0)
                for j in nodes:
                    T_mtz[k, j, v] = LpVariable(f"T_{k}_{j}_{v}",
                                                  lowBound=0, upBound=M_big)

        # CVaR
        eta, zeta = None, None
        if self.risk_aversion > 0 and K > 1:
            eta  = LpVariable("eta", lowBound=0)
            zeta = LpVariable.dicts("zeta", range(K), lowBound=0)

        # ── Objective ──────────────────────────────────────────────────────
        s1_var = lpSum(
            self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
            for s in self.suppliers for p in self.products
            if self.sp_available.get((s, p), False)
        )
        s1_fix = lpSum(
            self.supplier_fixed_cost[s] * y[s, p]
            for s in self.suppliers for p in self.products
            if self.sp_available.get((s, p), False)
        )

        s2_terms = []
        base_spoil = 0.04  # fraction of product cost spoiled per hour in transit

        for k, sc in enumerate(self.scenarios):
            prob  = sc.probability
            depot = depot_by_k[k]
            nodes = [depot] + self.stores
            ops   = operable_by_k[k]

            # [FLEET-2] Fixed vehicle deployment: F_v * departs[k,v]
            fixed_v = lpSum(
                self.fleet[v]["fixed_cost_vnd"]
                * lpSum(arc[k, depot, j, v] for j in self.stores
                        if (k, depot, j, v) in arc)
                for v in ops
            )

            # Variable transport cost L_v * d_ij * O_k  (Patel notation)
            vrp_var = lpSum(
                self.distance.get((i, j), 0)
                * (self.fleet[v]["cost_per_km"]
                   + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                * arc[k, i, j, v]
                for v in ops
                for i in nodes for j in nodes if i != j
                if (k, i, j, v) in arc
            )

            em_cost   = lpSum(2.0 * self.product_cost[p] * e[k, p] for p in self.products)
            pm        = min(10.0, 5.0 * sc.spoilage_multiplier)
            unmet_c   = lpSum(pm * self.product_cost[p] * u[k, p] for p in self.products)

            # Spoilage from inaccessible suppliers (Stage 1 opportunity cost)
            sp_s1 = lpSum(
                self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
                for p in self.products
                for s in self._inaccessible_suppliers(sc, p)
            )

            # [FLEET-4 / BUG-Q1 FIX] In-transit spoilage via arc (routing) variables
            # Root cause of previous bug: qty[k,r,p,v] had no lower bound → solver set
            # qty=0 to avoid sp_transit cost → sp_transit=0 → ref_truck advantage invisible.
            #
            # Fix: compute spoilage from arc[k,depot,r,v] (binary dispatch variable) ×
            # store_demand[r,p] (parameter). If vehicle v travels depot→store r, it is
            # carrying store r's demand, so spoilage = base_spoil × mult × (1-reduction)
            # × product_cost × transit_time × store_demand.
            # This is linear (arc binary × constant) and ref_trucks always show
            # spoilage_reduction=0.65 benefit whenever they are dispatched.
            sp_transit = lpSum(
                base_spoil
                * sc.spoilage_multiplier
                * (1.0 - self.fleet[v]["spoilage_reduction"])
                * self.product_cost[p]
                * (self.distance.get((depot, r), 0) / self._eff_speed(v, sc))
                * self.store_demand.get((r, p), 0)
                * arc[k, depot, r, v]
                for v in ops
                for r in self.stores for p in self.products
                if (k, depot, r, v) in arc
            )

            total_k = fixed_v + vrp_var + em_cost + unmet_c + sp_s1 + sp_transit
            s2_terms.append(prob * total_k)

        expected = s1_var + s1_fix + lpSum(s2_terms)

        if self.risk_aversion > 0 and eta is not None:
            cvar = eta + (1.0 / (1.0 - self.cvar_alpha)) * lpSum(
                self.scenarios[k].probability * zeta[k] for k in range(K)
            )
            model += (1 - self.risk_aversion) * expected + self.risk_aversion * cvar, "Obj"
        else:
            model += expected, "Obj"

        # ── Stage 1 Constraints ────────────────────────────────────────────
        for s in self.suppliers:
            model += (
                lpSum(x[s, p] * self.product_weight[p]
                      for p in self.products if self.sp_available.get((s, p), False))
                <= self.supplier_capacity[s], f"S1Cap_{s}"
            )
        for s in self.suppliers:
            for p in self.products:
                if self.sp_available.get((s, p), False):
                    moq = self.sp_moq.get((s, p), 0)
                    model += (x[s, p] >= moq * y[s, p], f"S1MOQlo_{s}_{p}")
                    model += (x[s, p] <= M_s1 * y[s, p], f"S1MOQhi_{s}_{p}")
        for p in self.products:
            d = self.total_demand.get(p, 0)
            if d > 0:
                all_x = lpSum(x[s, p] for s in self.suppliers
                              if self.sp_available.get((s, p), False))
                model += (all_x >= self.baseline_ratio * d, f"S1Base_{p}")
                model += (all_x <= 1.5 * d, f"S1Over_{p}")

        # ── Stage 2 Constraints ────────────────────────────────────────────
        for k, sc in enumerate(self.scenarios):
            depot = depot_by_k[k]
            nodes = [depot] + self.stores
            ops   = operable_by_k[k]

            for p in self.products:
                d      = self.total_demand.get(p, 0)
                em_cap = self.emergency_ratio * d * (1 if sc.emergency_feasible else 0)
                model += (e[k, p] <= em_cap, f"S2EmCap_{k}_{p}")

            for p in self.products:
                d = self.total_demand.get(p, 0)
                if d > 0:
                    acc = lpSum(x[s, p] for s in self._accessible_suppliers(sc, p))
                    model += (acc + e[k, p] + u[k, p] >= d, f"S2Dem_{k}_{p}")

            if not ops:
                continue  # all vehicles grounded; demand covered by emergency/unmet

            # Flow conservation
            for v in ops:
                for i in nodes:
                    in_f  = lpSum(arc[k, j, i, v] for j in nodes
                                  if j != i and (k, j, i, v) in arc)
                    out_f = lpSum(arc[k, i, j, v] for j in nodes
                                  if j != i and (k, i, j, v) in arc)
                    model += (in_f == out_f, f"VFlow_{k}_{i}_{v}")

            for v in ops:
                model += (
                    lpSum(arc[k, depot, j, v] for j in self.stores
                          if (k, depot, j, v) in arc) <= 1,
                    f"VDepart_{k}_{v}"
                )

            for r in self.stores:
                model += (
                    lpSum(arc[k, i, r, v]
                          for i in nodes for v in ops
                          if i != r and (k, i, r, v) in arc) >= 1,
                    f"VVisit_{k}_{r}"
                )

            # [FLEET-5] Per-vehicle weather-adjusted capacity
            for v in ops:
                eff_cap = self._eff_cap(v, sc.severity_level)
                model += (
                    lpSum(qty[k, r, p, v] * self.product_weight[p]
                          for r in self.stores for p in self.products
                          if (k, r, p, v) in qty)
                    <= eff_cap, f"VCap_{k}_{v}"
                )

            for r in self.stores:
                for p in self.products:
                    d_rp = self.store_demand.get((r, p), 0)
                    qty_vars = [qty[k, r, p, v] for v in ops if (k, r, p, v) in qty]
                    if d_rp > 0 and qty_vars:
                        model += (
                            lpSum(qty_vars) <= d_rp, f"VDelMax_{k}_{r}_{p}"
                        )
                    for v in ops:
                        if (k, r, p, v) not in qty: continue
                        vis = lpSum(arc[k, i, r, v] for i in nodes
                                    if i != r and (k, i, r, v) in arc)
                        model += (qty[k, r, p, v] <= M_qty * vis,
                                  f"VVisDel_{k}_{r}_{p}_{v}")

            # MTZ subtour elimination
            T_start = 5.0
            for v in ops:
                if (k, depot, v) in T_mtz:
                    model += (T_mtz[k, depot, v] == T_start, f"MTZdep_{k}_{v}")
            spd = {v: self._eff_speed(v, sc) for v in ops}
            for i in nodes:
                for j in self.stores:
                    if i == j: continue
                    t_ij = self.distance.get((i, j), 0)
                    for v in ops:
                        if (k, i, j, v) not in arc: continue
                        if (k, i, v) not in T_mtz or (k, j, v) not in T_mtz: continue
                        model += (
                            T_mtz[k, j, v] >= T_mtz[k, i, v] + t_ij / spd[v]
                            - M_big * (1 - arc[k, i, j, v]),
                            f"MTZ_{k}_{i}_{j}_{v}"
                        )

            if self.risk_aversion > 0 and zeta is not None:
                sc_approx = (
                    s1_var + s1_fix
                    + lpSum(self.fleet[v]["fixed_cost_vnd"]
                            * lpSum(arc[k, depot, j, v] for j in self.stores
                                    if (k, depot, j, v) in arc)
                            for v in ops)
                    + lpSum(
                        self.distance.get((i, j), 0)
                        * (self.fleet[v]["cost_per_km"]
                           + self.fleet[v]["cost_per_hour"] / spd.get(v, 40))
                        * arc[k, i, j, v]
                        for v in ops for i in nodes for j in nodes
                        if i != j and (k, i, j, v) in arc
                    )
                    + lpSum(2.0 * self.product_cost[p] * e[k, p] for p in self.products)
                    + lpSum(pm * self.product_cost[p] * u[k, p] for p in self.products)
                )
                model += (zeta[k] >= sc_approx - eta, f"CVaR_{k}")

        print(f"  ✓ Variables: {model.numVariables()} | Constraints: {model.numConstraints()}")
        return model, {
            "x": x, "y": y, "e": e, "u": u,
            "arc": arc, "qty": qty, "T_mtz": T_mtz,
            "depot_by_k": depot_by_k,
            "operable_by_k": operable_by_k,
            "eta": eta, "zeta": zeta,
        }

    # ── solve ──────────────────────────────────────────────────────────────
    def solve(self, time_limit: int = 1800,
              gap_tolerance: float = 0.05) -> Tuple[str, Dict]:
        model, vd = self.build_model()
        solver    = PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap_tolerance, msg=1)
        print(f"\nSolving (limit={time_limit}s, gap={gap_tolerance*100:.0f}%)…")
        t0 = time.time()
        model.solve(solver)
        elapsed = time.time() - t0
        status  = LpStatus[model.status]
        print(f"  Status: {status}  ({elapsed:.1f}s)")

        if status in ("Optimal", "Feasible"):
            obj = value(model.objective)
            print(f"  Objective: {obj:,.0f} VND")
            solution = self._extract_solution(vd)
            solution.update({
                "objective_value":  obj,
                "solve_time":       elapsed,
                "status":           status,
                "scenario_costs":   self._compute_scenario_costs(vd),
            })
            return status, solution
        return status, {}

    # ── extract solution ───────────────────────────────────────────────────
    def _extract_solution(self, vd: Dict) -> Dict:
        x, y, e, u = vd["x"], vd["y"], vd["e"], vd["u"]
        K = len(self.scenarios)

        stage1 = [
            {
                "supplier_id":    s,
                "product_id":     p,
                "quantity_units": round(value(x[s, p]) or 0, 2),
                "cost_vnd":       round((value(x[s, p]) or 0)
                                        * self.sp_cost.get((s, p), self.product_cost[p]), 0),
            }
            for s in self.suppliers for p in self.products
            if (value(x[s, p]) or 0) > 0.01
        ]

        scenario_routes = {}
        for k, sc in enumerate(self.scenarios):
            depot = vd["depot_by_k"][k]
            ops   = vd["operable_by_k"][k]
            routes = []
            for v in ops:
                stops, cur, vis = [], depot, {depot}
                while True:
                    nxt = next(
                        (j for j in self.stores
                         if j not in vis
                         and (value(vd["arc"].get((k, cur, j, v))) or 0) > 0.5),
                        None)
                    if nxt is None: break
                    deliveries = [
                        {"product_id": p, "quantity":
                         round(value(vd["qty"].get((k, nxt, p, v))) or 0, 2)}
                        for p in self.products
                        if (value(vd["qty"].get((k, nxt, p, v))) or 0) > 0.01
                    ]
                    if deliveries:
                        stops.append({"location": nxt, "deliveries": deliveries,
                                      "vehicle_type": self.fleet[v]["type_id"]})
                    vis.add(nxt); cur = nxt
                if stops:
                    routes.append({
                        "vehicle_id":    v,
                        "vehicle_type":  self.fleet[v]["type_id"],
                        "refrigerated":  self.fleet[v]["refrigerated"],
                        "stops":         stops,
                    })
            scenario_routes[sc.name] = routes

        return {
            "stage1_procurement": pd.DataFrame(stage1),
            "scenario_routes":    scenario_routes,
        }

    def _compute_scenario_costs(self, vd: Dict) -> pd.DataFrame:
        x, y, e, u = vd["x"], vd["y"], vd["e"], vd["u"]
        K = len(self.scenarios)
        base_spoil = 0.04

        s1_var = sum(
            (value(x[s, p]) or 0) * self.sp_cost.get((s, p), self.product_cost[p])
            for s in self.suppliers for p in self.products
            if self.sp_available.get((s, p), False)
        )
        s1_fix = sum(
            (value(y[s, p]) or 0) * self.supplier_fixed_cost[s]
            for s in self.suppliers for p in self.products
            if self.sp_available.get((s, p), False)
        )
        s1_total = s1_var + s1_fix

        rows = []
        for k, sc in enumerate(self.scenarios):
            depot = vd["depot_by_k"][k]
            nodes = [depot] + self.stores
            ops   = vd["operable_by_k"][k]

            # [FLEET-2] Fixed vehicle cost
            fix_v = sum(
                self.fleet[v]["fixed_cost_vnd"]
                * (1 if any(
                    (value(vd["arc"].get((k, depot, j, v))) or 0) > 0.5
                    for j in self.stores
                ) else 0)
                for v in ops
            )
            # Variable VRP cost
            var_v = sum(
                self.distance.get((i, j), 0)
                * (self.fleet[v]["cost_per_km"]
                   + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                * (value(vd["arc"].get((k, i, j, v))) or 0)
                for v in ops for i in nodes for j in nodes
                if i != j and (k, i, j, v) in vd["arc"]
            )
            em_c  = sum(2.0 * (value(e[k, p]) or 0) * self.product_cost[p]
                        for p in self.products)
            pm    = min(10.0, 5.0 * sc.spoilage_multiplier)
            unm_c = sum(pm * (value(u[k, p]) or 0) * self.product_cost[p]
                        for p in self.products)
            sp_s1 = sum(
                (value(x[s, p]) or 0) * self.sp_cost.get((s, p), self.product_cost[p])
                for p in self.products for s in self._inaccessible_suppliers(sc, p)
            )
            sp_tr = sum(
                base_spoil * sc.spoilage_multiplier
                * (1.0 - self.fleet[v]["spoilage_reduction"])
                * self.product_cost[p]
                * (self.distance.get((depot, r), 0) / self._eff_speed(v, sc))
                * self.store_demand.get((r, p), 0)
                * ((value(vd["arc"].get((k, depot, r, v))) or 0))
                for v in ops for r in self.stores for p in self.products
                if (k, depot, r, v) in vd["arc"]
            )

            rows.append({
                "scenario_name":        sc.name,
                "severity_level":       sc.severity_level,
                "probability":          sc.probability,
                "stage1_cost":          s1_total,
                "vrp_fixed_cost":       fix_v,
                "vrp_variable_cost":    var_v,
                "vrp_cost":             fix_v + var_v,
                "emergency_cost":       em_c,
                "spoilage_cost":        sp_s1 + sp_tr,
                "penalty_cost":         unm_c,
                "total_cost":           s1_total + fix_v + var_v + em_c
                                        + sp_s1 + sp_tr + unm_c,
                "n_operable_vehicles":    len(ops),
                "n_refrigerated_active":  sum(1 for v in ops if self.fleet[v]["refrigerated"]
                                              and any(
                                                  (value(vd["arc"].get((k, depot, j, v))) or 0) > 0.5
                                                  for j in self.stores
                                              )),
            })
        return pd.DataFrame(rows)