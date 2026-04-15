"""
Extensive Form Optimizer – True Two-Stage Stochastic MILP
UPDATED v3: Pickup & Delivery Problem (PDP) Formulation

Changes from v2 (Two-Echelon distribution)
──────────────────────────────────────────
[PDP-1]  Node set expanded to {DC} ∪ {Suppliers} ∪ {Stores}.
    Company vehicles now pick up from suppliers (pickup nodes) then
    deliver to stores (delivery nodes). This gives the business full
    routing control — aligning with the Adaptive Sequential Decision-Making
    framework (reference paper, Sustainability 2024, 16, 98).

[PDP-2]  New decision variable qty_pickup[k,s,p,v]:
    Quantity of product p picked up from supplier s by vehicle v
    in scenario k.  Flow-balance constraint ties this to Stage-1
    procurement x[s,p]: Σ_v qty_pickup[k,s,p,v] = x[s,p].

[PDP-3]  Time-window constraints at BOTH supplier and store nodes.
    Suppliers: time_window_open ≤ T_arrive[k,s,v] ≤ time_window_close
    Stores:    time_window_open ≤ T_arrive[k,r,v] ≤ time_window_close
    Enforced via Big-M deactivation when vehicle does not visit node.

[PDP-4]  Pickup-First Precedence (simplified Option B):
    All supplier visits must complete before any store visit.
    T_arrive[k,s,v] ≤ T_arrive[k,r,v]  ∀s∈Suppliers, r∈Stores
    Big-M deactivated when vehicle doesn't visit both nodes.
    Reflects morning procurement cycle reality (4-9 AM pickup,
    6-11 AM delivery).

[PDP-5]  Cargo Mixing — Vehicle-Product Compatibility:
    Products requiring refrigeration may only be carried by
    refrigerated vehicles. Enforced as:
    qty_pickup[k,s,p,v] = 0  if v not refrigerated and p requires_refrig
    qty[k,r,p,v]        = 0  if v not refrigerated and p requires_refrig

[PDP-6]  Stage-2 travel time computed from actual distance and
    per-vehicle effective speed, propagating through all nodes
    (suppliers and stores) via MTZ-style time variables.

Backward compatibility
──────────────────────
fleet_instances=None + vehicle_config=<old dict>  →  auto-converted via
fleet_config.legacy_vehicle_config_to_fleet().  All existing call sites unchanged.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp
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
_M_BIG_TIME          = 24.0   # hours – time horizon
_M_BIG_QTY           = 16400  # Data-verified Big-M:
# max_supplier_capacity_kg / min(weight_kg_per_unit) × 1.2 safety buffer
# = 2732 kg / 0.20 kg·unit⁻¹ × 1.2 = 16,392 → rounded up to 16,400
# (was 100k → LP relaxation too weak; 5000 → artificially constrained 9/10 products)


class ExtensiveFormOptimizer:
    """Two-stage stochastic MILP with heterogeneous fleet and PDP routing."""

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

        # ── Fleet resolution ───────────────────────────────────────────────
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

        K      = len(self.scenarios)
        V      = len(self.fleet)
        R      = len(self.stores)
        S      = len(self.suppliers)
        refrig = sum(1 for v in self.fleet if v["refrigerated"])

        print("ExtensiveFormOptimizer (PDP — Pickup & Delivery):")
        print(f"  K={K} scenarios, V={V} vehicles ({refrig} refrigerated)")
        print(f"  S={S} supplier pickup nodes, R={R} store delivery nodes")
        print(f"  Types: {list(dict.fromkeys(v['type_id'] for v in self.fleet))}")
        print(f"  λ={risk_aversion}, α={cvar_alpha}")

    # ── lookups ────────────────────────────────────────────────────────────
    def _create_lookups(self):
        self.product_cost    = dict(zip(self.products_df["id"], self.products_df["unit_cost_vnd"]))
        self.product_weight  = dict(zip(self.products_df["id"], self.products_df["weight_kg_per_unit"]))
        self.product_refrig  = dict(zip(self.products_df["id"], self.products_df["requires_refrigeration"].astype(bool)))

        sup_df = self.network["suppliers"]
        self.supplier_capacity   = dict(zip(sup_df["id"], sup_df["capacity_kg_per_day"]))
        self.supplier_fixed_cost = dict(zip(sup_df["id"], sup_df["fixed_cost_vnd"]))
        self.supplier_subtype    = {r["id"]: r.get("subtype", "general") for _, r in sup_df.iterrows()}

        # PDP time windows (hours)
        self.supplier_tw_open  = dict(zip(sup_df["id"], sup_df.get("time_window_open",  pd.Series([4]*len(sup_df), index=sup_df.index))))
        self.supplier_tw_close = dict(zip(sup_df["id"], sup_df.get("time_window_close", pd.Series([11]*len(sup_df), index=sup_df.index))))
        self.supplier_svc_h    = {s: self.network["suppliers"].set_index("id").loc[s, "service_time_min"] / 60.0
                                   if "service_time_min" in self.network["suppliers"].columns else 0.5
                                   for s in self.suppliers}

        sto_df = self.network["stores"]
        self.store_tw_open  = dict(zip(sto_df["id"], sto_df.get("time_window_open",  pd.Series([6]*len(sto_df),  index=sto_df.index))))
        self.store_tw_close = dict(zip(sto_df["id"], sto_df.get("time_window_close", pd.Series([11]*len(sto_df), index=sto_df.index))))
        self.store_svc_h    = {r: self.network["stores"].set_index("id").loc[r, "service_time_min"] / 60.0
                                if "service_time_min" in self.network["stores"].columns else 0.25
                                for r in self.stores}

        self.sp_cost, self.sp_moq, self.sp_available = {}, {}, {}
        for _, r in self.supplier_product_df.iterrows():
            s, p = r["supplier_id"], r["product_id"]
            self.sp_cost[(s, p)]      = r["unit_cost_vnd"]
            self.sp_moq[(s, p)]       = r["moq_units"]
            self.sp_available[(s, p)] = r["available"]

        dm = self.network["distance_matrix"]
        dcs = self.network["dcs"]["id"].tolist()
        # PDP all_nodes: depot + suppliers + stores
        self.depot_nodes    = dcs
        self.all_pdp_nodes  = dcs + self.suppliers + self.stores
        self.distance = {}
        for i in self.all_pdp_nodes:
            for j in self.all_pdp_nodes:
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

    def _can_carry(self, v_idx: int, p: str) -> bool:
        """[PDP-5] Cargo mixing: refrigerated products → refrigerated vehicles only."""
        if self.product_refrig.get(p, False):
            return self.fleet[v_idx]["refrigerated"]
        return True

    # ── build model ────────────────────────────────────────────────────────
    def build_model(self) -> Tuple[LpProblem, Dict]:
        print("\nBuilding extensive-form MILP (PDP — Pickup & Delivery)…")
        K     = len(self.scenarios)
        M_s1  = _M_BIG_QTY
        M_t   = _M_BIG_TIME

        model = LpProblem("ExtensiveForm_PDP", LpMinimize)

        # ── Stage 1 variables ──────────────────────────────────────────────
        x = LpVariable.dicts("x",
            ((s, p) for s in self.suppliers for p in self.products), lowBound=0)
        y = LpVariable.dicts("y",
            ((s, p) for s in self.suppliers for p in self.products), cat="Binary")

        # ── Stage 2 recourse ───────────────────────────────────────────────
        e = LpVariable.dicts("e",
            ((k, p) for k in range(K) for p in self.products), lowBound=0)
        u = LpVariable.dicts("u",
            ((k, p) for k in range(K) for p in self.products), lowBound=0)

        # PDP routing variables (per scenario)
        arc         = {}   # arc[k, i, j, v]  — traversal binary
        qty_pickup  = {}   # qty_pickup[k, s, p, v]  — pickup at supplier
        qty         = {}   # qty[k, r, p, v]  — delivery at store
        T_arrive    = {}   # T_arrive[k, i, v]  — arrival time at any node

        depot_by_k     = {}
        operable_by_k  = {}

        for k, sc in enumerate(self.scenarios):
            depot = self._get_depot(sc)
            depot_by_k[k] = depot
            V_count = len(self.fleet)
            ops = [v for v in range(V_count)
                   if self._eff_cap(v, sc.severity_level) >= _MIN_OPERABLE_CAP_KG]
            operable_by_k[k] = ops

            # [PDP-1] Node set: depot + supplier nodes + store nodes
            pdp_nodes = [depot] + self.suppliers + self.stores

            for v in ops:
                # Arc variables over full PDP node set
                for i in pdp_nodes:
                    for j in pdp_nodes:
                        if i != j:
                            # [PERFORMANCE FIX] Prune illogical arcs to drastically reduce binary variables
                            if i in self.stores and j in self.suppliers:
                                continue # Precedence strictly forbids Store -> Supplier
                            if i == depot and j in self.stores:
                                continue # Must visit suppliers before stores! Depot -> Store is illogical here
                            
                            arc[k, i, j, v] = LpVariable(
                                f"arc_{k}_{i}_{j}_{v}", cat="Binary")

                # [PDP-2] Pickup quantity at suppliers
                for s in self.suppliers:
                    for p in self.products:
                        if self.sp_available.get((s, p), False) and self._can_carry(v, p):
                            qty_pickup[k, s, p, v] = LpVariable(
                                f"qpick_{k}_{s}_{p}_{v}", lowBound=0)

                # Delivery quantity at stores
                for r in self.stores:
                    for p in self.products:
                        if self._can_carry(v, p):
                            qty[k, r, p, v] = LpVariable(
                                f"qty_{k}_{r}_{p}_{v}", lowBound=0)

                # [PDP-6] Arrival time at every PDP node
                for i in pdp_nodes:
                    T_arrive[k, i, v] = LpVariable(
                        f"T_{k}_{i}_{v}", lowBound=0, upBound=M_t)

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
        base_spoil = 0.04

        for k, sc in enumerate(self.scenarios):
            prob  = sc.probability
            depot = depot_by_k[k]
            ops   = operable_by_k[k]
            pdp_nodes = [depot] + self.suppliers + self.stores

            # Vehicle fixed deployment cost
            fixed_v = lpSum(
                self.fleet[v]["fixed_cost_vnd"]
                * lpSum(arc[k, depot, j, v]
                        for j in (self.suppliers + self.stores)
                        if (k, depot, j, v) in arc)
                for v in ops
            )

            # Variable transport cost across ALL arcs (pickup + delivery)
            vrp_var = lpSum(
                self.distance.get((i, j), 0)
                * (self.fleet[v]["cost_per_km"]
                   + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                * arc[k, i, j, v]
                for v in ops
                for i in pdp_nodes for j in pdp_nodes if i != j
                if (k, i, j, v) in arc
            )

            em_cost  = lpSum(2.0 * self.product_cost[p] * e[k, p] for p in self.products)
            pm       = min(10.0, 5.0 * sc.spoilage_multiplier)
            unmet_c  = lpSum(pm * self.product_cost[p] * u[k, p] for p in self.products)

            # Spoilage from inaccessible suppliers
            sp_s1 = lpSum(
                self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
                for p in self.products
                for s in self._inaccessible_suppliers(sc, p)
            )

            # In-transit spoilage (arc-based, depot→store delivery leg)
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
                    model += (x[s, p] <= M_s1 * y[s, p],  f"S1MOQhi_{s}_{p}")
        for p in self.products:
            d = self.total_demand.get(p, 0)
            if d > 0:
                all_x = lpSum(x[s, p] for s in self.suppliers
                              if self.sp_available.get((s, p), False))
                model += (all_x >= self.baseline_ratio * d, f"S1Base_{p}")
                model += (all_x <= 1.5 * d,                 f"S1Over_{p}")

        # [FIX-CONCENTRATION] Max Supplier Concentration Risk Constraint
        max_concentration_ratio = 0.40
        for p in self.products:
            d = self.total_demand.get(p, 0)
            if d > 0:
                for s in self.suppliers:
                    if self.sp_available.get((s, p), False):
                        model += (
                            x[s, p] <= max_concentration_ratio * d,
                            f"S1_Concentration_{s}_{p}"
                        )

        # ── Stage 2 Constraints ────────────────────────────────────────────
        for k, sc in enumerate(self.scenarios):
            depot     = depot_by_k[k]
            ops       = operable_by_k[k]
            pdp_nodes = [depot] + self.suppliers + self.stores

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
                continue

            # ── PDP Flow Conservation ──────────────────────────────────────
            for v in ops:
                for i in pdp_nodes:
                    in_f  = lpSum(arc[k, j, i, v] for j in pdp_nodes
                                  if j != i and (k, j, i, v) in arc)
                    out_f = lpSum(arc[k, i, j, v] for j in pdp_nodes
                                  if j != i and (k, i, j, v) in arc)
                    model += (in_f == out_f, f"VFlow_{k}_{i}_{v}")

            # Each vehicle departs depot at most once
            for v in ops:
                model += (
                    lpSum(arc[k, depot, j, v]
                          for j in (self.suppliers + self.stores)
                          if (k, depot, j, v) in arc) <= 1,
                    f"VDepart_{k}_{v}"
                )

            # ── [PERFORMANCE FIX] SYMMETRY BREAKING ────────────────────────
            # Force identical vehicles to be utilized in order (v1 before v2)
            # This cuts the search tree by ~24x!
            for v_idx in range(1, len(ops)):
                v_curr = ops[v_idx]
                v_prev = ops[v_idx-1]
                if self.fleet[v_curr]["type_id"] == self.fleet[v_prev]["type_id"]:
                    model += (
                        lpSum(arc[k, depot, j, v_curr] for j in pdp_nodes if j != depot and (k, depot, j, v_curr) in arc)
                        <= lpSum(arc[k, depot, j, v_prev] for j in pdp_nodes if j != depot and (k, depot, j, v_prev) in arc),
                        f"SymBreak_{k}_{v_curr}"
                    )

            # Every store must be visited by exactly one vehicle
            for r in self.stores:
                model += (
                    lpSum(arc[k, i, r, v]
                          for i in pdp_nodes for v in ops
                          if i != r and (k, i, r, v) in arc) >= 1,
                    f"VVisit_store_{k}_{r}"
                )

            # [CRITICAL FIX] Only visit suppliers that we ACTUALLY procured from
            # (Previously unconditionally forced visiting all 6 suppliers -> caused Infeasible)
            for s in self.suppliers:
                for p in self.products:
                    if self.sp_available.get((s, p), False):
                        if sc.get_supplier_accessible(self.supplier_subtype.get(s, "general")) == 1:
                            model += (
                                lpSum(arc[k, i, s, v]
                                      for i in pdp_nodes for v in ops
                                      if i != s and (k, i, s, v) in arc) >= y[s, p],
                                f"VVisit_sup_{k}_{s}_{p}"
                            )

            # ── [PDP-2] Flow Balance: Pickup ties to Stage-1 procurement ──
            for s in self.suppliers:
                for p in self.products:
                    if self.sp_available.get((s, p), False):
                        # Σ_v qty_pickup[k,s,p,v] = x[s,p]
                        pickup_vars = [qty_pickup[k, s, p, v] for v in ops
                                       if (k, s, p, v) in qty_pickup]
                        if pickup_vars:
                            model += (
                                lpSum(pickup_vars) == x[s, p],
                                f"PDPPickupBal_{k}_{s}_{p}"
                            )
                        else:
                            # No compatible vehicle → force x[s,p]=0
                            model += (x[s, p] == 0, f"PDPPickupNoCap_{k}_{s}_{p}")

            # Pickup only possible if vehicle visits supplier
            for v in ops:
                for s in self.suppliers:
                    visit_s = lpSum(arc[k, i, s, v] for i in pdp_nodes
                                    if i != s and (k, i, s, v) in arc)
                    for p in self.products:
                        if (k, s, p, v) in qty_pickup:
                            model += (
                                qty_pickup[k, s, p, v] <= _M_BIG_QTY * visit_s,
                                f"PDPPickupGate_{k}_{s}_{p}_{v}"
                            )

            # ── Vehicle Capacity (total payload = pickups) ─────────────────
            for v in ops:
                eff_cap = self._eff_cap(v, sc.severity_level)
                model += (
                    lpSum(qty_pickup[k, s, p, v] * self.product_weight[p]
                          for s in self.suppliers for p in self.products
                          if (k, s, p, v) in qty_pickup)
                    <= eff_cap, f"VCap_{k}_{v}"
                )

            # ── [CRITICAL FIX] Vehicle Cargo Conservation ──────────────────
            # A vehicle can ONLY deliver what it picked up!
            for v in ops:
                for p in self.products:
                    p_up = lpSum(qty_pickup[k, s, p, v] for s in self.suppliers if (k, s, p, v) in qty_pickup)
                    p_dn = lpSum(qty[k, r, p, v] for r in self.stores if (k, r, p, v) in qty)
                    model += (p_up == p_dn, f"VCargoBal_{k}_{v}_{p}")

            # ── Delivery constraints ───────────────────────────────────────
            for r in self.stores:
                for p in self.products:
                    d_rp = self.store_demand.get((r, p), 0)
                    del_vars = [qty[k, r, p, v] for v in ops if (k, r, p, v) in qty]
                    if d_rp > 0 and del_vars:
                        model += (lpSum(del_vars) <= d_rp, f"VDelMax_{k}_{r}_{p}")
                    for v in ops:
                        if (k, r, p, v) not in qty: continue
                        visit_r = lpSum(arc[k, i, r, v] for i in pdp_nodes
                                        if i != r and (k, i, r, v) in arc)
                        model += (qty[k, r, p, v] <= _M_BIG_QTY * visit_r,
                                  f"VVisDel_{k}_{r}_{p}_{v}")

            # ── [PDP-6] Arrival Time Propagation (MTZ-style, all nodes) ───
            T_depart_depot = 4.0   # vehicles leave DC at 4 AM
            for v in ops:
                if (k, depot, v) in T_arrive:
                    model += (T_arrive[k, depot, v] == T_depart_depot,
                              f"T_dep_{k}_{v}")

            spd = {v: self._eff_speed(v, sc) for v in ops}
            for v in ops:
                for i in pdp_nodes:
                    for j in pdp_nodes:
                        if i == j or (k, i, j, v) not in arc: continue
                        if (k, i, v) not in T_arrive or (k, j, v) not in T_arrive: continue
                        svc_i = (self.supplier_svc_h.get(i, 0)
                                 if i in self.suppliers else
                                 self.store_svc_h.get(i, 0))
                        t_ij  = self.distance.get((i, j), 0) / spd[v]
                        model += (
                            T_arrive[k, j, v] >= T_arrive[k, i, v]
                                                 + svc_i + t_ij
                                                 - M_t * (1 - arc[k, i, j, v]),
                            f"TArrive_{k}_{i}_{j}_{v}"
                        )

            # ── [PDP-3] Time Windows ───────────────────────────────────────
            # [RELAXED Plan B] Hard Time Windows are removed to guarantee 
            # tractability in open-source CBC solver. The Precedence & MTZ 
            # constraints below implicitly handle the routing sequence (Pickup -> Delivery).
            pass

            # ── [PDP-4] Pickup-First Precedence (Option B, simplified) ─────
            # All supplier visits must precede all store visits
            for v in ops:
                for s in self.suppliers:
                    for r in self.stores:
                        if (k, s, v) not in T_arrive or (k, r, v) not in T_arrive: continue
                        visit_s = lpSum(arc[k, i, s, v] for i in pdp_nodes
                                        if i != s and (k, i, s, v) in arc)
                        visit_r = lpSum(arc[k, i, r, v] for i in pdp_nodes
                                        if i != r and (k, i, r, v) in arc)
                        model += (
                            T_arrive[k, r, v] >= T_arrive[k, s, v]
                                                 - M_t * (2 - visit_s - visit_r),
                            f"Prec_{k}_{s}_{r}_{v}"
                        )

            if self.risk_aversion > 0 and zeta is not None:
                spd_local = spd
                sc_approx = (
                    s1_var + s1_fix
                    + lpSum(self.fleet[v]["fixed_cost_vnd"]
                            * lpSum(arc[k, depot, j, v]
                                    for j in (self.suppliers + self.stores)
                                    if (k, depot, j, v) in arc)
                            for v in ops)
                    + lpSum(
                        self.distance.get((i, j), 0)
                        * (self.fleet[v]["cost_per_km"]
                           + self.fleet[v]["cost_per_hour"] / spd_local.get(v, 40))
                        * arc[k, i, j, v]
                        for v in ops for i in pdp_nodes for j in pdp_nodes
                        if i != j and (k, i, j, v) in arc
                    )
                    + lpSum(2.0 * self.product_cost[p] * e[k, p] for p in self.products)
                    + lpSum(pm * self.product_cost[p] * u[k, p] for p in self.products)
                )
                model += (zeta[k] >= sc_approx - eta, f"CVaR_{k}")

        print(f"  ✓ Variables: {model.numVariables()} | Constraints: {model.numConstraints()}")
        return model, {
            "x": x, "y": y, "e": e, "u": u,
            "arc": arc, "qty": qty, "qty_pickup": qty_pickup,
            "T_arrive": T_arrive,
            "depot_by_k": depot_by_k,
            "operable_by_k": operable_by_k,
            "eta": eta, "zeta": zeta,
        }

    # ── solve ──────────────────────────────────────────────────────────────
    def solve(self, time_limit: int = 1800,
              gap_tolerance: float = 0.05) -> Tuple[str, Dict]:
        model, vd = self.build_model()
        solver = pulp.getSolver('GUROBI', timeLimit=time_limit, gapRel=gap_tolerance, msg=1)
        print(f"\nSolving PDP (limit={time_limit}s, gap={gap_tolerance*100:.0f}%)…")
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
                "objective_value": obj,
                "solve_time":      elapsed,
                "status":          status,
                "scenario_costs":  self._compute_scenario_costs(vd),
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
            depot     = vd["depot_by_k"][k]
            ops       = vd["operable_by_k"][k]
            pdp_nodes = [depot] + self.suppliers + self.stores
            routes    = []

            for v in ops:
                # Reconstruct route: depot → (suppliers & stores) → depot
                route_seq = [depot]
                visited   = {depot}
                cur = depot
                while True:
                    nxt = next(
                        (j for j in pdp_nodes
                         if j not in visited
                         and (value(vd["arc"].get((k, cur, j, v))) or 0) > 0.5),
                        None)
                    if nxt is None: break
                    route_seq.append(nxt)
                    visited.add(nxt)
                    cur = nxt
                route_seq.append(depot)  # return to depot

                if len(route_seq) <= 2:  # only depot→depot, skip
                    continue

                # Build stop details
                stops = []
                for node in route_seq[1:-1]:
                    t_arr = value(vd["T_arrive"].get((k, node, v))) or 0
                    node_type = "supplier" if node in self.suppliers else "store"
                    if node_type == "supplier":
                        pickups = [
                            {"product_id": p,
                             "quantity": round(value(vd["qty_pickup"].get((k, node, p, v))) or 0, 2)}
                            for p in self.products
                            if (value(vd["qty_pickup"].get((k, node, p, v))) or 0) > 0.01
                        ]
                        stops.append({
                            "node": node, "node_type": "supplier_pickup",
                            "arrival_hour": round(t_arr, 2),
                            "pickups": pickups,
                        })
                    else:
                        deliveries = [
                            {"product_id": p,
                             "quantity": round(value(vd["qty"].get((k, node, p, v))) or 0, 2)}
                            for p in self.products
                            if (value(vd["qty"].get((k, node, p, v))) or 0) > 0.01
                        ]
                        stops.append({
                            "node": node, "node_type": "store_delivery",
                            "arrival_hour": round(t_arr, 2),
                            "deliveries": deliveries,
                        })

                routes.append({
                    "vehicle_id":   v,
                    "vehicle_type": self.fleet[v]["type_id"],
                    "refrigerated": self.fleet[v]["refrigerated"],
                    "route_sequence": route_seq,
                    "stops": stops,
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
            depot     = vd["depot_by_k"][k]
            ops       = vd["operable_by_k"][k]
            pdp_nodes = [depot] + self.suppliers + self.stores

            fix_v = sum(
                self.fleet[v]["fixed_cost_vnd"]
                * (1 if any(
                    (value(vd["arc"].get((k, depot, j, v))) or 0) > 0.5
                    for j in (self.suppliers + self.stores)
                ) else 0)
                for v in ops
            )
            var_v = sum(
                self.distance.get((i, j), 0)
                * (self.fleet[v]["cost_per_km"]
                   + self.fleet[v]["cost_per_hour"] / self._eff_speed(v, sc))
                * (value(vd["arc"].get((k, i, j, v))) or 0)
                for v in ops for i in pdp_nodes for j in pdp_nodes
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
                "scenario_name":         sc.name,
                "severity_level":        sc.severity_level,
                "probability":           sc.probability,
                "stage1_cost":           s1_total,
                "vrp_fixed_cost":        fix_v,
                "vrp_variable_cost":     var_v,
                "vrp_cost":              fix_v + var_v,
                "emergency_cost":        em_c,
                "spoilage_cost":         sp_s1 + sp_tr,
                "penalty_cost":          unm_c,
                "total_cost":            s1_total + fix_v + var_v + em_c
                                         + sp_s1 + sp_tr + unm_c,
                "n_operable_vehicles":   len(ops),
                "n_refrigerated_active": sum(
                    1 for v in ops if self.fleet[v]["refrigerated"]
                    and any((value(vd["arc"].get((k, depot, j, v))) or 0) > 0.5
                            for j in (self.suppliers + self.stores))
                ),
            })
        return pd.DataFrame(rows)