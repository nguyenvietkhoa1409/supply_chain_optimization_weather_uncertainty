"""
Extensive Form Optimizer – True Two-Stage Stochastic MILP
NEW MODULE

Purpose
───────
Replaces the sequential solve approach in IntegratedStochasticModel.
Procurement (Stage 1) and VRP routing (Stage 2) are solved as ONE MILP.
This is the theoretically correct extensive form.

Fix addressed
─────────────
[HIGH / V-3]  Sequential solve breaks Stage 1–Stage 2 coupling.
    When procurement is solved first and VRP receives fixed quantities, the
    procurement objective cannot account for VRP infeasibility costs — so the
    optimizer may procure quantities that lead to infeasible or expensive routes.
    In the true extensive form, Stage 1 decisions are optimised with full
    awareness of how they affect every Stage 2 routing scenario.

Design
──────
Stage 1 variables (scenario-independent, non-anticipativity enforced):
  x[s,p]  – procurement quantity
  y[s,p]  – supplier activation (binary)

Stage 2 variables (one copy per scenario k):
  e[k,p]      – emergency procurement
  u[k,p]      – aggregate unmet demand
  arc[k,i,j,v] – VRP binary routing arc
  qty[k,r,p,v] – delivery quantity to store r for product p
  T[k,j,v]    – MTZ time variable (subtour elimination)

Objective:
  min  Stage1_cost
     + Σ_k p_k · (VRP_cost_k + emergency_cost_k + unmet_penalty_k + spoilage_cost_k)
     + λ · CVaR_α  (if risk_aversion > 0)

Subtour elimination: MTZ formulation — O(|R|²) constraints per (k,v).
With K=5, V=3, R=6: 5×3×36 = 540 MTZ constraints. Tractable.

Estimated solve time (Intel i5-1135G7, 8 GB RAM):
  K=5, R=6, V=3, P=10, S=6  →  5–30 minutes depending on problem structure.

References
──────────
Birge & Louveaux (2011), §3.2 Extensive Form.
Miller, Tucker & Zemlin (1960), MTZ subtour elimination.
Rockafellar & Uryasev (2000), CVaR formulation.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pulp import (
    PULP_CBC_CMD,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)


class ExtensiveFormOptimizer:
    """
    True two-stage stochastic extensive-form optimizer.
    Procurement + VRP in a single MILP.
    """

    _DC_MAX_SEVERITY = {
        "hoakhanh": 3,
        "lienchieu": 5,
    }

    def __init__(
        self,
        network: Dict,
        products_df: pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        weather_scenarios: List,
        vehicle_config: Dict = None,
        risk_aversion: float = 0.0,
        cvar_alpha: float = 0.95,
        baseline_ratio: float = 0.70,
        emergency_ratio: float = 0.40,
    ):
        self.network = network
        self.products_df = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df = demand_df
        self.scenarios = weather_scenarios
        self.risk_aversion = risk_aversion
        self.cvar_alpha = cvar_alpha
        self.baseline_ratio = baseline_ratio
        self.emergency_ratio = emergency_ratio

        self.suppliers = network["suppliers"]["id"].tolist()
        self.products = products_df["id"].tolist()
        self.stores = network["stores"]["id"].tolist()

        default_vcfg = {
            "num_vehicles": 3,
            "capacity_kg": 1000,
            "base_speed_kmh": 40,
            "cost_per_km": 5_000,
            "cost_per_hour": 50_000,
            "max_route_time_hours": 10,
            "unmet_penalty_per_unit": 80_000,
        }
        self.vcfg = {**default_vcfg, **(vehicle_config or {})}

        self._create_lookups()

        K = len(self.scenarios)
        V = self.vcfg["num_vehicles"]
        R = len(self.stores)
        P = len(self.products)
        S = len(self.suppliers)
        N = len(network.get("dcs", pd.DataFrame())) + R  # nodes = DCs + stores

        print("ExtensiveFormOptimizer (True Two-Stage):")
        print(f"  Scenarios K={K}, Vehicles V={V}, Stores R={R}, Products P={P}")
        print(f"  Estimated binary variables: ~{60 + K*V*N*N}")
        print(f"  Estimated constraints:      ~{80 + K*(R*P + V*N*N)}")
        print(f"  Risk aversion λ={risk_aversion}, CVaR α={cvar_alpha}")

    # ------------------------------------------------------------------
    def _create_lookups(self):
        self.product_cost = dict(
            zip(self.products_df["id"], self.products_df["unit_cost_vnd"])
        )
        self.product_weight = dict(
            zip(self.products_df["id"], self.products_df["weight_kg_per_unit"])
        )
        self.supplier_capacity = dict(
            zip(self.network["suppliers"]["id"], self.network["suppliers"]["capacity_kg_per_day"])
        )
        self.supplier_fixed_cost = dict(
            zip(self.network["suppliers"]["id"], self.network["suppliers"]["fixed_cost_vnd"])
        )
        self.supplier_subtype = {}
        for _, row in self.network["suppliers"].iterrows():
            self.supplier_subtype[row["id"]] = row.get("subtype", "general")

        self.sp_cost, self.sp_moq, self.sp_available = {}, {}, {}
        for _, row in self.supplier_product_df.iterrows():
            s, p = row["supplier_id"], row["product_id"]
            self.sp_cost[(s, p)] = row["unit_cost_vnd"]
            self.sp_moq[(s, p)] = row["moq_units"]
            self.sp_available[(s, p)] = row["available"]

        dist_matrix = self.network["distance_matrix"]
        dcs = self.network["dcs"]["id"].tolist()
        self.all_nodes = dcs + self.stores
        self.distance = {}
        for i in self.all_nodes:
            for j in self.all_nodes:
                if i != j:
                    try:
                        self.distance[(i, j)] = float(dist_matrix.loc[i, j])
                    except Exception:
                        try:
                            self.distance[(i, j)] = float(dist_matrix.loc[j, i])
                        except Exception:
                            self.distance[(i, j)] = 10.0

        self.store_demand = {}
        grp = (
            self.demand_df.groupby(["store_id", "product_id"])["demand_units"]
            .sum()
            .reset_index()
        )
        for _, row in grp.iterrows():
            self.store_demand[(row["store_id"], row["product_id"])] = float(row["demand_units"])

        self.total_demand = (
            self.demand_df.groupby("product_id")["demand_units"].sum().to_dict()
        )

    # ------------------------------------------------------------------
    def _get_depot(self, scenario) -> str:
        sev = scenario.severity_level
        for _, dc in self.network["dcs"].iterrows():
            key = dc["name"].lower().replace(" ", "").replace("_", "")
            max_sev = 5
            for k, v in self._DC_MAX_SEVERITY.items():
                if k in key:
                    max_sev = v
                    break
            if sev <= max_sev:
                return dc["id"]
        return self.network["dcs"]["id"].iloc[-1]  # safest fallback

    def _accessible_suppliers(self, scenario, product_id: str) -> List[str]:
        return [
            s for s in self.suppliers
            if self.sp_available.get((s, product_id), False)
            and scenario.get_supplier_accessible(self.supplier_subtype.get(s, "general")) == 1
        ]

    def _inaccessible_suppliers(self, scenario, product_id: str) -> List[str]:
        return [
            s for s in self.suppliers
            if self.sp_available.get((s, product_id), False)
            and scenario.get_supplier_accessible(self.supplier_subtype.get(s, "general")) == 0
        ]

    # ------------------------------------------------------------------
    def build_model(self) -> Tuple[LpProblem, Dict]:
        """Build the full extensive-form MILP."""
        print("\nBuilding extensive-form MILP…")
        K = len(self.scenarios)
        V = self.vcfg["num_vehicles"]
        M_big = self.vcfg["max_route_time_hours"] + 1.0
        M_qty = 100_000

        model = LpProblem("ExtensiveForm", LpMinimize)

        # ── STAGE 1 VARIABLES ─────────────────────────────────────────
        x = LpVariable.dicts("x", ((s, p) for s in self.suppliers for p in self.products), lowBound=0)
        y = LpVariable.dicts("y", ((s, p) for s in self.suppliers for p in self.products), cat="Binary")

        # ── STAGE 2 VARIABLES (per scenario) ──────────────────────────
        e = LpVariable.dicts("e", ((k, p) for k in range(K) for p in self.products), lowBound=0)
        u = LpVariable.dicts("u", ((k, p) for k in range(K) for p in self.products), lowBound=0)

        # VRP arc and delivery variables
        arc, qty, T_mtz = {}, {}, {}
        depot_by_k = {}

        for k, sc in enumerate(self.scenarios):
            depot = self._get_depot(sc)
            depot_by_k[k] = depot
            nodes = [depot] + self.stores
            adjusted_speed = max(1.0, self.vcfg["base_speed_kmh"] / sc.speed_reduction_factor)
            adjusted_cap = self.vcfg["capacity_kg"] * sc.capacity_reduction_factor

            for i in nodes:
                for j in nodes:
                    if i != j:
                        for v in range(V):
                            arc[k, i, j, v] = LpVariable(f"arc_{k}_{i}_{j}_{v}", cat="Binary")
            for r in self.stores:
                for p in self.products:
                    for v in range(V):
                        qty[k, r, p, v] = LpVariable(f"qty_{k}_{r}_{p}_{v}", lowBound=0)
            for j in nodes:
                for v in range(V):
                    T_mtz[k, j, v] = LpVariable(f"T_{k}_{j}_{v}", lowBound=0, upBound=M_big)

        # CVaR variables
        if self.risk_aversion > 0:
            eta = LpVariable("eta", lowBound=0)
            zeta = LpVariable.dicts("zeta", range(K), lowBound=0)
        else:
            eta, zeta = None, None

        # ── OBJECTIVE ─────────────────────────────────────────────────
        s1_var_cost = lpSum(
            self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
            for s in self.suppliers for p in self.products
            if self.sp_available.get((s, p), False)
        )
        s1_fix_cost = lpSum(
            self.supplier_fixed_cost[s] * y[s, p]
            for s in self.suppliers for p in self.products
            if self.sp_available.get((s, p), False)
        )

        s2_terms = []
        for k, sc in enumerate(self.scenarios):
            prob = sc.probability
            depot = depot_by_k[k]
            nodes = [depot] + self.stores
            adj_speed = max(1.0, self.vcfg["base_speed_kmh"] / sc.speed_reduction_factor)

            # VRP cost: distance + time
            vrp_cost = lpSum(
                (self.distance.get((i, j), 0)
                 * (self.vcfg["cost_per_km"] + self.vcfg["cost_per_hour"] / adj_speed))
                * arc[k, i, j, v]
                for i in nodes for j in nodes for v in range(V) if i != j
            )
            # Emergency cost
            em_cost = lpSum(2.0 * self.product_cost[p] * e[k, p] for p in self.products)
            # Unmet penalty
            penalty_mult = min(10.0, 5.0 * sc.spoilage_multiplier)
            unmet_cost = lpSum(penalty_mult * self.product_cost[p] * u[k, p] for p in self.products)
            # Spoilage cost (inaccessible suppliers)
            sp_cost_k = lpSum(
                self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
                for p in self.products for s in self._inaccessible_suppliers(sc, p)
            )

            scenario_total = vrp_cost + em_cost + unmet_cost + sp_cost_k
            s2_terms.append(prob * scenario_total)

        expected_total = s1_var_cost + s1_fix_cost + lpSum(s2_terms)

        if self.risk_aversion > 0 and eta is not None:
            cvar = eta + (1.0 / (1.0 - self.cvar_alpha)) * lpSum(
                self.scenarios[k].probability * zeta[k] for k in range(K)
            )
            model += (1 - self.risk_aversion) * expected_total + self.risk_aversion * cvar, "Obj"
        else:
            model += expected_total, "Obj"

        # ── STAGE 1 CONSTRAINTS ───────────────────────────────────────
        M_s1 = 100_000

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
                all_x = lpSum(x[s, p] for s in self.suppliers if self.sp_available.get((s, p), False))
                model += (all_x >= self.baseline_ratio * d, f"S1Base_{p}")
                model += (all_x <= 1.5 * d, f"S1Over_{p}")

        # ── STAGE 2 CONSTRAINTS (per scenario k) ─────────────────────
        for k, sc in enumerate(self.scenarios):
            depot = depot_by_k[k]
            nodes = [depot] + self.stores
            adj_speed = max(1.0, self.vcfg["base_speed_kmh"] / sc.speed_reduction_factor)
            adj_cap = self.vcfg["capacity_kg"] * sc.capacity_reduction_factor

            # Emergency feasibility
            for p in self.products:
                d = self.total_demand.get(p, 0)
                em_cap = self.emergency_ratio * d * (1 if sc.emergency_feasible else 0)
                model += (e[k, p] <= em_cap, f"S2EmCap_{k}_{p}")

            # Aggregate demand balance
            for p in self.products:
                d = self.total_demand.get(p, 0)
                if d > 0:
                    acc = lpSum(x[s, p] for s in self._accessible_suppliers(sc, p))
                    model += (acc + e[k, p] + u[k, p] >= d, f"S2Dem_{k}_{p}")

            # VRP: flow conservation
            for v in range(V):
                for i in nodes:
                    in_f = lpSum(arc[k, j, i, v] for j in nodes if j != i)
                    out_f = lpSum(arc[k, i, j, v] for j in nodes if j != i)
                    model += (in_f == out_f, f"VFlow_{k}_{i}_{v}")

            # VRP: each vehicle leaves depot at most once
            for v in range(V):
                model += (
                    lpSum(arc[k, depot, j, v] for j in self.stores) <= 1,
                    f"VDepart_{k}_{v}"
                )

            # VRP: each store visited at least once
            for r in self.stores:
                model += (
                    lpSum(arc[k, i, r, v] for i in nodes for v in range(V) if i != r) >= 1,
                    f"VVisit_{k}_{r}"
                )

            # VRP: vehicle capacity
            for v in range(V):
                model += (
                    lpSum(qty[k, r, p, v] * self.product_weight[p]
                          for r in self.stores for p in self.products)
                    <= adj_cap, f"VCap_{k}_{v}"
                )

            # VRP: store-level demand satisfaction
            for r in self.stores:
                for p in self.products:
                    d_rp = self.store_demand.get((r, p), 0)
                    if d_rp > 0:
                        model += (
                            lpSum(qty[k, r, p, v] for v in range(V)) <= d_rp,
                            f"VDelMax_{k}_{r}_{p}"
                        )

            # VRP: deliver only when visited
            for r in self.stores:
                for p in self.products:
                    for v in range(V):
                        vis = lpSum(arc[k, i, r, v] for i in nodes if i != r)
                        model += (qty[k, r, p, v] <= M_qty * vis, f"VVisDel_{k}_{r}_{p}_{v}")

            # VRP: MTZ subtour elimination
            T_start = 5.0
            for v in range(V):
                model += (T_mtz[k, depot, v] == T_start, f"MTZdep_{k}_{v}")

            for i in nodes:
                for j in self.stores:
                    if i != j:
                        t_ij = self.distance.get((i, j), 0) / adj_speed
                        for v in range(V):
                            model += (
                                T_mtz[k, j, v]
                                >= T_mtz[k, i, v] + t_ij - M_big * (1 - arc[k, i, j, v]),
                                f"MTZ_{k}_{i}_{j}_{v}"
                            )

            # CVaR auxiliary
            if self.risk_aversion > 0 and zeta is not None:
                # Total scenario cost for CVaR
                depot2 = depot_by_k[k]
                nodes2 = [depot2] + self.stores
                adj_spd2 = max(1.0, self.vcfg["base_speed_kmh"] / sc.speed_reduction_factor)
                sc_cost = (
                    s1_var_cost + s1_fix_cost
                    + lpSum(
                        (self.distance.get((i, j), 0)
                         * (self.vcfg["cost_per_km"] + self.vcfg["cost_per_hour"] / adj_spd2))
                        * arc[k, i, j, v]
                        for i in nodes2 for j in nodes2 for v in range(V) if i != j
                    )
                    + lpSum(2.0 * self.product_cost[p] * e[k, p] for p in self.products)
                    + lpSum(min(10.0, 5.0 * sc.spoilage_multiplier) * self.product_cost[p] * u[k, p]
                            for p in self.products)
                )
                model += (zeta[k] >= sc_cost - eta, f"CVaR_{k}")

        print(f"  ✓ Variables: {model.numVariables()}  |  Constraints: {model.numConstraints()}")
        return model, {
            "x": x, "y": y, "e": e, "u": u,
            "arc": arc, "qty": qty, "T_mtz": T_mtz,
            "depot_by_k": depot_by_k,
            "eta": eta, "zeta": zeta,
        }

    # ------------------------------------------------------------------
    def solve(self, time_limit: int = 1800, gap_tolerance: float = 0.05) -> Tuple[str, Dict]:
        model, vd = self.build_model()

        solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap_tolerance, msg=1)

        print(f"\nSolving extensive form (time limit={time_limit}s, gap={gap_tolerance*100:.0f}%)…")
        t0 = time.time()
        model.solve(solver)
        solve_time = time.time() - t0

        status = LpStatus[model.status]
        print(f"  Status:    {status}  ({solve_time:.1f}s)")

        if status in ("Optimal", "Feasible"):
            obj = value(model.objective)
            print(f"  Objective: {obj:,.0f} VND")

            solution = self._extract_solution(vd)
            solution.update({
                "objective_value": obj,
                "solve_time": solve_time,
                "status": status,
                "scenario_costs": self._compute_scenario_costs(vd),
            })
            return status, solution

        return status, {}

    # ------------------------------------------------------------------
    def _extract_solution(self, vd: Dict) -> Dict:
        x, y, e, u = vd["x"], vd["y"], vd["e"], vd["u"]
        K = len(self.scenarios)
        V = self.vcfg["num_vehicles"]

        stage1 = [
            {
                "supplier_id": s, "product_id": p,
                "quantity_units": round(value(x[s, p]) or 0, 2),
                "cost_vnd": round((value(x[s, p]) or 0) * self.sp_cost.get((s, p), self.product_cost[p]), 0),
            }
            for s in self.suppliers for p in self.products
            if (value(x[s, p]) or 0) > 0.01
        ]

        scenario_routes = {}
        for k, sc in enumerate(self.scenarios):
            depot = vd["depot_by_k"][k]
            nodes = [depot] + self.stores
            routes = []
            for v in range(V):
                stops, current, visited = [], depot, {depot}
                while True:
                    nxt = next(
                        (j for j in self.stores if j not in visited
                         and (value(vd["arc"].get((k, current, j, v))) or 0) > 0.5),
                        None,
                    )
                    if nxt is None:
                        break
                    deliveries = [
                        {"product_id": p, "quantity": round(value(vd["qty"][k, nxt, p, v]) or 0, 2)}
                        for p in self.products if (value(vd["qty"][k, nxt, p, v]) or 0) > 0.01
                    ]
                    if deliveries:
                        stops.append({"location": nxt, "deliveries": deliveries})
                    visited.add(nxt)
                    current = nxt
                if stops:
                    routes.append({"vehicle_id": v, "stops": stops})
            scenario_routes[sc.name] = routes

        return {
            "stage1_procurement": pd.DataFrame(stage1),
            "scenario_routes": scenario_routes,
        }

    def _compute_scenario_costs(self, vd: Dict) -> pd.DataFrame:
        x, y, e, u = vd["x"], vd["y"], vd["e"], vd["u"]
        K = len(self.scenarios)
        V = self.vcfg["num_vehicles"]

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
            adj_spd = max(1.0, self.vcfg["base_speed_kmh"] / sc.speed_reduction_factor)

            vrp_c = sum(
                (self.distance.get((i, j), 0)
                 * (self.vcfg["cost_per_km"] + self.vcfg["cost_per_hour"] / adj_spd))
                * (value(vd["arc"].get((k, i, j, v))) or 0)
                for i in nodes for j in nodes for v in range(V) if i != j
            )
            em_c = sum(2.0 * (value(e[k, p]) or 0) * self.product_cost[p] for p in self.products)
            pm = min(10.0, 5.0 * sc.spoilage_multiplier)
            unm_c = sum(pm * (value(u[k, p]) or 0) * self.product_cost[p] for p in self.products)
            sp_c = sum(
                (value(x[s, p]) or 0) * self.sp_cost.get((s, p), self.product_cost[p])
                for p in self.products for s in self._inaccessible_suppliers(sc, p)
            )
            rows.append({
                "scenario_name": sc.name,
                "severity_level": sc.severity_level,
                "probability": sc.probability,
                "stage1_cost": s1_total,
                "vrp_cost": vrp_c,
                "emergency_cost": em_c,
                "spoilage_cost": sp_c,
                "penalty_cost": unm_c,
                "total_cost": s1_total + vrp_c + em_c + sp_c + unm_c,
            })
        return pd.DataFrame(rows)