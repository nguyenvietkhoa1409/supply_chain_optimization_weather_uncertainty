"""
Weather-Aware Vehicle Routing Problem (VRP) – Standalone Module
FIXED VERSION

Fixes applied
─────────────
[CRITICAL / V-1]  VRP no longer returns inf cost.
    Root cause: Under Level 5 (capacity_factor=0.10), effective vehicle capacity
    = 1000 × 0.10 = 100 kg × 3 vehicles = 300 kg — less than weekly demand.
    Old code: hard capacity constraint → solver returns Infeasible → Python assigns
    np.inf → cascade through generate_report().

    Fix (two parts):
    a) Unmet demand variable u[i,p] with penalty cost (already existed but was
       not correctly preventing infeasibility). Strengthened by ensuring
       u[i,p] has no upper bound — any shortfall is always feasible at a cost.
    b) When VRP solver returns Infeasible or timeout, fall back to a penalty-only
       cost estimate rather than np.inf. The fallback cost =
       Σ demand × penalty_per_unit, clearly finite and signals the severity.

[HIGH / V-4]  Depot no longer hardcoded to dcs.iloc[0].
    OLD: self.depot = network['dcs']['id'].iloc[0]
    NEW: each solve() call receives the scenario and selects the accessible DC.
         DC accessibility logic:
           - DC_001 (Hòa Khánh): low-lying industrial zone → inaccessible Level 4+
           - DC_002 (Liên Chiểu): elevated → accessible all levels
         If no DC is accessible (should not happen; DC_002 is always accessible),
         fall back to DC_001 with a warning.

[V-2 note]  Arc-level spoilage cost added to VRP objective.
    cost_spoil[i,j,k,p] = c_sp_p × Q[i,p,v] × (1 − exp(−k_i(T_k) · t_{ij}(k)))
    Linearised as: α_spoil × c_sp_p × Q[i,p,v] × t_{ij}(k)
    where α_spoil is a small rate constant (0.05/h baseline, scaled by temperature).
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


class WeatherAwareVRP:
    """
    Vehicle Routing Problem with weather scenario integration.

    Designed to be called per scenario from IntegratedStochasticModel.
    Always returns a finite cost (inf-free guarantee).
    """

    # DC accessibility: maps DC name substring → max_severity_level accessible
    _DC_MAX_SEVERITY = {
        "Hoa Khanh": 3,   # floods at Level 4+
        "Lien Chieu": 5,  # elevated, always accessible
    }

    def __init__(
        self,
        network: Dict,
        products_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        procurement_solution: pd.DataFrame,
        weather_scenarios: List,
        vehicle_config: Dict = None,
    ):
        self.network = network
        self.products_df = products_df
        self.demand_df = demand_df
        self.procurement = procurement_solution
        self.scenarios = weather_scenarios

        default_cfg = {
            "num_vehicles": 3,
            "capacity_kg": 1000,
            "base_speed_kmh": 40,
            "cost_per_km": 5_000,
            "cost_per_hour": 50_000,
            "max_route_time_hours": 10,
            "unmet_penalty_per_unit": 80_000,  # VND – larger than transport cost to discourage
        }
        self.vehicle_config = {**default_cfg, **(vehicle_config or {})}

        self.stores = network["stores"]["id"].tolist()
        self.products = products_df["id"].tolist()

        self._create_lookups()

        print("WeatherAwareVRP (FIXED):")
        print(f"  Stores: {len(self.stores)}, Products: {len(self.products)}")
        print(f"  Vehicles: {self.vehicle_config['num_vehicles']}")

    # ------------------------------------------------------------------
    def _create_lookups(self):
        dist_matrix = self.network["distance_matrix"]

        dcs = self.network["dcs"]["id"].tolist()
        all_locs = dcs + self.stores

        self.distance = {}
        for i in all_locs:
            for j in all_locs:
                if i != j:
                    try:
                        self.distance[(i, j)] = float(dist_matrix.loc[i, j])
                    except Exception:
                        try:
                            self.distance[(i, j)] = float(dist_matrix.loc[j, i])
                        except Exception:
                            self.distance[(i, j)] = 10.0

        # Store-level demand
        self.demand = {}
        grp = (
            self.demand_df.groupby(["store_id", "product_id"])["demand_units"]
            .sum()
            .reset_index()
        )
        for _, row in grp.iterrows():
            self.demand[(row["store_id"], row["product_id"])] = float(row["demand_units"])

        self.product_weight = dict(
            zip(self.products_df["id"], self.products_df["weight_kg_per_unit"])
        )
        self.product_cost = dict(
            zip(self.products_df["id"], self.products_df["unit_cost_vnd"])
        )

        # Spoilage rate constant: base α = 0.05 /h (linearised Arrhenius, T_ref=26°C)
        # Scaled by temperature in solve()
        self.base_spoilage_rate = 0.05   # /h, per unit quantity

    # ------------------------------------------------------------------
    def _select_depot(self, scenario) -> str:
        """
        [V-4 FIX] Select accessible DC for this scenario.

        Hòa Khánh DC → accessible for Level ≤ 3
        Liên Chiểu DC → always accessible

        Returns DC id string.
        """
        dcs_df = self.network["dcs"]
        severity = scenario.severity_level

        for _, dc in dcs_df.iterrows():
            dc_name = dc["name"]
            max_sev = 5  # default: always accessible
            for key, val in self._DC_MAX_SEVERITY.items():
                if key.lower().replace(" ", "") in dc_name.lower().replace(" ", ""):
                    max_sev = val
                    break
            if severity <= max_sev:
                return dc["id"]

        # Fallback (should not reach here if Liên Chiểu DC is in dataset)
        print(f"  ⚠ No accessible DC found for severity {severity}; using first DC as fallback")
        return dcs_df["id"].iloc[0]

    # ------------------------------------------------------------------
    def build_model(self, scenario_id: int) -> Tuple[LpProblem, Dict]:
        sc = self.scenarios[scenario_id]
        depot = self._select_depot(sc)  # [V-4 FIX]

        speed_factor = 1.0 / sc.speed_reduction_factor   # convert travel-time factor to speed factor
        capacity_factor = sc.capacity_reduction_factor
        adjusted_speed = self.vehicle_config["base_speed_kmh"] * speed_factor
        adjusted_capacity = self.vehicle_config["capacity_kg"] * capacity_factor

        # [V-2] Temperature-scaled spoilage rate (Q10-based, simplified)
        T_k = sc.temperature_celsius
        T_ref = 26.0
        alpha_spoil = self.base_spoilage_rate * (2.0 ** ((T_k - T_ref) / 10.0))

        print(f"\n  Scenario [{sc.name}]: depot={depot}, cap={adjusted_capacity:.0f}kg, speed={adjusted_speed:.1f}km/h")

        model = LpProblem(f"VRP_{sc.name.replace(' ', '_')}", LpMinimize)

        locations = [depot] + self.stores
        vehicles = list(range(self.vehicle_config["num_vehicles"]))

        # ── VARIABLES ─────────────────────────────────────────────────
        arc = LpVariable.dicts(
            "arc",
            ((i, j, v) for i in locations for j in locations for v in vehicles if i != j),
            cat="Binary",
        )
        qty = LpVariable.dicts(
            "qty",
            ((r, p, v) for r in self.stores for p in self.products for v in vehicles),
            lowBound=0,
        )
        # [V-1 FIX] Unbounded unmet demand — always feasible
        unmet = LpVariable.dicts(
            "unmet",
            ((r, p) for r in self.stores for p in self.products),
            lowBound=0,
        )
        # MTZ position variables for subtour elimination
        T_mtz = LpVariable.dicts(
            "T_mtz",
            ((j, v) for j in locations for v in vehicles),
            lowBound=0,
            upBound=float(self.vehicle_config["max_route_time_hours"]),
        )

        # ── OBJECTIVE ─────────────────────────────────────────────────
        dist_cost = lpSum(
            self.distance.get((i, j), 0)
            * self.vehicle_config["cost_per_km"]
            * arc[i, j, v]
            for i in locations
            for j in locations
            for v in vehicles
            if i != j
        )
        time_cost = lpSum(
            (self.distance.get((i, j), 0) / max(adjusted_speed, 1.0))
            * self.vehicle_config["cost_per_hour"]
            * arc[i, j, v]
            for i in locations
            for j in locations
            for v in vehicles
            if i != j
        )
        # [V-2] Arc-level spoilage cost (linearised)
        spoil_cost = lpSum(
            alpha_spoil
            * (self.distance.get((i, r), 0) / max(adjusted_speed, 1.0))
            * self.product_cost[p]
            * qty[r, p, v]
            for i in locations
            for r in self.stores
            for p in self.products
            for v in vehicles
            if i != r
        )
        # [V-1 FIX] Unmet demand penalty (large but finite → never inf)
        penalty = lpSum(
            self.vehicle_config["unmet_penalty_per_unit"] * unmet[r, p]
            for r in self.stores
            for p in self.products
        )

        model += dist_cost + time_cost + spoil_cost + penalty, "Total_VRP_Cost"

        # ── CONSTRAINTS ───────────────────────────────────────────────
        M_big = self.vehicle_config["max_route_time_hours"] + 1.0

        # 1. Flow conservation at each node for each vehicle
        for v in vehicles:
            for i in locations:
                in_flow = lpSum(arc[j, i, v] for j in locations if j != i)
                out_flow = lpSum(arc[i, j, v] for j in locations if j != i)
                model += (in_flow == out_flow, f"Flow_{i}_{v}")

        # 2. Each vehicle leaves depot at most once
        for v in vehicles:
            model += (
                lpSum(arc[depot, j, v] for j in self.stores) <= 1,
                f"Depart_{v}",
            )

        # 3. Each store visited at least once (across all vehicles)
        for r in self.stores:
            model += (
                lpSum(arc[i, r, v] for i in locations for v in vehicles if i != r) >= 1,
                f"Visit_{r}",
            )

        # 4. Vehicle load capacity [V-1 FIX: adjusted_capacity may be very small;
        #    unmet demand absorbs the gap, so this constraint will not cause infeasibility]
        for v in vehicles:
            model += (
                lpSum(
                    qty[r, p, v] * self.product_weight[p]
                    for r in self.stores
                    for p in self.products
                )
                <= adjusted_capacity,
                f"Cap_{v}",
            )

        # 5. Demand satisfaction (with unmet safety valve)
        for r in self.stores:
            for p in self.products:
                d = self.demand.get((r, p), 0)
                if d > 0:
                    model += (
                        lpSum(qty[r, p, v] for v in vehicles) + unmet[r, p] >= d,
                        f"Demand_{r}_{p}",
                    )

        # 6. Deliver only when visited
        M_qty = 10_000
        for r in self.stores:
            for p in self.products:
                for v in vehicles:
                    visited = lpSum(arc[i, r, v] for i in locations if i != r)
                    model += (qty[r, p, v] <= M_qty * visited, f"Vis_{r}_{p}_{v}")

        # 7. MTZ subtour elimination (Miller–Tucker–Zemlin)
        # t_{ij}(k) = d_{ij} / adjusted_speed
        T_start = 5.0  # 05:00 departure from depot
        for v in vehicles:
            model += (T_mtz[depot, v] == T_start, f"MTZ_dep_{v}")

        for i in locations:
            for j in self.stores:
                if i != j:
                    t_ij = self.distance.get((i, j), 0) / max(adjusted_speed, 1.0)
                    for v in vehicles:
                        model += (
                            T_mtz[j, v]
                            >= T_mtz[i, v] + t_ij - M_big * (1 - arc[i, j, v]),
                            f"MTZ_{i}_{j}_{v}",
                        )

        return model, {"arc": arc, "qty": qty, "unmet": unmet, "T_mtz": T_mtz, "depot": depot}

    # ------------------------------------------------------------------
    def solve(self, scenario_id: int, time_limit: int = 300) -> Tuple[str, Dict]:
        """
        Solve VRP for given scenario.

        [V-1 FIX] Always returns a finite objective value.
        If solver returns Infeasible or errors, a penalty-only fallback cost is
        returned so the caller never receives np.inf.
        """
        # Compute fallback cost (all demand unmet)
        sc = self.scenarios[scenario_id]
        fallback_cost = sum(
            self.demand.get((r, p), 0) * self.vehicle_config["unmet_penalty_per_unit"]
            for r in self.stores
            for p in self.products
        )

        try:
            model, vars_dict = self.build_model(scenario_id)
        except Exception as ex:
            print(f"  ⚠ VRP build failed for scenario {sc.name}: {ex}")
            return "Fallback", {
                "objective_value": fallback_cost,
                "solve_time": 0.0,
                "status": "Fallback",
                "routes": [],
                "unmet_demand": fallback_cost / self.vehicle_config["unmet_penalty_per_unit"],
            }

        solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=0.10, msg=0)

        t0 = time.time()
        try:
            model.solve(solver)
        except Exception as ex:
            print(f"  ⚠ Solver error for {sc.name}: {ex}")
            return "Fallback", {
                "objective_value": fallback_cost,
                "solve_time": time.time() - t0,
                "status": "Fallback",
                "routes": [],
                "unmet_demand": fallback_cost / self.vehicle_config["unmet_penalty_per_unit"],
            }

        solve_time = time.time() - t0
        status = LpStatus[model.status]

        if status in ("Optimal", "Feasible"):
            obj = value(model.objective)
            if obj is None or (not np.isfinite(obj)):
                # Paranoia guard: should never happen but handle defensively
                obj = fallback_cost

            unmet_total = sum(
                value(vars_dict["unmet"][r, p]) or 0
                for r in self.stores
                for p in self.products
            )

            solution = self._extract_solution(vars_dict)
            solution.update(
                {
                    "objective_value": obj,
                    "solve_time": solve_time,
                    "status": status,
                    "unmet_demand": unmet_total,
                    "depot": vars_dict["depot"],
                }
            )
            print(f"    ✓ {sc.name}: {obj:,.0f} VND  (unmet={unmet_total:.1f} units, {solve_time:.1f}s)")
            return status, solution

        # Infeasible / not solved: return finite fallback
        print(f"    ⚠ VRP infeasible/timeout for {sc.name}; returning fallback cost")
        return "Fallback", {
            "objective_value": fallback_cost,
            "solve_time": solve_time,
            "status": "Fallback",
            "routes": [],
            "unmet_demand": fallback_cost / self.vehicle_config["unmet_penalty_per_unit"],
        }

    # ------------------------------------------------------------------
    def _extract_solution(self, vars_dict: Dict) -> Dict:
        arc, qty = vars_dict["arc"], vars_dict["qty"]
        depot = vars_dict["depot"]
        vehicles = list(range(self.vehicle_config["num_vehicles"]))
        locations = [depot] + self.stores

        routes = []
        for v in vehicles:
            route_stops = []
            current = depot
            visited_set = {depot}

            while True:
                nxt = None
                for j in self.stores:
                    if j not in visited_set and (current, j, v) in arc:
                        if (value(arc[current, j, v]) or 0) > 0.5:
                            nxt = j
                            break
                if nxt is None:
                    break
                deliveries = [
                    {"product_id": p, "quantity": round(value(qty[nxt, p, v]) or 0, 2)}
                    for p in self.products
                    if (value(qty[nxt, p, v]) or 0) > 0.01
                ]
                if deliveries:
                    route_stops.append({"location": nxt, "deliveries": deliveries})
                visited_set.add(nxt)
                current = nxt

            if route_stops:
                routes.append({"vehicle_id": v, "depot": depot, "stops": route_stops})

        return {"routes": routes}