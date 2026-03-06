"""
Weather-Aware Vehicle Routing Problem (VRP) – UPDATED v2
Heterogeneous Fleet with Dual-Capacity Constraints

Changes from v1
───────────────
[F-1] Heterogeneous fleet — each vehicle has its own payload, volume, cost, speed.
[F-2] use_vehicle[v] binary per solve — fixed cost only when dispatched.
[F-3] Dual capacity: weight (payload_kg) AND cubic (volume_m3) per vehicle.
[F-4] Weather capacity reduction differentiated per vehicle type via get_effective_capacity().
[F-5] Refrigeration spoilage premium for perishables in non-refrigerated vehicles.

All previous fixes inherited:
[V-1] Always returns finite cost (unmet demand fallback).
[V-4] Depot selection by weather severity.
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

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_generation.fleet_config import (
    VEHICLE_TYPES,
    expand_fleet,
    get_effective_capacity,
    DEFAULT_VOLUME_M3_PER_UNIT,
)


class WeatherAwareVRP:
    """
    Weather-aware VRP with heterogeneous fleet.
    Always returns finite cost.
    """

    _DC_MAX_SEVERITY = {"hoa khanh": 3, "lien chieu": 5}

    def __init__(
        self,
        network: Dict,
        products_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        procurement_solution: pd.DataFrame,
        weather_scenarios: List,
        vehicle_config: Dict = None,      # legacy single-fleet (backward compat)
        fleet_config: List[Dict] = None,  # NEW heterogeneous fleet
        refrig_penalty_factor: float = 1.5,
    ):
        self.network              = network
        self.products_df          = products_df
        self.demand_df            = demand_df
        self.procurement          = procurement_solution
        self.scenarios            = weather_scenarios
        self.refrig_penalty_factor = refrig_penalty_factor

        # Fleet setup
        if fleet_config is not None:
            self.fleet_vehicles = expand_fleet(fleet_config)
        elif vehicle_config is not None and "types" in vehicle_config:
            self.fleet_vehicles = expand_fleet(vehicle_config["types"])
        else:
            self.fleet_vehicles = self._legacy_to_fleet(vehicle_config)

        self.stores   = network["stores"]["id"].tolist()
        self.products = products_df["id"].tolist()

        self._create_lookups()

        V = len(self.fleet_vehicles)
        from collections import Counter
        fleet_str = ", ".join(f"{k}×{v}" for k, v in Counter(
            veh["type_id"] for veh in self.fleet_vehicles).items())
        print(f"WeatherAwareVRP v2 (heterogeneous fleet):")
        print(f"  Stores: {len(self.stores)}, Products: {len(self.products)}")
        print(f"  Fleet: {V} vehicles ({fleet_str})")

    # ------------------------------------------------------------------
    @staticmethod
    def _legacy_to_fleet(vehicle_config: Dict) -> List[Dict]:
        vcfg = vehicle_config or {}
        n    = vcfg.get("num_vehicles", 3)
        vtype = {
            "type_id": "legacy", "name_vn": "Legacy", "count": n,
            "payload_kg": vcfg.get("capacity_kg", 1000),
            "volume_m3": vcfg.get("capacity_kg", 1000) * 0.003,
            "fixed_cost_vnd": 0,
            "cost_per_km_vnd": vcfg.get("cost_per_km", 5_000),
            "cost_per_hour_vnd": vcfg.get("cost_per_hour", 50_000),
            "base_speed_kmh": vcfg.get("base_speed_kmh", 40),
            "refrigerated": False,
            "capacity_weather_penalty": 1.0,
            "speed_weather_penalty": 1.0,
            "max_severity_operable": 5,
        }
        return expand_fleet([vtype])

    # ------------------------------------------------------------------
    def _create_lookups(self):
        dist_matrix    = self.network["distance_matrix"]
        dcs            = self.network["dcs"]["id"].tolist()
        all_locs       = dcs + self.stores
        self.distance  = {}
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

        self.demand = {}
        grp = self.demand_df.groupby(["store_id", "product_id"])["demand_units"].sum().reset_index()
        for _, row in grp.iterrows():
            self.demand[(row["store_id"], row["product_id"])] = float(row["demand_units"])

        self.product_weight = dict(zip(self.products_df["id"], self.products_df["weight_kg_per_unit"]))
        self.product_cost   = dict(zip(self.products_df["id"], self.products_df["unit_cost_vnd"]))

        if "volume_m3_per_unit" in self.products_df.columns:
            self.product_volume = dict(zip(self.products_df["id"], self.products_df["volume_m3_per_unit"]))
        else:
            self.product_volume = {p: DEFAULT_VOLUME_M3_PER_UNIT for p in self.products}

        if "requires_refrigeration" in self.products_df.columns:
            self.product_refrig = dict(zip(self.products_df["id"], self.products_df["requires_refrigeration"]))
        else:
            self.product_refrig = {p: False for p in self.products}

        # Fallback per-unit penalty (VND) for unmet demand
        self.unmet_penalty = 80_000

    # ------------------------------------------------------------------
    def _select_depot(self, scenario) -> str:
        dcs_df   = self.network["dcs"]
        severity = scenario.severity_level
        for _, dc in dcs_df.iterrows():
            dc_name = dc["name"].lower().replace(" ", "")
            max_sev = 5
            for k, v in self._DC_MAX_SEVERITY.items():
                if k.replace(" ", "") in dc_name:
                    max_sev = v
                    break
            if severity <= max_sev:
                return dc["id"]
        return dcs_df["id"].iloc[-1]

    # ------------------------------------------------------------------
    def build_model(self, scenario_id: int) -> Tuple[LpProblem, Dict]:
        sc        = self.scenarios[scenario_id]
        depot     = self._select_depot(sc)
        locations = [depot] + self.stores
        V         = len(self.fleet_vehicles)
        vehicles  = list(range(V))
        M_big     = 25.0
        M_qty     = 10_000

        print(f"\n  Scenario [{sc.name}]:  depot={depot}")

        model = LpProblem(f"VRP_{sc.name.replace(' ', '_')}_v2", LpMinimize)

        # ── VARIABLES ─────────────────────────────────────────────────
        # [F-2] Vehicle activation
        use_vehicle = LpVariable.dicts("use_v", vehicles, cat="Binary")

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
        unmet = LpVariable.dicts(
            "unmet",
            ((r, p) for r in self.stores for p in self.products),
            lowBound=0,
        )
        T_mtz = LpVariable.dicts(
            "T",
            ((j, v) for j in locations for v in vehicles),
            lowBound=0, upBound=M_big,
        )

        # ── OBJECTIVE ─────────────────────────────────────────────────
        # [F-2] Fleet fixed cost
        fleet_fixed = lpSum(
            self.fleet_vehicles[v]["fixed_cost_vnd"] * use_vehicle[v]
            for v in vehicles
        )

        # Transport cost (per-vehicle distance + time)
        dist_cost = lpSum(
            self.distance.get((i, j), 0)
            * self.fleet_vehicles[v]["cost_per_km_vnd"]
            * arc[i, j, v]
            for i in locations for j in locations for v in vehicles if i != j
        )
        eff_speeds = {
            v: max(1.0, get_effective_capacity(self.fleet_vehicles[v], sc)["speed_kmh"])
            for v in vehicles
        }
        time_cost = lpSum(
            (self.distance.get((i, j), 0) / eff_speeds[v])
            * self.fleet_vehicles[v]["cost_per_hour_vnd"]
            * arc[i, j, v]
            for i in locations for j in locations for v in vehicles if i != j
        )

        # [F-5] Refrigeration spoilage premium
        refrig_premium = lpSum(
            (self.distance.get((i, r), 0) / eff_speeds[v])
            * 0.05
            * (self.refrig_penalty_factor - 1.0)
            * self.product_cost[p]
            * qty[r, p, v]
            for i in locations for r in self.stores
            for p in self.products for v in vehicles
            if i != r
            and self.product_refrig.get(p, False)
            and not self.fleet_vehicles[v]["refrigerated"]
        )

        penalty = lpSum(
            self.unmet_penalty * unmet[r, p]
            for r in self.stores for p in self.products
        )

        model += fleet_fixed + dist_cost + time_cost + refrig_premium + penalty, "Obj"

        # ── CONSTRAINTS ───────────────────────────────────────────────
        # [F-4] Disable vehicles exceeding max severity
        for v, veh in enumerate(self.fleet_vehicles):
            if sc.severity_level > veh["max_severity_operable"]:
                model += (use_vehicle[v] == 0, f"VehDisabled_{v}")
                model += (
                    lpSum(arc[depot, j, v] for j in self.stores) == 0,
                    f"VehNoDepart_{v}"
                )

        # Flow conservation
        for v in vehicles:
            for i in locations:
                model += (
                    lpSum(arc[j, i, v] for j in locations if j != i)
                    == lpSum(arc[i, j, v] for j in locations if j != i),
                    f"Flow_{i}_{v}"
                )

        # Depart at most once → linked to use_vehicle [F-2]
        for v in vehicles:
            model += (
                lpSum(arc[depot, j, v] for j in self.stores) <= use_vehicle[v],
                f"Depart_{v}"
            )

        # Each store visited at least once
        for r in self.stores:
            model += (
                lpSum(arc[i, r, v] for i in locations for v in vehicles if i != r) >= 1,
                f"Visit_{r}"
            )

        # [F-3a] Payload capacity per vehicle
        for v, veh in enumerate(self.fleet_vehicles):
            eff    = get_effective_capacity(veh, sc)
            cap_kg = eff["payload_kg"]
            model += (
                lpSum(qty[r, p, v] * self.product_weight[p]
                      for r in self.stores for p in self.products)
                <= cap_kg, f"CapKg_{v}"
            )

        # [F-3b] Cubic load capacity per vehicle
        for v, veh in enumerate(self.fleet_vehicles):
            eff    = get_effective_capacity(veh, sc)
            cap_m3 = eff["volume_m3"]
            model += (
                lpSum(qty[r, p, v] * self.product_volume.get(p, DEFAULT_VOLUME_M3_PER_UNIT)
                      for r in self.stores for p in self.products)
                <= cap_m3, f"CapM3_{v}"
            )

        # Demand satisfaction
        for r in self.stores:
            for p in self.products:
                d = self.demand.get((r, p), 0)
                if d > 0:
                    model += (
                        lpSum(qty[r, p, v] for v in vehicles) + unmet[r, p] >= d,
                        f"Dem_{r}_{p}"
                    )

        # Deliver only when visited
        for r in self.stores:
            for p in self.products:
                for v in vehicles:
                    vis = lpSum(arc[i, r, v] for i in locations if i != r)
                    model += (qty[r, p, v] <= M_qty * vis, f"VisDel_{r}_{p}_{v}")

        # MTZ subtour elimination
        for v in vehicles:
            model += (T_mtz[depot, v] == 5.0, f"MTZdep_{v}")

        for i in locations:
            for j in self.stores:
                if i != j:
                    t_ij = self.distance.get((i, j), 0) / max(1.0, eff_speeds[0])
                    for v in vehicles:
                        model += (
                            T_mtz[j, v]
                            >= T_mtz[i, v] + t_ij - M_big * (1 - arc[i, j, v]),
                            f"MTZ_{i}_{j}_{v}"
                        )

        return model, {
            "arc": arc, "qty": qty, "unmet": unmet, "T_mtz": T_mtz,
            "use_vehicle": use_vehicle, "depot": depot,
        }

    # ------------------------------------------------------------------
    def solve(self, scenario_id: int, time_limit: int = 300) -> Tuple[str, Dict]:
        sc            = self.scenarios[scenario_id]
        fallback_cost = sum(
            self.demand.get((r, p), 0) * self.unmet_penalty
            for r in self.stores for p in self.products
        )

        try:
            model, vd = self.build_model(scenario_id)
        except Exception as ex:
            print(f"  ⚠ VRP build failed [{sc.name}]: {ex}")
            return "Fallback", {"objective_value": fallback_cost, "solve_time": 0.0,
                                "status": "Fallback", "routes": [], "unmet_demand": 0.0}

        solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=0.10, msg=0)
        t0 = time.time()
        try:
            model.solve(solver)
        except Exception as ex:
            print(f"  ⚠ Solver error [{sc.name}]: {ex}")
            return "Fallback", {"objective_value": fallback_cost, "solve_time": time.time() - t0,
                                "status": "Fallback", "routes": [], "unmet_demand": 0.0}

        solve_time = time.time() - t0
        status     = LpStatus[model.status]

        if status in ("Optimal", "Feasible"):
            obj = value(model.objective)
            if obj is None or not np.isfinite(obj):
                obj = fallback_cost

            V       = len(self.fleet_vehicles)
            n_used  = sum(1 for v in range(V) if (value(vd["use_vehicle"][v]) or 0) > 0.5)
            unmet_t = sum(value(vd["unmet"][r, p]) or 0
                          for r in self.stores for p in self.products)

            # Fleet utilisation report
            fleet_info = [
                f"{self.fleet_vehicles[v]['vehicle_id']} "
                f"(cap={self.fleet_vehicles[v]['payload_kg']}kg"
                f"{'★' if self.fleet_vehicles[v]['refrigerated'] else ''})"
                for v in range(V) if (value(vd["use_vehicle"][v]) or 0) > 0.5
            ]
            print(f"    ✓ [{sc.name}]:  {obj:,.0f} VND  "
                  f"vehicles_used={n_used}/{V}  unmet={unmet_t:.0f}  ({solve_time:.1f}s)")
            if fleet_info:
                print(f"      Dispatched: {', '.join(fleet_info)}")

            sol = self._extract_solution(vd)
            sol.update({
                "objective_value":  obj,
                "solve_time":       solve_time,
                "status":           status,
                "unmet_demand":     unmet_t,
                "depot":            vd["depot"],
                "vehicles_used":    n_used,
            })
            return status, sol

        print(f"    ⚠ VRP infeasible/timeout [{sc.name}] → fallback cost")
        return "Fallback", {"objective_value": fallback_cost, "solve_time": solve_time,
                            "status": "Fallback", "routes": [], "unmet_demand": 0.0}

    # ------------------------------------------------------------------
    def _extract_solution(self, vd: Dict) -> Dict:
        arc, qty  = vd["arc"], vd["qty"]
        depot     = vd["depot"]
        V         = len(self.fleet_vehicles)
        vehicles  = list(range(V))
        locations = [depot] + self.stores

        routes = []
        for v in vehicles:
            if (value(vd["use_vehicle"][v]) or 0) < 0.5:
                continue
            stops, current, visited = [], depot, {depot}
            while True:
                nxt = next(
                    (j for j in self.stores if j not in visited
                     and (current, j, v) in arc
                     and (value(arc[current, j, v]) or 0) > 0.5),
                    None,
                )
                if nxt is None:
                    break
                deliveries = [
                    {"product_id": p, "quantity": round(value(qty[nxt, p, v]) or 0, 2)}
                    for p in self.products if (value(qty[nxt, p, v]) or 0) > 0.01
                ]
                if deliveries:
                    stops.append({
                        "location":         nxt,
                        "deliveries":       deliveries,
                        "vehicle_type":     self.fleet_vehicles[v]["type_id"],
                        "refrigerated":     self.fleet_vehicles[v]["refrigerated"],
                    })
                visited.add(nxt)
                current = nxt
            if stops:
                routes.append({
                    "vehicle_id":       v,
                    "vehicle_type":     self.fleet_vehicles[v]["type_id"],
                    "vehicle_name":     self.fleet_vehicles[v]["vehicle_id"],
                    "payload_kg":       self.fleet_vehicles[v]["payload_kg"],
                    "refrigerated":     self.fleet_vehicles[v]["refrigerated"],
                    "depot":            depot,
                    "stops":            stops,
                })
        return {"routes": routes}