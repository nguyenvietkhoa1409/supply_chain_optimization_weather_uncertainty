"""
Stochastic Procurement Optimization – Two-Stage MILP
FIXED VERSION

Fixes applied
─────────────
[CRITICAL / W-3]  Non-anticipativity violation removed.
    OLD: x[s,p] * survival_rate  (Stage 1 variable multiplied by scenario-dependent scalar
         inside a Stage 2 constraint → x effectively differs across scenarios).
    NEW: Σ_{s : a_{k,s}=1} x[s,p]  (same Stage 1 variable, only the summation set changes
         per scenario; x itself is scenario-independent ✓).
    Implementation: accessible_supply_k = sum of x[s,p] over suppliers whose subtype
    is accessible under scenario k (supplier_accessibility dict from WeatherScenario).

[CRITICAL / P-1]  Stage 1 demand constraint no longer forces full demand satisfaction.
    OLD: Σ x[s,p] ≥ D_p  →  collapses to deterministic (no stochastic value).
    NEW: Σ x[s,p] ≥ β·D_p  with β ∈ [0.5, 0.8] (configurable, default 0.70).
    Full demand satisfaction is enforced at Stage 2 through dispatch + emergency + unmet.

[HIGH / P-3]  Flow balance now includes spoilage loss term.
    sp_{k,p} = Σ_{s: a_{k,s}=0} x[s,p]  (goods from inaccessible suppliers are lost).
    Stage 2 balance: accessible_supply + emergency + unmet_demand ≥ D_p
    Spoilage tracked separately and added to objective as opportunity cost.

[MEDIUM / P-5]  Explicit spoilage cost term in objective.
    + Σ_k p_k · Σ_p c_sp_p · sp_{k,p}
    c_sp_p = unit_cost_vnd (opportunity cost of spoiled inventory).

[HIGH / V-1 prep]  Emergency procurement bounded by scenario feasibility flag.
    e[k,p] = 0  when scenario.emergency_feasible = False  (Level 5 typhoon).

[MEDIUM / P-2]  Safety stock z_i removed; replaced by overstock prevention bound.
    Σ x[s,p] ≤ (1 + δ_p)·D_p  where δ_p is shelf-life-derived overstock fraction.

References
──────────
Birge & Louveaux (2011), Introduction to Stochastic Programming, 2nd ed.
"""

import time
from typing import Dict, List, Tuple

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

# from procurement_base import ProcurementOptimizer  # noqa: F401 (kept for import compat)


class StochasticProcurementModel:
    """
    Two-Stage Stochastic MILP for Weather-Aware Procurement.

    Stage 1 (before weather realises):  x[s,p], y[s,p]
    Stage 2 (after scenario k realises): e[k,p], u[k,p]
    Derived (no extra variable):         accessible_supply, sp_{k,p}
    """

    def __init__(
        self,
        network: Dict,
        products_df: pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        weather_scenarios: List,
        risk_aversion: float = 0.0,
        baseline_ratio: float = 0.70,     # β: fraction of demand covered at Stage 1
        overstock_delta: float = None,    # δ: max overstock ratio; None → shelf-life derived
        emergency_ratio: float = 0.40,   # EC_p as fraction of demand
    ):
        self.network = network
        self.products_df = products_df
        self.supplier_product_df = supplier_product_df
        self.demand_df = demand_df
        self.scenarios = weather_scenarios
        self.risk_aversion = risk_aversion
        self.baseline_ratio = baseline_ratio
        self.emergency_ratio = emergency_ratio

        self.suppliers = network["suppliers"]["id"].tolist()
        self.products = products_df["id"].tolist()

        self._create_lookup_dicts()

        self.total_demand = (
            demand_df.groupby("product_id")["demand_units"].sum().to_dict()
        )

        # δ_p: max overstock fraction per product (shelf-life-derived)
        # A product with SL=1 day can hold at most ~10% extra before spoiling;
        # SL=7 days can hold up to 50% extra.
        if overstock_delta is None:
            self.overstock_delta = {}
            for _, prod in products_df.iterrows():
                sl = prod.get("shelf_life_days", 2.0)
                # δ = min(0.5, (SL - 1) / SL)  capped at 50%
                self.overstock_delta[prod["id"]] = min(0.50, max(0.05, (sl - 1) / sl))
        else:
            self.overstock_delta = {p: overstock_delta for p in self.products}

        print("Stochastic Procurement Model (FIXED):")
        print(f"  Suppliers: {len(self.suppliers)}")
        print(f"  Products:  {len(self.products)}")
        print(f"  Scenarios: {len(self.scenarios)}")
        print(f"  β (baseline ratio):   {self.baseline_ratio}")
        print(f"  λ (risk aversion):    {self.risk_aversion}")

    # ------------------------------------------------------------------
    def _create_lookup_dicts(self):
        self.product_cost = dict(
            zip(self.products_df["id"], self.products_df["unit_cost_vnd"])
        )
        self.product_weight = dict(
            zip(self.products_df["id"], self.products_df["weight_kg_per_unit"])
        )
        self.supplier_capacity = dict(
            zip(
                self.network["suppliers"]["id"],
                self.network["suppliers"]["capacity_kg_per_day"],
            )
        )
        self.supplier_fixed_cost = dict(
            zip(
                self.network["suppliers"]["id"],
                self.network["suppliers"]["fixed_cost_vnd"],
            )
        )

        # Lookup: supplier subtype (for accessibility check)
        self.supplier_subtype = {}
        for _, row in self.network["suppliers"].iterrows():
            self.supplier_subtype[row["id"]] = row.get("subtype", "general")

        self.sp_cost = {}
        self.sp_moq = {}
        self.sp_available = {}
        for _, row in self.supplier_product_df.iterrows():
            s, p = row["supplier_id"], row["product_id"]
            self.sp_cost[(s, p)] = row["unit_cost_vnd"]
            self.sp_moq[(s, p)] = row["moq_units"]
            self.sp_available[(s, p)] = row["available"]

    # ------------------------------------------------------------------
    def _get_accessible_suppliers(self, scenario, product_id: str) -> List[str]:
        """
        Return list of suppliers that are ACCESSIBLE under 'scenario'
        AND can supply 'product_id'.

        Non-anticipativity is preserved: x[s,p] is the same Stage 1
        variable; only the summation set varies by scenario.
        """
        return [
            s
            for s in self.suppliers
            if self.sp_available.get((s, product_id), False)
            and scenario.get_supplier_accessible(self.supplier_subtype.get(s, "general"))
            == 1
        ]

    def _get_inaccessible_suppliers(self, scenario, product_id: str) -> List[str]:
        """Suppliers that are inaccessible under 'scenario' but were procured from."""
        return [
            s
            for s in self.suppliers
            if self.sp_available.get((s, product_id), False)
            and scenario.get_supplier_accessible(self.supplier_subtype.get(s, "general"))
            == 0
        ]

    # ------------------------------------------------------------------
    def build_model(self) -> Tuple[LpProblem, Dict]:
        """
        Build the extensive-form two-stage stochastic MILP.

        Variable count (approx for default scale):
          Stage 1: 60 continuous (x) + 60 binary (y)
          Stage 2: 5 × 10 = 50 each for e, u

        Constraint count (approx):
          Stage 1: ~80 (capacity + MOQ + baseline + overstock)
          Stage 2: ~100 per scenario × 5 = ~500

        Estimated solve time on i5-1135G7: < 1 min.
        """
        print("\nBuilding two-stage stochastic MILP (FIXED)…")

        model = LpProblem("Stochastic_Procurement_Fixed", LpMinimize)

        # ── STAGE 1 VARIABLES ─────────────────────────────────────────
        x = LpVariable.dicts(
            "x",
            ((s, p) for s in self.suppliers for p in self.products),
            lowBound=0,
            cat="Continuous",
        )
        y = LpVariable.dicts(
            "y",
            ((s, p) for s in self.suppliers for p in self.products),
            cat="Binary",
        )

        # ── STAGE 2 VARIABLES ─────────────────────────────────────────
        e = LpVariable.dicts(
            "e",  # emergency procurement
            (
                (k, p)
                for k in range(len(self.scenarios))
                for p in self.products
            ),
            lowBound=0,
            cat="Continuous",
        )
        u = LpVariable.dicts(
            "u",  # unmet demand
            (
                (k, p)
                for k in range(len(self.scenarios))
                for p in self.products
            ),
            lowBound=0,
            cat="Continuous",
        )

        # ── OBJECTIVE ─────────────────────────────────────────────────
        # Stage 1 variable procurement cost
        stage1_var = lpSum(
            self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False)
        )

        # Stage 1 fixed ordering cost
        stage1_fix = lpSum(
            self.supplier_fixed_cost[s] * y[s, p]
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False)
        )

        stage2_terms = []
        for k, sc in enumerate(self.scenarios):
            prob = sc.probability

            # Emergency cost: 2× normal unit cost
            em_cost = lpSum(
                2.0 * self.product_cost[p] * e[k, p]
                for p in self.products
            )

            # Unmet demand penalty: 10× unit cost
            penalty_mult = min(10.0, 5.0 * sc.spoilage_multiplier)
            unmet_cost = lpSum(
                penalty_mult * self.product_cost[p] * u[k, p]
                for p in self.products
            )

            # [P-5 FIX] Spoilage cost: goods from inaccessible suppliers
            # sp_{k,p} = Σ_{s: a_{k,s}=0} x[s,p]  (Stage 1 variables only)
            spoilage_cost = lpSum(
                self.sp_cost.get((s, p), self.product_cost[p]) * x[s, p]
                for p in self.products
                for s in self._get_inaccessible_suppliers(sc, p)
            )

            stage2_terms.append(prob * (em_cost + unmet_cost + spoilage_cost))

        model += stage1_var + stage1_fix + lpSum(stage2_terms), "Total_Expected_Cost"

        # ── STAGE 1 CONSTRAINTS ───────────────────────────────────────
        M = 100_000

        # Supplier weight capacity
        for s in self.suppliers:
            model += (
                lpSum(
                    x[s, p] * self.product_weight[p]
                    for p in self.products
                    if self.sp_available.get((s, p), False)
                )
                <= self.supplier_capacity[s],
                f"S1_Cap_{s}",
            )

        # MOQ logic (binary activation)
        for s in self.suppliers:
            for p in self.products:
                if self.sp_available.get((s, p), False):
                    moq = self.sp_moq.get((s, p), 0)
                    model += (x[s, p] >= moq * y[s, p], f"S1_MOQ_lo_{s}_{p}")
                    model += (x[s, p] <= M * y[s, p], f"S1_MOQ_hi_{s}_{p}")

        # [P-1 FIX] Baseline procurement: Σ x[s,p] ≥ β·D_p
        for p in self.products:
            demand = self.total_demand.get(p, 0)
            if demand > 0:
                model += (
                    lpSum(
                        x[s, p]
                        for s in self.suppliers
                        if self.sp_available.get((s, p), False)
                    )
                    >= self.baseline_ratio * demand,
                    f"S1_Baseline_{p}",
                )
                # [P-2 FIX] Overstock prevention (replaces z_i safety stock)
                model += (
                    lpSum(
                        x[s, p]
                        for s in self.suppliers
                        if self.sp_available.get((s, p), False)
                    )
                    <= (1 + self.overstock_delta.get(p, 0.20)) * demand,
                    f"S1_Overstock_{p}",
                )

        # ── STAGE 2 CONSTRAINTS ───────────────────────────────────────
        for k, sc in enumerate(self.scenarios):
            # Emergency capacity bound
            # [V-1 prep FIX] e=0 when emergency is physically impossible (Level 5)
            for p in self.products:
                demand = self.total_demand.get(p, 0)
                em_cap = self.emergency_ratio * demand * (1 if sc.emergency_feasible else 0)
                model += (e[k, p] <= em_cap, f"S2_EmCap_{k}_{p}")

            # [W-3 FIX / P-3 FIX] Demand satisfaction via accessible supply only
            # accessible_supply_k_p = Σ_{s: a_{k,s}=1} x[s,p]  (Stage 1 vars, subsetted)
            # sp_{k,p}              = Σ_{s: a_{k,s}=0} x[s,p]  (implicit, in objective)
            # Balance: accessible_supply + e + u ≥ D_p
            for p in self.products:
                demand = self.total_demand.get(p, 0)
                if demand > 0:
                    accessible_suppliers = self._get_accessible_suppliers(sc, p)
                    accessible_supply = lpSum(x[s, p] for s in accessible_suppliers)

                    model += (
                        accessible_supply + e[k, p] + u[k, p] >= demand,
                        f"S2_Demand_{k}_{p}",
                    )

        print(
            f"  ✓ Variables: {model.numVariables()}  |  Constraints: {model.numConstraints()}"
        )
        return model, {"x": x, "y": y, "e": e, "u": u}

    # ------------------------------------------------------------------
    def solve(
        self,
        time_limit: int = 600,
        gap_tolerance: float = 0.02,
    ) -> Tuple[str, Dict]:
        model, vars_dict = self.build_model()

        solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap_tolerance, msg=1)

        print(f"\nSolving (time limit={time_limit}s, gap={gap_tolerance*100:.0f}%)…")
        t0 = time.time()
        model.solve(solver)
        solve_time = time.time() - t0

        status = LpStatus[model.status]
        print(f"  Status:    {status}")
        print(f"  Solve time: {solve_time:.2f}s")

        if status in ("Optimal", "Feasible"):
            obj = value(model.objective)
            print(f"  Objective: {obj:,.0f} VND")

            solution = self._extract_solution(vars_dict)
            solution["objective_value"] = obj
            solution["solve_time"] = solve_time
            solution["status"] = status
            solution["scenario_costs"] = self._compute_scenario_costs(vars_dict)
            return status, solution

        print(f"  ⚠ Solver status: {status}")
        return status, {}

    # ------------------------------------------------------------------
    def _extract_solution(self, vars_dict: Dict) -> Dict:
        x, y, e, u = vars_dict["x"], vars_dict["y"], vars_dict["e"], vars_dict["u"]

        stage1 = []
        for s in self.suppliers:
            for p in self.products:
                qty = value(x[s, p])
                if qty and qty > 0.01:
                    stage1.append(
                        {
                            "supplier_id": s,
                            "product_id": p,
                            "quantity_units": round(qty, 2),
                            "cost_vnd": round(
                                qty * self.sp_cost.get((s, p), self.product_cost[p]), 0
                            ),
                        }
                    )

        stage2_recourse = {}
        for k, sc in enumerate(self.scenarios):
            em_list, unmet_list = [], []
            for p in self.products:
                eq = value(e[k, p])
                uq = value(u[k, p])
                if eq and eq > 0.01:
                    em_list.append({"product_id": p, "quantity_units": round(eq, 2)})
                if uq and uq > 0.01:
                    unmet_list.append({"product_id": p, "unmet_quantity": round(uq, 2)})
            stage2_recourse[sc.name] = {
                "scenario_id": sc.scenario_id,
                "severity_level": sc.severity_level,
                "probability": sc.probability,
                "emergency_procurement": pd.DataFrame(em_list) if em_list else pd.DataFrame(),
                "unmet_demand": pd.DataFrame(unmet_list) if unmet_list else pd.DataFrame(),
            }

        supplier_usage = []
        for s in self.suppliers:
            used = [p for p in self.products if value(y[s, p]) and value(y[s, p]) > 0.5]
            if used:
                supplier_usage.append(
                    {
                        "supplier_id": s,
                        "products_supplied": used,
                        "num_products": len(used),
                    }
                )

        return {
            "stage1_procurement": pd.DataFrame(stage1),
            "stage2_recourse": stage2_recourse,
            "supplier_usage": pd.DataFrame(supplier_usage),
        }

    def _compute_scenario_costs(self, vars_dict: Dict) -> pd.DataFrame:
        x, y, e, u = vars_dict["x"], vars_dict["y"], vars_dict["e"], vars_dict["u"]

        # Stage 1 costs are scenario-independent
        s1_var = sum(
            (value(x[s, p]) or 0) * self.sp_cost.get((s, p), self.product_cost[p])
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False)
        )
        s1_fix = sum(
            (value(y[s, p]) or 0) * self.supplier_fixed_cost[s]
            for s in self.suppliers
            for p in self.products
            if self.sp_available.get((s, p), False)
        )
        s1_total = s1_var + s1_fix

        rows = []
        for k, sc in enumerate(self.scenarios):
            em_cost = sum(
                2.0 * (value(e[k, p]) or 0) * self.product_cost[p]
                for p in self.products
            )
            penalty_mult = min(10.0, 5.0 * sc.spoilage_multiplier)
            unmet_cost = sum(
                penalty_mult * (value(u[k, p]) or 0) * self.product_cost[p]
                for p in self.products
            )
            # Spoilage loss (opportunity cost)
            sp_cost = sum(
                (value(x[s, p]) or 0)
                * self.sp_cost.get((s, p), self.product_cost[p])
                for p in self.products
                for s in self._get_inaccessible_suppliers(sc, p)
            )
            rows.append(
                {
                    "scenario_name": sc.name,
                    "severity_level": sc.severity_level,
                    "probability": sc.probability,
                    "stage1_cost": s1_total,
                    "emergency_cost": em_cost,
                    "spoilage_cost": sp_cost,
                    "penalty_cost": unmet_cost,
                    "total_cost": s1_total + em_cost + sp_cost + unmet_cost,
                }
            )
        return pd.DataFrame(rows)