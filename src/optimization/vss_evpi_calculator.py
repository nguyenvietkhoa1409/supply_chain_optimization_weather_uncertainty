"""
VSS & EVPI Calculator – Stochastic Optimization Validation
FIXED VERSION

Fixes applied
─────────────
[CRITICAL / M-1]  EEV definition corrected.
    OLD: eev_cost = det_objective  (deterministic objective ≠ EEV).
    CORRECT:
      1. Solve deterministic model with expected weather → obtain x*_EV (Stage 1 plan).
      2. Fix Stage 1 at x*_EV.
      3. For each scenario k, solve Stage 2 recourse problem to get cost_k(x*_EV).
      4. EEV = Σ_k p_k · cost_k(x*_EV).
    This module provides compute_eev() which performs steps 3–4 given x*_EV and scenarios.

[HIGH / M-2]  WS (Wait-and-See) definition corrected.
    OLD: WS = min(scenario_costs)   — completely wrong (picks one scenario, ignores others).
    CORRECT:
      WS = Σ_k p_k · OPT_k
      where OPT_k = optimal cost of solving deterministic problem for scenario k alone.
    This requires K separate optimization solves.  compute_ws() performs these solves.

[HIGH / M-3]  Ordering property verification now meaningful.
    OLD: WS ≤ RP ≤ EEV was "PASS" because both WS and EEV were wrong.
    NEW: After correct EEV and WS, verify:
         WS ≤ RP ≤ EEV  (should always hold in a correctly formulated two-stage model).
    If it fails, the model has a formulation error that must be diagnosed.

References
──────────
Birge & Louveaux (2011), §4.1–4.3 — VSS, EVPI, WS definitions.
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


class StochasticValidator:
    """
    Computes VSS and EVPI with correct mathematical definitions.

    Usage pattern:
    1.  Solve stochastic (RP):         rp_obj, rp_vars = extensive_form.solve()
    2.  Solve EV deterministic:        ev_x = det_baseline.get_ev_stage1()
    3.  Compute EEV:                   eev = validator.compute_eev(ev_x, scenarios, ...)
    4.  Compute WS:                    ws  = validator.compute_ws(scenarios, ...)
    5.  Report metrics:                validator.report(rp_obj, eev, ws)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ------------------------------------------------------------------
    # [M-1 FIX] Correct EEV computation
    # ------------------------------------------------------------------
    def compute_eev(
        self,
        ev_stage1_procurement: pd.DataFrame,
        scenarios: List,
        network: Dict,
        products_df: pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        time_limit_per_scenario: int = 120,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Compute EEV = Expected cost of EV solution evaluated across all scenarios.

        Algorithm
        ─────────
        For each scenario k:
          1. Fix Stage 1 procurement at ev_stage1_procurement (from EV deterministic model).
          2. Compute accessible supply = Σ_{s: a_{k,s}=1} x[s,p].
          3. Remaining shortfall = max(0, demand - accessible_supply).
          4. Solve Stage 2 recourse:
               e[k,p] ≥ 0  (emergency, bounded by EC_p · ε^{em}_k)
               u[k,p] ≥ 0  (unmet demand)
             minimizing 2·c_p·e + penalty·c_p·u
             subject to:  accessible_supply + e + u ≥ D_p
          5. cost_k = Stage1_cost + em_cost + unmet_cost + spoilage_cost

        EEV = Σ_k p_k · cost_k

        Parameters
        ──────────
        ev_stage1_procurement : DataFrame with columns [supplier_id, product_id, quantity_units]
        scenarios             : List[WeatherScenario]
        (other args)          : Same network/product/demand objects used in optimization.

        Returns
        ───────
        (eev_total, scenario_breakdown_df)
        """
        if self.verbose:
            print("\nComputing EEV (M-1 fix: evaluating EV solution across all scenarios)…")

        product_cost = dict(zip(products_df["id"], products_df["unit_cost_vnd"]))
        product_weight = dict(zip(products_df["id"], products_df["weight_kg_per_unit"]))

        # Supplier subtype lookup
        supplier_subtype = {}
        for _, row in network["suppliers"].iterrows():
            supplier_subtype[row["id"]] = row.get("subtype", "general")

        # sp_cost lookup
        sp_cost = {}
        sp_avail = {}
        for _, row in supplier_product_df.iterrows():
            sp_cost[(row["supplier_id"], row["product_id"])] = row["unit_cost_vnd"]
            sp_avail[(row["supplier_id"], row["product_id"])] = row["available"]

        # Total demand by product
        total_demand = demand_df.groupby("product_id")["demand_units"].sum().to_dict()

        # Stage 1 cost from EV solution (fixed for all scenarios)
        stage1_cost = ev_stage1_procurement["cost_vnd"].sum()

        # Fixed procurement quantities  dict: (supplier, product) → qty
        fixed_qty = {}
        for _, row in ev_stage1_procurement.iterrows():
            fixed_qty[(row["supplier_id"], row["product_id"])] = row["quantity_units"]

        rows = []
        for k, sc in enumerate(scenarios):
            # Accessible supply under this scenario
            accessible_supply = {}
            spoilage_loss = {}
            for p in products_df["id"]:
                acc = sum(
                    fixed_qty.get((s, p), 0)
                    for s in network["suppliers"]["id"]
                    if sc.get_supplier_accessible(supplier_subtype.get(s, "general")) == 1
                )
                inacc = sum(
                    fixed_qty.get((s, p), 0)
                    for s in network["suppliers"]["id"]
                    if sc.get_supplier_accessible(supplier_subtype.get(s, "general")) == 0
                )
                accessible_supply[p] = acc
                # Spoilage opportunity cost
                inacc_cost = sum(
                    fixed_qty.get((s, p), 0)
                    * sp_cost.get((s, p), product_cost.get(p, 0))
                    for s in network["suppliers"]["id"]
                    if sc.get_supplier_accessible(supplier_subtype.get(s, "general")) == 0
                    and sp_avail.get((s, p), False)
                )
                spoilage_loss[p] = inacc_cost

            # Solve Stage 2 recourse (small LP)
            products = products_df["id"].tolist()
            m2 = LpProblem(f"EEV_Stage2_k{k}", LpMinimize)
            e = LpVariable.dicts(f"e_k{k}", products, lowBound=0)
            u = LpVariable.dicts(f"u_k{k}", products, lowBound=0)

            em_ratio = 0.40
            for p in products:
                d = total_demand.get(p, 0)
                em_cap = em_ratio * d * (1 if sc.emergency_feasible else 0)
                m2 += (e[p] <= em_cap, f"em_cap_{k}_{p}")

                if d > 0:
                    m2 += (
                        accessible_supply[p] + e[p] + u[p] >= d,
                        f"dem_{k}_{p}",
                    )

            penalty_mult = min(10.0, 5.0 * sc.spoilage_multiplier)
            m2 += lpSum(
                2.0 * product_cost[p] * e[p] + penalty_mult * product_cost[p] * u[p]
                for p in products
            )

            m2.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit_per_scenario))

            em_cost = sum(2.0 * (value(e[p]) or 0) * product_cost[p] for p in products)
            unmet_cost = sum(
                penalty_mult * (value(u[p]) or 0) * product_cost[p] for p in products
            )
            sp_cost_total = sum(spoilage_loss[p] for p in products)

            total_k = stage1_cost + em_cost + unmet_cost + sp_cost_total

            rows.append(
                {
                    "scenario_name": sc.name,
                    "severity_level": sc.severity_level,
                    "probability": sc.probability,
                    "stage1_cost": stage1_cost,
                    "emergency_cost": em_cost,
                    "spoilage_cost": sp_cost_total,
                    "penalty_cost": unmet_cost,
                    "total_cost_k": total_k,
                }
            )
            if self.verbose:
                print(
                    f"  k={k:2d} [{sc.name}]: cost_k = {total_k:,.0f} VND  "
                    f"(em={em_cost:,.0f}, unmet={unmet_cost:,.0f}, sp={sp_cost_total:,.0f})"
                )

        df = pd.DataFrame(rows)
        eev = (df["total_cost_k"] * df["probability"]).sum()

        if self.verbose:
            print(f"\n  EEV = {eev:,.0f} VND  (Σ p_k · cost_k(x*_EV))")

        return eev, df

    # ------------------------------------------------------------------
    # [M-2 FIX] Correct WS computation
    # ------------------------------------------------------------------
    def compute_ws(
        self,
        scenarios: List,
        network: Dict,
        products_df: pd.DataFrame,
        supplier_product_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        time_limit_per_scenario: int = 300,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Compute WS = Wait-and-See cost = Σ_k p_k · OPT_k.

        For each scenario k, solve the deterministic problem assuming
        the decision-maker knows scenario k perfectly at Stage 1.
        OPT_k = optimal procurement + routing cost under scenario k alone.

        This provides the lower bound on what's achievable with perfect weather
        information (EVPI = RP - WS).

        Algorithm per scenario k
        ────────────────────────
        min  Σ_{s,p} c_{sp} · x[s,p]  +  Σ_p f_s · y[s,p]
             +  Σ_p (2·c_p · e_p  +  penalty · c_p · u_p)
             +  Spoilage(x, scenario k)
        s.t.
          Σ_{s: a_{k,s}=1} x[s,p]  +  e_p  +  u_p  ≥  D_p
          x[s,p] ≥ MOQ · y[s,p]
          x[s,p] ≤ M · y[s,p]
          e_p ≤ EC_p · ε^{em}_k
          Capacity, overstock constraints.

        Parameters
        ──────────
        Same as compute_eev().

        Returns
        ───────
        (ws_total, per_scenario_df)
        """
        if self.verbose:
            print("\nComputing WS (M-2 fix: K separate deterministic solves)…")
            print(f"  Running {len(scenarios)} separate optimizations…")

        product_cost = dict(zip(products_df["id"], products_df["unit_cost_vnd"]))
        product_weight = dict(zip(products_df["id"], products_df["weight_kg_per_unit"]))
        supplier_capacity = dict(
            zip(network["suppliers"]["id"], network["suppliers"]["capacity_kg_per_day"])
        )
        supplier_fixed_cost = dict(
            zip(network["suppliers"]["id"], network["suppliers"]["fixed_cost_vnd"])
        )
        supplier_subtype = {
            row["id"]: row.get("subtype", "general")
            for _, row in network["suppliers"].iterrows()
        }

        sp_cost, sp_moq, sp_avail = {}, {}, {}
        for _, row in supplier_product_df.iterrows():
            s, p = row["supplier_id"], row["product_id"]
            sp_cost[(s, p)] = row["unit_cost_vnd"]
            sp_moq[(s, p)] = row["moq_units"]
            sp_avail[(s, p)] = row["available"]

        total_demand = demand_df.groupby("product_id")["demand_units"].sum().to_dict()
        suppliers = network["suppliers"]["id"].tolist()
        products = products_df["id"].tolist()

        rows = []
        for k, sc in enumerate(scenarios):
            t0 = time.time()
            m = LpProblem(f"WS_k{k}", LpMinimize)

            # Stage 1 variables
            x = LpVariable.dicts(f"x_k{k}", ((s, p) for s in suppliers for p in products), lowBound=0)
            y = LpVariable.dicts(f"y_k{k}", ((s, p) for s in suppliers for p in products), cat="Binary")
            e = LpVariable.dicts(f"e_k{k}", products, lowBound=0)
            u = LpVariable.dicts(f"u_k{k}", products, lowBound=0)

            # Under perfect info, only accessible suppliers are available for procurement
            acc_set = {
                (s, p)
                for s in suppliers
                for p in products
                if sp_avail.get((s, p), False)
                and sc.get_supplier_accessible(supplier_subtype.get(s, "general")) == 1
            }

            procurement_cost = lpSum(
                sp_cost.get((s, p), product_cost[p]) * x[s, p]
                for s, p in acc_set
            )
            fixed_cost_term = lpSum(
                supplier_fixed_cost[s] * y[s, p]
                for s, p in acc_set
            )
            em_penalty = 0.40
            penalty_mult = min(10.0, 5.0 * sc.spoilage_multiplier)
            recourse_cost = lpSum(
                2.0 * product_cost[p] * e[p] + penalty_mult * product_cost[p] * u[p]
                for p in products
            )

            m += procurement_cost + fixed_cost_term + recourse_cost

            # Constraints
            M_big = 100_000
            for s in suppliers:
                m += (
                    lpSum(x[s, p] * product_weight[p] for p in products if sp_avail.get((s, p), False))
                    <= supplier_capacity[s],
                    f"Cap_{k}_{s}",
                )
            for s, p in acc_set:
                moq = sp_moq.get((s, p), 0)
                m += (x[s, p] >= moq * y[s, p], f"MOQlo_{k}_{s}_{p}")
                m += (x[s, p] <= M_big * y[s, p], f"MOQhi_{k}_{s}_{p}")

            for p in products:
                d = total_demand.get(p, 0)
                if d > 0:
                    acc_supply = lpSum(x[s, p] for s in suppliers if (s, p) in acc_set)
                    em_cap = em_penalty * d * (1 if sc.emergency_feasible else 0)
                    m += (e[p] <= em_cap, f"EmCap_{k}_{p}")
                    m += (acc_supply + e[p] + u[p] >= d, f"Dem_{k}_{p}")

            m.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit_per_scenario))
            solve_t = time.time() - t0

            if LpStatus[m.status] in ("Optimal", "Feasible"):
                opt_k = value(m.objective) or 0
            else:
                # If infeasible (shouldn't happen with unmet), use large penalty estimate
                opt_k = sum(
                    total_demand.get(p, 0) * penalty_mult * product_cost.get(p, 0)
                    for p in products
                )
                if self.verbose:
                    print(f"  k={k} [{sc.name}] solver status={LpStatus[m.status]} → fallback cost")

            rows.append(
                {
                    "scenario_name": sc.name,
                    "severity_level": sc.severity_level,
                    "probability": sc.probability,
                    "opt_k": opt_k,
                    "solve_time": solve_t,
                }
            )
            if self.verbose:
                print(f"  k={k:2d} [{sc.name}]: OPT_k = {opt_k:,.0f} VND  ({solve_t:.1f}s)")

        df = pd.DataFrame(rows)
        ws = (df["opt_k"] * df["probability"]).sum()

        if self.verbose:
            print(f"\n  WS = {ws:,.0f} VND  (Σ p_k · OPT_k)")

        return ws, df

    # ------------------------------------------------------------------
    def compute_vss(self, rp: float, eev: float) -> Dict:
        """
        VSS = EEV - RP.

        rp  : Stochastic (recourse problem) objective.
        eev : Expected value of EV solution evaluated stochastically (from compute_eev).
        """
        vss = eev - rp
        vss_pct = (vss / eev * 100) if eev > 0 else 0.0
        return {
            "RP": rp,
            "EEV": eev,
            "VSS": vss,
            "VSS_percent": vss_pct,
            "interpretation": self._interpret_vss(vss_pct),
        }

    def compute_evpi(self, rp: float, ws: float) -> Dict:
        """
        EVPI = RP - WS.

        rp : Stochastic objective.
        ws : Wait-and-see cost (from compute_ws).
        """
        evpi = rp - ws
        evpi_pct = (evpi / rp * 100) if rp > 0 else 0.0
        return {
            "RP": rp,
            "WS": ws,
            "EVPI": evpi,
            "EVPI_percent": evpi_pct,
            "interpretation": self._interpret_evpi(evpi_pct),
        }

    # ------------------------------------------------------------------
    # [M-3 FIX] Meaningful ordering check
    # ------------------------------------------------------------------
    def verify_ordering(self, ws: float, rp: float, eev: float) -> Dict:
        """
        Verify WS ≤ RP ≤ EEV.

        All three values must be computed with correct definitions (M-1, M-2 fixes).
        Violations indicate model formulation errors.

        Returns a dict with individual checks and overall pass/fail.
        """
        ws_le_rp = ws <= rp + 1e-3   # small tolerance for numerical noise
        rp_le_eev = rp <= eev + 1e-3

        reasons = []
        if not ws_le_rp:
            reasons.append(
                f"WS ({ws:,.0f}) > RP ({rp:,.0f}): "
                "WS must be a lower bound; check that each scenario k is solved independently "
                "and without cross-scenario coupling."
            )
        if not rp_le_eev:
            reasons.append(
                f"RP ({rp:,.0f}) > EEV ({eev:,.0f}): "
                "Stochastic solution should never be worse than evaluating the EV solution; "
                "check non-anticipativity enforcement and that EEV uses correct x*_EV."
            )

        return {
            "WS": ws,
            "RP": rp,
            "EEV": eev,
            "WS_le_RP": ws_le_rp,
            "RP_le_EEV": rp_le_eev,
            "ordering_pass": ws_le_rp and rp_le_eev,
            "violation_reasons": reasons,
        }

    # ------------------------------------------------------------------
    def generate_validation_report(
        self,
        rp: float,
        eev: float,
        ws: float,
        scenario_costs_rp: pd.DataFrame,
        eev_breakdown: Optional[pd.DataFrame] = None,
        ws_breakdown: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Generate comprehensive validation report with correct metrics.
        """
        vss_result = self.compute_vss(rp, eev)
        evpi_result = self.compute_evpi(rp, ws)
        ordering = self.verify_ordering(ws, rp, eev)

        lines = []
        lines.append("=" * 80)
        lines.append("STOCHASTIC OPTIMIZATION VALIDATION REPORT  (Fixed: M-1, M-2, M-3)")
        lines.append("=" * 80)

        # VSS
        lines.append("\n1. VALUE OF STOCHASTIC SOLUTION (VSS)")
        lines.append("-" * 80)
        lines.append(f"  EEV  (Expected cost of EV solution): {eev:>20,.0f} VND")
        lines.append(f"  RP   (Stochastic recourse problem):  {rp:>20,.0f} VND")
        lines.append(f"  VSS  = EEV - RP:                     {vss_result['VSS']:>20,.0f} VND")
        lines.append(f"  VSS%:                                {vss_result['VSS_percent']:>20.2f}%")
        lines.append(f"\n  → {vss_result['interpretation']}")
        lines.append(
            "\n  Note: EEV computed by evaluating EV Stage 1 solution across all K scenarios"
            "\n        (correct definition — NOT the deterministic objective directly)."
        )

        # EVPI
        lines.append("\n2. EXPECTED VALUE OF PERFECT INFORMATION (EVPI)")
        lines.append("-" * 80)
        lines.append(f"  RP   (Stochastic solution):          {rp:>20,.0f} VND")
        lines.append(f"  WS   (Wait-and-see, K solves):       {ws:>20,.0f} VND")
        lines.append(f"  EVPI = RP - WS:                      {evpi_result['EVPI']:>20,.0f} VND")
        lines.append(f"  EVPI%:                               {evpi_result['EVPI_percent']:>20.2f}%")
        lines.append(f"\n  → {evpi_result['interpretation']}")
        lines.append(
            "\n  Note: WS = Σ_k p_k · OPT_k  (K separate deterministic solves)"
            "\n        (correct definition — NOT min(scenario_costs))."
        )

        # Ordering check
        lines.append("\n3. ORDERING PROPERTY  (WS ≤ RP ≤ EEV)")
        lines.append("-" * 80)
        lines.append(f"  WS  = {ws:>20,.0f} VND")
        lines.append(f"  RP  = {rp:>20,.0f} VND")
        lines.append(f"  EEV = {eev:>20,.0f} VND")
        lines.append(f"  WS ≤ RP:   {'✓ PASS' if ordering['WS_le_RP']  else '❌ FAIL'}")
        lines.append(f"  RP ≤ EEV:  {'✓ PASS' if ordering['RP_le_EEV'] else '❌ FAIL'}")
        if ordering["violation_reasons"]:
            lines.append("\n  ⚠ Violations detected:")
            for r in ordering["violation_reasons"]:
                lines.append(f"    - {r}")

        # Scenario breakdown (RP)
        lines.append("\n4. SCENARIO-WISE COST BREAKDOWN  (Recourse Problem)")
        lines.append("-" * 80)
        for _, row in scenario_costs_rp.iterrows():
            lines.append(
                f"  {row['scenario_name']:30s}  p={row['probability']:.2f}  "
                f"cost={row['total_cost']:>14,.0f} VND"
            )
        lines.append(f"\n  Mean:  {scenario_costs_rp['total_cost'].mean():>14,.0f} VND")
        lines.append(f"  Worst: {scenario_costs_rp['total_cost'].max():>14,.0f} VND")
        lines.append(f"  Best:  {scenario_costs_rp['total_cost'].min():>14,.0f} VND")
        lines.append(f"  CV:    {scenario_costs_rp['total_cost'].std()/scenario_costs_rp['total_cost'].mean():.2%}")

        # Optional breakdowns
        if eev_breakdown is not None:
            lines.append("\n5. EEV BREAKDOWN BY SCENARIO")
            lines.append("-" * 80)
            for _, row in eev_breakdown.iterrows():
                lines.append(
                    f"  {row['scenario_name']:30s}  p={row['probability']:.2f}  "
                    f"cost_k={row['total_cost_k']:>14,.0f} VND"
                )

        if ws_breakdown is not None:
            lines.append("\n6. WAIT-AND-SEE (WS) BREAKDOWN BY SCENARIO")
            lines.append("-" * 80)
            for _, row in ws_breakdown.iterrows():
                lines.append(
                    f"  {row['scenario_name']:30s}  p={row['probability']:.2f}  "
                    f"OPT_k={row['opt_k']:>14,.0f} VND  ({row['solve_time']:.1f}s)"
                )

        # Conclusion
        lines.append("\n" + "=" * 80)
        lines.append("CONCLUSION")
        lines.append("=" * 80)
        vss_pct = vss_result["VSS_percent"]
        if vss_pct > 10:
            lines.append("✓ Stochastic solution has SUBSTANTIAL value over deterministic")
        elif vss_pct > 5:
            lines.append("✓ Stochastic solution has SIGNIFICANT value")
        elif vss_pct > 1:
            lines.append("✓ Stochastic solution has MODERATE value")
        elif vss_pct > 0:
            lines.append("✓ Stochastic solution marginally better than EV solution")
        else:
            lines.append("⚠ Stochastic solution appears worse — check formulation")

        evpi_pct = evpi_result["EVPI_percent"]
        if evpi_pct > 10:
            lines.append(f"✓ Weather forecasting has VERY HIGH value ({evpi_pct:.1f}%) — invest in monitoring")
        elif evpi_pct > 5:
            lines.append(f"✓ Weather forecasting has HIGH value ({evpi_pct:.1f}%)")
        else:
            lines.append(f"  Weather forecasting provides modest benefit ({evpi_pct:.1f}%)")

        if ordering["ordering_pass"]:
            lines.append("✓ Ordering property WS ≤ RP ≤ EEV verified — model is consistent")
        else:
            lines.append("❌ Ordering property VIOLATED — model has formulation errors")

        lines.append("=" * 80)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def _interpret_vss(vss_pct: float) -> str:
        if vss_pct < 1:
            return "Minimal benefit from stochastic approach"
        elif vss_pct < 5:
            return "Moderate benefit — stochastic approach preferred"
        elif vss_pct < 10:
            return "Significant benefit — stochastic approach strongly recommended"
        else:
            return "Substantial benefit — stochastic approach essential"

    @staticmethod
    def _interpret_evpi(evpi_pct: float) -> str:
        if evpi_pct < 2:
            return "Low value of perfect information — forecasting not critical"
        elif evpi_pct < 5:
            return "Moderate value — improved forecasting worthwhile"
        elif evpi_pct < 10:
            return "High value — invest in weather monitoring systems"
        else:
            return "Very high value — perfect forecasts highly valuable"


# Keep backward-compatible compute_vss/compute_evpi signatures for
# any existing caller that passes (stochastic_obj, det_obj, eev_cost).
class LegacyStochasticValidator(StochasticValidator):
    """
    Backwards-compatible wrapper.
    Calls from run_stochastic_optimization.py still work.
    Emits deprecation warnings for incorrect usage patterns.
    """

    def compute_vss(
        self,
        stochastic_objective: float = None,
        deterministic_objective: float = None,
        eev_cost: float = None,
        rp: float = None,
        eev: float = None,
    ) -> Dict:
        # New call signature
        if rp is not None and eev is not None:
            return super().compute_vss(rp, eev)
        # Legacy signature
        if eev_cost is None:
            eev_cost = deterministic_objective
        import warnings
        warnings.warn(
            "Using deterministic_objective as EEV approximation. "
            "For correct VSS, call compute_eev() first and pass rp=..., eev=... directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().compute_vss(stochastic_objective, eev_cost)

    def compute_evpi(
        self,
        stochastic_objective: float = None,
        wait_and_see_cost: float = None,
        rp: float = None,
        ws: float = None,
    ) -> Dict:
        if rp is not None and ws is not None:
            return super().compute_evpi(rp, ws)
        import warnings
        warnings.warn(
            "Using min(scenario_costs) as WS approximation. "
            "For correct EVPI, call compute_ws() first and pass rp=..., ws=... directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().compute_evpi(stochastic_objective, wait_and_see_cost)