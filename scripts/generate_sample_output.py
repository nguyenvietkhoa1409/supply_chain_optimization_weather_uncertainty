"""
generate_sample_output.py
──────────────────────────────────────────────────────────────────────────────
Reads optimisation results from results/ and produces three human-readable
output files that demonstrate the full PDP solution for the thesis.

Outputs
───────
results/sample_procurement_plan.txt   – Stage-1 procurement per supplier
                                        + Herfindahl Index (HHI)
results/sample_vrp_routes.txt         – PDP route comparison:
                                        Normal Day vs Typhoon scenario
results/sample_recourse_comparison.csv– Cost-uplift table across all scenarios

Run AFTER run_stochastic_optimization.py.
"""

import json
import os
import sys

import pandas as pd


# ── paths ──────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.join(BASE, "..")
RES   = os.path.join(PROJ, "results")
DATA  = os.path.join(PROJ, "data", "synthetic")

PROC_CSV    = os.path.join(RES, "stochastic_procurement_fixed.csv")
SC_CSV      = os.path.join(RES, "scenario_costs_fixed.csv")
ROUTES_JSON = os.path.join(RES, "scenario_routes_fixed.json")
SUP_CSV     = os.path.join(DATA, "suppliers.csv")
PROD_CSV    = os.path.join(DATA, "product_catalog.csv")


def load_data():
    proc       = pd.read_csv(PROC_CSV)
    sc_costs   = pd.read_csv(SC_CSV)
    suppliers  = pd.read_csv(SUP_CSV)
    products   = pd.read_csv(PROD_CSV)

    routes = {}
    if os.path.exists(ROUTES_JSON):
        with open(ROUTES_JSON, encoding="utf-8") as f:
            routes = json.load(f)
    else:
        print(f"  ⚠  {ROUTES_JSON} not found — skipping VRP route report")

    sup_map  = suppliers.set_index("id")[["name", "subtype"]].to_dict("index")
    prod_map = {}
    for _, r in products.iterrows():
        pid = r.get("id") or r.get("product_id")
        nm  = r.get("name") or r.get("product_name") or pid
        prod_map[pid] = nm

    return proc, sc_costs, routes, sup_map, prod_map


# ── Output A: Procurement Plan ──────────────────────────────────────────────
def build_procurement_plan(proc: pd.DataFrame, sc_costs: pd.DataFrame,
                           sup_map: dict, prod_map: dict) -> str:
    lines = []
    W = 70
    lines.append("═" * W)
    lines.append("STAGE 1 PROCUREMENT PLAN  (RP Solution — Daily Cycle 2024-10-01)")
    lines.append("Two-Stage Stochastic MILP with PDP Routing")
    lines.append("═" * W)

    total_qty  = proc["quantity_units"].sum()
    total_cost = proc["cost_vnd"].sum()

    for sid, grp in proc.groupby("supplier_id"):
        meta     = sup_map.get(sid, {"name": sid, "subtype": "unknown"})
        s_qty    = grp["quantity_units"].sum()
        s_cost   = grp["cost_vnd"].sum()
        s_share  = 100 * s_qty / total_qty if total_qty > 0 else 0
        label    = "[General/Wholesale]" if meta.get("subtype") == "general" else \
                   f"[{meta.get('subtype','').capitalize()} Specialist]"
        lines.append(f"\n  Supplier: {sid} — {meta.get('name', sid)}  {label}")
        lines.append(f"  {'─'*65}")
        for _, row in grp.iterrows():
            pname = prod_map.get(row["product_id"], row["product_id"])
            lines.append(
                f"    {row['product_id']:<10} {pname:<28}"
                f"  {row['quantity_units']:>8.2f} units   {row['cost_vnd']:>14,.0f} VND"
            )
        lines.append(f"  {'─'*65}")
        lines.append(
            f"  Subtotal: {s_qty:>8.2f} units | {s_cost:>16,.0f} VND | "
            f"Share: {s_share:.1f}%"
        )

    # HHI
    shares = proc.groupby("supplier_id")["quantity_units"].sum() / total_qty
    hhi    = (shares ** 2).sum()
    if   hhi < 0.15: hhi_rating = "Highly diversified"
    elif hhi < 0.25: hhi_rating = "Moderately diversified"
    elif hhi < 0.40: hhi_rating = "Moderately concentrated"
    else:            hhi_rating = "Highly concentrated (risk!)"

    lines.append("\n" + "─" * W)
    lines.append("DIVERSIFICATION SUMMARY")
    lines.append("─" * W)
    lines.append(f"  Active Suppliers:         {proc['supplier_id'].nunique()}")
    lines.append(f"  Total Procured (units):   {total_qty:.2f}")
    lines.append(f"  Total Procurement Cost:   {total_cost:,.0f} VND")
    dom = proc.groupby("supplier_id")["quantity_units"].sum().idxmax()
    lines.append(
        f"  Max Single-Supplier Qty:  "
        f"{proc.groupby('supplier_id')['quantity_units'].sum().max():.2f} units "
        f"({sup_map.get(dom,{}).get('name', dom)})"
    )
    lines.append(f"  Herfindahl Index (HHI):   {hhi:.4f}  → {hhi_rating}")
    lines.append("  [HHI < 0.15 = Highly diversified | < 0.25 = Moderate | > 0.4 = Concentrated]")
    lines.append("═" * W)
    return "\n".join(lines)


# ── Output B: VRP Route Comparison ────────────────────────────────────────
def build_vrp_routes(routes: dict, proc: pd.DataFrame, prod_map: dict) -> str:
    if not routes:
        return "(No route data — run run_stochastic_optimization.py first)"

    W = 72
    lines = []
    lines.append("═" * W)
    lines.append("VRP PICKUP-AND-DELIVERY (PDP) ROUTE REPORT — SCENARIO COMPARISON")
    lines.append("Architecture: Company vehicles DC → Suppliers (pickup) → Stores (deliver) → DC")
    lines.append("═" * W)

    # Pick best (normal) and worst (typhoon) scenarios
    best_key     = next((k for k in routes if "Normal" in k or "normal" in k), None)
    typhoon_key  = next((k for k in routes if "Typhoon" in k or "Tropical" in k), None)
    chosen_keys  = [k for k in [best_key, typhoon_key] if k is not None]
    if not chosen_keys:
        chosen_keys = list(routes.keys())[:2]

    for sc_name in chosen_keys:
        sc_routes = routes[sc_name]
        sev_icon  = "✅" if "Normal" in sc_name or "Light" in sc_name else "⚠️ "
        lines.append(f"\n{sev_icon} SCENARIO: {sc_name}")
        lines.append(f"   {'─'*68}")

        if not sc_routes:
            lines.append("   🚫 NO ACTIVE VEHICLES — all fleet grounded (weather severity too high)")
            lines.append("      → All stores: UNSERVED | Penalty cost applies")
            lines.append(f"   {'─'*68}")
            continue

        lines.append(f"   Active vehicles: {len(sc_routes)}")
        for route in sc_routes:
            vid    = route["vehicle_id"]
            vtype  = route["vehicle_type"]
            refrig = "❄ Refrigerated" if route["refrigerated"] else "Standard"
            seq    = " → ".join(route["route_sequence"])
            lines.append(f"\n   Vehicle V-{vid:02d} [{vtype.capitalize()} | {refrig}]")
            lines.append(f"     Route: {seq}")
            lines.append(f"     Stops:")
            for stop in route["stops"]:
                arr = f"  (arrive {stop['arrival_hour']:.2f}h)" if stop.get("arrival_hour") else ""
                if stop["node_type"] == "supplier_pickup":
                    items = ", ".join(
                        f"{prod_map.get(it['product_id'], it['product_id'])}: {it['quantity']:.1f}"
                        for it in stop.get("pickups", [])
                    )
                    lines.append(f"       📦 PICKUP  @ {stop['node']}{arr}")
                    lines.append(f"              {items}")
                else:
                    items = ", ".join(
                        f"{prod_map.get(it['product_id'], it['product_id'])}: {it['quantity']:.1f}"
                        for it in stop.get("deliveries", [])
                    )
                    lines.append(f"       🏪 DELIVER @ {stop['node']}{arr}")
                    lines.append(f"              {items}")
        lines.append(f"\n   {'─'*68}")

    lines.append("═" * W)
    return "\n".join(lines)


# ── Output C: Recourse Comparison CSV ─────────────────────────────────────
def build_recourse_csv(sc_costs: pd.DataFrame) -> pd.DataFrame:
    df = sc_costs.copy()
    # Baseline = cheapest scenario (first Normal scenario)
    baseline_cost = df.loc[df["scenario_name"].str.contains("Normal", na=False), "total_cost"].min()
    if pd.isna(baseline_cost):
        baseline_cost = df["total_cost"].min()

    total_vehicles = df["n_operable_vehicles"].max()
    df["n_grounded_vehicles"] = total_vehicles - df["n_operable_vehicles"]
    df["cost_vs_baseline_pct"] = (
        (df["total_cost"] - baseline_cost) / baseline_cost * 100
    ).round(2)
    df["cost_uplift_label"] = df["cost_vs_baseline_pct"].apply(
        lambda v: f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"
    )

    out_cols = [
        "scenario_name", "severity_level", "probability",
        "n_operable_vehicles", "n_grounded_vehicles",
        "vrp_fixed_cost", "vrp_variable_cost", "vrp_cost",
        "emergency_cost", "spoilage_cost", "penalty_cost", "total_cost",
        "cost_vs_baseline_pct", "cost_uplift_label",
    ]
    return df[[c for c in out_cols if c in df.columns]]


# ── main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SAMPLE OUTPUT GENERATOR — PDP Supply Chain Optimization")
    print("=" * 70)

    proc, sc_costs, routes, sup_map, prod_map = load_data()

    # Output A
    print("\n[1/3] Generating procurement plan…")
    plan_txt = build_procurement_plan(proc, sc_costs, sup_map, prod_map)
    outA = os.path.join(RES, "sample_procurement_plan.txt")
    with open(outA, "w", encoding="utf-8") as f:
        f.write(plan_txt)
    print(f"  ✓ Saved → {outA}")
    print(plan_txt)

    # Output B
    print("\n[2/3] Generating VRP route comparison…")
    route_txt = build_vrp_routes(routes, proc, prod_map)
    outB = os.path.join(RES, "sample_vrp_routes.txt")
    with open(outB, "w", encoding="utf-8") as f:
        f.write(route_txt)
    print(f"  ✓ Saved → {outB}")
    print(route_txt)

    # Output C
    print("\n[3/3] Generating recourse comparison CSV…")
    recourse_df = build_recourse_csv(sc_costs)
    outC = os.path.join(RES, "sample_recourse_comparison.csv")
    recourse_df.to_csv(outC, index=False)
    print(f"  ✓ Saved → {outC}")
    print(recourse_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
