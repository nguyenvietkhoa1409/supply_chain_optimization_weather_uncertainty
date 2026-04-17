import pandas as pd
import json
import os

results = r'D:\Food chain optimization\results'

# 1. Stage 1 Procurement
proc = pd.read_csv(os.path.join(results, 'tp_stochastic_procurement.csv'))
print('=== STAGE 1 PROCUREMENT (Here-and-Now Decisions) ===')
print(proc.to_string(index=False))
print(f'\nTotal procurement cost (VND): {proc["cost_vnd"].sum():,.0f}')
print(f'Total quantity (units):       {proc["quantity_units"].sum():,.1f}')
print(f'Suppliers used:               {proc["supplier_id"].nunique()}')
print(f'Products covered:             {proc["product_id"].nunique()}')

# Concentration check
for pid in proc["product_id"].unique():
    sub = proc[proc["product_id"] == pid]
    total_q = sub["quantity_units"].sum()
    for _, row in sub.iterrows():
        ratio = row["quantity_units"] / total_q if total_q > 0 else 0
        flag = "  ← >40% CONCENTRATION" if ratio > 0.40 else ""
        print(f'  {pid} from {row["supplier_id"]}: {row["quantity_units"]:.1f} units ({ratio:.1%}){flag}')

# 2. Scenario Costs
sc = pd.read_csv(os.path.join(results, 'tp_scenario_costs.csv'))
print('\n=== SCENARIO COST BREAKDOWN ===')
cols = ['scenario_name','severity_level','probability',
        'stage1_cost','proc_vrp_cost','dist_vrp_cost',
        'spoilage_cost','emergency_cost','penalty_cost',
        'total_cost','n_operable_vehicles']
print(sc[cols].to_string(index=False))

print('\n--- Cost Component Analysis ---')
weighted_total = (sc['probability'] * sc['total_cost']).sum()
print(f'Expected Total Cost (E[TC]):       {weighted_total:,.0f} VND')
print(f'Weighted penalty cost:             {(sc["probability"] * sc["penalty_cost"]).sum():,.0f} VND')
print(f'Weighted proc VRP cost:            {(sc["probability"] * sc["proc_vrp_cost"]).sum():,.0f} VND')
print(f'Weighted dist VRP cost:            {(sc["probability"] * sc["dist_vrp_cost"]).sum():,.0f} VND')
print(f'Weighted spoilage cost:            {(sc["probability"] * sc["spoilage_cost"]).sum():,.0f} VND')
print(f'Weighted emergency cost:           {(sc["probability"] * sc["emergency_cost"]).sum():,.0f} VND')

# Scenario severity analysis
for _, row in sc.iterrows():
    pct_penalty = row["penalty_cost"] / row["total_cost"] * 100 if row["total_cost"] > 0 else 0
    pct_vrp = (row["proc_vrp_cost"] + row["dist_vrp_cost"]) / row["total_cost"] * 100 if row["total_cost"] > 0 else 0
    print(f'  [{row["scenario_name"][:30]}] sev={row["severity_level"]} ops={row["n_operable_vehicles"]} veh | penalty={pct_penalty:.1f}% | vrp={pct_vrp:.1f}%')

# 3. Routes
with open(os.path.join(results, 'tp_scenario_routes.json'), encoding='utf-8') as f:
    routes = json.load(f)

print('\n=== ROUTE STRUCTURE (Phase 2A + 2B) ===')
for sc_name, data in routes.items():
    inv = data.get('inventory', {})
    pr  = data.get('procurement_routes', [])
    dr  = data.get('distribution_routes', [])
    total_inv = sum(v for v in inv.values() if isinstance(v, (int, float)) and v > 0)
    print(f'\n  Scenario: {sc_name}')
    print(f'  Phase 2A   : {len(pr)} procurement vehicle(s) | DC inventory: {total_inv:.1f} units')
    for r in pr:
        route_str = ' -> '.join(r.get('route', []))
        pickups_flat = {p: q for s, prods in r.get('pickups', {}).items()
                        for p, q in prods.items()}
        print(f'    [{r.get("vehicle_type","?")}] {route_str}')
        print(f'      pickups: {pickups_flat}')
    print(f'  Phase 2B   : {len(dr)} distribution vehicle(s)')
    for r in dr:
        route_str = ' -> '.join(r.get('route', []))
        print(f'    [{r.get("vehicle_type","?")}] {route_str}')

print('\n=== INVENTORY BALANCE CHECK (Phase 2A → Phase 2B) ===')
print('(If inv=0 but dist routes exist → inventory-less delivery = gap in model)')
for sc_name, data in routes.items():
    inv = data.get('inventory', {})
    dr  = data.get('distribution_routes', [])
    non_zero_inv = {p: v for p, v in inv.items() if isinstance(v, (int, float)) and v > 0}
    print(f'  {sc_name}: inventory products={len(non_zero_inv)} | dist_vehicles={len(dr)}')
