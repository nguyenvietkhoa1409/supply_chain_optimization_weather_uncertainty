"""
diagnostics/feasibility_check.py
Chẩn đoán data-level infeasibility độc lập với solver.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')

sup  = pd.read_csv(f'{DATA}/suppliers.csv')
sto  = pd.read_csv(f'{DATA}/stores.csv')
prod = pd.read_csv(f'{DATA}/products.csv')
dem  = pd.read_csv(f'{DATA}/daily_demand.csv')
sp   = pd.read_csv(f'{DATA}/supplier_product_matrix.csv')
dist = pd.read_csv(f'{DATA}/distance_matrix.csv', index_col=0)
dc   = pd.read_csv(f'{DATA}/distribution_centers.csv')

dem1 = dem[dem['date'] == '2024-10-01'].copy()

FLEET = [
    {'type': 'mini_van',    'count': 3, 'cap_kg': 300,  'refrig': False, 'max_sev': 3},
    {'type': 'light_truck', 'count': 2, 'cap_kg': 1000, 'refrig': False, 'max_sev': 4},
    {'type': 'ref_truck',   'count': 2, 'cap_kg': 1500, 'refrig': True,  'max_sev': 4},
    {'type': 'heavy_truck', 'count': 1, 'cap_kg': 3000, 'refrig': False, 'max_sev': 3},
]

W = 76
issues = []

print('=' * W)
print('DATA FEASIBILITY DIAGNOSTIC REPORT  (PDP Supply Chain)')
print('=' * W)

# ─────────────────────────────────────────────────────────────────────────────
# [1] Total demand vs total fleet capacity
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] TOTAL DEMAND vs FLEET CAPACITY (Normal weather)')
print('-' * W)

d_kg_merged = dem1.merge(prod[['id', 'weight_kg_per_unit']], left_on='product_id', right_on='id')
total_demand_kg  = (d_kg_merged['demand_units'] * d_kg_merged['weight_kg_per_unit']).sum()
total_fleet_cap  = sum(v['count'] * v['cap_kg'] for v in FLEET)
refrig_fleet_cap = sum(v['count'] * v['cap_kg'] for v in FLEET if v['refrig'])

ref_prod_ids = prod[prod['requires_refrigeration'] == True]['id'].tolist()
dem_ref      = dem1[dem1['product_id'].isin(ref_prod_ids)].merge(
    prod[['id', 'weight_kg_per_unit']], left_on='product_id', right_on='id')
ref_demand_kg = (dem_ref['demand_units'] * dem_ref['weight_kg_per_unit']).sum()

cap_ratio = total_demand_kg / total_fleet_cap
ref_ratio = ref_demand_kg   / refrig_fleet_cap

def flag(r, warn=0.80, err=1.0):
    if r >= err:  return '❌ EXCEEDS CAPACITY — INFEASIBLE'
    if r >= warn: return '⚠  TIGHT (>80% utilization)'
    return '✅ OK'

print(f'  Total demand (all products): {total_demand_kg:,.1f} kg')
print(f'  Total fleet capacity:        {total_fleet_cap:,} kg')
print(f'  Utilization:                 {cap_ratio:.1%}  {flag(cap_ratio)}')
print()
print(f'  Refrigerated demand only:    {ref_demand_kg:,.1f} kg')
print(f'  Refrigerated fleet cap:      {refrig_fleet_cap:,} kg (2 × ref_truck)')
print(f'  Ref utilization:             {ref_ratio:.1%}  {flag(ref_ratio, 0.80, 1.0)}')

if cap_ratio >= 1.0:
    issues.append('CRITICAL: Total demand exceeds total fleet capacity!')
if ref_ratio >= 1.0:
    issues.append('CRITICAL: Refrigerated demand exceeds refrigerated fleet capacity!')

# ─────────────────────────────────────────────────────────────────────────────
# [2] Per-product supply vs demand
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] SUPPLIER SUPPLY CAPACITY vs DEMAND — per product')
print('-' * W)
print(f"  {'Product':<28} {'Demand':>8} {'Dem_kg':>8} {'Sup_cap_kg':>11} {'Ratio':>7}  Status")
print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*11} {'-'*7}  ------")

for _, p in prod.iterrows():
    pid  = p['id']
    name = p['name'][:27]
    w    = p['weight_kg_per_unit']
    d    = dem1[dem1['product_id'] == pid]['demand_units'].sum()
    d_kg = d * w

    sup_ids = sp[sp['product_id'] == pid]['supplier_id'].tolist() \
              if 'product_id' in sp.columns else []
    avail_kg = sup[sup['id'].isin(sup_ids)]['capacity_kg_per_day'].sum()
    n_sups  = len(sup_ids)

    ratio = d_kg / avail_kg if avail_kg > 0 else float('inf')
    f = flag(ratio, warn=0.7, err=1.0)
    print(f'  {name:<28} {d:>8.1f} {d_kg:>8.1f} {avail_kg:>11.1f} {ratio:>7.2f}  {f}  (n_sup={n_sups})')

    if ratio >= 1.0:
        issues.append(f'CRITICAL: Supply < Demand for {pid} ({name}) — ratio={ratio:.2f}')
    if n_sups == 1:
        issues.append(f'RISK: Single-source supply for {pid} — if supplier inaccessible → infeasible!')

# ─────────────────────────────────────────────────────────────────────────────
# [3] concentration constraint vs supply
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3] CONCENTRATION CONSTRAINT (max 40% from one supplier) vs SUPPLY')
print('-' * W)
print('  If total supply available × 40% < demand → concentration prevents meeting demand!')
print()

for _, p in prod.iterrows():
    pid  = p['id']
    name = p['name'][:27]
    w    = p['weight_kg_per_unit']
    d    = dem1[dem1['product_id'] == pid]['demand_units'].sum()
    d_kg = d * w

    sup_ids = sp[sp['product_id'] == pid]['supplier_id'].tolist() \
              if 'product_id' in sp.columns else []
    sup_sub = sup[sup['id'].isin(sup_ids)].copy()

    # With concentration max 40% of total demand per supplier
    max_from_one = 0.40 * d  # units
    total_available = 0
    for _, s in sup_sub.iterrows():
        max_units = s['capacity_kg_per_day'] / w
        total_available += min(max_units, max_from_one)

    can_meet = total_available >= d
    f = '✅' if can_meet else '❌ CONCENTRATION MAKES THIS INFEASIBLE'
    print(f'  {pid} {name:<27}  demand={d:.0f}u  max_per_sup={max_from_one:.0f}u  '
          f'total_avail={total_available:.0f}u  {f}')

    if not can_meet:
        issues.append(f'CRITICAL: Concentration constraint prevents meeting demand for {pid}!')

# ─────────────────────────────────────────────────────────────────────────────
# [4] Store coverage vs fleet in bad weather
# ─────────────────────────────────────────────────────────────────────────────
print('\n[4] FLEET AVAILABLE per WEATHER SEVERITY')
print('-' * W)
n_stores = len(sto)

for sev in range(1, 6):
    operable = [v for v in FLEET if v['max_sev'] >= sev]
    n_veh    = sum(v['count'] for v in operable)
    # typical capacity reduction at each severity
    cap_factor = {1: 1.0, 2: 0.9, 3: 0.75, 4: 0.60, 5: 0.40}.get(sev, 1.0)
    eff_cap    = sum(v['count'] * v['cap_kg'] * cap_factor for v in operable)
    ref_eff    = sum(v['count'] * v['cap_kg'] * cap_factor for v in operable if v['refrig'])
    veh_ok = '✅' if n_veh >= n_stores else '⚠  Fewer vehicles than stores'
    print(f'  Sev {sev}: {n_veh} vehicles  eff_cap={eff_cap:,.0f}kg  ref_eff={ref_eff:,.0f}kg'
          f'  {veh_ok}')
    if n_veh > 0 and n_veh < n_stores:
        issues.append(f'WARNING: At sev={sev}, only {n_veh} vehicles for {n_stores} stores — '
                      f'must backtrack or combine routes!')

# ─────────────────────────────────────────────────────────────────────────────
# [5] Distance matrix completeness
# ─────────────────────────────────────────────────────────────────────────────
print('\n[5] DISTANCE MATRIX COMPLETENESS')
print('-' * W)
dc_ids  = dc['id'].tolist()
sup_ids = sup['id'].tolist()
sto_ids = sto['id'].tolist()
all_nodes = dc_ids + sup_ids + sto_ids

missing = [n for n in all_nodes if n not in dist.index]
print(f'  Expected nodes: {len(all_nodes)}  |  In matrix: {len(dist.index)}')
if missing:
    print(f'  ❌ MISSING from matrix: {missing}')
    issues.append(f'CRITICAL: Nodes missing from distance matrix: {missing}')
else:
    print(f'  ✅ All nodes present in distance matrix')

# Check for NaN or zero distances between different nodes
zero_pairs = [(i, j) for i in all_nodes for j in all_nodes
              if i != j and i in dist.index and j in dist.columns
              and dist.loc[i, j] == 0]
if zero_pairs:
    print(f'  ⚠  Zero-distance pairs (excluding self): {zero_pairs[:5]}...')
    issues.append(f'WARNING: {len(zero_pairs)} zero-distance pairs between different nodes')
else:
    print(f'  ✅ No spurious zero-distances found')

# ─────────────────────────────────────────────────────────────────────────────
# [6] PDP-specific: overload when all pickups precede all deliveries
# ─────────────────────────────────────────────────────────────────────────────
print('\n[6] PDP CARGO OVERLOAD: All pickups loaded before ANY delivery')
print('-' * W)
print('  In Pickup-First PDP, vehicle must carry ALL collected goods until first drop-off.')
print()

# Each vehicle: worst case carries all pickups at once before delivering
# max single-route demand if one vehicle covers all stores for one product
for _, p in prod.iterrows():
    pid  = p['id']
    name = p['name'][:26]
    w    = p['weight_kg_per_unit']
    d    = dem1[dem1['product_id'] == pid]['demand_units'].sum()
    d_kg = d * w

    refrig = p['requires_refrigeration']
    best_v = max((v for v in FLEET if (not refrig or v['refrig'])),
                 key=lambda v: v['cap_kg'], default=None)

    if best_v is None:
        print(f'  {pid} {name:<26}  ❌ NO COMPATIBLE VEHICLE (refrig mismatch!)')
        issues.append(f'CRITICAL: No compatible vehicle for {pid} (refrigeration mismatch)')
        continue

    if d_kg > best_v['cap_kg']:
        # needs multiple vehicles
        n_trips = int(np.ceil(d_kg / best_v['cap_kg']))
        print(f'  {pid} {name:<26}  {d_kg:>7.1f}kg  best_v_cap={best_v["cap_kg"]}kg  '
              f'needs {n_trips} trips/vehicles  ⚠ MULTI-VEHICLE REQUIRED')
    else:
        print(f'  {pid} {name:<26}  {d_kg:>7.1f}kg  best_v_cap={best_v["cap_kg"]}kg  ✅ fits one vehicle')

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * W)
print('SUMMARY OF ISSUES FOUND')
print('=' * W)
if not issues:
    print('  ✅ No data-level infeasibility detected — complexity is the only issue.')
else:
    for i, issue in enumerate(issues, 1):
        print(f'  [{i}] {issue}')
print('=' * W)
