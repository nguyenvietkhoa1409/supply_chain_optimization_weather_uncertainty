"""Quick diagnostic of supplier/product/demand profiles."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'd:/Food chain optimization/src')

sup = pd.read_csv('d:/Food chain optimization/data/synthetic/suppliers.csv')
prod = pd.read_csv('d:/Food chain optimization/data/synthetic/products.csv')
sp = pd.read_csv('d:/Food chain optimization/data/synthetic/supplier_product_matrix.csv')
dem_full = pd.read_csv('d:/Food chain optimization/data/synthetic/daily_demand.csv')
dem = dem_full[dem_full['date'] == '2024-10-01']
dist = pd.read_csv('d:/Food chain optimization/data/synthetic/distance_matrix.csv', index_col=0)

print('=== PRODUCTS PER SUPPLIER ===')
for _, row in sup.iterrows():
    n = len(sp[sp['supplier_id'] == row['id']])
    print(f"  {row['id']} ({row['subtype']:12s}): {n:2d} products, cap={row['capacity_kg_per_day']:.0f} kg/day")

print('\n=== DEMAND vs CAPACITY per product ===')
for _, p in prod.iterrows():
    d_units = dem[dem['product_id'] == p['id']]['demand_units'].sum()
    d_kg = d_units * p['weight_kg_per_unit']
    sup_for_p = sp[sp['product_id'] == p['id']]['supplier_id'].tolist()
    total_cap = sum(sup[sup['id'] == s]['capacity_kg_per_day'].values[0] for s in sup_for_p)
    ratio = total_cap / d_kg if d_kg > 0 else 999
    pname = p['name'][:22]
    print(f"  {pname:22s}: demand={d_units:.0f}u={d_kg:.1f}kg  "
          f"n_sup={len(sup_for_p)}  total_cap={total_cap:.0f}kg  ratio={ratio:.1f}x")

print('\n=== VEHICLE CAPACITY vs TOTAL DEMAND ===')
dm = dem.merge(prod[['id', 'weight_kg_per_unit']], left_on='product_id', right_on='id')
total_kg = (dm['demand_units'] * dm['weight_kg_per_unit']).sum()
total_units = dem['demand_units'].sum()
total_fleet_kg = 3*300 + 2*1000 + 2*1500 + 3000
print(f"  Total demand: {total_units:.0f} units = {total_kg:.1f} kg")
print(f"  Total fleet:  {total_fleet_kg} kg")
print(f"  Utilization:  {total_kg/total_fleet_kg*100:.1f}%")
print(f"  Trucks needed (avg 1000kg/truck): {total_kg/1000:.1f}")

print('\n=== DISTANCE DISTRIBUTION (km) ===')
sup_cols = [c for c in dist.columns if str(c).startswith('SUP')]
sto_cols = [c for c in dist.columns if str(c).startswith('STORE')]
dc_rows  = [r for r in dist.index if str(r).startswith('DC')]
d_sup = dist.loc[dc_rows, sup_cols].values.flatten()
d_sto = dist.loc[dc_rows, sto_cols].values.flatten()
d_sup = d_sup[d_sup > 0]
d_sto = d_sto[d_sto > 0]
print(f"  DC-to-supplier: min={d_sup.min():.1f} max={d_sup.max():.1f} mean={d_sup.mean():.1f} km")
print(f"  DC-to-store:    min={d_sto.min():.1f} max={d_sto.max():.1f} mean={d_sto.mean():.1f} km")

# Supplier-to-supplier distances
d_ss = dist.loc[sup_cols, sup_cols].values.flatten()
d_ss = d_ss[d_ss > 0]
print(f"  Supplier-to-sup: min={d_ss.min():.1f} max={d_ss.max():.1f} mean={d_ss.mean():.1f} km")

print('\n=== SINGLE SUPPLIER DOMINANCE CHECK ===')
# If one supplier had ALL products, what % of total demand could it cover?
for _, row in sup.iterrows():
    prods_avail = sp[sp['supplier_id'] == row['id']]['product_id'].tolist()
    d_sup_kg = sum(
        dem[dem['product_id'] == p]['demand_units'].sum()
        * prod[prod['id'] == p]['weight_kg_per_unit'].values[0]
        for p in prods_avail
        if len(prod[prod['id'] == p]) > 0
    )
    pct_demand = d_sup_kg / total_kg * 100
    cap_util = row['capacity_kg_per_day'] / d_sup_kg * 100 if d_sup_kg > 0 else 0
    print(f"  {row['id']} ({row['subtype']:12s}): products={len(prods_avail)} "
          f"demand_covered={pct_demand:.1f}%  cap/demand={cap_util:.0f}%")
