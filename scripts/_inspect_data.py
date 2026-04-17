import pandas as pd
import os
import sys
sys.path.insert(0, r'D:\Food chain optimization\src')
from data_generation.fleet_config import VEHICLE_TYPES, expand_fleet

data_dir = r'D:\Food chain optimization\data\synthetic'

print('=== VEHICLE_TYPES ===')
for vt in VEHICLE_TYPES:
    print(f"  {vt['type_id']:25s} count={vt['count']} cap={vt['capacity_kg']}kg "
          f"refrig={vt.get('refrigerated', False)} fixed={vt.get('fixed_cost_vnd',0):,}")

print('\n=== EXPANDED FLEET (all vehicles) ===')
fleet = expand_fleet(VEHICLE_TYPES)
for v in fleet:
    print(f"  {v['vehicle_id']:10s} type={v['type_id']:25s} cap={v['capacity_kg']}kg "
          f"refrig={v.get('refrigerated', False)}")

print('\n=== supplier_product_matrix (available only) ===')
sp = pd.read_csv(os.path.join(data_dir, 'supplier_product_matrix.csv'))
avail = sp[sp['available'] == True]
pivot = avail.pivot_table(
    index='supplier_id', columns='product_id',
    values='moq_units', aggfunc='first'
).fillna(0).astype(int)
print(pivot.to_string())

print('\n=== daily_demand sample (2024-10-01) ===')
demand = pd.read_csv(os.path.join(data_dir, 'daily_demand.csv'))
day = demand[demand['date'] == '2024-10-01']
pivot2 = day.pivot_table(
    index='store_id', columns='product_id',
    values='demand_units', aggfunc='sum'
).fillna(0)
print(pivot2.to_string())
print(f"\nTotal demand units: {day['demand_units'].sum():.1f}")
