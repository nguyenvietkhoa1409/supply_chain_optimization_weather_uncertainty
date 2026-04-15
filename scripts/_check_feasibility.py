import pandas as pd, sys
sys.path.insert(0, 'd:/Food chain optimization/src')

sp = pd.read_csv('d:/Food chain optimization/data/synthetic/supplier_product_matrix.csv')
prod = pd.read_csv('d:/Food chain optimization/data/synthetic/products.csv')

CONCENTRATION_MAX = 0.40
print(f'CONCENTRATION_MAX = {CONCENTRATION_MAX}')
print(f'Min suppliers needed = ceil(1/{CONCENTRATION_MAX}) = 3')
print()
print('Product Coverage Feasibility:')
infeasible_prods = []
for _, p in prod.iterrows():
    n_sup = len(sp[sp['product_id'] == p['id']])
    max_pct = n_sup * CONCENTRATION_MAX * 100
    ok = max_pct >= 100
    status = 'OK' if ok else '*** INFEASIBLE ***'
    pname = p['name']
    sups = sp[sp['product_id'] == p['id']]['supplier_id'].tolist()
    print(f"  [{status}] {pname[:25]:25s}: {n_sup} sup x 40% = {max_pct:.0f}%  suppliers={sups}")
    if not ok:
        infeasible_prods.append(p['id'])

print(f'\nConclusion: {len(infeasible_prods)} products cause master infeasibility:')
for pid in infeasible_prods:
    print(f'  -> {pid}: only {len(sp[sp["product_id"]==pid])} suppliers (need 3 with CONCENTRATION_MAX=0.40)')
print('\nFix: add 3rd supplier for each infeasible product, OR raise CONCENTRATION_MAX to 0.51')
