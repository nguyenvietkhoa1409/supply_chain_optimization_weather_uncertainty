content = open('d:/Food chain optimization/scripts/run_benders_optimization.py', encoding='utf-8').read()

old = (
    '    model, arc, pickup_bal_constrs, ops, extras, build_status = result\n'
    '    depot, active_sups, sto_ids, prod_ids, pdp_nodes, valid_arcs, qp_vars = extras\n'
    '\n'
    '    if build_status == "NoVehicles":\n'
    '        total = sum(x_sol.get((s, p), 0) for s in suppliers["id"] for p in products["id"])\n'
    '        return total * UNMET_PENALTY_VND, "NoVehicles", {}'
)
new = (
    '    model, arc, pickup_bal_constrs, ops, extras, build_status = result\n'
    '\n'
    '    # MUST check build_status BEFORE unpacking extras (extras is None when NoVehicles)\n'
    '    if build_status == "NoVehicles":\n'
    '        total = sum(x_sol.get((s, p), 0) for s in suppliers["id"] for p in products["id"])\n'
    '        return total * UNMET_PENALTY_VND, "NoVehicles", {}\n'
    '\n'
    '    depot, active_sups, sto_ids, prod_ids, pdp_nodes, valid_arcs, qp_vars = extras'
)

if old in content:
    content = content.replace(old, new, 1)
    open('d:/Food chain optimization/scripts/run_benders_optimization.py', 'w', encoding='utf-8').write(content)
    print('OK: fix applied successfully')
else:
    print('Pattern not found — checking surrounding context:')
    idx = content.find('depot, active_sups, sto_ids')
    print(repr(content[idx-300:idx+200]))
