"""Fix incorrect PuLP GUROBI solver API calls."""
with open('d:/Food chain optimization/scripts/run_benders_optimization.py', encoding='utf-8') as f:
    content = f.read()

# LP solver fix: options=[("Method", 2)] → Method=2
old_lp = 'solver = pulp.getSolver(\'GUROBI\', msg=False, timeLimit=LP_SUB_TIME, options=[("Method", 2)])  # Force Barrier for duals'
new_lp = 'solver = pulp.getSolver("GUROBI", msg=False, timeLimit=LP_SUB_TIME, Method=2)'

# MIP solver fix: options=[(...)] → kwargs
old_mip = ('solver = pulp.getSolver(\'GUROBI\', msg=verbose,\n'
           '                            timeLimit=MIP_SUB_TIME, gapRel=MIP_SUB_GAP,\n'
           '                            options=[("MIPFocus", 1), ("Heuristics", 0.8), ("Presolve", 2)])')
new_mip_v2 = ('solver = pulp.getSolver("GUROBI", msg=verbose,\n'
              '                            timeLimit=MIP_SUB_TIME, gapRel=SUB_GAP,\n'
              '                            MIPFocus=1, Heuristics=0.8, Presolve=2)')

old_mip2 = ('solver = pulp.getSolver(\'GUROBI\', msg=verbose, timeLimit=MIP_SUB_TIME, gapRel=SUB_GAP, '
            'options=[("MIPFocus", 1), ("Presolve", 2), ("Heuristics", 0.8)])')
new_mip2 = ('solver = pulp.getSolver("GUROBI", msg=verbose, timeLimit=MIP_SUB_TIME, gapRel=SUB_GAP, '
            'MIPFocus=1, Presolve=2, Heuristics=0.8)')

changed = 0
if old_lp in content:
    content = content.replace(old_lp, new_lp, 1)
    changed += 1
    print("Fixed LP solver call")

if old_mip in content:
    content = content.replace(old_mip, new_mip_v2, 1)
    changed += 1
    print("Fixed MIP solver call (variant 1)")
elif old_mip2 in content:
    content = content.replace(old_mip2, new_mip2, 1)
    changed += 1
    print("Fixed MIP solver call (variant 2)")

if changed < 2:
    # Fallback: find and show current getSolver calls
    import re
    matches = [(m.start(), m.group()) for m in re.finditer(r'getSolver\([^\)]+\)', content)]
    print(f"\nOnly fixed {changed}/2. Remaining getSolver calls:")
    for pos, m in matches:
        lineno = content[:pos].count('\n') + 1
        print(f"  Line {lineno}: {m[:80]}")
else:
    with open('d:/Food chain optimization/scripts/run_benders_optimization.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("\nSUCCESS: Both solver calls fixed.")
