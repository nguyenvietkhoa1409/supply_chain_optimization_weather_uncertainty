import sys
with open('d:/Food chain optimization/scripts/run_benders_optimization.py', encoding='utf-8') as f:
    content = f.read()

content = content.replace('SUB_TIME            = 300   # 5 mins per subproblem max\nSUB_GAP             = 0.05', 'LP_SUB_TIME         = 30\nMIP_SUB_TIME        = 90\nSUB_GAP             = 0.10')

old_solver_call = 'solver = get_solver(SUB_TIME, SUB_GAP, verbose)'

new_lp = 'solver = pulp.getSolver("GUROBI", msg=verbose, timeLimit=LP_SUB_TIME)'
new_mip = 'solver = pulp.getSolver("GUROBI", msg=verbose, timeLimit=MIP_SUB_TIME, gapRel=SUB_GAP, options=[("MIPFocus", 1), ("Presolve", 2), ("Heuristics", 0.8)])'

parts = content.split(old_solver_call)
if len(parts) == 3: # 1st occurrence (LP), 2nd occurrence (MIP)
    content = parts[0] + new_lp + parts[1] + new_mip + parts[2]
    with open('d:/Food chain optimization/scripts/run_benders_optimization.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS')
else:
    print('FAILED length', len(parts))
