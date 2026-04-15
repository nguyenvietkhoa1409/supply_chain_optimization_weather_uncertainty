"""Revert aggressive heuristic settings that inflated UB 3.5x."""
import re

with open('d:/Food chain optimization/scripts/run_benders_optimization.py', encoding='utf-8') as f:
    content = f.read()

# Fix 1: SUB_GAP back to 5%, MIP_SUB_TIME to 120s
old_consts = 'LP_SUB_TIME              = 30       # 30s LP relaxation\nMIP_SUB_TIME             = 90       # 90s MIP routing subproblem\nSUB_GAP                  = 0.10     # 10% Subproblem Gap (faster early iterations)'
new_consts = 'LP_SUB_TIME              = 30       # 30s LP relaxation (Barrier)\nMIP_SUB_TIME             = 120      # 120s MIP routing subproblem (was 300s, balanced)\nSUB_GAP                  = 0.05     # 5% MIP gap — must be tight or UB inflates'

if old_consts in content:
    content = content.replace(old_consts, new_consts, 1)
    print('Fixed constants: MIP_SUB_TIME=120, SUB_GAP=0.05')
else:
    print('WARNING: constants not found — checking actual content...')
    for line in content.split('\n'):
        if 'MIP_SUB_TIME' in line or 'SUB_GAP' in line or 'LP_SUB_TIME' in line:
            print(' ', repr(line))

# Fix 2: Remove Heuristics=0.8, keep MIPFocus=1, keep Presolve=2
old_mip = ('solver = pulp.getSolver("GUROBI", msg=verbose,\n'
           '                            timeLimit=MIP_SUB_TIME, gapRel=SUB_GAP,\n'
           '                            MIPFocus=1, Heuristics=0.8, Presolve=2)')
new_mip = ('solver = pulp.getSolver("GUROBI", msg=verbose,\n'
           '                            timeLimit=MIP_SUB_TIME, gapRel=SUB_GAP,\n'
           '                            MIPFocus=1, Presolve=2)')

if old_mip in content:
    content = content.replace(old_mip, new_mip, 1)
    print('Fixed MIP solver: removed Heuristics=0.8')
else:
    # Try finding getSolver around MIPFocus
    idx = content.find('MIPFocus=1, Heuristics=0.8')
    if idx != -1:
        start = content.rfind('\n', 0, idx) + 1
        end = content.find('\n', idx) + 1
        print('MIP solver line:', repr(content[start:end]))
    else:
        print('No Heuristics=0.8 found — may already be clean')

with open('d:/Food chain optimization/scripts/run_benders_optimization.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('\nDone. Final getSolver calls:')
for m in re.finditer(r'getSolver\([^\)]+\)', content):
    lineno = content[:m.start()].count('\n') + 1
    print(f'  L{lineno}: {m.group()[:90]}')
