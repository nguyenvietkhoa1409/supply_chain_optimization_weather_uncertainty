"""Revert to original working solver settings — just reduce time limit to 180s."""
import re

with open('d:/Food chain optimization/scripts/run_benders_optimization.py', encoding='utf-8') as f:
    content = f.read()

# Fix constants block
old_consts = ('LP_SUB_TIME              = 30       # 30s LP relaxation (Barrier)\n'
              'MIP_SUB_TIME             = 120      # 120s MIP routing subproblem (was 300s, balanced)\n'
              'SUB_GAP                  = 0.05     # 5% MIP gap — must be tight or UB inflates')
new_consts  = ('LP_SUB_TIME              = 30       # LP barrier solve (fast)\n'
               'MIP_SUB_TIME             = 180      # 3 min per MIP subproblem (balanced)\n'
               'SUB_GAP                  = 0.05     # 5% MIP gap (must keep tight — loose gap inflates UB!)')

# Fix MIP solver call (remove MIPFocus and Presolve overrides)
old_mip = ('solver = pulp.getSolver("GUROBI", msg=verbose,\n'
           '                            timeLimit=MIP_SUB_TIME, gapRel=SUB_GAP,\n'
           '                            MIPFocus=1, Presolve=2)')
new_mip  = ('solver = pulp.getSolver("GUROBI", msg=verbose,\n'
            '                            timeLimit=MIP_SUB_TIME, gapRel=SUB_GAP)')

changed = 0
if old_consts in content:
    content = content.replace(old_consts, new_consts, 1)
    changed += 1
    print("✅ Fixed constants: MIP_SUB_TIME=180s, SUB_GAP=0.05")
else:
    print("⚠ Constants not found")

if old_mip in content:
    content = content.replace(old_mip, new_mip, 1)
    changed += 1
    print("✅ Fixed MIP solver: removed MIPFocus=1, Presolve=2 (back to Gurobi defaults)")
else:
    print("⚠ MIP solver call not found")

with open('d:/Food chain optimization/scripts/run_benders_optimization.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nChanged {changed}/2 items.")
print("Final getSolver calls:")
for m in re.finditer(r'getSolver\([^\)]+\)', content):
    lineno = content[:m.start()].count('\n') + 1
    print(f"  L{lineno}: {m.group()[:100]}")
