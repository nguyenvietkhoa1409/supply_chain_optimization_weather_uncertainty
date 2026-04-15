import os
import re

benders_file = 'd:/Food chain optimization/scripts/run_benders_optimization.py'
with open(benders_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Add pandas import if missing
if 'import pandas as pd' not in content:
    content = content.replace('import numpy as np', 'import numpy as np\nimport pandas as pd\nimport os', 1)

# Add CSV export logical block
export_code = '''
    # --- EXPORT TO CSV FOR VISUALIZATION ---
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Procurement CSV
    proc_rows = []
    prod_info = products.set_index("id")
    for (s, p), qty in best_x.items():
        if qty > 0.01:
            unit_cost = prod_info.loc[p, "unit_cost_vnd"] if p in prod_info.index else 0
            pname = prod_info.loc[p, "name"] if p in prod_info.index else p
            proc_rows.append({
                "supplier_id": s,
                "product_id": p,
                "product_name": pname,
                "quantity_units": qty,
                "unit_cost_vnd": unit_cost,
                "total_cost_vnd": qty * unit_cost
            })
    pd.DataFrame(proc_rows).to_csv(os.path.join(results_dir, "benders_procurement_fixed.csv"), index=False)
    
    # 2. Scenario Costs CSV
    # Re-calculate costs
    stage1_cost = sum(r["total_cost_vnd"] for r in proc_rows)
    scen_rows = []
    for k, sc, Q_k, sub_status, routes in sub_results:
        # If Optimal/Feasible -> routing cost = Q_k, penalty = 0
        # If NoVehicles -> routing cost = 0, penalty = Q_k
        if sub_status in ("Optimal", "Feasible"):
            routing = Q_k
            penalty = 0.0
        else:
            routing = 0.0
            penalty = Q_k
            
        total_cost = stage1_cost + routing + penalty
        scen_rows.append({
            "scenario_name": sc.name,
            "severity_level": sc.severity_level,
            "probability": sc.probability,
            "stage1_cost": stage1_cost,
            "routing_cost": routing,
            "penalty_cost": penalty,
            "spoilage_cost": 0.0, # Not explicitly tracked in simple output
            "emergency_cost": 0.0,
            "total_cost": total_cost,
            "status": sub_status
        })
    pd.DataFrame(scen_rows).to_csv(os.path.join(results_dir, "benders_scenario_costs_fixed.csv"), index=False)
    print(f"\\n  💾 Saved results to results/benders_procurement_fixed.csv and benders_scenario_costs_fixed.csv!")

    print("\\n" + "=" * 80)
    print("ACADEMIC NOTES (v2)")
'''

if 'import pandas as pd' not in content:
    content = content.replace('import numpy as np', 'import numpy as np\nimport pandas as pd\nimport os', 1)

if '# --- EXPORT TO CSV FOR VISUALIZATION ---' not in content:
    content = content.replace('    print("\\n" + "=" * 80)\n    print("ACADEMIC NOTES (v2)")', export_code)

with open(benders_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched run_benders_optimization.py for CSV export.")
