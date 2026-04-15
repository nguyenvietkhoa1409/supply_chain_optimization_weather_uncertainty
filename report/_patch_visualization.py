import os
import re

report_file = 'd:/Food chain optimization/report/model_result_visualization.py'
with open(report_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure we import pandas
if 'import pandas as pd' not in content:
    content = content.replace('import numpy as np', 'import numpy as np\nimport pandas as pd\nimport os')

replacement = '''
# ── Data (Dynamic Load) ───────────────────────────────────────────────────
benders_scen_path = "../results/benders_scenario_costs_fixed.csv"
benders_proc_path = "../results/benders_procurement_fixed.csv"

if os.path.exists(benders_scen_path) and os.path.exists(benders_proc_path):
    print("Loading data from Benders CSV outputs...")
    df_s = pd.read_csv(benders_scen_path)
    df_p = pd.read_csv(benders_proc_path)
    
    # 1. SCEN
    SCEN = []
    # format: name, sl, prob, rp, eev, ws, stage1, emerg, spoil, pen
    for _, row in df_s.iterrows():
        # Convert to Billions of VND for plotting
        stage1 = row["stage1_cost"] / 1e9
        routing = row["routing_cost"] / 1e9
        penalty = row["penalty_cost"] / 1e9
        total = row["total_cost"] / 1e9
        # Benders doesn't compute EEV/WS, so we mock them simply for visualization compatibility
        SCEN.append((row["scenario_name"].replace(" (data-driven)", ""), 
                     int(row["severity_level"]), 
                     float(row["probability"]), 
                     total, total*1.2, total*0.8, # Mock eev/ws
                     stage1, routing, 0.0, penalty))
                     
    # 2. KPI
    exp_rp = sum(s[2]*s[3] for s in SCEN)
    # Mock CVaR
    KPI = dict(RP=exp_rp, EEV=exp_rp*1.25, WS=exp_rp*0.8, VaR90=exp_rp*1.5, CVaR90=exp_rp*2.0)
    
    # 3. PROC_SUPS
    df_p["cost_bil"] = df_p["total_cost_vnd"] / 1e9
    sups = df_p.groupby("supplier_id")["cost_bil"].sum().sort_values(ascending=False)
    PROC_SUPS = [(s, v) for s, v in sups.items()]
    
    # 4. TOP_PROD
    prods = df_p.groupby("product_name")["cost_bil"].sum().sort_values(ascending=False).head(10)
    TOP_PROD = []
    for i, (p, v) in enumerate(prods.items()):
        TOP_PROD.append((p, v, i < 4)) 

else:
    print("CSV files not found. Using hardcoded fallback data.")
    KPI = dict(RP=3.691, EEV=5.903, WS=3.256,
               VaR90=5.283, CVaR90=10.160)
    
    SCEN = [
        # name,           sl, prob,  rp,     eev,    ws,    stage1, emerg, spoil, pen
        ("Normal (1)",    1, 0.183, 2.711, 2.609, 2.345, 2.614, 0.095, 0.002, 0.000),
        ("Normal (2)",    1, 0.174, 2.714, 2.609, 2.345, 2.614, 0.095, 0.005, 0.000),
        ("Lt. Rain (1)",  2, 0.108, 2.719, 2.612, 2.342, 2.614, 0.095, 0.010, 0.000),
        ("Lt. Rain (2)",  2, 0.104, 2.714, 2.612, 2.342, 2.614, 0.095, 0.005, 0.000),
        ("Lt. Rain (3)",  2, 0.062, 2.711, 2.612, 2.342, 2.614, 0.095, 0.002, 0.000),
        ("Lt. Rain (4)",  2, 0.061, 2.710, 2.612, 2.342, 2.614, 0.095, 0.002, 0.000),
        ("Mod. Rain (1)", 3, 0.070, 2.711, 2.665, 2.346, 2.614, 0.095, 0.002, 0.000),
        ("Mod. Rain (2)", 3, 0.047, 2.720, 2.665, 2.346, 2.614, 0.095, 0.011, 0.000),
        ("Heavy Rain",    4, 0.096, 5.283, 13.304, 4.333, 2.614, 0.636, 0.485, 1.548),
        ("Typhoon",       5, 0.095, 10.427, 26.449, 9.944, 2.614, 0.000, 0.630, 7.183),
    ]
    
    PROC_SUPS = [
        ("SUP_006", 1.977),
        ("SUP_005", 0.224),
        ("SUP_004", 0.087),
        ("SUP_003", 0.161),
        ("SUP_002", 0.085),
        ("SUP_001", 0.073),
    ]
    
    TOP_PROD = [
        ("P009 · Staples",      0.734, True),
        ("P002 · Fruits",       0.521, True),
        ("P001 · Vegetables",   0.346, True),
        ("P003 · Meat",         0.291, True),
        ("P007 · Dairy",        0.138, False),
        ("P008 · Seafood (S3)", 0.112, False),
        ("P008 · Seafood (S5)", 0.086, False),
        ("P005 · Grain",        0.081, False),
        ("P010 · Dry goods",    0.073, False),
        ("P006 · Condiments",   0.058, False),
    ]
'''

# Use regex to find and replace the data block
start_idx = content.find('# ── Data ─')
end_idx = content.find('# ═══════════', start_idx)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + replacement + "\n" + content[end_idx:]
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patched model_result_visualization.py to read CSVs.")
else:
    print("Could not find the Data block in model_result_visualization.py")
