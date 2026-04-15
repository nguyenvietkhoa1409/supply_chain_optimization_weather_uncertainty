import os

report_file = 'd:/Food chain optimization/report/model_result_visualization.py'
with open(report_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Add the unpacking logic that was deleted
unpacking_logic = '''
names  = [s[0] for s in SCEN]
sl     = [s[1] for s in SCEN]
probs  = [s[2] for s in SCEN]
rp     = [s[3] for s in SCEN]
eev    = [s[4] for s in SCEN]
ws     = [s[5] for s in SCEN]
stage1 = [s[6] for s in SCEN]
emerg  = [s[7] for s in SCEN]
spoil  = [s[8] for s in SCEN]
pen    = [s[9] for s in SCEN]

# ══════════════════════════════════════════════════════════════════════════
'''

# Replace the figure 1 header with the unpacking + figure 1
old_header = '# ══════════════════════════════════════════════════════════════════════════\n# FIGURE 1'
new_header = unpacking_logic + old_header

# Fix one issue with the legend in model_result_visualization
if 'names  = ' not in content:
    content = content.replace(old_header, new_header)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed unpacking logic in model_result_visualization.py")
