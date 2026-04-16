path = r'd:\Food chain optimization\scripts\run_stochastic_optimization.py'
content = open(path, encoding='utf-8').read()

# Detect line ending
if '\r\n' in content:
    le = '\r\n'
    print("Line ending: CRLF")
else:
    le = '\n'
    print("Line ending: LF")

old = (
    f'    season = input("\\nSeason (1=Dry, 2=Monsoon): ").strip(){le}'
    f'    if season == "1":{le}'
    f'        scenarios   = ManualWeatherScenarios.create_dry_season_scenarios(){le}'
    f'        season_name = "Dry Season"{le}'
    f'    else:{le}'
    f'        # scenarios   = ManualWeatherScenarios.create_monsoon_season_scenarios(){le}'
    f'        # season_name = "Monsoon Season"{le}'
    f'        scenarios = get_data_driven_scenarios(season="monsoon", target_count=5){le}'
    f'        season_name = "Monsoon Season"{le}'
)
new = (
    f'    # [SCALED] Fixed to monsoon, K=4 scenarios (was interactive: target_count=5){le}'
    f'    scenarios   = get_data_driven_scenarios(season="monsoon", target_count=4){le}'
    f'    season_name = "Monsoon Season"{le}'
)

if old in content:
    content = content.replace(old, new)
    open(path, 'w', encoding='utf-8').write(content)
    print("SUCCESS: target_count updated to 4, interactive input removed")
    # verify
    assert 'target_count=4' in content
    assert 'target_count=5' not in content
    assert 'input(' not in content
    print("Verified: K=4, no input()")
else:
    print("ERROR: pattern still not found — manual check needed")
    # Show raw around target_count=5
    idx = content.find('target_count=5')
    print(repr(content[max(0,idx-120):idx+60]))
