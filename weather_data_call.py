import pandas as pd

df = pd.read_csv("data/weather_cache/danang_2014-01-01_2023-12-31.csv", 
                 parse_dates=["date"])

print("=== VALIDATION ===")
print(f"Total records: {len(df)}")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print(f"Expected: 2014-01-01 → 2023-12-31 ({365*10} days ~= 3650)")
print()

# Kiểm tra tháng 10 - đây là tháng mưa nhiều nhất Đà Nẵng
oct_data = df[df['date'].dt.month == 10]
print(f"October mean rainfall: {oct_data['rainfall_mm'].mean():.1f} mm/day")
print(f"Expected từ literature: ~21 mm/day (650mm/30days)")
print()

# Kiểm tra ngày typhoon cụ thể đã biết
# Bão Noru đổ bộ Đà Nẵng 27/09/2022
noru = df[df['date'] == '2022-09-27']
print(f"Bão Noru (27/09/2022): {noru['rainfall_mm'].values} mm")
print(f"Expected: >100mm nếu là real ERA5 data")
print()

# Kiểm tra wind speed trong typhoon
print(f"Max wind ever recorded: {df['wind_max_kmh'].max():.1f} km/h")
print(f"Expected: >80 km/h nếu có typhoon events")