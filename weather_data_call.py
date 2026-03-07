import sys
sys.path.insert(0, "src")
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from weather.weather_data import DaNangWeatherData

wd = DaNangWeatherData(cache_dir="data/weather_cache")

# Fetch fresh từ Open-Meteo API (bỏ qua cache vì đã rename)
df = wd.fetch_historical_data("2014-01-01", "2023-12-31", use_cache=False)

print(f"\nSource: {df.attrs['source']}")
print(f"Records: {len(df)}")
assert df.attrs["source"] == "api", "API call failed — kiểm tra internet"
assert len(df) >= 3650, "Thiếu records"

# Fit và save distributions với source=api đúng
wd.fit_seasonal_distributions()
wd.save_distributions("data/danang_distribution_parameters.json")

print("\n" + wd.summary())
print("\n✓ DONE — danang_distribution_parameters.json đã được regenerate từ real data")