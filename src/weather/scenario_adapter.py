"""
scenario_adapter.py — Drop-in Bridge for run_stochastic_optimization.py
═══════════════════════════════════════════════════════════════════════════

Converts data-driven scenarios (GeneratedWeatherScenario) to the interface
expected by ExtensiveFormOptimizer, vss_evpi_calculator, and related modules.

One-line integration in run_stochastic_optimization.py
───────────────────────────────────────────────────────
REPLACE:
    from weather.manual_scenarios import ManualWeatherScenarios
    scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()

WITH:
    from weather.scenario_adapter import get_data_driven_scenarios
    scenarios = get_data_driven_scenarios(season="monsoon")

No other changes required. All downstream code works without modification.
"""

from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Dict, List, Optional

from weather.weather_data import DaNangWeatherData
from weather.scenario_generator import GeneratedWeatherScenario, WeatherScenarioGenerator

logger = logging.getLogger(__name__)


class DataDrivenWeatherScenario(GeneratedWeatherScenario):
    """Thin subclass satisfying isinstance checks for WeatherScenario-typed code."""

    # ── Attribute aliases để tương thích với manual_scenarios.py ──────────
    @property
    def temperature_celsius(self) -> float:
        return self.temperature_c

    @property
    def rainfall_mm_per_day(self) -> float:
        return self.rainfall_mm

    @property
    def wind_speed_kmh(self) -> float:
        return self.wind_kmh

    @property
    def road_closure_probability(self) -> float:
        return self.road_closure_prob


def get_data_driven_scenarios(
    season:         str   = "monsoon",
    n_samples:      int   = 600,
    target_count:   int   = 10,
    seed:           int   = 2024,
    cache_dir:      Optional[str] = "data/weather_cache",
    dist_cache:     Optional[str] = "data/danang_distribution_parameters.json",
    force_extreme:  bool  = True,
    merge_duplicates: bool = False,
    api_start:      str   = "2014-01-01",
    api_end:        str   = "2023-12-31",
    verbose:        bool  = True,
) -> List[DataDrivenWeatherScenario]:
    """
    One-call entry point: fetch/load distributions → generate scenarios.

    Caching:
      1. Load dist_cache JSON if it exists (fast, no API call needed)
      2. Otherwise: call Open-Meteo API, fit distributions, save cache
      3. LHS + FFS + force_extreme → return scenarios

    Returns
    -------
    List[DataDrivenWeatherScenario] — drop-in for ManualWeatherScenarios output.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    wd = DaNangWeatherData(cache_dir=cache_dir)

    if dist_cache and Path(dist_cache).exists():
        logger.info(f"Loading cached distributions: {dist_cache}")
        wd.load_distributions(dist_cache)
    else:
        logger.info("Fetching historical weather + fitting distributions…")
        wd.fetch_historical_data(api_start, api_end)
        wd.fit_seasonal_distributions()
        if dist_cache:
            Path(dist_cache).parent.mkdir(parents=True, exist_ok=True)
            wd.save_distributions(dist_cache)

    if verbose:
        print(wd.summary())

    gen = WeatherScenarioGenerator(wd)
    scenarios, quality = gen.generate_scenarios(
        season=season, n_samples=n_samples, target_count=target_count,
        seed=seed, force_extreme=force_extreme,
        merge_duplicates=merge_duplicates
    )

    if verbose:
        print(f"\n{quality.summary()}")

    return [DataDrivenWeatherScenario(**s.__dict__) for s in scenarios]


def get_scenario_summary_table(
    scenarios: List[DataDrivenWeatherScenario],
) -> "pd.DataFrame":
    """Returns the same format as ManualWeatherScenarios for reporting."""
    import pandas as pd
    return pd.DataFrame([{
        "ID":           s.scenario_id,
        "Name":         s.name,
        "Level":        s.severity_level,
        "Rainfall (mm)": f"{s.rainfall_mm:.1f}",
        "Temp (°C)":    f"{s.temperature_c:.1f}",
        "Speed Factor": f"{s.speed_reduction_factor:.2f}x",
        "Capacity":     f"{s.capacity_reduction_factor:.0%}",
        "Spoilage":     f"{s.spoilage_multiplier:.2f}x",
        "Accessible Suppliers": ", ".join(
            k for k,v in s.supplier_accessibility.items() if v
        ),
        "Emergency OK": "Yes" if s.emergency_feasible else "NO",
        "Probability":  f"{s.probability:.1%}",
    } for s in scenarios])