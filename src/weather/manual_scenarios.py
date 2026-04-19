"""
Manual Weather Scenarios - Da Nang Fresh Food Supply Chain
FIXED VERSION

Changes from original:
  [W-2] road_closure_probability kept as float for reporting only;
        actual arc accessibility is handled via arc_accessibility_factor
  [W-3 prep] Added supplier_accessibility dict: maps supplier subtype -> binary (1/0)
             This feeds into StochasticProcurementModel to gate Stage 2 dispatch.
             Rationale: Da Nang geography
               - seafood suppliers (Thọ Quang port, Nam Ô port): coastal/low, flood Level 4+
               - vegetable farms (Hòa Vang district): rural low-lying, flood Level 4+
               - meat suppliers (Hòa Khánh Industrial Zone): elevated, accessible to Level 4
               - general wholesale (Da Nang Market, city center): always accessible
  [V-1 prep] Added emergency_feasible: False for Level 5 (typhoon),
             prevents VRP/procurement models from relying on emergency supply
             that physically cannot operate
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List
import os 

@dataclass
class WeatherScenario:
    """Represents a single weather scenario with operational impacts."""

    scenario_id: int
    name: str
    severity_level: int        # 1–5 (clear to typhoon)
    description: str

    # Raw weather parameters
    rainfall_mm_per_day: float
    temperature_celsius: float
    wind_speed_kmh: float

    # Operational impact factors
    speed_reduction_factor: float       # Travel-time multiplier (1.0 = normal, 2.2 = 2.2× slower)
    capacity_reduction_factor: float    # Fraction of vehicle capacity available [0,1]
    spoilage_multiplier: float          # Global baseline spoilage acceleration (used only for
                                        # reporting; product-specific μ_{k,i} is computed by
                                        # the model via Q10 rule from temperature_celsius)
    road_closure_probability: float     # For reporting only; arc-level binary o_{k,ij}
                                        # is derived geographically in the VRP model

    probability: float

    # -----------------------------------------------------------------------
    # NEW FIELDS (fixes W-3 prep, V-1 prep)
    # -----------------------------------------------------------------------

    # supplier_accessibility: maps supplier subtype → 1 (accessible) / 0 (inaccessible)
    # Used by StochasticProcurementModel to determine which Stage 1 procurement
    # can actually be dispatched under scenario k (non-anticipativity fix).
    supplier_accessibility: Dict[str, int] = field(
        default_factory=lambda: {
            "seafood": 1,
            "vegetables": 1,
            "meat": 1,
            "general": 1,
        }
    )

    # emergency_feasible: whether emergency procurement is physically possible.
    # Set to False for Level 5 (typhoon) — wholesale markets are closed,
    # no delivery vehicles operate.
    emergency_feasible: bool = True

    # demand_reduction_factor: scales demand based on weather severity
    demand_reduction_factor: float = 1.0

    def get_supplier_accessible(self, supplier_subtype: str) -> int:
        """Convenience accessor; returns 1 if subtype accessible under this scenario."""
        return self.supplier_accessibility.get(supplier_subtype, 1)

    def to_dict(self):
        return asdict(self)


class ManualWeatherScenarios:
    """
    Defines realistic weather scenarios for Da Nang.

    Sources:
    - FHWA (2023) weather-road impact studies
    - Da Nang flood records (Hòa Vang, Hòa Khánh, coastal zones)
    - Labuza (1984) food spoilage kinetics
    """

    # -----------------------------------------------------------------------
    # Supplier accessibility templates per severity level
    # Level 1–3: all accessible
    # Level 4:   seafood ports and vegetable farms flooded; meat (Hòa Khánh)
    #             and general (city market) remain accessible
    # Level 5:   only general wholesale market (central, elevated) accessible
    # -----------------------------------------------------------------------
    _ACCESSIBILITY = {
        1: {"seafood": 1, "vegetables": 1, "meat": 1, "general": 1},
        2: {"seafood": 1, "vegetables": 1, "meat": 1, "general": 1},
        3: {"seafood": 1, "vegetables": 1, "meat": 1, "general": 1},
        4: {"seafood": 0, "vegetables": 0, "meat": 1, "general": 1},
        5: {"seafood": 0, "vegetables": 0, "meat": 0, "general": 1},
    }

    # -----------------------------------------------------------------------
    @classmethod
    def create_dry_season_scenarios(cls) -> List[WeatherScenario]:
        """
        Dry season (February–August).
        Main risk: extreme heat accelerating spoilage; no route disruption.
        Probabilities sum to 1.0.
        """
        scenarios = [
            WeatherScenario(
                scenario_id=1,
                name="Clear/Sunny",
                severity_level=1,
                description="Typical dry season day – clear skies, hot temperature",
                rainfall_mm_per_day=0.0,
                temperature_celsius=32.0,
                wind_speed_kmh=15.0,
                speed_reduction_factor=1.00,
                capacity_reduction_factor=1.00,
                spoilage_multiplier=1.20,
                road_closure_probability=0.00,
                probability=0.60,
                supplier_accessibility=cls._ACCESSIBILITY[1],
                emergency_feasible=True,
                demand_reduction_factor=1.00,
            ),
            WeatherScenario(
                scenario_id=2,
                name="Hot & Humid",
                severity_level=2,
                description="Extreme heat day (>35 °C) with high humidity",
                rainfall_mm_per_day=2.0,
                temperature_celsius=36.0,
                wind_speed_kmh=12.0,
                speed_reduction_factor=1.05,
                capacity_reduction_factor=0.95,
                spoilage_multiplier=1.40,
                road_closure_probability=0.00,
                probability=0.25,
                supplier_accessibility=cls._ACCESSIBILITY[2],
                emergency_feasible=True,
                demand_reduction_factor=0.95,
            ),
            WeatherScenario(
                scenario_id=3,
                name="Light Scattered Showers",
                severity_level=2,
                description="Afternoon showers typical in tropical climate",
                rainfall_mm_per_day=8.0,
                temperature_celsius=30.0,
                wind_speed_kmh=20.0,
                speed_reduction_factor=1.10,
                capacity_reduction_factor=0.92,
                spoilage_multiplier=1.10,
                road_closure_probability=0.01,
                probability=0.15,
                supplier_accessibility=cls._ACCESSIBILITY[2],
                emergency_feasible=True,
                demand_reduction_factor=0.95,
            ),
        ]
        return scenarios

    @classmethod
    def create_monsoon_season_scenarios(cls) -> List[WeatherScenario]:
        """
        Monsoon season (September–December).
        Main risks: flooding, typhoon, supply-chain disruption.
        Probabilities sum to 1.0.
        """
        scenarios = [
            WeatherScenario(
                scenario_id=11,
                name="Normal Monsoon Day",
                severity_level=1,
                description="Typical monsoon day – intermittent rain, comfortable temperature",
                rainfall_mm_per_day=12.0,
                temperature_celsius=26.0,
                wind_speed_kmh=18.0,
                speed_reduction_factor=1.08,
                capacity_reduction_factor=0.95,
                spoilage_multiplier=1.00,
                road_closure_probability=0.02,
                probability=0.30,
                supplier_accessibility=cls._ACCESSIBILITY[1],
                emergency_feasible=True,
                demand_reduction_factor=1.00,
            ),
            WeatherScenario(
                scenario_id=12,
                name="Light Rain",
                severity_level=2,
                description="Steady light rain throughout the day",
                rainfall_mm_per_day=18.0,
                temperature_celsius=25.0,
                wind_speed_kmh=22.0,
                speed_reduction_factor=1.15,
                capacity_reduction_factor=0.90,
                spoilage_multiplier=1.05,
                road_closure_probability=0.05,
                probability=0.25,
                supplier_accessibility=cls._ACCESSIBILITY[2],
                emergency_feasible=True,
                demand_reduction_factor=0.95,
            ),
            WeatherScenario(
                scenario_id=13,
                name="Moderate Rain",
                severity_level=3,
                description="Heavy rain periods, localized flooding possible",
                rainfall_mm_per_day=45.0,
                temperature_celsius=24.0,
                wind_speed_kmh=30.0,
                speed_reduction_factor=1.25,
                capacity_reduction_factor=0.80,
                spoilage_multiplier=1.15,
                road_closure_probability=0.10,
                probability=0.20,
                supplier_accessibility=cls._ACCESSIBILITY[3],
                emergency_feasible=True,
                demand_reduction_factor=0.80,
            ),
            WeatherScenario(
                scenario_id=14,
                name="Heavy Rain",
                severity_level=4,
                description="Intense rainfall, widespread flooding, travel hazardous",
                rainfall_mm_per_day=85.0,
                temperature_celsius=23.0,
                wind_speed_kmh=40.0,
                speed_reduction_factor=1.55,
                capacity_reduction_factor=0.60,
                spoilage_multiplier=1.30,
                road_closure_probability=0.35,
                probability=0.15,
                supplier_accessibility=cls._ACCESSIBILITY[4],  # seafood/veg inaccessible
                emergency_feasible=True,   # partial: wholesale market still operates
                demand_reduction_factor=0.55,
            ),
            WeatherScenario(
                scenario_id=15,
                name="Tropical Storm/Typhoon",
                severity_level=5,
                description="Typhoon conditions – severe disruption, most routes closed",
                rainfall_mm_per_day=180.0,
                temperature_celsius=22.0,
                wind_speed_kmh=85.0,
                speed_reduction_factor=2.20,
                capacity_reduction_factor=0.10,  # corrected from 0.15 → aligns with framework doc
                spoilage_multiplier=2.00,
                road_closure_probability=0.85,
                probability=0.10,
                supplier_accessibility=cls._ACCESSIBILITY[5],  # only general accessible
                emergency_feasible=False,  # NO emergency procurement possible
                demand_reduction_factor=0.15,
            ),
        ]
        return scenarios

    @classmethod
    def create_all_scenarios(cls) -> List[WeatherScenario]:
        """Combine dry + monsoon scenarios (unnormalized; caller must renormalize)."""
        return cls.create_dry_season_scenarios() + cls.create_monsoon_season_scenarios()

    # -----------------------------------------------------------------------
    @staticmethod
    def save_scenarios(
        scenarios: List[WeatherScenario],
        filename: str = "manual_weather_scenarios.json",
        output_dir: str = "../data/weather/scenarios",
    ):
        import os
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            json.dump([s.to_dict() for s in scenarios], f, indent=2)
        print(f"✓ Saved {len(scenarios)} scenarios to {filepath}")

    @staticmethod
    def load_scenarios(
        filename: str = "manual_weather_scenarios.json",
        data_dir: str = "../data/weather/scenarios",
    ) -> List[WeatherScenario]:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            dicts = json.load(f)
        return [WeatherScenario(**d) for d in dicts]

    @staticmethod
    def get_scenario_summary_table(scenarios: List[WeatherScenario]) -> pd.DataFrame:
        data = []
        for s in scenarios:
            accessible = [k for k, v in s.supplier_accessibility.items() if v == 1]
            data.append(
                {
                    "ID": s.scenario_id,
                    "Name": s.name,
                    "Level": s.severity_level,
                    "Rainfall (mm)": s.rainfall_mm_per_day,
                    "Temp (°C)": s.temperature_celsius,
                    "Speed Factor": f"{s.speed_reduction_factor:.2f}x",
                    "Capacity": f"{s.capacity_reduction_factor * 100:.0f}%",
                    "Spoilage": f"{s.spoilage_multiplier:.2f}x",
                    "Demand": f"{s.demand_reduction_factor * 100:.0f}%",
                    "Accessible Suppliers": ", ".join(accessible),
                    "Emergency OK": "Yes" if s.emergency_feasible else "NO",
                    "Probability": f"{s.probability * 100:.1f}%",
                }
            )
        return pd.DataFrame(data)


if __name__ == "__main__":
    print("DRY SEASON")
    dry = ManualWeatherScenarios.create_dry_season_scenarios()
    print(ManualWeatherScenarios.get_scenario_summary_table(dry).to_string(index=False))
    print(f"\nΣp = {sum(s.probability for s in dry):.3f}")

    print("\n\nMONSOON SEASON")
    mon = ManualWeatherScenarios.create_monsoon_season_scenarios()
    print(ManualWeatherScenarios.get_scenario_summary_table(mon).to_string(index=False))
    print(f"\nΣp = {sum(s.probability for s in mon):.3f}")