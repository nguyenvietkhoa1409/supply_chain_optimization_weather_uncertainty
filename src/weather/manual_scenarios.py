"""
Manual Weather Scenarios - Realistic weather conditions for Da Nang
Creates 5 severity levels based on typical meteorological patterns

This is a simplified version for initial testing.
Later we'll replace with statistical scenario generation from historical data.
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class WeatherScenario:
    """Represents a single weather scenario with operational impacts"""
    scenario_id: int
    name: str
    severity_level: int  # 1-5 (clear to typhoon)
    description: str
    
    # Weather parameters
    rainfall_mm_per_day: float
    temperature_celsius: float
    wind_speed_kmh: float
    
    # Operational impact factors
    speed_reduction_factor: float  # Multiplier for travel time (1.0 = no impact, 2.0 = 2x slower)
    capacity_reduction_factor: float  # Fraction of normal capacity available (1.0 = full, 0.5 = half)
    spoilage_multiplier: float  # Acceleration of food degradation (1.0 = normal, 2.0 = 2x faster)
    road_closure_probability: float  # Probability of route being closed (0.0-1.0)
    
    # Scenario probability
    probability: float
    
    def to_dict(self):
        return asdict(self)


class ManualWeatherScenarios:
    """
    Defines realistic weather scenarios for Da Nang based on:
    - Historical climate patterns
    - Transportation research (FHWA weather impact studies)
    - Food science (temperature-dependent spoilage rates)
    """
    
    @staticmethod
    def create_dry_season_scenarios() -> List[WeatherScenario]:
        """
        Dry season (February-August) scenarios
        
        Characteristics:
        - Low rainfall (<115mm/month)
        - High temperatures (29-35°C)
        - Minimal disruption risk
        - Main concern: Heat-induced spoilage
        """
        
        scenarios = [
            WeatherScenario(
                scenario_id=1,
                name="Clear/Sunny",
                severity_level=1,
                description="Typical dry season day - clear skies, hot temperature",
                rainfall_mm_per_day=0.0,
                temperature_celsius=32.0,
                wind_speed_kmh=15.0,
                speed_reduction_factor=1.00,  # No speed impact
                capacity_reduction_factor=1.00,  # Full capacity
                spoilage_multiplier=1.20,  # 20% faster spoilage due to heat
                road_closure_probability=0.00,
                probability=0.60  # 60% of days
            ),
            
            WeatherScenario(
                scenario_id=2,
                name="Hot & Humid",
                severity_level=2,
                description="Extreme heat day (>35°C) with high humidity",
                rainfall_mm_per_day=2.0,
                temperature_celsius=36.0,
                wind_speed_kmh=12.0,
                speed_reduction_factor=1.05,  # Slight slowdown due to heat
                capacity_reduction_factor=0.95,  # Minor capacity reduction
                spoilage_multiplier=1.40,  # 40% faster spoilage
                road_closure_probability=0.00,
                probability=0.25  # 25% of days
            ),
            
            WeatherScenario(
                scenario_id=3,
                name="Light Scattered Showers",
                severity_level=2,
                description="Afternoon showers typical in tropical climate",
                rainfall_mm_per_day=8.0,
                temperature_celsius=30.0,
                wind_speed_kmh=20.0,
                speed_reduction_factor=1.10,  # 10% slower
                capacity_reduction_factor=0.92,
                spoilage_multiplier=1.10,
                road_closure_probability=0.01,
                probability=0.15  # 15% of days
            )
        ]
        
        return scenarios
    
    @staticmethod
    def create_monsoon_season_scenarios() -> List[WeatherScenario]:
        """
        Monsoon season (September-December) scenarios
        
        Characteristics:
        - High rainfall (300-650mm/month)
        - Peak typhoon risk (September-November)
        - Significant operational disruptions
        - Road flooding common in October
        """
        
        scenarios = [
            WeatherScenario(
                scenario_id=11,
                name="Normal Monsoon Day",
                severity_level=1,
                description="Typical monsoon day - intermittent rain, comfortable temperature",
                rainfall_mm_per_day=12.0,
                temperature_celsius=26.0,
                wind_speed_kmh=18.0,
                speed_reduction_factor=1.08,
                capacity_reduction_factor=0.95,
                spoilage_multiplier=1.00,  # Cooler temp offsets humidity
                road_closure_probability=0.02,
                probability=0.30  # 30% of monsoon days
            ),
            
            WeatherScenario(
                scenario_id=12,
                name="Light Rain",
                severity_level=2,
                description="Steady light rain throughout the day",
                rainfall_mm_per_day=18.0,
                temperature_celsius=25.0,
                wind_speed_kmh=22.0,
                speed_reduction_factor=1.15,  # 15% slower
                capacity_reduction_factor=0.90,
                spoilage_multiplier=1.05,
                road_closure_probability=0.05,
                probability=0.25
            ),
            
            WeatherScenario(
                scenario_id=13,
                name="Moderate Rain",
                severity_level=3,
                description="Heavy rain periods, localized flooding possible",
                rainfall_mm_per_day=45.0,
                temperature_celsius=24.0,
                wind_speed_kmh=30.0,
                speed_reduction_factor=1.25,  # 25% slower
                capacity_reduction_factor=0.80,
                spoilage_multiplier=1.15,
                road_closure_probability=0.10,
                probability=0.20
            ),
            
            WeatherScenario(
                scenario_id=14,
                name="Heavy Rain",
                severity_level=4,
                description="Intense rainfall, widespread flooding, travel hazardous",
                rainfall_mm_per_day=85.0,
                temperature_celsius=23.0,
                wind_speed_kmh=40.0,
                speed_reduction_factor=1.55,  # 55% slower (nearly 40% reduction in speed)
                capacity_reduction_factor=0.60,
                spoilage_multiplier=1.30,
                road_closure_probability=0.35,
                probability=0.15
            ),
            
            WeatherScenario(
                scenario_id=15,
                name="Tropical Storm/Typhoon",
                severity_level=5,
                description="Typhoon conditions - severe disruption, routes closed",
                rainfall_mm_per_day=180.0,
                temperature_celsius=22.0,
                wind_speed_kmh=85.0,
                speed_reduction_factor=2.20,  # Extremely slow travel if possible
                capacity_reduction_factor=0.15,  # Only 15% capacity operational
                spoilage_multiplier=2.00,  # 2x spoilage due to refrigeration failures
                road_closure_probability=0.85,  # Most routes closed
                probability=0.10  # ~10% of monsoon days (Oct peak)
            )
        ]
        
        return scenarios
    
    @staticmethod
    def create_all_scenarios() -> List[WeatherScenario]:
        """Combine all scenarios (dry + monsoon)"""
        dry = ManualWeatherScenarios.create_dry_season_scenarios()
        monsoon = ManualWeatherScenarios.create_monsoon_season_scenarios()
        return dry + monsoon
    
    @staticmethod
    def save_scenarios(scenarios: List[WeatherScenario], 
                      filename: str = 'manual_weather_scenarios.json',
                      output_dir: str = '../data/weather/scenarios'):
        """Save scenarios to JSON file"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to list of dicts
        scenario_dicts = [s.to_dict() for s in scenarios]
        
        # Save
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(scenario_dicts, f, indent=2)
        
        print(f"✓ Saved {len(scenarios)} scenarios to {filepath}")
    
    @staticmethod
    def load_scenarios(filename: str = 'manual_weather_scenarios.json',
                      data_dir: str = '../data/weather/scenarios') -> List[WeatherScenario]:
        """Load scenarios from JSON file"""
        filepath = os.path.join(data_dir, filename)
        
        with open(filepath, 'r') as f:
            scenario_dicts = json.load(f)
        
        scenarios = [WeatherScenario(**s) for s in scenario_dicts]
        return scenarios
    
    @staticmethod
    def get_scenario_summary_table(scenarios: List[WeatherScenario]) -> pd.DataFrame:
        """Create summary table of scenarios"""
        data = []
        for s in scenarios:
            data.append({
                'ID': s.scenario_id,
                'Name': s.name,
                'Level': s.severity_level,
                'Rainfall (mm)': s.rainfall_mm_per_day,
                'Temp (°C)': s.temperature_celsius,
                'Speed Factor': f"{s.speed_reduction_factor:.2f}x",
                'Capacity': f"{s.capacity_reduction_factor*100:.0f}%",
                'Spoilage': f"{s.spoilage_multiplier:.2f}x",
                'Closure Risk': f"{s.road_closure_probability*100:.0f}%",
                'Probability': f"{s.probability*100:.1f}%"
            })
        return pd.DataFrame(data)


if __name__ == "__main__":
    print("="*70)
    print("MANUAL WEATHER SCENARIOS FOR DA NANG")
    print("="*70)
    
    # Create dry season scenarios
    print("\n\nDRY SEASON SCENARIOS (Feb-Aug)")
    print("-" * 70)
    dry_scenarios = ManualWeatherScenarios.create_dry_season_scenarios()
    dry_table = ManualWeatherScenarios.get_scenario_summary_table(dry_scenarios)
    print(dry_table.to_string(index=False))
    
    # Validate probabilities sum to ~1.0
    dry_prob_sum = sum(s.probability for s in dry_scenarios)
    print(f"\nTotal probability: {dry_prob_sum:.2f} (should be ~1.0)")
    
    # Create monsoon season scenarios
    print("\n\n" + "="*70)
    print("MONSOON SEASON SCENARIOS (Sep-Dec)")
    print("-" * 70)
    monsoon_scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    monsoon_table = ManualWeatherScenarios.get_scenario_summary_table(monsoon_scenarios)
    print(monsoon_table.to_string(index=False))
    
    monsoon_prob_sum = sum(s.probability for s in monsoon_scenarios)
    print(f"\nTotal probability: {monsoon_prob_sum:.2f} (should be ~1.0)")
    
    # Save both
    ManualWeatherScenarios.save_scenarios(
        dry_scenarios, 
        'dry_season_scenarios.json'
    )
    
    ManualWeatherScenarios.save_scenarios(
        monsoon_scenarios,
        'monsoon_season_scenarios.json'
    )
    
    # Save combined
    all_scenarios = ManualWeatherScenarios.create_all_scenarios()
    ManualWeatherScenarios.save_scenarios(
        all_scenarios,
        'all_scenarios.json'
    )
    
    print("\n\n✓ All scenario files saved successfully!")
    
    # Display operational impact comparison
    print("\n\n" + "="*70)
    print("OPERATIONAL IMPACT COMPARISON")
    print("="*70)
    print("\nSeverity Level Analysis:")
    
    for level in range(1, 6):
        level_scenarios = [s for s in all_scenarios if s.severity_level == level]
        if level_scenarios:
            avg_speed = np.mean([s.speed_reduction_factor for s in level_scenarios])
            avg_capacity = np.mean([s.capacity_reduction_factor for s in level_scenarios])
            avg_spoilage = np.mean([s.spoilage_multiplier for s in level_scenarios])
            avg_closure = np.mean([s.road_closure_probability for s in level_scenarios])
            
            print(f"\nLevel {level}: {len(level_scenarios)} scenarios")
            print(f"  Speed impact: {avg_speed:.2f}x slower")
            print(f"  Capacity: {avg_capacity*100:.0f}% available")
            print(f"  Spoilage: {avg_spoilage:.2f}x faster")
            print(f"  Road closure risk: {avg_closure*100:.0f}%")
