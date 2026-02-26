"""
Unit tests for Weather-Aware VRP
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimization.weather_vrp import WeatherAwareVRP
from data_generation.network_generator import DaNangNetworkGenerator
from data_generation.product_generator import ProductCatalogGenerator
from data_generation.demand_generator import DemandPatternGenerator
from weather.manual_scenarios import ManualWeatherScenarios


@pytest.fixture
def test_data():
    """Generate minimal test dataset"""
    network_gen = DaNangNetworkGenerator(seed=42)
    network = network_gen.generate_network(n_suppliers=3, n_dcs=1, n_stores=3)
    
    product_gen = ProductCatalogGenerator(seed=42)
    products = product_gen.generate_products(n_products=3)
    
    demand_gen = DemandPatternGenerator(seed=42)
    daily_demand = demand_gen.generate_demand_plan(
        network['stores'], products, planning_horizon_days=7
    )
    weekly_demand = demand_gen.aggregate_to_weekly(daily_demand)
    
    procurement = pd.DataFrame({
        'supplier_id': ['SUP_001'] * 3,
        'product_id': products['id'].tolist(),
        'quantity_units': [50] * 3
    })
    
    scenarios = ManualWeatherScenarios.create_monsoon_season_scenarios()
    
    return network, products, weekly_demand, procurement, scenarios


def test_vrp_initialization(test_data):
    """Test VRP model initialization"""
    network, products, demand, procurement, scenarios = test_data
    
    vrp = WeatherAwareVRP(
        network=network,
        products_df=products,
        demand_df=demand,
        procurement_solution=procurement,
        weather_scenarios=scenarios,
        vehicle_config={'num_vehicles': 2}
    )
    
    assert len(vrp.stores) == 3
    assert len(vrp.products) == 3
    assert vrp.vehicle_config['num_vehicles'] == 2


def test_vrp_model_building(test_data):
    """Test VRP model construction"""
    network, products, demand, procurement, scenarios = test_data
    
    vrp = WeatherAwareVRP(
        network=network,
        products_df=products,
        demand_df=demand,
        procurement_solution=procurement,
        weather_scenarios=scenarios
    )
    
    model, vars_dict = vrp.build_model(scenario_id=0)
    
    assert model is not None
    assert 'x' in vars_dict
    assert 't' in vars_dict
    assert 'q' in vars_dict
    assert model.numVariables() > 0
    assert model.numConstraints() > 0


def test_vrp_solve(test_data):
    """Test VRP solving (may be slow)"""
    network, products, demand, procurement, scenarios = test_data
    
    vrp = WeatherAwareVRP(
        network=network,
        products_df=products,
        demand_df=demand,
        procurement_solution=procurement,
        weather_scenarios=scenarios,
        vehicle_config={'num_vehicles': 2}
    )
    
    status, solution = vrp.solve(scenario_id=0, time_limit=60)
    
    assert status in ['Optimal', 'Feasible', 'Not Solved']
    
    if status == 'Optimal':
        assert 'objective_value' in solution
        assert solution['objective_value'] > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])