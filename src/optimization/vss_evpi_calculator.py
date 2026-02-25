"""
VSS & EVPI Calculator
Validates the value of stochastic solution and perfect information

Metrics:
- VSS (Value of Stochastic Solution) = EEV - RP
- EVPI (Expected Value of Perfect Information) = RP - WS
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os


class StochasticValidator:
    """
    Computes validation metrics for stochastic optimization
    
    Key metrics:
    - VSS: How much better is stochastic vs. deterministic?
    - EVPI: What's the upper bound value of perfect forecasts?
    - Out-of-sample robustness: Performance on unseen scenarios
    """
    
    def __init__(self):
        pass
    
    def compute_vss(self,
                   stochastic_objective: float,
                   deterministic_objective: float,
                   eev_cost: float = None) -> Dict:
        """
        Compute Value of Stochastic Solution
        
        VSS = EEV - RP
        where:
        - EEV: Expected result when using deterministic solution across scenarios
        - RP: Recourse Problem (stochastic solution objective)
        
        Args:
            stochastic_objective: Objective of stochastic model (RP)
            deterministic_objective: Objective of deterministic model
            eev_cost: If provided, the actual EEV (expected cost of det solution)
        
        Returns:
            Dict with VSS metrics
        """
        
        # If EEV not provided, use deterministic objective as approximation
        if eev_cost is None:
            eev_cost = deterministic_objective
        
        rp = stochastic_objective
        vss = eev_cost - rp
        vss_pct = (vss / eev_cost * 100) if eev_cost > 0 else 0
        
        result = {
            'EEV': eev_cost,
            'RP': rp,
            'VSS': vss,
            'VSS_percent': vss_pct,
            'interpretation': self._interpret_vss(vss_pct)
        }
        
        return result
    
    def compute_evpi(self,
                    stochastic_objective: float,
                    wait_and_see_cost: float) -> Dict:
        """
        Compute Expected Value of Perfect Information
        
        EVPI = RP - WS
        where:
        - RP: Recourse Problem (stochastic solution)
        - WS: Wait-and-See (perfect information bound)
        
        Args:
            stochastic_objective: Objective of stochastic model (RP)
            wait_and_see_cost: Expected cost with perfect information
        
        Returns:
            Dict with EVPI metrics
        """
        
        rp = stochastic_objective
        ws = wait_and_see_cost
        evpi = rp - ws
        evpi_pct = (evpi / rp * 100) if rp > 0 else 0
        
        result = {
            'RP': rp,
            'WS': ws,
            'EVPI': evpi,
            'EVPI_percent': evpi_pct,
            'interpretation': self._interpret_evpi(evpi_pct)
        }
        
        return result
    
    def _interpret_vss(self, vss_pct: float) -> str:
        """Interpret VSS percentage"""
        if vss_pct < 1:
            return "Minimal benefit from stochastic approach"
        elif vss_pct < 5:
            return "Moderate benefit - stochastic approach preferred"
        elif vss_pct < 10:
            return "Significant benefit - stochastic approach strongly recommended"
        else:
            return "Substantial benefit - stochastic approach essential"
    
    def _interpret_evpi(self, evpi_pct: float) -> str:
        """Interpret EVPI percentage"""
        if evpi_pct < 2:
            return "Low value of perfect information - forecasting not critical"
        elif evpi_pct < 5:
            return "Moderate value - improved forecasting worthwhile"
        elif evpi_pct < 10:
            return "High value - invest in weather monitoring systems"
        else:
            return "Very high value - perfect forecasts highly valuable"
    
    def generate_validation_report(self,
                                   vss_result: Dict,
                                   evpi_result: Dict,
                                   scenario_costs: pd.DataFrame) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            vss_result: Output from compute_vss()
            evpi_result: Output from compute_evpi()
            scenario_costs: DataFrame with per-scenario cost breakdown
        
        Returns:
            Formatted report string
        """
        
        report = []
        report.append("="*80)
        report.append("STOCHASTIC OPTIMIZATION VALIDATION REPORT")
        report.append("="*80)
        
        # VSS Analysis
        report.append("\n1. VALUE OF STOCHASTIC SOLUTION (VSS)")
        report.append("-" * 80)
        report.append(f"  Expected Cost (Deterministic): {vss_result['EEV']:>20,.0f} VND")
        report.append(f"  Stochastic Solution Cost:      {vss_result['RP']:>20,.0f} VND")
        report.append(f"  VSS (Cost Savings):            {vss_result['VSS']:>20,.0f} VND")
        report.append(f"  VSS (%):                       {vss_result['VSS_percent']:>20.2f}%")
        report.append(f"\n  → {vss_result['interpretation']}")
        
        # EVPI Analysis
        if evpi_result:
            report.append("\n2. EXPECTED VALUE OF PERFECT INFORMATION (EVPI)")
            report.append("-" * 80)
            report.append(f"  Stochastic Solution Cost:      {evpi_result['RP']:>20,.0f} VND")
            report.append(f"  Perfect Information Bound:     {evpi_result['WS']:>20,.0f} VND")
            report.append(f"  EVPI (Forecast Value):         {evpi_result['EVPI']:>20,.0f} VND")
            report.append(f"  EVPI (%):                      {evpi_result['EVPI_percent']:>20.2f}%")
            report.append(f"\n  → {evpi_result['interpretation']}")
        
        # Scenario Analysis
        report.append("\n3. SCENARIO-WISE COST BREAKDOWN")
        report.append("-" * 80)
        
        # Summary statistics
        report.append(f"  Mean Cost:     {scenario_costs['total_cost'].mean():>15,.0f} VND")
        report.append(f"  Std Dev:       {scenario_costs['total_cost'].std():>15,.0f} VND")
        report.append(f"  Min Cost:      {scenario_costs['total_cost'].min():>15,.0f} VND")
        report.append(f"  Max Cost:      {scenario_costs['total_cost'].max():>15,.0f} VND")
        report.append(f"  CV:            {scenario_costs['total_cost'].std() / scenario_costs['total_cost'].mean():>15.2%}")
        
        # Worst/Best scenarios
        worst_idx = scenario_costs['total_cost'].idxmax()
        best_idx = scenario_costs['total_cost'].idxmin()
        
        report.append(f"\n  Worst Scenario: {scenario_costs.loc[worst_idx, 'scenario_name']}")
        report.append(f"    Cost: {scenario_costs.loc[worst_idx, 'total_cost']:,.0f} VND")
        report.append(f"    Severity: Level {scenario_costs.loc[worst_idx, 'severity_level']}")
        
        report.append(f"\n  Best Scenario: {scenario_costs.loc[best_idx, 'scenario_name']}")
        report.append(f"    Cost: {scenario_costs.loc[best_idx, 'total_cost']:,.0f} VND")
        report.append(f"    Severity: Level {scenario_costs.loc[best_idx, 'severity_level']}")
        
        # Conclusion
        report.append("\n" + "="*80)
        report.append("CONCLUSION")
        report.append("="*80)
        
        if vss_result['VSS_percent'] > 5:
            report.append("✓ Stochastic optimization provides SIGNIFICANT value")
            report.append(f"  Cost savings: {vss_result['VSS_percent']:.1f}% compared to deterministic approach")
        elif vss_result['VSS_percent'] > 0:
            report.append("✓ Stochastic optimization provides MODERATE value")
            report.append(f"  Cost savings: {vss_result['VSS_percent']:.1f}%")
        else:
            report.append("⚠ Limited benefit from stochastic approach")
        
        if evpi_result and evpi_result['EVPI_percent'] > 5:
            report.append(f"✓ Weather forecasting has HIGH value ({evpi_result['EVPI_percent']:.1f}%)")
            report.append("  Recommendation: Invest in weather monitoring systems")
        
        report.append("="*80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    validator = StochasticValidator()
    
    # Mock data
    vss = validator.compute_vss(
        stochastic_objective=2_300_000_000,
        deterministic_objective=2_500_000_000,
        eev_cost=2_500_000_000
    )
    
    evpi = validator.compute_evpi(
        stochastic_objective=2_300_000_000,
        wait_and_see_cost=2_100_000_000
    )
    
    # Mock scenario costs
    scenario_costs = pd.DataFrame({
        'scenario_name': ['Clear', 'Light Rain', 'Heavy Rain', 'Typhoon'],
        'severity_level': [1, 2, 4, 5],
        'probability': [0.3, 0.4, 0.2, 0.1],
        'total_cost': [2_200_000_000, 2_300_000_000, 2_500_000_000, 2_800_000_000]
    })
    
    report = validator.generate_validation_report(vss, evpi, scenario_costs)
    print(report)