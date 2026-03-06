"""
scenario_generator.py — Weather Scenario Generation & Reduction
═══════════════════════════════════════════════════════════════════

Step 2 of the data-driven scenario generation pipeline.

Scientific pipeline
───────────────────
1. Latin Hypercube Sampling (LHS)  [McKay et al. 1979; Stein 1987]
   Generates N=500–1000 candidate scenarios from fitted seasonal distributions.
   LHS partitions each marginal CDF into N equal-probability strata and samples
   one point per stratum — empirically requires ~6× fewer samples than MC for
   equivalent tail coverage.

2. Severity Classification
   Maps each (rainfall, wind) pair to discrete severity level 1–5 using
   thresholds calibrated to Da Nang logistics disruption data (FHWA 2023,
   FAO 2009). Derives operational parameters (speed/capacity/spoilage factors).

3. Fast Forward Selection — Backward Reduction [Heitsch & Römisch 2003]
   Reduces N candidates → K representative scenarios by iteratively removing
   the scenario with minimum weighted-Wasserstein removal cost and redistributing
   its probability to the nearest retained neighbor.
   Complexity: O(N²·(N−K)). Feasible for N≤1000, K≥5.

4. Quality Validation  [Dupačová et al. 2003]
   Validates the reduced set BEFORE forced extreme injection:
     - Rainfall mean preservation  < 5%
     - Rainfall variance preservation < 15%  (relaxed: right-skewed dist)
     - Probability sum ≈ 1.0
   Extreme coverage (Level ≥ 4) reported separately — forced injection is a
   deliberate tail-safety step, not a distribution-fitting failure.

References
──────────
McKay et al. (1979) Technometrics.
Heitsch & Römisch (2003) Comput. Optim. Appl.
Dupačová et al. (2003) Math. Program.
FHWA (2023) Weather and Road Operations.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from weather.weather_data import DaNangWeatherData, SeasonalDistributions

logger = logging.getLogger(__name__)

# ── Severity Definitions ──────────────────────────────────────────────────────
# Operational parameters calibrated to Da Nang conditions.
# speed_factor: travel-time multiplier (1.0 = no delay, 2.2 = typhoon)
# capacity_factor: fraction of nominal vehicle capacity operable [0,1]
# spoilage_multiplier: Arrhenius-based degradation rate multiplier

SEVERITY_DEFINITIONS = [
    {
        "level": 1, "name": "Normal Monsoon Day",
        "rain_max": 5.0, "wind_max": 25.0,
        "speed_factor": 1.08, "capacity_factor": 0.95, "spoilage_multiplier": 1.00,
        "supplier_accessibility": {"seafood":1,"vegetables":1,"meat":1,"general":1},
        "emergency_feasible": True, "road_closure_prob": 0.00,
    },
    {
        "level": 2, "name": "Light Rain",
        "rain_max": 20.0, "wind_max": 40.0,
        "speed_factor": 1.15, "capacity_factor": 0.90, "spoilage_multiplier": 1.05,
        "supplier_accessibility": {"seafood":1,"vegetables":1,"meat":1,"general":1},
        "emergency_feasible": True, "road_closure_prob": 0.01,
    },
    {
        "level": 3, "name": "Moderate Rain",
        "rain_max": 50.0, "wind_max": 60.0,
        "speed_factor": 1.25, "capacity_factor": 0.80, "spoilage_multiplier": 1.15,
        "supplier_accessibility": {"seafood":1,"vegetables":1,"meat":1,"general":1},
        "emergency_feasible": True, "road_closure_prob": 0.07,
    },
    {
        "level": 4, "name": "Heavy Rain",
        "rain_max": 100.0, "wind_max": 90.0,
        "speed_factor": 1.55, "capacity_factor": 0.60, "spoilage_multiplier": 1.30,
        "supplier_accessibility": {"seafood":0,"vegetables":0,"meat":1,"general":1},
        "emergency_feasible": True, "road_closure_prob": 0.40,
    },
    {
        "level": 5, "name": "Tropical Storm/Typhoon",
        "rain_max": float("inf"), "wind_max": float("inf"),
        "speed_factor": 2.20, "capacity_factor": 0.10, "spoilage_multiplier": 2.00,
        "supplier_accessibility": {"seafood":0,"vegetables":0,"meat":0,"general":1},
        "emergency_feasible": False, "road_closure_prob": 0.90,
    },
]


@dataclass
class GeneratedWeatherScenario:
    """
    One data-driven weather scenario with full operational parameterisation.
    Interface is drop-in compatible with existing project WeatherScenario.
    """
    scenario_id:    int
    name:           str
    probability:    float
    rainfall_mm:    float
    temperature_c:  float
    wind_kmh:       float
    severity_level: int
    speed_reduction_factor:    float
    capacity_reduction_factor: float
    spoilage_multiplier:       float
    supplier_accessibility:    Dict[str, int]
    emergency_feasible:        bool
    road_closure_prob:         float
    source_season:  str = ""
    generation_method: str = "LHS+FFS"

    def get_supplier_accessible(self, subtype: str) -> int:
        return self.supplier_accessibility.get(subtype, 1)

    def get_dc_accessible(self, dc_id: str) -> int:
        """Hoa Khanh inaccessible at severity ≥ 4; Lien Chieu always accessible."""
        if self.severity_level >= 4:
            return 1 if "lienchieu" in dc_id.lower() else 0
        return 1


@dataclass
class ScenarioQualityReport:
    """Quality metrics for a reduced scenario set (evaluated before force_extreme)."""
    n_original: int
    n_reduced:  int
    season:     str
    # Moment preservation vs original LHS distribution
    rain_mean_original:  float
    rain_mean_reduced:   float
    rain_mean_error_pct: float
    rain_var_original:   float
    rain_var_reduced:    float
    rain_var_error_pct:  float
    rain_p90_original:   float
    # Probability & extremes (after force_extreme)
    prob_sum:             float
    has_extreme_scenario: bool   # any scenario severity >= 4
    level_dist:           Dict[int, float]
    wasserstein_distance: float
    # Pass/fail (distribution fit only — extreme coverage checked separately)
    dist_fit_passes: bool = False

    def summary(self) -> str:
        fit_status = "✓ PASS" if self.dist_fit_passes else "⚠ WARN"
        return (
            f"Scenario Reduction Quality [{fit_status}]\n"
            f"  {self.n_original} → {self.n_reduced} scenarios  ({self.season})\n"
            f"  Rainfall mean: {self.rain_mean_original:.2f}→{self.rain_mean_reduced:.2f}mm "
            f"  Δ={self.rain_mean_error_pct:+.1f}% (threshold ±5%)\n"
            f"  Rainfall var:  {self.rain_var_original:.1f}→{self.rain_var_reduced:.1f} "
            f"  Δ={self.rain_var_error_pct:+.1f}% (threshold ±15%)\n"
            f"  Prob sum: {self.prob_sum:.8f} | "
            f"Extreme (L4/L5): {'Yes' if self.has_extreme_scenario else 'No (forced)'}\n"
            f"  Wasserstein dist: {self.wasserstein_distance:.4f}\n"
            f"  Level probs: "
            + "  ".join(f"L{k}={v:.0%}" for k,v in sorted(self.level_dist.items()))
        )


class WeatherScenarioGenerator:
    """
    Generates representative weather scenarios via LHS + Fast Forward Selection.
    """

    def __init__(self, weather_data: DaNangWeatherData):
        self.wd = weather_data
        if not weather_data.distributions:
            raise ValueError("Call weather_data.fit_seasonal_distributions() first")

    # ── 1. LHS Sampling ───────────────────────────────────────────────────────

    def generate_lhs_samples(
        self, n_samples: int, season: str, seed: int = 2024
    ) -> pd.DataFrame:
        """
        Generate n_samples candidates via 3D Latin Hypercube Sampling in
        (rainfall, temperature, wind_speed) space.

        Rainfall dimension uses a zero-inflated Gamma:
          P(rain=0)  = zero_day_prob  (estimated from historical data)
          P(rain>0)  = Gamma(shape, scale) rescaled to (1−zero_day_prob)
        This faithfully represents the bimodal rainfall distribution
        (many dry days + heavy rain events) without mixing distributions.
        """
        if season not in self.wd.distributions:
            raise ValueError(f"No distributions for season '{season}'")
        d: SeasonalDistributions = self.wd.distributions[season]

        sampler = qmc.LatinHypercube(d=3, scramble=True, seed=seed)
        u = sampler.random(n=n_samples)   # [0,1]^3

        # Dimension 0: Zero-inflated Gamma for rainfall
        zdp = d.rain_zero_day_prob
        rain = np.where(
            u[:, 0] <= zdp,
            0.0,
            stats.gamma.ppf(
                (u[:, 0] - zdp) / max(1.0 - zdp, 1e-8),
                a=d.rain_shape, scale=d.rain_scale
            )
        )
        rain = np.clip(rain, 0.0, None)

        # Dimension 1: Normal temperature
        temp = stats.norm.ppf(u[:, 1], loc=d.temp_loc, scale=d.temp_scale)

        # Dimension 2: Weibull wind speed
        wind = np.clip(
            stats.weibull_min.ppf(u[:, 2], c=d.wind_c, loc=0, scale=d.wind_scale),
            0.0, None
        )

        df = pd.DataFrame({"rainfall_mm": rain, "temperature_c": temp, "wind_kmh": wind})
        df["severity_level"] = df.apply(
            lambda r: self.classify_severity(r["rainfall_mm"], r["wind_kmh"]), axis=1
        )

        level_dist = df["severity_level"].value_counts(normalize=True).sort_index()
        logger.info(
            f"LHS {season}: {n_samples} samples | mean_rain={rain.mean():.1f}mm | "
            + " ".join(f"L{l}={p:.0%}" for l,p in level_dist.items())
        )
        return df

    # ── 2. Severity Classification ────────────────────────────────────────────

    def classify_severity(self, rainfall_mm: float, wind_kmh: float = 0.0) -> int:
        if rainfall_mm > 100.0 or wind_kmh > 90.0:  return 5
        if rainfall_mm >  50.0 or wind_kmh > 60.0:  return 4
        if rainfall_mm >  20.0 or wind_kmh > 40.0:  return 3
        if rainfall_mm >   5.0 or wind_kmh > 25.0:  return 2
        return 1

    def _severity_params(self, level: int) -> Dict:
        for sd in SEVERITY_DEFINITIONS:
            if sd["level"] == level:
                return sd
        raise ValueError(f"Unknown severity level: {level}")

    # ── 3. Fast Forward Selection (Backward Reduction) ────────────────────────

    def reduce_scenarios(
        self,
        candidates_df:  pd.DataFrame,
        target_count:   int = 8,
        feature_cols:   Optional[List[str]] = None,
        weights:        Optional[Dict[str, float]] = None,
    ) -> List[GeneratedWeatherScenario]:
        """
        Heitsch & Römisch (2003) backward reduction: iteratively remove the
        scenario with minimum probability × distance-to-nearest-neighbor,
        redistributing its probability to that neighbor.

        Distance: weighted Euclidean in normalised feature space.
        Weights give severity_level 4× importance to preserve operational diversity
        (scenarios at different severity levels have qualitatively different fleet/
        supplier impacts — keeping them distinct matters more than rainfall mm accuracy).
        """
        if feature_cols is None:
            feature_cols = ["rainfall_mm", "temperature_c", "wind_kmh", "severity_level"]
        if weights is None:
            weights = {"rainfall_mm":3.0, "temperature_c":1.0,
                       "wind_kmh":2.0, "severity_level":4.0}

        N = len(candidates_df)
        if N <= target_count:
            target_count = N

        logger.info(f"Fast Forward Selection: {N} → {target_count} scenarios")

        X = candidates_df[feature_cols].values.astype(float)
        w = np.array([weights.get(c, 1.0) for c in feature_cols])
        xr = X.max(0) - X.min(0); xr[xr == 0] = 1.0
        X_norm = (X - X.min(0)) / xr * w

        probs     = np.ones(N) / N
        remaining = set(range(N))
        D         = self._pairwise_distance(X_norm)

        for step in range(N - target_count):
            rem = sorted(remaining)
            best_cost, rm_idx, nn_idx = float("inf"), -1, -1
            for i in rem:
                nn = min((j for j in rem if j != i), key=lambda j: D[i, j])
                cost = probs[i] * D[i, nn]
                if cost < best_cost:
                    best_cost, rm_idx = cost, i
                    nn_idx = nn
            probs[nn_idx] += probs[rm_idx]
            remaining.remove(rm_idx)

        final_idx   = sorted(remaining)
        final_probs = probs[final_idx] / probs[final_idx].sum()   # normalize

        scenarios = []
        for rank, (oi, prob) in enumerate(zip(final_idx, final_probs)):
            row = candidates_df.iloc[oi]
            lvl = int(row["severity_level"])
            sp  = self._severity_params(lvl)
            scenarios.append(GeneratedWeatherScenario(
                scenario_id=rank+1, name=sp["name"], probability=float(prob),
                rainfall_mm=float(row["rainfall_mm"]), temperature_c=float(row["temperature_c"]),
                wind_kmh=float(row["wind_kmh"]), severity_level=lvl,
                speed_reduction_factor=sp["speed_factor"],
                capacity_reduction_factor=sp["capacity_factor"],
                spoilage_multiplier=sp["spoilage_multiplier"],
                supplier_accessibility=sp["supplier_accessibility"].copy(),
                emergency_feasible=sp["emergency_feasible"],
                road_closure_prob=sp["road_closure_prob"],
            ))
        logger.info(f"  ✓ Reduction complete | prob_sum={sum(s.probability for s in scenarios):.8f}")
        return scenarios

    def _pairwise_distance(self, X: np.ndarray) -> np.ndarray:
        sq = np.sum(X**2, axis=1, keepdims=True)
        D2 = np.clip(sq + sq.T - 2.0 * (X @ X.T), 0.0, None)
        return np.sqrt(D2)

    # ── 4. Forced Extreme Scenario ────────────────────────────────────────────

    def force_extreme_scenario(
        self,
        scenarios: List[GeneratedWeatherScenario],
        season:    str,
        min_prob:  float = 0.05,
    ) -> List[GeneratedWeatherScenario]:
        """
        Guarantee at least one Level 4 AND one Level 5 scenario in the set.
        Uses historical typhoon frequency for probability assignment:
          Monsoon: L5 ≈ 10% (concentrated Oct–Nov), L4 ≈ 15%
          Dry:     L5 ≈ 2%,  L4 ≈ 3%
        Replaces lowest-probability scenarios and renormalises.
        This is a deliberate tail-safety step — quality validation runs BEFORE this.
        """
        hist = {"monsoon": {"L4": 0.15, "L5": 0.10},
                "dry":     {"L4": 0.03, "L5": 0.02}}
        hp = hist.get(season, {"L4": 0.05, "L5": 0.05})
        d  = self.wd.distributions.get(season)

        def _make_scenario(level: int, label: str, rain: float,
                           wind: float, temp: float, prob: float):
            sp = self._severity_params(level)
            return GeneratedWeatherScenario(
                scenario_id=-1, name=sp["name"] + f" ({label})",
                probability=prob, rainfall_mm=rain, temperature_c=temp, wind_kmh=wind,
                severity_level=level,
                speed_reduction_factor=sp["speed_factor"],
                capacity_reduction_factor=sp["capacity_factor"],
                spoilage_multiplier=sp["spoilage_multiplier"],
                supplier_accessibility=sp["supplier_accessibility"].copy(),
                emergency_feasible=sp["emergency_feasible"],
                road_closure_prob=sp["road_closure_prob"],
                source_season=season,
            )

        # Representative rain/wind from distribution p90/p99
        if d and d.rain_scale > 0:
            p90r = float(stats.gamma.ppf(0.90, d.rain_shape, scale=d.rain_scale))
            p99r = float(stats.gamma.ppf(0.99, d.rain_shape, scale=d.rain_scale))
            p90w = float(stats.weibull_min.ppf(0.90, d.wind_c, scale=d.wind_scale))
            p99w = float(stats.weibull_min.ppf(0.99, d.wind_c, scale=d.wind_scale))
            tref = d.temp_loc - 2.0
        else:
            p90r, p99r, p90w, p99w, tref = 65.0, 120.0, 55.0, 100.0, 23.0

        needed = []
        if not any(s.severity_level >= 5 for s in scenarios):
            needed.append(_make_scenario(5, "data-driven", max(p99r, 110.0),
                                         max(p99w, 95.0), tref, max(min_prob, hp["L5"])))
        if not any(s.severity_level >= 4 for s in scenarios):
            needed.append(_make_scenario(4, "data-driven", max(p90r, 55.0),
                                         max(p90w, 55.0), tref + 1.0, max(min_prob, hp["L4"])))

        if not needed:
            return scenarios

        # Remove lowest-probability non-extreme scenarios to make room
        non_ex = sorted([s for s in scenarios if s.severity_level < 4],
                        key=lambda s: s.probability)
        for forced in needed:
            if non_ex:
                removed = non_ex.pop(0)
                scenarios = [s for s in scenarios if s.scenario_id != removed.scenario_id]
            scenarios.append(forced)
            logger.info(f"  Injected forced L{forced.severity_level} scenario "
                        f"(p={forced.probability:.2f}, rain={forced.rainfall_mm:.0f}mm)")

        # Renormalise
        total = sum(s.probability for s in scenarios)
        for s in scenarios:
            s.probability /= total
        return scenarios

    # ── 5. Quality Validation ─────────────────────────────────────────────────

    def validate_scenario_quality(
        self,
        original_df: pd.DataFrame,
        scenarios:   List[GeneratedWeatherScenario],
    ) -> ScenarioQualityReport:
        """
        Validate reduced set against original LHS distribution.

        NOTE: Call this BEFORE force_extreme_scenario. Variance increase from
        forced L4/L5 injection is intentional tail safety, not fitting failure.

        Thresholds:
          Mean error  < 5%   (Dupačová et al. 2003)
          Variance err < 15% (relaxed from 10%; Gamma distribution has high
                              natural variance that FFS concentrates into few points)
        """
        rain_orig  = original_df["rainfall_mm"].values
        probs      = np.array([s.probability for s in scenarios])
        rain_red   = np.array([s.rainfall_mm  for s in scenarios])

        mu_o, mu_r    = float(rain_orig.mean()), float((probs * rain_red).sum())
        var_o         = float(rain_orig.var())
        var_r         = float((probs * (rain_red - mu_r)**2).sum())
        mu_err        = abs(mu_r - mu_o) / max(mu_o, 1e-6) * 100
        var_err       = abs(var_r - var_o) / max(var_o, 1e-6) * 100
        p90_o         = float(np.percentile(rain_orig, 90))
        prob_sum      = float(probs.sum())
        has_extreme   = any(s.severity_level >= 4 for s in scenarios)
        level_dist    = {}
        for s in scenarios:
            level_dist[s.severity_level] = level_dist.get(s.severity_level, 0.0) + s.probability
        ws = self._wasserstein1(rain_orig, rain_red, probs)

        passes = mu_err < 5.0 and var_err < 15.0 and abs(prob_sum - 1.0) < 1e-4

        report = ScenarioQualityReport(
            n_original=len(original_df), n_reduced=len(scenarios), season="",
            rain_mean_original=mu_o, rain_mean_reduced=mu_r, rain_mean_error_pct=mu_err,
            rain_var_original=var_o, rain_var_reduced=var_r, rain_var_error_pct=var_err,
            rain_p90_original=p90_o, prob_sum=prob_sum, has_extreme_scenario=has_extreme,
            level_dist=level_dist, wasserstein_distance=ws, dist_fit_passes=passes,
        )
        log = logger.info if passes else logger.warning
        log(f"\n{report.summary()}")
        if not passes:
            logger.warning(
                "  → Tip: increase target_count or n_samples to improve moment preservation"
            )
        return report

    def _wasserstein1(
        self, rain_orig: np.ndarray, rain_red: np.ndarray, probs: np.ndarray
    ) -> float:
        orig_sorted = np.sort(rain_orig)
        orig_cdf    = np.arange(1, len(orig_sorted)+1) / len(orig_sorted)
        order       = np.argsort(rain_red)
        red_sorted  = rain_red[order]
        red_cdf     = np.cumsum(probs[order])
        grid = np.union1d(orig_sorted, red_sorted)
        c1   = np.interp(grid, orig_sorted, orig_cdf, left=0, right=1)
        c2   = np.interp(grid, red_sorted,  red_cdf,  left=0, right=1)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(np.abs(c1 - c2), grid))

    # ── 6. Full Pipeline ──────────────────────────────────────────────────────

    def generate_scenarios(
        self,
        season:        str   = "monsoon",
        n_samples:     int   = 600,
        target_count:  int   = 8,
        seed:          int   = 2024,
        force_extreme: bool  = True,
        min_extreme_prob: float = 0.05,
    ) -> Tuple[List[GeneratedWeatherScenario], ScenarioQualityReport]:
        """
        Full pipeline: LHS → severity mapping → FFS reduction →
                       quality validation → force extreme → return.

        Quality validation runs on the pure FFS output (before forced injection)
        so the report reflects true distribution-fitting quality.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating {target_count} scenarios | {season.upper()} | "
                    f"LHS n={n_samples} | seed={seed}")
        logger.info(f"{'='*60}")

        candidates = self.generate_lhs_samples(n_samples, season, seed=seed)
        scenarios  = self.reduce_scenarios(candidates, target_count=target_count)
        for s in scenarios:
            s.source_season = season

        # ── Validate BEFORE forced injection ──
        quality        = self.validate_scenario_quality(candidates, scenarios)
        quality.season = season

        # ── Force extreme scenarios if needed ──
        if force_extreme:
            scenarios = self.force_extreme_scenario(scenarios, season, min_extreme_prob)

        # Sort by severity, reassign IDs
        scenarios.sort(key=lambda s: (s.severity_level, -s.probability))
        for i, sc in enumerate(scenarios):
            sc.scenario_id = i + 1

        logger.info(f"\n✓ Final {len(scenarios)} scenarios ({season})")
        self._print_table(scenarios)
        return scenarios, quality

    def _print_table(self, scenarios: List[GeneratedWeatherScenario]) -> None:
        hdr = (f"{'ID':>3}  {'Name':30}  {'p':>6}  {'Rain':>7}  {'Temp':>5}  "
               f"{'Wind':>5}  {'L':>2}  {'SpeedF':>7}  {'CapF':>5}  "
               f"{'Spoil':>5}  {'Emerg':>5}")
        logger.info(hdr)
        logger.info("-" * len(hdr))
        for s in scenarios:
            logger.info(
                f"{s.scenario_id:>3}  {s.name:30}  {s.probability:>6.3f}  "
                f"{s.rainfall_mm:>7.1f}  {s.temperature_c:>5.1f}  {s.wind_kmh:>5.1f}  "
                f"{s.severity_level:>2}  {s.speed_reduction_factor:>7.2f}  "
                f"{s.capacity_reduction_factor:>5.2f}  {s.spoilage_multiplier:>5.2f}  "
                f"{'Yes' if s.emergency_feasible else 'No':>5}"
            )

    def scenarios_to_dataframe(
        self, scenarios: List[GeneratedWeatherScenario]
    ) -> pd.DataFrame:
        return pd.DataFrame([{
            "scenario_id": s.scenario_id, "name": s.name, "probability": s.probability,
            "severity_level": s.severity_level, "rainfall_mm": s.rainfall_mm,
            "temperature_c": s.temperature_c, "wind_kmh": s.wind_kmh,
            "speed_reduction_factor": s.speed_reduction_factor,
            "capacity_reduction_factor": s.capacity_reduction_factor,
            "spoilage_multiplier": s.spoilage_multiplier,
            "emergency_feasible": s.emergency_feasible,
            "road_closure_prob": s.road_closure_prob,
            "acc_seafood":    s.supplier_accessibility.get("seafood", 1),
            "acc_vegetables": s.supplier_accessibility.get("vegetables", 1),
            "acc_meat":       s.supplier_accessibility.get("meat", 1),
            "acc_general":    s.supplier_accessibility.get("general", 1),
            "source_season":  s.source_season,
        } for s in scenarios])