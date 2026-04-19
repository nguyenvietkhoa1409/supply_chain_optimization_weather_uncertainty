"""
weather_data.py — Da Nang Historical Weather Data & Distribution Fitting
FIXED VERSION

Bug fixes applied:
  [FIX-1] Cache loading path now sets df.attrs["source"] = "api" by checking
           the _source column saved by the API path. Fixes KeyError on df.attrs['source'].
  [FIX-2] fit_seasonal_distributions() now detects source from BOTH attrs AND
           _source column in DataFrame, so distributions saved with correct source label
           even when loaded from CSV cache (which doesn't preserve attrs).
"""

from __future__ import annotations
import json, logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DANANG_LAT = 16.068
DANANG_LON = 108.221
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
MONSOON_MONTHS = {9, 10, 11, 12, 1}
DRY_MONTHS     = {2, 3, 4, 5, 6, 7, 8}
KS_FALLBACK_RATIO = 4.0

EMBEDDED_DISTRIBUTIONS: Dict[str, Dict] = {
    "dry": {
        "rainfall":    {"dist":"gamma",       "shape":0.52, "scale":7.8,  "loc":0.0, "zero_day_prob":0.40},
        "temperature": {"dist":"norm",        "loc":28.4,   "scale":2.3},
        "wind_speed":  {"dist":"weibull_min", "c":1.8,      "scale":12.0, "loc":0.0},
    },
    "monsoon": {
        "rainfall":    {"dist":"gamma",       "shape":0.68, "scale":28.5, "loc":0.0, "zero_day_prob":0.25},
        "temperature": {"dist":"norm",        "loc":25.1,   "scale":1.8},
        "wind_speed":  {"dist":"weibull_min", "c":1.6,      "scale":16.5, "loc":0.0},
    },
}

DANANG_MONTHLY_STATS = pd.DataFrame({
    "month":       list(range(1,13)),
    "month_name":  ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "mean_temp_c": [22.5,23.0,25.0,27.0,29.0,30.5,30.0,29.5,28.0,26.0,24.5,22.5],
    "rainfall_mm": [85,25,20,30,60,85,90,115,310,650,430,215],
    "rainy_days":  [12,6,4,5,9,10,11,11,14,20,20,18],
    "typhoon_prob":[0.03,0.01,0.01,0.01,0.03,0.03,0.05,0.05,0.15,0.28,0.18,0.05],
    "season":      ["monsoon","dry","dry","dry","dry","dry","dry","dry","monsoon","monsoon","monsoon","monsoon"],
})


@dataclass
class SeasonalDistributions:
    season: str
    n_days: int = 0
    n_rainy: int = 0
    date_range: Tuple[str, str] = ("", "")
    rain_shape: float = 0.0
    rain_scale: float = 0.0
    rain_zero_day_prob: float = 0.0
    rain_ks_stat: float = 0.0
    rain_ks_critical: float = 0.0
    rain_ks_ratio: float = 0.0
    rain_ks_pvalue: float = 0.0
    temp_loc: float = 0.0
    temp_scale: float = 0.0
    temp_ks_ratio: float = 0.0
    temp_ks_pvalue: float = 0.0
    wind_c: float = 0.0
    wind_scale: float = 0.0
    wind_ks_ratio: float = 0.0
    wind_ks_pvalue: float = 0.0
    rain_mean: float = 0.0
    rain_p95: float = 0.0
    rain_p99: float = 0.0
    temp_mean: float = 0.0
    wind_mean: float = 0.0
    source: str = "api"

    def to_dict(self) -> Dict: return asdict(self)

    def fit_quality(self) -> str:
        r = self.rain_ks_ratio
        if r == 0: return "unknown"
        if r < 2.0: return "excellent"
        if r < 4.0: return "good"
        return "poor"


class DaNangWeatherData:
    """Manages historical Da Nang weather data and distribution fitting."""

    def __init__(self, cache_dir: Optional[str] = None, request_timeout: int = 60):
        self.lat, self.lon = DANANG_LAT, DANANG_LON
        self.raw_df: Optional[pd.DataFrame] = None
        self.distributions: Dict[str, SeasonalDistributions] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._timeout = request_timeout

    # ── [FIX-1] Helper: detect data source from DataFrame ─────────────────
    @staticmethod
    def _detect_source(df: pd.DataFrame) -> str:
        """
        Detect whether a DataFrame originated from the API or synthetic generation.

        Priority order:
          1. df.attrs["source"]  — set in-memory by API/synthetic path (lost after CSV round-trip)
          2. df["_source"] column — persisted to CSV only by the API save path
          3. Fallback: "unknown"
        """
        # In-memory attrs (reliable when not loaded from CSV)
        if df.attrs.get("source") in ("api", "synthetic"):
            return df.attrs["source"]
        # Column persisted to CSV by API path: df["_source"] = "api"
        if "_source" in df.columns:
            unique_vals = df["_source"].dropna().unique()
            if "api" in unique_vals:
                return "api"
        return "unknown"

    def fetch_historical_data(
        self, start_date: str = "2014-01-01", end_date: str = "2023-12-31",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch daily ERA5 reanalysis data from Open-Meteo (free, no auth)."""

        # ── [FIX-1] Cache path: set attrs from _source column ─────────────
        if use_cache and self.cache_dir:
            cf = self.cache_dir / f"danang_{start_date}_{end_date}.csv"
            if cf.exists():
                logger.info(f"Loading cached data: {cf}")
                df = pd.read_csv(cf, parse_dates=["date"])
                # Detect and restore source tag lost during CSV serialisation
                detected = self._detect_source(df)
                df.attrs["source"] = detected
                logger.info(f"  Cache source detected: {detected} ({len(df)} records)")
                self.raw_df = df
                return self.raw_df

        try:
            import requests
            params = {
                "latitude": self.lat, "longitude": self.lon,
                "start_date": start_date, "end_date": end_date,
                "daily": ["precipitation_sum","temperature_2m_max",
                          "temperature_2m_min","windspeed_10m_max"],
                "timezone": "Asia/Ho_Chi_Minh",
            }
            logger.info(f"Fetching Open-Meteo: {start_date} → {end_date}")
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=self._timeout)
            resp.raise_for_status()
            daily = resp.json().get("daily", {})
            if not daily or "time" not in daily:
                raise ValueError("Empty API response")
            df = pd.DataFrame({
                "date":         pd.to_datetime(daily["time"]),
                "rainfall_mm":  daily.get("precipitation_sum", []),
                "temp_max_c":   daily.get("temperature_2m_max", []),
                "temp_min_c":   daily.get("temperature_2m_min", []),
                "wind_max_kmh": daily.get("windspeed_10m_max", []),
            })
            df = self._clean_and_enrich(df)
            df.attrs["source"] = "api"
            logger.info(f"✓ {len(df)} records ({len(df)/365:.1f} years)")
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                df["_source"] = "api"   # persist source tag to CSV
                df.to_csv(self.cache_dir / f"danang_{start_date}_{end_date}.csv", index=False)

        except Exception as exc:
            logger.warning(f"API unavailable ({exc.__class__.__name__}) — using synthetic data")
            df = self._build_synthetic_raw_df(start_date, end_date)

        self.raw_df = df
        return df

    def _clean_and_enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rainfall_mm"]  = df["rainfall_mm"].clip(lower=0.0).fillna(0.0)
        df["temp_max_c"]   = df["temp_max_c"].ffill()
        df["temp_min_c"]   = df["temp_min_c"].ffill()
        df["wind_max_kmh"] = df["wind_max_kmh"].fillna(df["wind_max_kmh"].median())
        df["temp_mean_c"]  = (df["temp_max_c"] + df["temp_min_c"]) / 2.0
        df["month"]        = df["date"].dt.month
        df["season"]       = df["month"].apply(lambda m: "monsoon" if m in MONSOON_MONTHS else "dry")
        return df

    def _build_synthetic_raw_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info("  Generating synthetic data from embedded statistics (offline mode)")
        rng = np.random.default_rng(42)
        dates = pd.date_range(start_date, end_date, freq="D")
        rows = []
        for d in dates:
            season = "monsoon" if d.month in MONSOON_MONTHS else "dry"
            ep = EMBEDDED_DISTRIBUTIONS[season]
            rain = 0.0 if rng.random() < ep["rainfall"]["zero_day_prob"] else \
                   max(0.0, float(rng.gamma(ep["rainfall"]["shape"], ep["rainfall"]["scale"])))
            temp = float(rng.normal(ep["temperature"]["loc"], ep["temperature"]["scale"]))
            wind = max(0.0, float(stats.weibull_min.rvs(
                ep["wind_speed"]["c"], scale=ep["wind_speed"]["scale"],
                random_state=int(rng.integers(0, 2**31)))))
            rows.append({"date":d,"rainfall_mm":rain,"temp_max_c":temp+2.5,
                         "temp_min_c":temp-2.5,"temp_mean_c":temp,
                         "wind_max_kmh":wind,"month":d.month,
                         "season":season})
        df = pd.DataFrame(rows)
        df.attrs["source"] = "synthetic"
        self.raw_df = df
        return df

    def fit_seasonal_distributions(self) -> Dict[str, SeasonalDistributions]:
        """
        Fit Gamma/Normal/Weibull distributions by season.
        [FIX-2] Source detection now uses _detect_source() covering both
                df.attrs and the _source column — fixes wrong "synthetic" label
                when data was loaded from CSV cache.
        """
        if self.raw_df is None:
            raise ValueError("No data — call fetch_historical_data() first")
        results = {}

        # ── [FIX-2] Robust source detection ───────────────────────────────
        detected_source = self._detect_source(self.raw_df)
        is_api = (detected_source == "api")
        logger.info(f"Distribution fitting: data source = '{detected_source}' "
                    f"(is_api={is_api})")

        for season in ["dry", "monsoon"]:
            sub = self.raw_df[self.raw_df["season"] == season].copy()
            n   = len(sub)
            logger.info(f"\nFitting {season.upper()} (n={n} days)")
            d = SeasonalDistributions(
                season=season, n_days=n, n_rainy=0,
                date_range=(str(sub["date"].min().date()), str(sub["date"].max().date())),
                source="api" if is_api else "synthetic",
            )

            # Rainfall — Gamma on rainy days
            rain_all = sub["rainfall_mm"].values
            rainy    = rain_all[rain_all > 0.1]
            d.n_rainy           = len(rainy)
            d.rain_zero_day_prob = float((rain_all <= 0.1).mean())
            d.rain_mean = float(rain_all.mean())
            d.rain_p95  = float(np.percentile(rain_all, 95))
            d.rain_p99  = float(np.percentile(rain_all, 99))

            if len(rainy) >= 50:
                shape, loc, scale = stats.gamma.fit(rainy, floc=0)
                ks_s, ks_p = stats.kstest(rainy, "gamma", args=(shape, loc, scale))
                crit = 1.358 / np.sqrt(len(rainy))
                ratio = ks_s / crit
                d.rain_shape=float(shape); d.rain_scale=float(scale)
                d.rain_ks_stat=float(ks_s); d.rain_ks_critical=float(crit)
                d.rain_ks_ratio=float(ratio); d.rain_ks_pvalue=float(ks_p)
                if ratio > KS_FALLBACK_RATIO:
                    logger.warning(f"  ⚠ Rainfall KS ratio={ratio:.2f} — poor fit, using embedded")
                    ep = EMBEDDED_DISTRIBUTIONS[season]["rainfall"]
                    d.rain_shape, d.rain_scale = ep["shape"], ep["scale"]
                    d.source = "embedded_fallback"
                else:
                    q = "excellent" if ratio < 2 else "good"
                    logger.info(f"  ✓ Rainfall Gamma(α={shape:.3f}, β={scale:.2f}mm) "
                                f"KS_ratio={ratio:.2f} [{q}] p={ks_p:.4f}")
            else:
                ep = EMBEDDED_DISTRIBUTIONS[season]["rainfall"]
                d.rain_shape, d.rain_scale = ep["shape"], ep["scale"]
                d.source = "embedded_fallback"

            # Temperature — Normal
            temp = sub["temp_mean_c"].values
            d.temp_mean = float(temp.mean())
            if len(temp) >= 30:
                lt, st = stats.norm.fit(temp)
                ks_t, pt = stats.kstest(temp, "norm", args=(lt, st))
                d.temp_loc=float(lt); d.temp_scale=float(st)
                d.temp_ks_ratio=float(ks_t/(1.358/np.sqrt(len(temp))))
                d.temp_ks_pvalue=float(pt)
                logger.info(f"  ✓ Temperature Normal(μ={lt:.1f}°C, σ={st:.2f}) "
                             f"KS_ratio={d.temp_ks_ratio:.2f}")
            else:
                ep = EMBEDDED_DISTRIBUTIONS[season]["temperature"]
                d.temp_loc, d.temp_scale = ep["loc"], ep["scale"]

            # Wind — Weibull
            wind = sub["wind_max_kmh"].values
            d.wind_mean = float(wind.mean())
            if len(wind) >= 30:
                try:
                    c, lw, sw = stats.weibull_min.fit(wind, floc=0)
                    ks_w, pw = stats.kstest(wind, "weibull_min", args=(c, lw, sw))
                    d.wind_c=float(c); d.wind_scale=float(sw)
                    d.wind_ks_ratio=float(ks_w/(1.358/np.sqrt(len(wind))))
                    d.wind_ks_pvalue=float(pw)
                    logger.info(f"  ✓ Wind Weibull(k={c:.2f}, λ={sw:.2f}km/h) "
                                 f"KS_ratio={d.wind_ks_ratio:.2f}")
                except Exception as e:
                    ep = EMBEDDED_DISTRIBUTIONS[season]["wind_speed"]
                    d.wind_c, d.wind_scale = ep["c"], ep["scale"]
                    logger.warning(f"  ⚠ Wind fit failed ({e}) — embedded")
            else:
                ep = EMBEDDED_DISTRIBUTIONS[season]["wind_speed"]
                d.wind_c, d.wind_scale = ep["c"], ep["scale"]

            logger.info(f"  rain_mean={d.rain_mean:.1f}mm p95={d.rain_p95:.0f}mm "
                        f"p99={d.rain_p99:.0f}mm zero_day={d.rain_zero_day_prob:.1%} "
                        f"fit={d.fit_quality()}")
            results[season] = d

        self.distributions = results
        return results

    def load_embedded_distributions(self) -> Dict[str, SeasonalDistributions]:
        results = {}
        for season in ["dry", "monsoon"]:
            ep = EMBEDDED_DISTRIBUTIONS[season]
            d = SeasonalDistributions(
                season=season, n_days=0, n_rainy=0,
                date_range=("embedded","embedded"),
                rain_shape=ep["rainfall"]["shape"], rain_scale=ep["rainfall"]["scale"],
                rain_zero_day_prob=ep["rainfall"]["zero_day_prob"],
                rain_ks_ratio=1.0,
                temp_loc=ep["temperature"]["loc"], temp_scale=ep["temperature"]["scale"],
                wind_c=ep["wind_speed"]["c"], wind_scale=ep["wind_speed"]["scale"],
                rain_mean=ep["rainfall"]["shape"]*ep["rainfall"]["scale"],
                rain_p95=float(stats.gamma.ppf(0.95,ep["rainfall"]["shape"],scale=ep["rainfall"]["scale"])),
                rain_p99=float(stats.gamma.ppf(0.99,ep["rainfall"]["shape"],scale=ep["rainfall"]["scale"])),
                temp_mean=ep["temperature"]["loc"],
                wind_mean=float(stats.weibull_min.mean(ep["wind_speed"]["c"],scale=ep["wind_speed"]["scale"])),
                source="embedded_fallback",
            )
            results[season] = d
        self.distributions = results
        return results

    def save_distributions(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({s:d.to_dict() for s,d in self.distributions.items()},
                                          indent=2, default=str))
        logger.info(f"Distributions saved: {path}")

    def load_distributions(self, path: str) -> Dict[str, SeasonalDistributions]:
        raw = json.loads(Path(path).read_text())
        self.distributions = {s: SeasonalDistributions(**v) for s,v in raw.items()}
        return self.distributions

    def get_monthly_statistics(self) -> pd.DataFrame:
        if self.raw_df is None:
            return DANANG_MONTHLY_STATS.copy()
        return self.raw_df.groupby("month").agg(
            mean_temp_c=("temp_mean_c","mean"),
            rainfall_mm=("rainfall_mm","sum"),
            rainy_days=("rainfall_mm",lambda x:(x>1.0).sum()),
            rain_p90=("rainfall_mm",lambda x:np.percentile(x,90)),
        ).reset_index()

    def summary(self) -> str:
        lines = ["="*72, "Da Nang Weather Distributions", "="*72]
        for s, d in self.distributions.items():
            lines += [
                f"\n{s.upper()} SEASON (n={d.n_days} days, source={d.source})",
                f"  Rainfall:    Gamma(α={d.rain_shape:.3f}, β={d.rain_scale:.2f}mm) "
                f"mean={d.rain_mean:.1f}mm p95={d.rain_p95:.0f}mm p99={d.rain_p99:.0f}mm "
                f"zero_day={d.rain_zero_day_prob:.1%} KS_ratio={d.rain_ks_ratio:.2f} [{d.fit_quality()}]",
                f"  Temperature: Normal(μ={d.temp_loc:.1f}°C, σ={d.temp_scale:.2f}) "
                f"mean={d.temp_mean:.1f}°C KS_ratio={d.temp_ks_ratio:.2f}",
                f"  Wind:        Weibull(k={d.wind_c:.2f}, λ={d.wind_scale:.2f}km/h) "
                f"mean={d.wind_mean:.1f}km/h KS_ratio={d.wind_ks_ratio:.2f}",
            ]
        lines.append("="*72)
        return "\n".join(lines)

    def compute_historical_severity_frequencies(
        self,
        season: str = "monsoon"
    ) -> Dict[int, Dict]:
        """
        Tính tần suất lịch sử của từng severity level từ ERA5 data.
        
        Sử dụng Wilson Score Interval để đánh giá Confidence Interval (95%).
        """
        if self.raw_df is None:
            raise ValueError("No data — call fetch_historical_data() first")
            
        import scipy.stats as stats
        
        sub = self.raw_df[self.raw_df["season"] == season].copy()
        n_total = len(sub)
        
        if n_total == 0:
            raise ValueError(f"No data for season '{season}'")
            
        def _classify(row):
            rain = row.get("rainfall_mm", 0)
            wind = row.get("wind_max_kmh", 0)
            if rain > 100.0 or wind > 90.0:  return 5
            if rain >  50.0 or wind > 60.0:  return 4
            if rain >  20.0 or wind > 40.0:  return 3
            if rain >   5.0 or wind > 25.0:  return 2
            return 1
            
        sub["severity"] = sub.apply(_classify, axis=1)
        
        result = {}
        z = 1.96  # 95% CI
        
        for level in [1, 2, 3, 4, 5]:
            sub_level = sub[sub["severity"] == level]
            n_k = len(sub_level)
            p = n_k / n_total
            
            center = (p + z**2 / (2*n_total)) / (1 + z**2 / n_total)
            margin = (z * (p*(1-p)/n_total + z**2/(4*n_total**2))**0.5) / (1 + z**2/n_total)
            ci_lo  = max(0.0, center - margin)
            ci_hi  = min(1.0, center + margin)
            
            if n_k > 0:
                rain_p50  = float(sub_level["rainfall_mm"].median())
                rain_p75  = float(sub_level["rainfall_mm"].quantile(0.75))
                temp_mean = float(sub_level["temp_mean_c"].mean())
                wind_mean = float(sub_level["wind_max_kmh"].mean())
            else:
                rain_p50 = rain_p75 = temp_mean = wind_mean = 0.0
                
            result[level] = {
                "prob":      round(p, 4),
                "ci_lo":     round(ci_lo, 4),
                "ci_hi":     round(ci_hi, 4),
                "n_days":    n_k,
                "n_total":   n_total,
                "rain_p50":  round(rain_p50, 1),
                "rain_p75":  round(rain_p75, 1),
                "temp_mean": round(temp_mean, 1),
                "wind_mean": round(wind_mean, 1),
            }
            
            logger.info(
                f"  L{level}: n={n_k:4d}/{n_total}  "
                f"p={p:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  "
                f"rain_p50={rain_p50:.1f}mm  wind={wind_mean:.1f}km/h"
            )
            
        total_p = sum(v["prob"] for v in result.values())
        logger.info(f"  Σp = {total_p:.6f} (should be exactly 1.0)")
        
        return result



# Patch: robust loader that ignores unknown/missing fields
def _load_distributions_robust(self, path: str):
    import dataclasses
    raw = json.loads(Path(path).read_text())
    valid_fields = {f.name for f in dataclasses.fields(SeasonalDistributions)}
    self.distributions = {
        s: SeasonalDistributions(**{k: v for k, v in d.items() if k in valid_fields})
        for s, d in raw.items()
    }
    return self.distributions

import types
DaNangWeatherData.load_distributions = _load_distributions_robust