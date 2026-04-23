"""
Da Nang Heterogeneous Fleet Configuration
==========================================
Adapted from Patel et al. (2024) Table 4 for Vietnamese urban logistics context.

Paper (Table 4):
  Type 1: 1,000 kg / 1.72 m³ / CAD 100 fixed / CAD 1.0/km
  Type 2: 3,000 kg / 5.16 m³ / CAD 120 fixed / CAD 1.2/km
  Type 3: 5,000 kg / 10.4 m³  / CAD 150 fixed / CAD 1.5/km
  Type 4: 10,000kg / 20.6 m³  / CAD 200 fixed / CAD 1.8/km

Da Nang Adaptation:
  - Scale reduced to match local fresh-food micro-logistics (6 stores, ~2,000 kg/day demand)
  - Vietnamese road conditions: mix of narrow city lanes + suburban roads
  - Added refrigeration tier (seafood/meat must use xe lạnh or pay spoilage premium)
  - Weather severity limit per vehicle type (xe tải mini flooded out at Level 4)

Key differences from current single-fleet:
  - 4 types × multiple instances = 8 total vehicles (heterogeneous)
  - Binary use_vehicle[k,v]: fixed cost charged ONLY when vehicle dispatched
  - Dual-capacity: payload_kg AND volume_m3 (prevents unrealistic over-packing)
  - Weather capacity penalised differently per type (small vehicles more flood-vulnerable)
  - Vehicle availability gated by weather severity (heavy truck disabled in typhoon)
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# Time-Window Configuration for Fresh Retail
# ──────────────────────────────────────────────────────────────────────────────

FRESH_RETAIL_TIME_WINDOWS = {
    # Phase 2A: Procurement tour
    "supplier": {
        "seafood":    {"open": 4.0,  "close": 8.5},   # cảng cá
        "vegetables": {"open": 5.0,  "close": 9.5},   # nông trại
        "meat":       {"open": 4.5,  "close": 8.5},   # lò mổ
        "general":    {"open": 5.0,  "close": 10.0},  # chợ đầu mối
    },
    # Phase 2B: Distribution tour — uniform cho tất cả fresh retail stores
    "store": {
        "default":    {"open": 10.0, "close": 13.0},  # 10:00–13:00 nhận hàng
    },
    # DC processing window
    "dc": {
        "receive_by":   9.5,   # Phase 2A phải về DC trước 9:30
        "dispatch_at": 10.0,   # Phase 2B bắt đầu từ 10:00
    },
    # Weather adjustments (per severity level)
    "weather_delay_h": {
        1: 0.0,  # normal
        2: 0.0,  # light rain
        3: 0.5,  # moderate rain → +30min
        4: 1.0,  # heavy rain → +1h (applicable cho supplier window only)
        5: 99.0, # typhoon → inaccessible
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Vehicle type definitions
# ──────────────────────────────────────────────────────────────────────────────

VEHICLE_TYPES = [
    {
        # Xe tải mini / pickup truck — nimble urban delivery
        # Analogous to Patel Type 1 (1,000 kg) but scaled down for Da Nang micro-routes
        "type_id":              "mini_van",
        "name_vn":              "Xe tải mini (300 kg)",
        "count":                3,                  # 3 units in fleet

        # ── Capacity ──────────────────────────────────────────────────
        "payload_kg":           300,                # maximum weight load
        "volume_m3":            0.9,                # cubic load capacity

        # ── Cost (VND) ────────────────────────────────────────────────
        "fixed_cost_vnd":       600_000,            # daily dispatch + driver daily wage
        "loading_cost_per_stop": 100_000,            # loading cost
        "cost_per_km_vnd":      12_000,              # fuel (RON95) + driver
        "cost_per_hour_vnd":    80_000,             # driver hourly rate

        # ── Performance ───────────────────────────────────────────────
        "base_speed_kmh":       50,                 # nimble in city traffic
        "refrigerated":         False,

        # ── Weather resilience ────────────────────────────────────────
        # capacity_weather_penalty: extra reduction ON TOP of scenario's global factor
        # 0.80 → in Level 4 flood, effective cap = base_factor(0.60) × 0.80 = 0.48
        "capacity_weather_penalty": 0.80,
        "speed_weather_penalty":    1.10,           # additional slowdown multiplier
        "max_severity_operable":    3,              # disabled at Level 4+ (flooded streets)
    },
    {
        # Xe tải nhỏ / light truck — standard city delivery
        # Analogous to Patel Type 1 (1,000 kg) — direct equivalent
        "type_id":              "light_truck",
        "name_vn":              "Xe tải nhỏ (1,000 kg)",
        "count":                2,

        "payload_kg":           1_000,
        "volume_m3":            3.0,

        "fixed_cost_vnd":       400_000,
        "cost_per_km_vnd":      14_000,
        "cost_per_hour_vnd":    90_000,

        "base_speed_kmh":       40,
        "refrigerated":         False,

        "capacity_weather_penalty": 0.92,
        "speed_weather_penalty":    1.05,
        "max_severity_operable":    4,              # disabled in typhoon (Level 5)
    },
    {
        # Xe tải lạnh / refrigerated truck — perishable-grade transport
        # No direct Patel equivalent; added as Da Nang extension for seafood/meat
        "type_id":              "ref_truck",
        "name_vn":              "Xe tải lạnh (1,500 kg)",
        "count":                2,

        "payload_kg":           1_500,
        "volume_m3":            5.0,

        "fixed_cost_vnd":       900_000,            # premium + cold chain daily
        "cold_chain_daily_cost": 200_000,           # cold chain maintenance
        "cost_per_km_vnd":      9_000,
        "cost_per_hour_vnd":    60_000,

        "base_speed_kmh":       35,
        "refrigerated":         True,               # halves spoilage rate for perishables

        "capacity_weather_penalty": 0.95,           # more robust (higher chassis)
        "speed_weather_penalty":    1.00,
        "max_severity_operable":    4,
    },
    {
        # Xe tải lớn / heavy truck — bulk procurement from suppliers → DC
        # Analogous to Patel Type 2/3 (3,000 kg range)
        "type_id":              "heavy_truck",
        "name_vn":              "Xe tải lớn (3,000 kg)",
        "count":                1,

        "payload_kg":           3_000,
        "volume_m3":            10.0,

        "fixed_cost_vnd":       1_200_000,
        "cost_per_km_vnd":      13_000,
        "cost_per_hour_vnd":    80_000,

        "base_speed_kmh":       30,                 # slow on narrow Da Nang roads
        "refrigerated":         False,

        "capacity_weather_penalty": 1.00,           # most flood-resistant chassis
        "speed_weather_penalty":    1.00,
        "max_severity_operable":    3,              # banned on flooded suburban roads (Level 4+)
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: expand types into individual vehicle instances
# ──────────────────────────────────────────────────────────────────────────────

def expand_fleet(vehicle_types: List[Dict] = None) -> List[Dict]:
    """
    Expand vehicle type definitions into a flat list of individual vehicle instances.

    Returns a list ordered as:
      [mini_van_0, mini_van_1, mini_van_2, light_truck_0, light_truck_1,
       ref_truck_0, ref_truck_1, heavy_truck_0]

    Each element is a dict with all type properties plus:
      - vehicle_id:  str  e.g. "mini_van_0"
      - type_id:     str  e.g. "mini_van"
      - instance_id: int  e.g. 0
    """
    if vehicle_types is None:
        vehicle_types = VEHICLE_TYPES

    fleet = []
    for vtype in vehicle_types:
        for i in range(vtype["count"]):
            v = dict(vtype)
            v["vehicle_id"] = f"{vtype['type_id']}_{i}"
            v["instance_id"] = i
            fleet.append(v)
    return fleet


def get_fleet_summary(vehicle_types: List[Dict] = None) -> str:
    """Return a formatted summary table for logging."""
    if vehicle_types is None:
        vehicle_types = VEHICLE_TYPES

    lines = [
        "=" * 90,
        "DA NANG HETEROGENEOUS FLEET CONFIGURATION  (adapted from Patel et al. Table 4)",
        "=" * 90,
        f"{'Type':<20} {'Count':>5} {'Payload':>9} {'Volume':>8} "
        f"{'Fixed(VND)':>12} {'Cost/km':>9} {'Speed':>7} {'Refrig':>7} {'MaxSev':>7}",
        "-" * 90,
    ]
    total = 0
    for v in vehicle_types:
        lines.append(
            f"{v['name_vn']:<20} {v['count']:>5} "
            f"{v['payload_kg']:>7} kg {v['volume_m3']:>6.1f}m³ "
            f"{v['fixed_cost_vnd']:>12,.0f} "
            f"{v['cost_per_km_vnd']:>8,.0f} "
            f"{v['base_speed_kmh']:>6}km/h "
            f"{'Yes' if v['refrigerated'] else 'No':>7} "
            f"{'≤'+str(v['max_severity_operable']):>7}"
        )
        total += v["count"]
    lines.append("-" * 90)
    lines.append(f"  Total fleet size: {total} vehicles")
    lines.append("=" * 90)
    return "\n".join(lines)


def get_effective_capacity(vehicle: Dict, scenario) -> Dict:
    """
    Compute weather-adjusted capacity for a specific vehicle under a scenario.

    Parameters
    ----------
    vehicle  : single vehicle dict from expand_fleet()
    scenario : WeatherScenario dataclass

    Returns
    -------
    dict with keys:
      available_bool    – False if vehicle disabled by severity
      payload_kg        – effective weight capacity
      volume_m3         – effective cubic capacity
      speed_kmh         – effective travel speed
    """
    # Vehicle availability gate
    if scenario.severity_level > vehicle["max_severity_operable"]:
        return {
            "available_bool": False,
            "payload_kg":     0.0,
            "volume_m3":      0.0,
            "speed_kmh":      0.0,
        }

    # Apply global weather factor then type-specific penalty
    global_factor = scenario.capacity_reduction_factor
    type_penalty  = vehicle["capacity_weather_penalty"]
    eff_capacity  = vehicle["payload_kg"] * global_factor * type_penalty
    eff_volume    = vehicle["volume_m3"]  * global_factor * type_penalty

    # Speed: global reduction factor × type-specific slowdown
    eff_speed = max(
        5.0,
        vehicle["base_speed_kmh"]
        / scenario.speed_reduction_factor
        / vehicle["speed_weather_penalty"],
    )

    return {
        "available_bool": True,
        "payload_kg":     max(0.0, eff_capacity),
        "volume_m3":      max(0.0, eff_volume),
        "speed_kmh":      eff_speed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Product volume reference (m³/unit) — for cubic load constraint
# Consistent with weight: volume_m3 ≈ weight_kg * 0.005~0.010
# ──────────────────────────────────────────────────────────────────────────────

PRODUCT_VOLUME_M3 = {
    # seafood — dense packing in ice/styrofoam boxes
    "Fresh Fish (Cá)":          0.003,   # 0.5 kg → 0.006 m³/kg
    "Fresh Shrimp (Tôm)":       0.002,   # 0.3 kg → 0.007 m³/kg
    "Squid (Mực)":              0.003,   # 0.4 kg
    # vegetables — loose stacking, bulkier per kg
    "Leafy Greens (Rau xanh)":  0.003,   # 0.2 kg → 0.015 m³/kg (bulky)
    "Tomatoes (Cà chua)":       0.003,   # 0.3 kg
    "Cabbage (Bắp cải)":        0.006,   # 0.5 kg → 0.012 m³/kg (bulky)
    # meat — vacuum-packed, compact
    "Chicken (Gà)":             0.005,   # 0.6 kg
    "Pork (Thịt heo)":          0.004,   # 0.5 kg
    "Beef (Thịt bò)":           0.003,   # 0.4 kg
    # fruit — medium packing density
    "Dragon Fruit (Thanh long)": 0.004,  # 0.4 kg
}

DEFAULT_VOLUME_M3_PER_UNIT = 0.004   # fallback for unknown products


# ──────────────────────────────────────────────────────────────────────────────
# Format conversion: data_generation → ExtensiveFormOptimizer
# ──────────────────────────────────────────────────────────────────────────────

def _linear_capacity_factor(sev: int, max_sev: int, worst_penalty: float) -> float:
    """
    Linear interpolation of weather capacity factor across severity levels.

    Level 1 (no weather):      factor = 1.0
    Level max_sev:             factor = worst_penalty
    Level > max_sev:           factor = 0.0  (vehicle disabled / grounded)
    Intermediate:              linearly interpolated
    """
    if sev > max_sev:
        return 0.0
    if max_sev <= 1:
        return float(worst_penalty)
    t = (sev - 1) / (max_sev - 1)          # 0.0 at L1 → 1.0 at L_max_sev
    return round(1.0 - t * (1.0 - worst_penalty), 4)


def to_optimizer_fleet(compact_fleet: List[Dict]) -> List[Dict]:
    """
    Convert fleet list from expand_fleet() (compact data_generation format)
    to the dict format expected by ExtensiveFormOptimizer.

    Field mapping
    ─────────────
    compact                             → optimizer
    payload_kg                          → capacity_kg
    cost_per_km_vnd                     → cost_per_km
    cost_per_hour_vnd                   → cost_per_hour
    capacity_weather_penalty +          → weather_capacity_factor {1→5: float}
      max_severity_operable
    speed_weather_penalty               → weather_speed_factor    {1→5: float}
    (refrigerated=True → 0.65 benefit)  → spoilage_reduction

    Pass-through: vehicle_id, type_id, fixed_cost_vnd, base_speed_kmh, refrigerated
    """
    result = []
    for v in compact_fleet:
        max_sev  = int(v.get("max_severity_operable", 5))
        cap_pen  = float(v.get("capacity_weather_penalty", 1.0))
        spd_pen  = float(v.get("speed_weather_penalty",    1.0))

        # Per-severity capacity fraction (0.0 when disabled)
        weather_cap = {
            sev: _linear_capacity_factor(sev, max_sev, cap_pen)
            for sev in range(1, 6)
        }

        # Per-severity speed fraction
        # speed_weather_penalty is a slowdown divisor → convert to multiplier
        # Disabled levels get 0.0 (won't be used but keeps dict complete)
        weather_spd = {
            sev: (round(1.0 / spd_pen, 4) if sev <= max_sev else 0.0)
            for sev in range(1, 6)
        }

        # Spoilage reduction: refrigerated trucks benefit from cold-chain
        # 65 % reduction mirrors Patel et al. Arrhenius cold-chain benefit
        spoilage_reduction = 0.65 if v.get("refrigerated", False) else 0.0

        result.append({
            # identity
            "vehicle_id":           v["vehicle_id"],
            "type_id":              v["type_id"],
            "name":                 v.get("name_vn", v["vehicle_id"]),
            # capacity
            "capacity_kg":          float(v["payload_kg"]),
            "volume_m3":            float(v.get("volume_m3", 0.0)),
            # costs
            "fixed_cost_vnd":       float(v["fixed_cost_vnd"]),
            "cost_per_km":          float(v["cost_per_km_vnd"]),
            "cost_per_hour":        float(v["cost_per_hour_vnd"]),
            # operations
            "base_speed_kmh":       float(v["base_speed_kmh"]),
            "refrigerated":         bool(v["refrigerated"]),
            "spoilage_reduction":   spoilage_reduction,
            # per-severity dicts
            "weather_capacity_factor": weather_cap,
            "weather_speed_factor":    weather_spd,
        })
    return result


if __name__ == "__main__":
    print(get_fleet_summary())
    fleet = expand_fleet()
    print(f"\nExpanded fleet: {len(fleet)} individual vehicles")
    for v in fleet:
        print(f"  {v['vehicle_id']:20s}  payload={v['payload_kg']:5} kg  "
              f"volume={v['volume_m3']:.1f} m³  refrig={v['refrigerated']}")