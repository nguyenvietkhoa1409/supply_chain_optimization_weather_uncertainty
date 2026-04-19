"""
spoilage_model.py
================================================================================
Arrhenius/Q10-based physical spoilage model for fresh food transportation.
Replaces flat scalar multipliers with temperature & time-dependent decaying rates.
"""

from typing import Dict

# Param baseline cho các nhóm sản phẩm tươi sống
# temp_ref_C: nhiệt độ chuẩn (để đạt base_days)
# q10: hệ số gia tốc spoilage khi nhiệt độ tăng thêm 10 độ C
PRODUCT_SHELF_LIFE_PARAMS: Dict[str, Dict[str, float]] = {
    "seafood":   {"base_days": 2.0, "temp_ref_C": 5.0,  "q10": 3.0},
    "meat":      {"base_days": 3.0, "temp_ref_C": 5.0,  "q10": 2.8},
    "vegetable": {"base_days": 5.0, "temp_ref_C": 20.0, "q10": 2.0},
    "fruit":     {"base_days": 7.0, "temp_ref_C": 20.0, "q10": 1.8},
}

def compute_spoilage_rate(
    category: str, 
    transport_time_h: float, 
    is_refrigerated: bool, 
    ambient_temp_c: float
) -> float:
    """
    Computes the fraction [0, 1] of product spoiled during a transport leg.
    Uses the Q10 rule (approximation of Arrhenius equation) to model thermal stress.
    
    For unrefrigerated transport (vulnerable to ambient heat), the spoilage
    rate accelerates exponentially according to the ambient temperature.
    """
    if category not in PRODUCT_SHELF_LIFE_PARAMS:
        category = "vegetable"
        
    params = PRODUCT_SHELF_LIFE_PARAMS[category]
    t_ref = params["temp_ref_C"]
    q10 = params["q10"]
    base_life_h = params["base_days"] * 24.0
    
    # Giả định: xe lạnh duy trì được cấu hình chuẩn t_ref.
    # Xe thường: chênh lệch nhiệt độ sẽ kích hoạt Q10.
    actual_temp = t_ref if is_refrigerated else ambient_temp_c
    
    # Tính hệ số gia tốc (Acceleration factor)
    # Nếu actual_temp > t_ref, acceleration lớn hơn 1 (nhanh hỏng).
    acceleration = q10 ** ((actual_temp - t_ref) / 10.0)
    
    # Spoilage bị mất đi theo thời gian vận tải
    spoil_fraction = (transport_time_h / base_life_h) * acceleration
    
    return min(1.0, spoil_fraction)

def compute_inventory_spoilage(
    category: str,
    wait_time_h: float,
    ambient_temp_c: float,
    weather_multiplier: float
) -> float:
    """
    Fallback cho inventory tại chỗ (vài chênh lệch môi trường).
    Sử dụng weather_multiplier để bồi đắp thiên tai/lũ lụt gây ẩm mốc.
    """
    base_fraction = compute_spoilage_rate(category, wait_time_h, False, ambient_temp_c)
    return min(1.0, base_fraction * weather_multiplier)
