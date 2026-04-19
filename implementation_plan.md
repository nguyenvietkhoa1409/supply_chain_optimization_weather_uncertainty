# Implementation Plan: Realism Improvements — Fresh Food Supply Chain Optimizer

## Bối cảnh & Đánh giá Tổng thể

Pipeline hiện tại là một **Two-Phase Two-Stage Stochastic MILP** hoàn chỉnh với:
- `two_phase_optimizer.py` — core MILP (976 dòng, đã stable)
- `fleet_config.py`, `demand_generator.py`, `manual_scenarios.py` — data generation
- `product_generator.py`, `supplier_product_matrix.py` — cost/product data

**Nhận xét chung:** Cả 5 cải thiện trong fixing plan đều **hợp lý về mặt học thuật và thực tiễn**. Không có cải thiện nào xung đột kiến trúc nghiêm trọng. Tuy nhiên, mức độ phức tạp tích hợp và rủi ro khác nhau rõ rệt, nên phân tier để tránh destabilizing mô hình đang hoạt động.

---

## Đánh giá Chi tiết từng Điểm

### ✅ Điểm 1 — VRP Cost Recalibration (Rất nên làm, Low Risk)

**Vấn đề xác nhận:**
Từ code `fleet_config.py` hiện tại: `mini_van` có `fixed_cost_vnd = 150_000` và `cost_per_km_vnd = 3_000`. Với khoảng cách Đà Nẵng 5-10 km, chi phí/chuyến ≈ 180,000 VND, không đáng kể so với Stage-1 procurement cost hàng trăm triệu.

**Bổ sung từ fixing plan:** Thêm `loading_cost_per_stop` và `driver_daily_wage` như fixed cost components, tăng `fixed_cost_vnd` của `mini_van` từ 150k → ~600k (bao gồm lương lái xe 250k/ngày), `ref_truck` từ 700k → ~900k (thêm `cold_chain_daily_cost` 200k).

**Đánh giá tích hợp:** ✅ **Rất an toàn.** Chỉ cần sửa `VEHICLE_TYPES` dict trong `fleet_config.py`. Không thay đổi MILP structure. `to_optimizer_fleet()` đã handle `fixed_cost_vnd` trực tiếp — chỉ cần cập nhật số.

---

### ✅ Điểm 2 — Tiered Unmet Penalty by Category (Nên làm, Medium Risk)

**Vấn đề xác nhận:**
`_UNMET_PENALTY_MULT = 10.0` (dòng 71, `two_phase_optimizer.py`) là hằng số flat áp dụng cho tất cả sản phẩm. Fixing plan đề xuất giảm xuống còn 2.0–4.5× tùy category, phản ánh thực tế thị trường tốt hơn.

> [!WARNING]
> **Rủi ro quan trọng cần xem xét:** `_UNMET_PENALTY_MULT = 10.0` hiện tại có vai trò kép: vừa là economic realism, vừa là **solver forcing mechanism** — đảm bảo optimizer không "skip" procurement. Nếu hạ penalty xuống còn 2.0-4.5×, cần kiểm tra lại xem solver có còn bị buộc phải dispatch Phase-2A hay không, đặc biệt ở các scenario nhẹ.

**Đánh giá tích hợp:** Feasible nhưng cần thêm một validation run sau khi sửa. Thay đổi tại `_UNMET_PENALTY_MULT` và objective function trong `two_phase_optimizer.py`.

---

### ✅ Điểm 3 — Arrhenius Spoilage Model (Học thuật cao, Medium-High Effort)

**Vấn đề xác nhận:**
Hiện tại `spoilage_multiplier` trong `WeatherScenario` là một scalar flat (e.g., 1.30× cho Heavy Rain), áp dụng globally. Code `product_generator.py` đã có `shelf_life_days` và `temperature_sensitivity` nhưng chưa được dùng trong tính toán spoilage chi tiết.

**Bổ sung từ fixing plan:** Thêm `shelf_life_base_days`, `activation_energy_Ea`, `ambient_temp_monsoon_C` vào product data, và hàm `compute_spoilage_rate()` theo Q10 rule. Phân biệt refrigerated vs non-refrigerated transport.

**Đánh giá tích hợp:** ⚠️ **Phức tạp hơn.** Arrhenius model đề xuất là để tính `spoilage_fraction` theo transport time — tuy nhiên trong MILP hiện tại, spoilage cost chỉ xuất hiện dưới dạng **opportunity cost từ inaccessible suppliers** (dòng 412-416), không phải per-unit spoilage during transport. 

**Giải pháp đề xuất:** Implement Arrhenius như một **pre-computation layer** để tạo ra `scenario_product_spoilage_rate[k, p]` thay thế `base_spoil = 0.04` (dòng 356) và `sc.spoilage_multiplier` trong refrigeration penalty. Không thay đổi constraint structure, chỉ thay thế coefficients — safe.

---

### ✅ Điểm 4 — Supplier Cost Calibration (Nên làm, Low Risk)

**Vấn đề xác nhận:**
`product_generator.py` dòng 248: `supplier_cost = base_cost * np.random.uniform(1.30, 1.40)` cho general supplier — tức 30-40% premium. Dữ liệu thực tế Đà Nẵng (fixing plan) cho thấy chỉ nên 15-25%. Hiện tại trong `supplier_product_matrix.csv`, SUP_006 (Da Nang Wholesale) cung cấp PROD_001 với 157,730 VND/unit so với SUP_003 ở 100,878 VND/unit — chênh lệch 57%.

**Đánh giá tích hợp:** ✅ **Rất an toàn.** Chỉ sửa `product_generator.py` (thay uniform(1.30, 1.40) → uniform(1.15, 1.25)) và **regenerate** `supplier_product_matrix.csv`. Cần chạy lại `generate_all_data.py` sau đó.

---

### ✅ Điểm 5 — Weather-Conditioned Demand (Quan trọng nhất về realism, Low-Medium Risk)

**Vấn đề xác nhận:**
`demand_generator.py` hiện tại hoàn toàn **không tham chiếu weather scenario**. `_get_base_demand()` và `generate_demand_plan()` không nhận scenario parameter. Điều này có nghĩa là `unmet_cost` trong Typhoon scenario (severity 5) vẫn tính trên full demand, tạo ra penalty 781M VND phi thực tế.

**Bổ sung từ fixing plan:** Thêm `demand_reduction_factor` vào `WeatherScenario` dataclass (ví dụ: Typhoon = 0.15), và dùng nó để scale `store_demand` trong objective/constraints.

**Đánh giá tích hợp:** ⚠️ **Ảnh hưởng đến MILP structure.** `store_demand` và `total_demand` hiện tại là **fixed lookups** trong `_build_lookups()`. Để thêm weather-conditioned demand, cần:
1. Thêm `demand_reduction_factor` vào `WeatherScenario`
2. Tạo `store_demand[k][r,p]` là dict per-scenario thay vì một dict flat
3. Cập nhật tất cả references trong constraints và objective

> [!IMPORTANT]
> Đây là thay đổi non-trivial nhất về phía MILP (ảnh hưởng tới `dDem`, `dDemFB` constraints và `unmet_cost` objective term), nhưng **giá trị realism rất cao** và logic cần thiết rõ ràng. Nên implement sau khi đã test Tier 1.

---

## Proposed Changes — Tiered Implementation

### 🏆 Tier 1 — Data & Parameters (Không chạm MILP — An toàn tuyệt đối)
*Thực hiện trước, validate bằng 1 run đơn giản*

---

#### [MODIFY] [fleet_config.py](file:///d:/Food%20chain%20optimization/src/data_generation/fleet_config.py)
- Tăng `fixed_cost_vnd` của `mini_van`: 150k → 600k (bao gồm `driver_daily_wage` 250k + khấu hao/bảo hiểm 350k)
- Tăng `cost_per_km_vnd` của `mini_van`: 3,000 → 5,500 (RON95 thực tế)
- Thêm `loading_cost_per_stop`: 25,000 VND vào dict (không dùng trong MILP ngay, nhưng documented)
- Tăng `fixed_cost_vnd` của `ref_truck`: 700k → 900k (thêm `cold_chain_daily_cost` 200k)
- Thêm comment documentation về cơ sở tính toán

#### [MODIFY] [product_generator.py](file:///d:/Food%20chain%20optimization/src/data_generation/product_generator.py)
- General supplier premium: `uniform(1.30, 1.40)` → `uniform(1.15, 1.25)`
- Specialist supplier range: từ `uniform(0.85, 1.15)` → `uniform(0.85, 1.10)` (tighten spread)

#### [NEW] Regenerate data files
- Chạy lại `generate_all_data.py` (hoặc tạo `recalibrate_costs.py`) để update `supplier_product_matrix.csv`

---

### 🏆 Tier 2 — Model Enrichment (Thêm vào MILP — Cần test kỹ)
*Thực hiện sau khi Tier 1 đã pass validation run*

---

#### [MODIFY] [manual_scenarios.py](file:///d:/Food%20chain%20optimization/src/weather/manual_scenarios.py)
- Thêm `demand_reduction_factor: float = 1.0` vào `WeatherScenario` dataclass
- Update tất cả scenario definitions:
  ```
  Normal Monsoon  → 1.00
  Light Rain      → 0.95
  Moderate Rain   → 0.80
  Heavy Rain      → 0.55
  Typhoon         → 0.15
  ```

#### [MODIFY] [two_phase_optimizer.py](file:///d:/Food%20chain%20optimization/src/optimization/two_phase_optimizer.py)
- **Weather-conditioned demand (Điểm 5):**
  - Trong `_build_lookups()`: tạo `self.store_demand_base` (dict gốc)
  - Trong `build_model()`: tính `self.effective_demand[k][r,p] = store_demand_base[r,p] * sc.demand_reduction_factor`
  - Update tất cả references đến `self.store_demand` trong `dDem`/`dDemFB` constraints và `unmet_cost` objective → dùng `effective_demand[k]`
  - Update `_scenario_costs()` để report thêm `demand_reduction_factor` per scenario

- **Tiered unmet penalty (Điểm 2):**
  - Thêm dict `UNMET_PENALTY_BY_CATEGORY = {"seafood": 4.5, "meat": 4.0, "vegetable": 2.5, "fruit": 2.0}` vào module constants
  - Thêm `GOODWILL_LOSS_FACTOR = 0.20` (20% surcharge for repeat-unmet)
  - Build `prod_penalty_mult[p]` lookup trong `_build_lookups()` từ product category
  - Replace `pm = _UNMET_PENALTY_MULT` → `pm = self.prod_penalty_mult[p]` trong objective  
  
  > [!WARNING]
  > Sau khi giảm penalty, bắt buộc phải validate rằng WS ≤ RP ≤ EEV vẫn đúng. Nếu RP tăng (solver skip pickup), phải raise penalty lại hoặc giữ floor tối thiểu 5.0 cho seafood/meat.

---

### 🏆 Tier 3 — Advanced Modeling (Optional, High Academic Value)
*Thực hiện sau khi Tier 1+2 stable, hoặc nếu có thời gian*

---

#### [NEW] `src/data_generation/spoilage_model.py`
- Implement `compute_spoilage_rate(product_id, transport_time_h, is_refrigerated, temperature_C)` theo Q10 rule
- Thêm `PRODUCT_SHELF_LIFE_PARAMS` dict với `shelf_life_base_days`, `temp_reference_C`, `activation_energy_Ea` cho từng product (theo fixing plan)
- Function trả về `spoilage_fraction ∈ [0, 1]`

#### [MODIFY] [two_phase_optimizer.py](file:///d:/Food%20chain%20optimization/src/optimization/two_phase_optimizer.py) (Arrhenius integration)
- Trong `build_model()`: pre-compute `scenario_product_mu[k, p]` = spoilage rate dùng `compute_spoilage_rate()`
- Thay `base_spoil = 0.04` → `base_spoil = scenario_product_mu[k, p]` trong refrigeration penalty term
- Thay `sc.spoilage_multiplier` trong `spoil_s1` → dùng `scenario_product_mu[k, p]` weighted
- Cập nhật column `spoilage_cost` trong `_scenario_costs()` để reflect per-product rates

---

## Open Questions

> [!IMPORTANT]
> **Q1 — Penalty floor:** Sau khi hạ `_UNMET_PENALTY_MULT` từ 10× xuống 2-4.5×, cần kiểm tra xem `_WASTE_MULT = 3.5` có còn đủ cao hơn penalty của một số SKU rẻ như vegetable (2.5×) không. Nếu waste_cost < penalty_cost thì solver sẽ prefer waste over pickup — đây là regression. Đề xuất: set floor tối thiểu cho penalty là `max(category_mult, 5.0)` cho items `requires_refrigeration=True`.

> [!IMPORTANT]
> **Q2 — Data regeneration:** Khi đổi supplier premium rate và regenerate `supplier_product_matrix.csv`, cần quyết định: (a) regenerate fresh với seed=42, hoặc (b) tạo script `recalibrate_costs.py` chỉ scale các entry của SUP_006 mà không thay đổi các supplier khác. Option (b) ít xáo trộn hơn.

**Q3 — Scope time-windows (Rec 1B):** Thêm time-window constraints cho stores sẽ cần thêm binary variables và tightening MILP significantly, có thể tăng solve time đáng kể. Bạn có muốn đưa vào scope này không? Nếu có, nó cần một task riêng.

---

## Verification Plan

### Automated Validation
Sau mỗi tier, chạy:
```bash
python scripts/run_two_phase_optimization.py
```
Và kiểm tra:
1. **WS ≤ RP ≤ EEV** ordering vẫn đúng
2. VRP cost / Total cost ratio ≥ 10% (Tier 1 target)
3. Typhoon penalty cost giảm ≥ 50% so với hiện tại (Tier 2 target)
4. `supplier_product_matrix.csv` premium SUP_006 vs specialist ≤ 30% (Tier 1 target)

### Regression Markers
| Metric | Hiện tại (baseline) | Target sau Tier 1+2 |
|--------|---------------------|---------------------|
| VRP% of total (Normal) | ~3% | ≥ 10% |
| Typhoon penalty | ~781M VND | ≤ 120M VND |
| SUP_006 premium vs specialist | ~57% | ≤ 25% |
| WS ≤ RP ≤ EEV | ✅ | ✅ (must hold) |
