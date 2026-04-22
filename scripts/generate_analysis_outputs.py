import json
import pandas as pd
import numpy as np
import os
import sys

# Add src to path if needed for scenarios
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_generation.fleet_config import VEHICLE_TYPES, expand_fleet
from data_generation.fleet_config import PRODUCT_VOLUME_M3, DEFAULT_VOLUME_M3_PER_UNIT

def main():
    print("Generating Analysis Outputs...")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

    with open(os.path.join(RESULTS_DIR, "tp_scenario_routes.json"), "r", encoding="utf-8") as f:
        routes = json.load(f)
    
    proc = pd.read_csv(os.path.join(RESULTS_DIR, "tp_stochastic_procurement.csv"))
    demand_full = pd.read_csv(os.path.join(DATA_DIR, "daily_demand.csv"))
    demand_df = demand_full[demand_full["date"] == "2024-10-01"]
    
    suppliers = pd.read_csv(os.path.join(DATA_DIR, "suppliers.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    dist_matrix = pd.read_csv(os.path.join(DATA_DIR, "distance_matrix.csv"), index_col=0)
    
    fleet_vehicles = {v["vehicle_id"]: v for v in expand_fleet(VEHICLE_TYPES)}

    # Hardcode scenario demand factors based on ManualWeatherScenarios
    scenario_info = {
        "Normal Monsoon Day": {"dem_factor": 1.0, "speed_factor": 1.0, "w_delay": 0.0},
        "Light Rain": {"dem_factor": 0.95, "speed_factor": 1.1, "w_delay": 0.0},
        "Moderate Rain": {"dem_factor": 0.80, "speed_factor": 1.3, "w_delay": 0.5},
        "Heavy Rain": {"dem_factor": 0.55, "speed_factor": 1.7, "w_delay": 1.0},
        "Tropical Storm/Typhoon": {"dem_factor": 0.00, "speed_factor": 2.5, "w_delay": 99.0}
    }

    # ---------------------------------------------------------
    # OUTPUT 1: Service Level Breakdown
    # ---------------------------------------------------------
    demand_lookup = {}
    for _, row in demand_df.iterrows():
        demand_lookup[(row["store_id"], row["product_id"])] = row["demand_units"]

    sl_rows = []
    for sc_name, data in routes.items():
        delivered = {}
        for route in data.get("distribution_routes", []):
            for store, prods in route.get("deliveries", {}).items():
                for prod, qty in prods.items():
                    delivered[(store, prod)] = delivered.get((store, prod), 0) + qty
        
        dem_factor = scenario_info.get(sc_name, {}).get("dem_factor", 1.0)
        
        for (store, prod), base_demand in demand_lookup.items():
            eff_demand = base_demand * dem_factor
            if eff_demand > 0:
                del_qty = delivered.get((store, prod), 0)
                fill_rate = min(1.0, del_qty / eff_demand)
                sl_rows.append({
                    "scenario": sc_name,
                    "store": store,
                    "product": prod,
                    "base_demand": base_demand,
                    "eff_demand": eff_demand,
                    "delivered": del_qty,
                    "fill_rate": fill_rate
                })
            else:
                sl_rows.append({
                    "scenario": sc_name,
                    "store": store,
                    "product": prod,
                    "base_demand": base_demand,
                    "eff_demand": 0,
                    "delivered": 0,
                    "fill_rate": 1.0
                })

    try:
        pd.DataFrame(sl_rows).to_csv(os.path.join(RESULTS_DIR, "analysis_service_level.csv"), index=False)
        print(" ✓ Created analysis_service_level.csv")
    except PermissionError:
        print(" ! Could not write analysis_service_level.csv")

    # ---------------------------------------------------------
    # OUTPUT 2: Time-Window Timeline (Arrivals)
    # ---------------------------------------------------------
    timeline_rows = []
    for sc_name, data in routes.items():
        speed_factor = scenario_info.get(sc_name, {}).get("speed_factor", 1.0)
        
        for p_idx, phase in enumerate(["procurement_routes", "distribution_routes"]):
            p_name = "2A" if phase == "procurement_routes" else "2B"
            start_t = 4.0 if phase == "procurement_routes" else 10.0
            
            for rt in data.get(phase, []):
                vid = rt["vehicle_id"]
                node_seq = rt.get("route", [])
                if not node_seq: continue
                
                veh = fleet_vehicles.get(vid)
                base_speed = veh["base_speed_kmh"] if veh else 40.0
                # Using general weather speed multiplier (just approximation for plotting)
                speed = base_speed / speed_factor
                
                t = start_t
                prev = node_seq[0]
                
                for stop in node_seq[1:]:
                    dist = 0
                    if prev in dist_matrix.index and stop in dist_matrix.columns:
                        dist = dist_matrix.loc[prev, stop]
                    elif prev in dist_matrix.columns and stop in dist_matrix.index:
                        dist = dist_matrix.loc[stop, prev]
                    
                    travel_h = dist / max(speed, 5.0)
                    t += travel_h
                    
                    service_h = 0.5 if phase == "procurement_routes" else 0.25
                    
                    timeline_rows.append({
                        "scenario": sc_name,
                        "phase": p_name,
                        "vehicle": vid,
                        "from_node": prev,
                        "node": stop,
                        "arrival_h": round(t, 2),
                        "departure_h": round(t + service_h, 2) if stop != node_seq[-1] else round(t, 2),
                        "distance_km": round(dist, 2),
                        "travel_time_h": round(travel_h, 2)
                    })
                    if stop != node_seq[-1]:
                        t += service_h
                    prev = stop

    try:
        pd.DataFrame(timeline_rows).to_csv(os.path.join(RESULTS_DIR, "analysis_timeline.csv"), index=False)
        print(" ✓ Created analysis_timeline.csv")
    except PermissionError:
        print(" ! Could not write analysis_timeline.csv (file open). Moving on.")

    # ---------------------------------------------------------
    # OUTPUT 3: Supplier Utilization & HHI
    # ---------------------------------------------------------
    prod_weight = dict(zip(products["id"], products["weight_kg_per_unit"]))
    
    sup_utilization = []
    for _, s_row in suppliers.iterrows():
        sid = s_row["id"]
        cap = s_row["capacity_kg_per_day"]
        ordered_df = proc[proc["supplier_id"] == sid]
        
        ordered_kg = 0
        for _, o_row in ordered_df.iterrows():
            ordered_kg += o_row["quantity_units"] * prod_weight.get(o_row["product_id"], 1.0)
            
        utilization = ordered_kg / max(cap, 1.0)
        sup_utilization.append({
            "supplier_id": sid,
            "name": s_row["name"],
            "capacity_kg": cap,
            "ordered_kg": ordered_kg,
            "utilization_pct": min(100.0, utilization * 100)
        })
    try:
        pd.DataFrame(sup_utilization).to_csv(os.path.join(RESULTS_DIR, "analysis_supplier_utilization.csv"), index=False)
    except Exception: pass
    
    hhi_rows = []
    for pid in products["id"]:
        orders = proc[proc["product_id"] == pid]["quantity_units"]
        total = orders.sum()
        if total > 0:
            shares = orders / total
            hhi = (shares**2).sum()
        else:
            hhi = 0
        hhi_rows.append({"product_id": pid, "hhi": hhi, "total_ordered": total})
    try:
        pd.DataFrame(hhi_rows).to_csv(os.path.join(RESULTS_DIR, "analysis_hhi.csv"), index=False)
        print(" ✓ Created analysis_supplier_utilization.csv and analysis_hhi.csv")
    except Exception: pass

    # ---------------------------------------------------------
    # OUTPUT 4: Vehicle Utilization
    # ---------------------------------------------------------
    veh_utilization = []
    timeline_df = pd.DataFrame(timeline_rows)
    
    for sc_name, data in routes.items():
        for phase in ["procurement_routes", "distribution_routes"]:
            for rt in data.get(phase, []):
                vid = rt["vehicle_id"]
                veh = fleet_vehicles.get(vid)
                cap_kg = veh["payload_kg"] if veh else 1.0
                
                loaded_kg = 0
                if phase == "procurement_routes":
                    for stop, items in rt.get("pickups", {}).items():
                        for p, q in items.items():
                            loaded_kg += q * prod_weight.get(p, 1.0)
                else:
                    for stop, items in rt.get("deliveries", {}).items():
                        for p, q in items.items():
                            loaded_kg += q * prod_weight.get(p, 1.0)
                            
                p_name_short = "2A" if phase == "procurement_routes" else "2B"
                dist_info = timeline_df[(timeline_df["scenario"] == sc_name) & 
                                        (timeline_df["vehicle"] == vid) & 
                                        (timeline_df["phase"] == p_name_short)]
                
                total_km = dist_info["distance_km"].sum() if not dist_info.empty else 0
                load_factor = (loaded_kg / cap_kg) * 100
                
                veh_utilization.append({
                    "scenario": sc_name,
                    "phase": p_name_short,
                    "vehicle_id": vid,
                    "vehicle_type": veh["type_id"] if veh else "unknown",
                    "total_km": round(total_km, 2),
                    "loaded_kg": round(loaded_kg, 2),
                    "capacity_kg": cap_kg,
                    "load_factor_pct": round(min(100.0, load_factor), 2)
                })

    try:
        pd.DataFrame(veh_utilization).to_csv(os.path.join(RESULTS_DIR, "analysis_vehicle_utilization.csv"), index=False)
        print(" ✓ Created analysis_vehicle_utilization.csv")
    except Exception:
        print(" ! Could not write analysis_vehicle_utilization.csv")

if __name__ == "__main__":
    main()
