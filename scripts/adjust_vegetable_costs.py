import pandas as pd
import numpy as np

np.random.seed(42)

products_path = "d:/Food chain optimization/data/synthetic/products.csv"
sp_matrix_path = "d:/Food chain optimization/data/synthetic/supplier_product_matrix.csv"

# Load data
df_prod = pd.read_csv(products_path)
df_sp = pd.read_csv(sp_matrix_path)

# Map for reduction targets
target_prices = {
    "PROD_004": 8500,
    "PROD_005": 12000,
    "PROD_006": 6000,
    "PROD_010": 15000
}

# Update products.csv (and keep margin constant)
for idx, row in df_prod.iterrows():
    pid = row["id"]
    if pid in target_prices:
        old_cost = row["unit_cost_vnd"]
        new_cost = target_prices[pid]
        
        # Calculate new price based on margin
        margin = row["margin_pct"]
        new_price = new_cost * (1 + margin / 100.0)
        
        df_prod.at[idx, "unit_cost_vnd"] = new_cost
        df_prod.at[idx, "unit_price_vnd"] = round(new_price, 0)
        
df_prod.to_csv(products_path, index=False)
print("Updated products.csv")

# Update supplier_product_matrix.csv
prod_cost_dict = df_prod.set_index("id")["unit_cost_vnd"].to_dict()

for idx, row in df_sp.iterrows():
    pid = row["product_id"]
    sid = row["supplier_id"]
    
    if pid in target_prices:
        base_cost = prod_cost_dict[pid]
        if sid == "SUP_006":
            new_cost = base_cost * np.random.uniform(1.15, 1.25)
        else:
            new_cost = base_cost * np.random.uniform(0.85, 1.10)
        df_sp.at[idx, "unit_cost_vnd"] = round(new_cost, 0)

df_sp.to_csv(sp_matrix_path, index=False)
print("Updated supplier_product_matrix.csv")
