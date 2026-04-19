import pandas as pd
import numpy as np
import os

np.random.seed(42)

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "synthetic"))
matrix_path = os.path.join(data_dir, "supplier_product_matrix.csv")
products_path = os.path.join(data_dir, "products.csv")

sp_df = pd.read_csv(matrix_path)
prod_df = pd.read_csv(products_path).set_index("id")

for idx, row in sp_df.iterrows():
    sid = row["supplier_id"]
    pid = row["product_id"]
    base_cost = prod_df.loc[pid, "unit_cost_vnd"]
    
    if sid == "SUP_006":
        new_cost = base_cost * np.random.uniform(1.15, 1.25)
        sp_df.at[idx, "unit_cost_vnd"] = round(new_cost, 0)
    else:
        new_cost = base_cost * np.random.uniform(0.85, 1.10)
        sp_df.at[idx, "unit_cost_vnd"] = round(new_cost, 0)

sp_df.to_csv(matrix_path, index=False)
print("Recalibrated supplier premiums in supplier_product_matrix.csv!")
