import os
import numpy as np
import pandas as pd


csv_path = r"E:\Data\doctor tao\max_and_min\max_min_train_test.csv"
pd_data = pd.read_csv(csv_path)
feature_name = pd_data.columns
original_index = ["CaseName", "label"]
for name in feature_name:
    if len(name) > 10:
        if name.split("_")[1] == "original":
            original_index.append(name)
store_path = os.path.dirname(csv_path)
pd_origin = pd_data[original_index]
pd_origin.to_csv(os.path.join(store_path, "original_new.csv"), index=None)