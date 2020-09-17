"""
pd.merge(P_2, Z_2, how='left', on="CaseName")

"""
import os
import numpy as np
import pandas as pd


class CominePC:
    def __init__(self):
        self.p_path = r"D:\Data\doctor_gao\jiazhuangxian\old_data\T2\T_L\selected_train_and_test.csv"
        self.c_path = r"D:\Data\doctor_gao\jiazhuangxian\old_data\T1C\T_L\selected_train_and_test.csv"
        self.store_path = r"D:\Data\doctor_gao\jiazhuangxian\old_data\T_L"

    def make_same_case_name(self, pd_data):
        # pd_list = list(pd_data["CaseName"])
        # pd_case_name_list = []
        # for case_name in pd_list:
        #     x = case_name.split("_")[0]
        #     pd_case_name_list.append(x)
        # pd_data["CaseName"] = pd_case_name_list
        # pd_new_data = pd_data
        return pd_data

    def change_row(self, pd_data, clas):
        pd_new_row = []
        pd_header_list = list(pd_data.loc[[]])
        for feature in pd_header_list:
            x = feature.split("roi")[-1]
            if x == "CaseName":
                pass
            else:
                x = clas + x
            pd_new_row.append(x)
        pd_data.columns = pd_new_row

        return pd_data

    def get_csv(self):
        pd_p = pd.read_csv(self.p_path)
        pd_c = pd.read_csv(self.c_path)
        # pd_p_new = self.change_row(self.make_same_case_name(pd_p), "MAX")
        # pd_c_new = self.change_row(self.make_same_case_name(pd_c), "MIN")
        pd_result = pd.merge(pd_c, pd_p, how="left", on="CaseName")
        pd_new_result = pd_result.dropna(how="any", axis=0)

        pd_new_result.to_csv(os.path.join(self.store_path, "max_min_train_test.csv"), index=None)

cobine = CominePC()
cobine.get_csv()


