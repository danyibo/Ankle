import os
import numpy as np
import pandas as pd

class GetTrainTest:
    def __init__(self):
        self.model_feature = r"X:\DYB\data_and_result\doctor tao\max_and_min\log_wave_original\selected\LDA_result\Mean\PCC\Relief_26\LDA\LDA_coef.csv"
        self.raw_feature = r"X:\DYB\data_and_result\doctor tao\random_feature\random_case\train_test.csv"
        self.pd_model_feature = pd.read_csv(self.model_feature)
        self.pd_raw_feature = pd.read_csv(self.raw_feature)

    def get_selected_feature(self):
        feature_name = self.pd_model_feature["Unnamed: 0"]
        new_feature_name = ["data_roi_" + name[4:] for name in feature_name]

        case_name = self.pd_raw_feature["CaseName"]
        label = self.pd_raw_feature["label"]
        selected_feature = self.pd_raw_feature[new_feature_name]

        result = pd.concat([case_name, label, selected_feature], axis=1)
        store_path = os.path.dirname(self.raw_feature)
        result.to_csv(os.path.join(store_path, "selected_feature.csv"), index=None)




if __name__ == '__main__':
    get = GetTrainTest()
    get.get_selected_feature()
