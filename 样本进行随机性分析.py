import os
import numpy as np
import pandas as pd
from Tool.FolderProcress import make_folder
import random
import shutil


class GetRandomFeature:
    def __init__(self):
        self.root_path = r"X:\DYB\data_and_result\doctor tao"
        self.roi_list = ["roi_1", "roi_2", "roi_3", "roi_4",
                         "roi_5", "roi_6", "roi_7", "roi_8"]
        self.store_path = os.path.join(self.root_path, "random_feature")
        make_folder(self.store_path)
        self.normal_data_path = r"X:\DYB\data_and_result\doctor tao\data\Normal control"
        self.store_random_normal_case = os.path.join(self.root_path, "Random Normal Control")
        make_folder(self.store_random_normal_case)

    def get_normal_feature(self):
        pd_all_normal = []
        for roi in self.roi_list:
            normal_path = os.path.join(self.root_path, roi, "normal.csv")
            pd_all_normal.append(pd.read_csv(normal_path))
        pd_all_normal = pd.concat(pd_all_normal)
        pd_all_normal.to_csv(os.path.join(self.store_path, "all_normal.csv"), index=None)

    def get_train_test(self):
        normal_path = os.path.join(self.store_path, "all_normal.csv")
        instability_path = os.path.join(self.store_path, "instability.csv")
        pd_normal = pd.read_csv(normal_path)
        pd_normal.insert(loc=1, column="label", value=0)
        pd_instability = pd.read_csv(instability_path)
        pd_instability.insert(loc=1, column="label", value=1)
        result = pd_normal.append(pd_instability)
        result.to_csv(os.path.join(self.store_path, "train_test.csv"), index=None)

    def get_random_case(self):
        for case in os.listdir(self.normal_data_path):
            index_roi = random.randint(1, 8)
            store_path = os.path.dirname(self.normal_data_path)
            roi_path = os.path.join(self.normal_data_path, case, "roi_" + str(index_roi) + ".nii")
            data_path = os.path.join(self.normal_data_path, case, "data.nii")
            store_case_path = os.path.join(store_path, "Random Normal Case", case)
            make_folder(store_case_path)
            shutil.copy(roi_path, store_case_path)
            shutil.copy(data_path, store_case_path)

    def rename(self):
        folder_path = os.path.join(os.path.dirname(self.normal_data_path), "Random Normal Case")
        for case in os.listdir(folder_path):
            for file in os.listdir(os.path.join(folder_path, case)):
                if file != "data.nii":
                    roi_path = os.path.join(folder_path, case, file)
                    os.rename(roi_path, os.path.join(folder_path, case, "roi.nii"))
                    print(case)


def get_train_test(feature_1, feature_2):
    pd_feature_1 = pd.read_csv(feature_1)  # 不稳定组
    pd_feature_2 = pd.read_csv(feature_2)  # 对照组
    pd_feature_1.insert(loc=1, column="label", value=1)
    pd_feature_2.insert(loc=1, column="label", value=0)
    result = pd_feature_1.append(pd_feature_2)
    store_path = os.path.dirname(feature_1)
    result.to_csv(os.path.join(store_path, "train_test.csv"), index=None)


feature_1 = r"X:\DYB\data_and_result\doctor tao\big_roi\instability.csv"
feature_2 = r"X:\DYB\data_and_result\doctor tao\big_roi\normal.csv"
get_train_test(feature_1, feature_2)

if __name__ == '__main__':
    get_random_feature = GetRandomFeature()
    # get_random_feature.get_normal_feature()
    # get_random_feature.get_train_test()
    # get_random_feature.get_random_case()
    # get_random_feature.rename()