import numpy as np
import os
import pandas as pd
from FAE.DataContainer.data_container import DataContainer

""" 
    test code 
    we want to get max value or min value in 8 features 
    example:
        set three array
        combine these array into an array with a new axis
        get max value in special axis 
"""

array_1 = [[1, 2, 3],
           [2, 2, 5],
           [2, 3, 9]]
array_2 = [[1, 2, 3],
           [7, 2, 5],
           [2, 9, 100]]
array_3 = [[1, 2, 3],
           [2, 90, 4],
           [50, 3, 9]]
array_1 = np.asarray(array_1)
array_2 = np.asarray(array_2)
array_3 = np.asarray(array_3)

new_array_1 = np.expand_dims(array_1, axis=2)
new_array_2 = np.expand_dims(array_2, axis=2)
new_array_3 = np.expand_dims(array_3, axis=2)


big_array = np.concatenate((new_array_1, new_array_2, new_array_3), axis=2)


class GetMaxMin:
    def __init__(self):
        self.root_path = r'E:\Data\doctor tao'
        self.folder_list = ["roi_1", "roi_2", "roi_3", "roi_4",
                            "roi_5", "roi_6", "roi_7", "roi_8"]

    def get_eight_features(self):
        pd_feature_list = []
        for folder in os.listdir(self.root_path):
            if folder in self.folder_list:
                feature_path = os.path.join(self.root_path, folder, "train_and_test.csv")
                data_continue = DataContainer()
                data_continue.load(feature_path)
                pd_feature_list.append(data_continue.get_array())

        return pd_feature_list

    def get_info(self):
        for folder in os.listdir(self.root_path):
            if folder in self.folder_list:
                feature_path = os.path.join(self.root_path, folder, "train_and_test.csv")
                data_continue = DataContainer()
                data_continue.load(feature_path)
                feature_name = data_continue.get_feature_name()
                label = data_continue.get_label()
                case_name = data_continue.get_case_name()
                return feature_name, label, case_name

    def get_eight_array(self):
        pd_feaure_list = self.get_eight_features()
        expend_dim = []
        for i in pd_feaure_list:
            new = np.expand_dims(i, axis=2)
            expend_dim.append(new)
        f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8 = expend_dim
        eight_feature_array = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8), axis=2)
        return eight_feature_array

    def get_max(self):
        feature_name, label, case_name = self.get_info()
        eight_array = self.get_eight_array()
        max_array = np.max(eight_array, axis=2)
        frame = np.concatenate((label[..., np.newaxis], max_array), axis=1)
        feature_name.insert(0, "label")
        pd_result = pd.DataFrame(data=frame, index=case_name, columns=feature_name)
        pd_result.to_csv(r'E:\Data\doctor tao\max.csv')

    def get_min(self):
        feature_name, label, case_name = self.get_info()
        eight_array = self.get_eight_array()
        min_array = np.min(eight_array, axis=2)
        frame = np.concatenate((label[..., np.newaxis], min_array), axis=1)
        feature_name.insert(0, "label")
        pd_result = pd.DataFrame(data=frame, index=case_name, columns=feature_name)
        pd_result.to_csv(r'E:\Data\doctor tao\min.csv')


if __name__ == '__main__':
    get_min = GetMaxMin()
    get_min.get_min()

    get_max = GetMaxMin()
    get_max.get_max()


