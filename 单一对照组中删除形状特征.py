import os
import pandas as pd


class DeletShape:
    def __init__(self):
        self.root_path = r"X:\DYB\data_and_result\doctor tao"
        # self.roi_list = ["roi_1", "roi_2", "roi_3", "roi_4",
        #                  "roi_5", "roi_6", "roi_7", "roi_8"]
        self.roi_list = ["max_and_min"]

    def remove_shape(self):
        for roi in self.roi_list:
            roi_folder_path = os.path.join(self.root_path, roi)
            feature_path = os.path.join(roi_folder_path, "max_min_train_test.csv")
            pd_feature = pd.read_csv(feature_path)
            feature_name = pd_feature.columns
            new_feature_name = [name for name in feature_name[2:] if name.split("_")[-2] != "shape"]
            new_feature_name = ["CaseName", "label"] + new_feature_name
            new_feature = pd_feature[new_feature_name]
            new_feature.to_csv(os.path.join(roi_folder_path, "new_train_test.csv"), index=None)


if __name__ == '__main__':
    delet_shape = DeletShape()
    delet_shape.remove_shape()