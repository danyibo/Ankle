import os
import pandas as pd


class CombineFeature:
    def __init__(self, data_path):
        self.data_path = data_path
        # self.instability_path = os.path.join(self.data_path, "instability.csv")
        # self.normal_path = os.path.join(self.data_path, "normal.csv")

        self.instability_train_path = os.path.join(self.data_path, "instability_train.csv")
        self.instability_test_path = os.path.join(self.data_path, "instability_test.csv")

        self.normal_train_path = os.path.join(self.data_path, "normal_train.csv")
        self.normal_test_path = os.path.join(self.data_path, "normal_test.csv")



    def get_feature(self):
        pd_inst_train = pd.read_csv(self.instability_train_path)
        pd_norm_train = pd.read_csv(self.normal_train_path)
        pd_inst_train.insert(loc=1, column="label", value=1)
        pd_norm_train.insert(loc=1, column="label", value=0)
        train_feature = pd_inst_train.append(pd_norm_train)

        pd_inst_test = pd.read_csv(self.instability_test_path)
        pd_norm_test = pd.read_csv(self.normal_test_path)
        pd_inst_test.insert(loc=1, column="label", value=1)
        pd_norm_test.insert(loc=1, column="label", value=0)
        test_feature = pd_inst_test.append(pd_norm_test)


        return train_feature, test_feature

    def save(self):
        train_feature, test_feature = self.get_feature()
        train_feature.to_csv(os.path.join(self.data_path, "train.csv"), index=None)
        test_feature.to_csv(os.path.join(self.data_path, "test.csv"), index=None)

class Combine:

    def __init__(self, data_path):
        self.data_path = data_path
        self.normal_path = os.path.join(self.data_path, "normal.csv")
        self.instability_path = os.path.join(self.data_path, "instability.csv")

    def get_feature(self):
        pd_inst = pd.read_csv(self.instability_path)
        pd_norm = pd.read_csv(self.normal_path)
        pd_inst.insert(loc=1, column="label", value=1)
        pd_norm.insert(loc=1, column="label", value=0)
        result = pd_inst.append(pd_norm)
        return result

    def save(self):
        result = self.get_feature()
        result.to_csv(os.path.join(self.data_path, "train_and_test.csv"), index=None)


def check_feature(feature_path):
    pd_feature = pd.read_csv(feature_path)
    print(pd_feature.shape)


if __name__ == '__main__':
    # combine = CombineFeature(data_path=r'E:\Data\doctor tao\data\lesions split\feature')
    # combine.save()
    root_path = r"E:\Data\doctor tao"
    roi_list = ["roi_1", "roi_2", "roi_3", "roi_4",
                "roi_5", "roi_6", "roi_7", "roi_8"]
    for roi in roi_list:
        folder_path = os.path.join(root_path, roi)
        combine = Combine(data_path=folder_path)
        combine.save()
        feature_path = os.path.join(folder_path, "train_and_test.csv")
        check_feature(feature_path=feature_path)