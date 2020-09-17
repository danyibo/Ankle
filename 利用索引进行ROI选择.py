import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import *
from Tool.FolderProcress import *

class CopyBuIndex:
    def __init__(self):
        self.root_folder_path = r"X:\DYB\data_and_result\doctor tao\data\Ankle instability"
        self.store_root_path = os.path.join(
            os.path.dirname(self.root_folder_path), "92new_instability")
        self.index_path = r"X:\DYB\data_and_result\doctor tao\max_and_min\log_wave_original\feature_index\case_name_index.csv"

    def copy_by_index(self):
        pd_index = pd.read_csv(self.index_path)
        case_name = pd_index["0"]
        index_number = pd_index["1"]
        for index_case, case, index in zip(case_name, os.listdir(self.root_folder_path), index_number):
            if index_case != case:
                print("数据有问题！")
            if index_case == case:
                target_roi_path = os.path.join(self.root_folder_path, case, "roi_"+str(index)+".nii")
                data_path = os.path.join(self.root_folder_path, case, "data.nii")
                store_case_path = os.path.join(self.store_root_path, case)
                make_folder(store_case_path)
                shutil.copy(target_roi_path, store_case_path)
                shutil.copy(data_path, store_case_path)
                print("copy case {} finished!".format(case))
    def rename(self):
        """
        当数据复制好后，需要对其重命名
        :return:
        """
        for case in os.listdir(self.store_root_path):
            case_path = os.path.join(self.store_root_path, case)
            for file in os.listdir(case_path):
                if file.split("_")[0] == "roi":
                    roi_path = os.path.join(case_path, file)
                    os.rename(roi_path, os.path.join(case_path, "roi.nii"))



if __name__ == '__main__':
    copy = CopyBuIndex()
    # copy.copy_by_index()
    copy.rename()