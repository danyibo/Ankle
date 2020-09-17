import os
import numpy as np
from MeDIT.Visualization import Imshow3DArray
import matplotlib.pyplot as plt
from scipy import signal
from Tool.DataProcress import get_array_from_path, standard
import shutil

plt.style.use("ggplot")

"""
    有很多case的ROI生成有问题
    生成的数量很少
 
"""


class CheckRoi:
    def __init__(self, case_path):
        self.case_path = case_path

    def get_roi(self):
        roi_folder = os.path.join(self.case_path, "result")
        roi_number = len(os.listdir(roi_folder))
        return roi_number

    def check_origin_data(self):
        for file in os.listdir(self.case_path):
            file_path = os.path.join(self.case_path, file)
            print(file_path)


def run(all_data_path):
    case_number = []
    roi_number_list = []
    for sub_folder in os.listdir(all_data_path):
        sub_folder_path = os.path.join(all_data_path, sub_folder)
        for case in os.listdir(sub_folder_path):

            case_number.append(case)
            case_path = os.path.join(sub_folder_path, case)
            check_roi = CheckRoi(case_path=case_path)
            roi_number = check_roi.get_roi()
            try:
                check_roi = CheckRoi(case_path=case_path)
                roi_number = check_roi.get_roi()
                if roi_number == 0:
                    print(case_path)
            except:
                pass
            """ 显示生成的ROI个数 """

            roi_number_list.append(roi_number)
    print("总的数据量为{}".format(len(case_number)))

    plt.hist(roi_number_list)
    plt.title("ROI numbers")
    plt.xlabel("roi_numbers")
    plt.ylabel("case_numbers")
    plt.show()


class CheckOriginData:
    def __init__(self, case_path):
        self.case_path = case_path

    def get_data_array(self):
        for file in os.listdir(self.case_path):
            if file.split("_")[-1] == "src.nii":
                data_path = os.path.join(self.case_path, file)
                data_array = get_array_from_path(data_path)
                data_array = standard(data_array)
                return data_array

    def get_roi_array(self):
        roi_array_list = []
        for file in os.listdir(self.case_path):

            if file.split("_")[-1] != "src.nii":
                roi_path = os.path.join(self.case_path, file)
                print(roi_path)
                roi_array = get_array_from_path(roi_path)
                roi_array_list.append(roi_array)
        return roi_array_list

    def show(self):
        data_array = self.get_data_array()
        roi_array = self.get_roi_array()
        Imshow3DArray(data_array, roi_array)


class Rename:
    def __init__(self, case_path):
        self.case_path = case_path

    def copy(self):
        for file in os.listdir(self.case_path):
            if file.split("_")[-1] == "src.nii":
                data_path = os.path.join(self.case_path, file)
                shutil.copy(data_path,os.path.join(self.case_path, "data.nii"))


def re_name(all_data_path):
    for sub_folder in os.listdir(all_data_path):
        sub_folder_path = os.path.join(all_data_path, sub_folder)
        for case in os.listdir(sub_folder_path):
            case_path = os.path.join(sub_folder_path, case)
            rename = Rename(case_path)
            rename.copy()


if __name__ == '__main__':
    all_data_path = r"H:\tao"
    # run(all_data_path=all_data_path)
    case_path = r"H:\tao\tao_20200715\10011763"
    check_origin = CheckOriginData(case_path=case_path)
    check_origin.show()
    # re_name(all_data_path)

