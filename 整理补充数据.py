import os
import numpy as np
import shutil
from Tool.FolderProcress import make_folder


class GetCase:
    """"

    处理补充的一些数据，补充的是踝关节稳定的对照组数据

    """
    def __init__(self, folder_path, store_path):
        self.folder_path = folder_path
        self.store_path = store_path

    def get_case(self):
        for case in os.listdir(self.folder_path):
            case_path = os.path.join(self.folder_path, case)
            store_case_path = os.path.join(self.store_path, case)
            make_folder(store_case_path)
            for file in os.listdir(case_path):
                if file != "SeriesRois":
                    file_path = os.path.join(case_path, file)
                    shutil.copy(file_path, store_case_path)


class ChangeName:
    """
    将之前改为data.nii的文件名字，改回去
    这样有利于数据的处理和数据的含义识别
    """
    def __init__(self, case_path):
        self.case_path = case_path  # 存放数据的路径，里面是case

    def get_name(self):
        """
        拿到改名的字符串
        :return:
        """
        for file in os.listdir(self.case_path):
            if file != "data.nii":
                name = file.split("_")[:5]
                name_2 = str(name[0]) + "_" + str(name[1]) \
                         + "_" + str(name[2]) + "_" + str(name[3]) + "_" + str(name[4])
                return name_2


    def re_name(self):
        name = self.get_name()
        # print(name)
        try:
            data_nii_path = os.path.join(self.case_path, "data.nii")
            name = self.get_name()
            os.rename(data_nii_path, os.path.join(self.case_path, name + "_src.nii"))
        except:
            print("已经改名！")








if __name__ == '__main__':
    data_path = r"E:\doctor tao\data\normal_add"
    store_path = r"E:\doctor tao\data\normal_add_store"
    get_case = GetCase(folder_path=data_path, store_path=store_path)
    # get_case.get_case()

    folder_path = r"E:\doctor tao\data\postoperative"

    def run_change_name(folder_path=folder_path):
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            change_name = ChangeName(case_path=case_path)
            change_name.re_name()

    run_change_name()

