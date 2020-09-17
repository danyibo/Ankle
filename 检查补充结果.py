import os
import numpy as np
import SimpleITK as sitk
import glob
from Tool.DataProcress import *
from MeDIT.Visualization import Imshow3DArray


class Check:
    """
    8月14号：检查最新补充的数据
    """
    def __init__(self):
        self.root_path = r"H:\tao\tao_20200715"

    def get_and_save(self):
        """
        将data进行改名并且保存
        将roi和数据进行显示检查
        :return:
        """
        for case in os.listdir(self.root_path):
            case_path = os.path.join(self.root_path, case)
            result_path = os.path.join(case_path, "result")
            from Tool.FolderProcress import remove_folder
            remove_folder(result_path)

            # roi_path = os.path.join(case_path, "roi_1.nii")
            # roi_array = get_array_from_path(roi_path)

            for file in os.listdir(case_path):
                if file.split("_")[-1] == "src.nii":
                    data_path = os.path.join(case_path, file)

                    """检查显示"""

                    # data_array = standard(get_array_from_path(data_path))
                    # Imshow3DArray(data_array, roi_array)

                    shutil.copy(data_path, os.path.join(case_path, "data.nii"))
                    print("Save {} finished!".format(case))

class Chose:
    """
    利用之前最大值建模的特征，进行补充数据建模
    """
    def __init__(self):
        feature_path = r""

if __name__ == '__main__':
    check_data = Check()
    check_data.get_and_save()
