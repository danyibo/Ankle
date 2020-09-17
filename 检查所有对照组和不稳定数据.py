import os
import numpy as np
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import *

class Check:
    """
    2020年8月24：
    由于增加了一些数据，竟然效果还不如之前，因此进行一下检查

    """
    def __init__(self):
        self.root_path = r"E:\Data\doctor tao\data"
        self.normal_path = os.path.join(self.root_path, "Normal control")
        self.instability_path = os.path.join(self.root_path, "Ankle instability")
        self.roi_list = ["roi_1", "roi_2", "roi_3", "roi_4", "roi_5", "roi_6",
                         "roi_7", "roi_8"]

    def check_roi(self):
        """
        检查roi是否对应上，即检查层数即可
        :return:
        """

        for case in os.listdir(self.normal_path):
            case_path = os.path.join(self.normal_path, case)
            roi_path = os.path.join(case_path, self.roi_list[0]+".nii")
            roi_array = get_array_from_path(roi_path)
            data_path = os.path.join(case_path, "data.nii")
            data_array = standard(get_array_from_path(data_path))
            index = []
            for i in range(roi_array.shape[-1]):
                if np.sum(roi_array[..., i]) != 0:
                    index.append(i)
            if index[0] < 6:
                Imshow3DArray(data_array, roi_array)





if __name__ == '__main__':
    check = Check()
    check.check_roi()
