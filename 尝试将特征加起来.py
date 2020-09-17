import os
import numpy as np
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
from MeDIT.SaveAndLoad import SaveNumpyToImageByRef
import SimpleITK as sitk


class AddRoi:
    """将8个ROI加起来看看效果如何"""
    def __init__(self):
        self.root_path = r"X:\DYB\data_and_result\doctor tao\data"
        self.normal_and_instability = ["Normal control", "Ankle instability"]

    def get_roi(self):
        for group in self.normal_and_instability:
            for case in os.listdir(os.path.join(self.root_path, group)):
                data_path = os.path.join(self.root_path, group, case, "data.nii")
                data_array = standard(get_array_from_path(data_path))
                data = sitk.ReadImage(data_path)
                big_roi = 0
                for i in range(1, 9):
                    roi_path = os.path.join(self.root_path, group, case, "roi_" + str(i) + ".nii")

                    roi_array = get_array_from_path(roi_path)
                    big_roi += roi_array
                Imshow3DArray(data_array, big_roi)
                # SaveNumpyToImageByRef(store_path=os.path.join(self.root_path, group, case, "big_roi.nii"),
                #                       data=big_roi,
                #                       ref_image=data)
                # print("case {} is saved!".format(case))





add = AddRoi()
add.get_roi()
