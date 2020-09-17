import os
import numpy as np
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.SaveAndLoad import SaveNumpyToImageByRef
import SimpleITK as sitk


class GetSingleRoi:
    def __init__(self, case_path):
        self.case_path = case_path
        self.all_roi_path = os.path.join(case_path, "result")

    def get_roi_array_list(self):
        roi_array_list = []
        roi_name_list = []
        for file in os.listdir(self.all_roi_path):
            roi_path = os.path.join(self.all_roi_path, file)
            roi_array = get_array_from_path(roi_path)
            roi_array_list.append(roi_array)
            roi_name_list.append(file)
        return roi_array_list, roi_name_list

    def get_data_array(self):
        for file in os.listdir(self.case_path):
            if file.split("_")[-1] == "src.nii":
                data_path = os.path.join(self.case_path, file)
                data = sitk.ReadImage(data_path)
                data_array = get_array_from_path(data_path)
                data_array = standard(data_array)
                return data_array, data

    def check_roi(self, roi_array):
        roi_list = []
        for i in roi_array:
            for j in i:
                for k in j:
                    roi_list.append(k)
        return set(roi_list)

    def get_single_roi(self):
        roi_array_list, roi_name_list = self.get_roi_array_list()

        roi_1 = roi_array_list[0]  # 里面是 1 和 9 两个数值
        roi_2 = roi_array_list[1]
        roi_3 = roi_array_list[2]
        roi_4 = roi_array_list[3]
        roi_5 = roi_array_list[4]
        roi_6 = roi_array_list[5]
        roi_7 = roi_array_list[6]
        roi_8 = roi_array_list[7]

        data_array, data = self.get_data_array()

        roi_1_bone = np.where(roi_1 == 9, 1, 0)   # 距下关节外侧跟骨面
        roi_2_bone = np.where(roi_2 == 10, 1, 0)  # 距下关节外侧距骨面
        roi_3_bone = np.where(roi_3 == 11, 1, 0)  # 胫距关节外侧距骨面
        roi_4_bone = np.where(roi_4 == 12, 1, 0)  # 胫距关节外侧胫骨面
        roi_5_bone = np.where(roi_5 == 13, 1, 0)  # 距下关节内侧跟骨面
        roi_6_bone = np.where(roi_6 == 14, 1, 0)  # 距下关节内侧距骨面
        roi_7_bone = np.where(roi_7 == 15, 1, 0)  # 胫距关节内侧距骨面
        roi_8_bone = np.where(roi_8 == 16, 1, 0)  # 胫距关节内侧胫骨面

        # 保存8个roi
        SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "roi_1.nii"),
                              data=roi_1_bone,
                              ref_image=data)
        SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "roi_2.nii"),
                              data=roi_2_bone,
                              ref_image=data)
        SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "roi_3.nii"),
                              data=roi_3_bone,
                              ref_image=data)
        SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "roi_4.nii"),
                              data=roi_4_bone,
                              ref_image=data)
        SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "roi_5.nii"),
                              data=roi_5_bone,
                              ref_image=data)
        SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "roi_6.nii"),
                              data=roi_6_bone,
                              ref_image=data)
        SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "roi_7.nii"),
                              data=roi_7_bone,
                              ref_image=data)
        SaveNumpyToImageByRef(store_path=os.path.join(self.case_path, "roi_8.nii"),
                              data=roi_8_bone,
                              ref_image=data)

        # Imshow3DArray(data_array, roi_8_bone)

class CheckRoi:
    """
    显示修最终的ROI，让医生进行命名
    """
    def __init__(self, case_path):
        self.case_path = case_path

    def get_roi_list(self):
        roi_array_list = []
        roi_name = ["roi_1.nii", "roi_2.nii", "roi_3.nii", "roi_4.nii",
                    "roi_5.nii", "roi_6.nii", "roi_7.nii", "roi_8.nii"]
        for name in roi_name:
            roi_path = os.path.join(self.case_path, name)
            roi_array = get_array_from_path(roi_path)
            roi_array_list.append(roi_array)
        return roi_array_list

    def show(self):
        for file in os.listdir(self.case_path):
            if file.split("_")[-1] == "src.nii":
                data_path = os.path.join(self.case_path, file)
                data_array = standard(get_array_from_path(data_path))
                roi_array_list = self.get_roi_list()
                Imshow3DArray(data_array, roi_array_list)



def run(all_data_path):
    for folder in os.listdir(all_data_path):
        folder_path = os.path.join(all_data_path, folder)
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)

            """获取单一的ROI并进行保存"""

            try:
                get_single = GetSingleRoi(case_path)
                get_single.get_single_roi()
            except:
                print(case_path)

            """显示ROI希望重新命名"""
            # check_roi = CheckRoi(case_path)
            # check_roi.show()



if __name__ == '__main__':
    all_data_path = r"H:\tao"
    run(all_data_path)