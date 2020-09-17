import os
from Tool.FolderProcress import make_folder
import shutil


class Split:
    def __init__(self, src_folder_path):
        self.src_folder_path = src_folder_path  # 里面存放了train,test
        self.train_path = os.path.join(self.src_folder_path, "train")
        self.test_path = os.path.join(self.src_folder_path, "test")
        self.store_trian_path = os.path.join(self.src_folder_path, "new_train")
        make_folder(self.store_trian_path)
        self.store_test_path = os.path.join(self.src_folder_path, "new_test")
        make_folder(self.store_test_path)


    def get_roi_number(self, folder_path):
        roi_list = []
        data_path_list = []
        roi_path_list = []
        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)
            for file in os.listdir(case_path):
                data_path = os.path.join(case_path, "data.nii")
                if file.split("_")[0] == "roi":
                    roi_list.append(file)
                    roi_path = os.path.join(case_path, file)
                    roi_path_list.append(roi_path)
                    data_path_list.append(data_path)

        return len(roi_list)+1, data_path_list, roi_path_list


    def move_train(self, store_name):
        roi_number, data_path_list, roi_path_list = self.get_roi_number(self.train_path)
        for data, roi, count in zip(data_path_list, roi_path_list, range(1, roi_number)):
            store_data_path = os.path.join(self.store_trian_path, store_name+"_train_" + str(count))
            make_folder(store_data_path)
            shutil.copy(data, store_data_path)
            shutil.copy(roi, store_data_path)

    def move_test(self, store_name):
        roi_number, data_path_list, roi_path_list = self.get_roi_number(self.test_path)
        for data, roi, count in zip(data_path_list, roi_path_list, range(1, roi_number)):
            store_data_path = os.path.join(self.store_test_path, store_name+"_test_" + str(count))
            make_folder(store_data_path)
            shutil.copy(data, store_data_path)
            shutil.copy(roi, store_data_path)

    def run(self, store_name):
        self.move_train(store_name)
        self.move_test(store_name)

def get_roi_and_rename(folder_path):
    for case in os.listdir(folder_path):
        case_path = os.path.join(folder_path, case)
        for file in os.listdir(case_path):
            if file.split("_")[0] == "roi":
                roi_path = os.path.join(case_path, file)
                os.rename(roi_path, os.path.join(case_path, "roi.nii"))

def rename(src_folder_path):
    train_path = os.path.join(src_folder_path, "new_train")
    test_path = os.path.join(src_folder_path, "new_test")
    get_roi_and_rename(folder_path=train_path)
    get_roi_and_rename(folder_path=test_path)


src_folder_path = r"E:\Data\doctor tao\data\lesions split\稳定"
split = Split(src_folder_path=src_folder_path)
# split.run(store_name="instability")
# rename(src_folder_path)
