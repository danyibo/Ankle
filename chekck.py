import os
import shutil
from Tool.FolderProcress import make_folder, remove_folder

store_path = r"E:\Data\doctor_xie\新建文件夹 (2)"
data_path = r"E:\Data\doctor_xie\新建文件夹"
for case in os.listdir(data_path):
    store_case_path = os.path.join(store_path, case)
    make_folder(store_case_path)
    case_path = os.path.join(data_path, case)
    for file in os.listdir(case_path):
        if file != "data_1.nii" and file != "roi.nii.gz":
            file_path = os.path.join(case_path, file)
            shutil.move(file_path, store_case_path)