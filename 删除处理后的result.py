import os
from Tool.FolderProcress import remove_folder
import os
# for root, dirs, files in os.walk(dir_path):
#     if not os.listdir(root):
#         os.rmdir(root)


all_data_path = r"E:\Data\doctor tao\data\Normal control"
for case in os.listdir(all_data_path):
    case_path = os.path.join(all_data_path, case)
    result_path = os.path.join(case_path, "result")
    remove_folder(result_path)
