import os
import numpy as np
import pandas as pd
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import *


class Check:
    def __init__(self):
        self.root_path = r'H:\tao_20200715'

    def get_roi(self):
        for case in os.listdir(self.root_path):
            case_path = os.path.join(self.root_path, case)
            roi_folder = os.path.join(case_path, "result")
            for file in os.listdir(case_path):
               if file.split("_")[-1] == "src.nii":
                   data_path = os.path.join(case_path, file)


check_data = Check()
check_data.get_roi()
