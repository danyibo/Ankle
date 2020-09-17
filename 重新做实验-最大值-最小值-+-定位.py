import os
import numpy as np
import pandas as pd

########################################
# 先处理最大值和最小值的特征表格删除形状特征  #
########################################


def get_new_max_min():
    old_max_path = r"E:\Data\doctor tao\max\max.csv"
    old_min_path = r"E:\Data\doctor tao\min\min.csv"

    pd_max = pd.read_csv(old_max_path)
    pd_min = pd.read_csv(old_min_path)

    feature_name = pd_max.columns
    new_feature_name = [name for name in feature_name
                        if len(name) > 10
                        if name.split("_")[-2] != "shape"]
    new_feature_name = ["CaseName", "label"] + new_feature_name  # 去除掉形状特征的列索引
    new_max = pd_max[new_feature_name]
    new_min = pd_min[new_feature_name]

    new_max.to_csv(os.path.join(os.path.dirname(old_max_path), "new_max.csv"), index=None)
    new_min.to_csv(os.path.join(os.path.dirname(old_min_path), "new_min.csv"), index=None)

# get_new_max_min()

############################################################
# 按照固定的测试集进行数据拆分：以最后定位比较好的测试为准，进行拆分 #
############################################################



def split_data():
    """
    FAE
    :return:
    """
    pass

####################################
# 利用最大值最小值建模的特征进行合并建模 #
####################################

#: min 选出了11个特征
#: max 选出了
min_feature_path = r"E:\Data\doctor tao\min\min_result\Mean\PCC\KW_11\LR\LR_coef.csv"


