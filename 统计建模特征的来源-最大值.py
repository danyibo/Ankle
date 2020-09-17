"""

    现在建模的特征来自于最大值
    我们的假设是：8个roi并不是都有病，而是其中的某些roi有病
    那么我们可以利用这些建模的特征来返回，看这些最大值来自于几个roi
    如果最大值来自于一个roi，那就完全验证了我们的想法
    如果来自于2-3个也是可以验证
    但是，如果来自于7个roi，那就是随机性的东西

"""

import pandas as pd
import os
import numpy as np




"""实现的思路，先一个case一个case进行统计"""


"""此段是利用最小值和最大值建模后选择出的特征"""

feature_path = r"X:\DYB\data_and_result\doctor tao\max_and_min\log_wave_original\selected\LDA_result\Mean\PCC\Relief_26\LDA\LDA_coef.csv"
feature = pd.read_csv(feature_path)
feature = list(feature["Unnamed: 0"])

store_path =r"X:\DYB\data_and_result\doctor tao\max_and_min\log_wave_original\feature_index"
max_feature = []
min_feature = []
for i in feature:
    if i.split("_")[0] == "MAX":
        max_feature.append(i)
    elif i.split("_")[0] == "MIN":
        min_feature.append(i)
pd_max = pd.DataFrame(data={"max_feature_name": max_feature})
pd_min = pd.DataFrame(data={"min_feature_name": min_feature})
pd_max.to_csv(os.path.join(store_path, "max_index.csv"), index=None)
pd_min.to_csv(os.path.join(store_path, "min_index.csv"), index=None)

selected_feature_name = []
for i in pd_max["max_feature_name"]:
    selected_feature_name.append("data_roi_"+i[4:])

# print(len(selected_feature_name))
# print(selected_feature_name)


class GetFeatureSource:
    def __init__(self):
        self.root_path = r"X:\DYB\data_and_result\doctor tao"
        self.roi_folder = ["roi_1", "roi_2", "roi_3", "roi_4",
                           "roi_5", "roi_6", "roi_7", "roi_8"]
        self.feature_csv_name = "instability.csv"

    def get_case(self):

        pd_new = []
        for roi in self.roi_folder:
            pd_path = os.path.join(self.root_path, roi, self.feature_csv_name)
            pd_feature = pd.read_csv(pd_path)
            case_name = pd_feature["CaseName"]
            pd_selected = pd_feature[selected_feature_name]
            pd_selected_feature = pd.concat((case_name, pd_selected), axis=1)
            pd_new.append(pd_selected_feature)
        return pd_new

    def check(self):
        """
        将添加的表格进行检查
        :return:
        """
        pd_new = self.get_case()
        for i in range(0, 8):
            pd_i = pd.read_csv(os.path.join(self.root_path,
                                            self.roi_folder[i],
                                            self.feature_csv_name))[selected_feature_name[0]][0]

            pd_j = pd_new[i][selected_feature_name[0]][0]
            if pd_i != pd_j:
                print("表格有问题！")

    def get_max(self):
        big_index_list = []
        for i in range(len(selected_feature_name)):
            pd_new = self.get_case()
            case_name = pd_new[0]["CaseName"]
            feature_1 = pd_new[0][selected_feature_name[i]]
            feature_2 = pd_new[1][selected_feature_name[i]]
            feature_3 = pd_new[2][selected_feature_name[i]]
            feature_4 = pd_new[3][selected_feature_name[i]]
            feature_5 = pd_new[4][selected_feature_name[i]]
            feature_6 = pd_new[5][selected_feature_name[i]]
            feature_7 = pd_new[6][selected_feature_name[i]]
            feature_8 = pd_new[7][selected_feature_name[i]]
            # 这是第一个特征在八个roi里面组成的表格
            result_1 = pd.concat((case_name, feature_1, feature_2, feature_3,feature_4,
                                  feature_5, feature_6, feature_7, feature_8,), axis=1)
            # print(result_1)

            index_list = []
            for i in range(len(case_name)):
                value = list(result_1.iloc[i][1:]).index(max(list(result_1.iloc[i][1:])))

                index_list.append(value)

            big_index_list.append(index_list)
        print(len(big_index_list))

        pd_result = pd.DataFrame(data=zip(big_index_list[0], big_index_list[1], big_index_list[2],
                                          big_index_list[3], big_index_list[4], big_index_list[5],
                                          big_index_list[6], big_index_list[7], big_index_list[8]
                                          ), columns=selected_feature_name)
        pd_result.to_csv(os.path.join(store_path, "new_max_index.csv"), index=None)











get = GetFeatureSource()
# get.get_case()
# get.check()
# get.get_max()

result = pd.read_csv(os.path.join(store_path, "new_max_index.csv"))
feature_name = result.columns
value = result.values

repet_list = []
from collections import Counter
print("特征来源索引", "    重复次数", "占有比例")
for x in value:
    # 重复元素个数为
    number = Counter(x).most_common(1)
    for n in number:
        _, reput_number = n
        precent = reput_number / len(x)
        # print(len(x))
        # if precent > 0.4:
        print(x, "   ", reput_number, "   ",round(reput_number/len(x), 1))
        repet_list.append(precent)
# print(feature_name)
import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.hist(repet_list)
plt.show()



# import matplotlib.mlab as mlab
# plt.hist(result[feature_name[8]], range=(0, 8), bins=8)
# plt.xlabel("roi_index")
# plt.ylabel("counts")
# plt.title("model roi index")
# plt.show()




