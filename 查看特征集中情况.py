"""
  检查一下特征集中的情况如何
  再进一步处理了对照组的特征情况，进行比较

"""
import pandas as pd

max_min_path = r"X:\DYB\data_and_result\doctor tao\max_and_min\log_wave_original\feature_index\new_min_index.csv"
pd_max_min = pd.read_csv(max_min_path)


value = pd_max_min.values

percent_list = []
from collections import Counter
print("特征来源索引", "    重复次数", "占有比例")
for x in value:
    print(x[1:])
#     # 重复元素个数为
    number = Counter(x[1:]).most_common(1)
    for n in number:
        _, reput_number = n
        precent = reput_number / len(x[1:])
        percent_list.append(precent)
#         # print(len(x))
#         # if precent > 0.4:
        print(x[1:], "   ", reput_number, "   ",round(reput_number/len(x), 1))

import matplotlib.pyplot as plt
plt.hist(percent_list)
plt.show()