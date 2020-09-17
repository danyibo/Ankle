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
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import matplotlib.mlab as mlab


root_path = r'X:\DYB\data_and_result\doctor tao\max_and_min\log_wave_original\feature_index'


csv_path = r"X:\DYB\data_and_result\doctor tao\max_and_min\log_wave_original\feature_index\max_and_min_index.csv"
# result = pd.read_csv(csv_path)
# feature_name = result.columns[1:]
# case_name = result["CaseName"]
# value = result.values
# index_list = []
# for i in value:
#     case, *index = i
#     index_list.append(max(index, key=index.count)+1)
# result = pd.DataFrame(data=zip(case_name, index_list))
# result.to_csv(os.path.join(os.path.dirname(csv_path), "case_name_index.csv"), index=None)

repeat_list = []
result = pd.read_csv(csv_path)
feature_name = result.columns
value = result.values
from collections import Counter
print("特征来源索引", "                                                    重复次数", "     占有比例")
for x in value:
    # 重复元素个数为

    number = Counter(x).most_common(1)
    for n in number:
        _, reput_number = n
        precent = reput_number / len(x[1:])
        # print(len(x))
        # if precent > 0.4:
        repeat_list.append(precent)
        print(x[1:], "              ", reput_number, "    ",round(reput_number/len(x), 1))

plt.hist(repeat_list, edgecolor="black", range=(0, 1), alpha=0.9)
plt.title("MAX and MIN ROI: Proportion of repeated ROI")
plt.xlabel("Proportion")
plt.ylabel("Counts")
plt.show()


# for i in range(len(feature_name)):
#     plt.hist(result[feature_name[i]], edgecolor="black", range=(0, 8), bins=8)
#     plt.xlabel("roi_index")
#     plt.ylabel("counts")
#     plt.title(feature_name[i])
    # plt.show()
