"""本文件是为了写踝关节摘要"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from MeDIT.Visualization import Imshow3DArray
from Tool.DataProcress import get_array_from_path
from sklearn.metrics import roc_curve, roc_auc_score

""""

    摘要显示图的问题：
    （1）：将半个软骨，软骨下骨的数值用表格的形式显示出来，一张表
    （2）：将最终的软骨、软骨下骨、软骨+软骨下骨的AUC图显示出来
    （3）：画一个流程图
    （4）：画一个关节的图

"""


##############################################
# 显示ROC曲线
# 将8个软骨，软骨下骨模型的ROC曲线显示到一个图中
##############################################


#######################
#  ROC 画曲线的函数
#######################

# color_list = sns.color_palette('deep') + sns.color_palette('bright')
# def DrawROCList(pred_list, label_list, name_list='', store_path='', is_show=True, fig=plt.figure()):
#     '''
#     To Draw the ROC curve.
#     :param pred_list: The list of the prediction.
#     :param label_list: The list of the label.
#     :param name_list: The list of the legend name.
#     :param store_path: The store path. Support jpg and eps.
#     :return: None
#
#     Apr-28-18, Yang SONG [yang.song.91@foxmail.com]
#     '''
#     if not isinstance(pred_list, list):
#         pred_list = [pred_list]
#     if not isinstance(label_list, list):
#         label_list = [label_list]
#     if not isinstance(name_list, list):
#         name_list = [name_list]
#
#     fig.clear()
#     axes = fig.add_subplot(1, 1, 1)
#
#     for index in range(len(pred_list)):
#         fpr, tpr, threshold = roc_curve(label_list[index], pred_list[index])
#         auc = roc_auc_score(label_list[index], pred_list[index])
#         name_list[index] = name_list[index] + (' (AUC = %0.3f)' % auc)
#
#         axes.plot(fpr, tpr, color=color_list[index], label='ROC curve (AUC = %0.3f)' % auc,linewidth=3)
#
#     axes.plot([0, 1], [0, 1], color='navy', linestyle='--')
#     axes.set_xlim(0.0, 1.0)
#     axes.set_ylim(0.0, 1.05)
#     axes.set_xlabel('False Positive Rate')
#     axes.set_ylabel('True Positive Rate')
#     axes.set_title('Receiver operating characteristic curve')
#     axes.legend(name_list, loc="lower right")
#     if store_path:
#         fig.set_tight_layout(True)
#         if store_path[-3:] == 'jpg':
#             fig.savefig(store_path, dpi=1200, format='jpeg')
#         elif store_path[-3:] == 'eps':
#             fig.savefig(store_path, dpi=1200, format='eps')
#
#     # if is_show:
#     #     plt.show()
#
#     return axes

########################
#  得到画AUC曲线的数值
########################

def get_plot(pred_list, label_list, name_list='', store_path='', is_show=True, fig=plt.figure()):
    '''
    To Draw the ROC curve.
    :param pred_list: The list of the prediction.
    :param label_list: The list of the label.
    :param name_list: The list of the legend name.
    :param store_path: The store path. Support jpg and eps.
    :return: None

    Apr-28-18, Yang SONG [yang.song.91@foxmail.com]
    '''
    if not isinstance(pred_list, list):
        pred_list = [pred_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if not isinstance(name_list, list):
        name_list = [name_list]
    for index in range(len(pred_list)):
        fpr, tpr, threshold = roc_curve(label_list[index], pred_list[index])
        auc = roc_auc_score(label_list[index], pred_list[index])
        name_list[index] = name_list[index] + (' (AUC = %0.3f)' % auc)
        return fpr, tpr, auc

###################
#  显示一张图
###################


def show_roc(train_prediction_path, test_prediction_path, val_prediction_path, title_name):
    pd_train = pd.read_csv(train_prediction_path)
    pd_test = pd.read_csv(test_prediction_path)
    pd_val = pd.read_csv(val_prediction_path)
    train_pred = pd_train["Pred"]; train_label = pd_train["Label"]
    val_pred = pd_val["Pred"]; val_label = pd_val["Label"]
    test_pred = pd_test["Pred"]; test_label = pd_test["Label"]

    fpr_train, tpr_train, auc_train = get_plot(pred_list=train_pred, label_list=train_label)
    fpr_test, tpr_test, auc_test = get_plot(pred_list=test_pred, label_list=test_label)
    fpr_val, tpr_val, auc_val = get_plot(pred_list=val_pred, label_list=val_label)

    plt.plot(fpr_train, tpr_train, label='ROC curve (AUC = %0.3f)' % auc_train, linewidth=3)  # 画出train
    plt.plot(fpr_val, tpr_val, label='ROC curve (AUC = %0.3f)' % auc_val, linewidth=3)  # 画出val
    plt.plot(fpr_test, tpr_test, linewidth=3)  # 画出test
    plt.legend(["train (AUC = %0.3f)" % auc_train, "val (AUC = %0.3f)" % auc_val, "test (AUC = %0.3f)" % auc_test], loc="lower right")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_name+' ROC curve')
    # plt.show()


# show_roc()

# 显示三个模型的ROC曲线图
def show_model_roc():

    name_list = ["train_prediction.csv", "cv_val_prediction.csv", "test_prediction.csv"]
    # 先绘制软骨部位的ROC曲线
    ruan_model_path = r"Y:\DYB\data_and_result\doctor tao\new_ruan_1--8\gbdt\result\Mean\PCC\RFE_28\SVM"
    ruan_train_path = os.path.join(ruan_model_path, name_list[0])
    ruan_val_path = os.path.join(ruan_model_path, name_list[1])
    ruan_test_path = os.path.join(ruan_model_path, name_list[2])

    # 绘制软骨下骨部位的ROC曲线
    xia_model_path = r"Y:\DYB\data_and_result\doctor tao\xia_1---8\result\Mean\PCC\RFE_20\SVM"
    xia_train_path = os.path.join(xia_model_path, name_list[0])
    xia_val_path = os.path.join(xia_model_path, name_list[1])
    xia_test_path = os.path.join(xia_model_path, name_list[2])

    # 绘制软骨+软骨下骨的ROC曲线
    ruan_and_xia_model_path = r"Y:\DYB\data_and_result\doctor tao\ruan_combine_xia\new_result_11-6\result\Mean\PCC\Relief_38\SVM"
    ruan_and_xia_train_path = os.path.join(ruan_and_xia_model_path, name_list[0])
    ruan_and_xia_val_path = os.path.join(ruan_and_xia_model_path, name_list[1])
    ruan_and_xia_test_path = os.path.join(ruan_and_xia_model_path, name_list[2])


    plt.subplot(2, 3, 1)
    # 显示软骨
    show_roc(train_prediction_path=ruan_train_path, val_prediction_path=ruan_val_path,
             test_prediction_path=ruan_test_path, title_name="cartilage")
    plt.subplot(2, 3, 2)
    # 显示下骨
    show_roc(train_prediction_path=xia_train_path, val_prediction_path=xia_val_path,
             test_prediction_path=xia_test_path, title_name="subchondral")
    plt.subplot(2, 3, 3)
    # 显示软骨和软骨下骨的ROC曲线
    show_roc(train_prediction_path=ruan_and_xia_train_path,
             val_prediction_path=ruan_and_xia_val_path,
             test_prediction_path=ruan_and_xia_test_path,title_name="cartilage and subchondral")
    plt.show()

# show_model_roc()


####################
# 显示关节图
####################
from Tool.DataProcress import standard

def crop_roi(data_array):
    """对图像进行一次裁剪"""
    new_data_array = data_array[90:, 40:300,...]
    return new_data_array
#
# data_path = r"Y:\DYB\data_and_result\doctor tao\data\all_data\Normal control\10007807\data.nii"
# roi_path = r"Y:\DYB\data_and_result\doctor tao\data\all_data\Normal control\10007807\new_ruan_roi_1.nii"
# crop_roi(data_path, roi_path)


def show_ankle_figure(case_path, ruan, xia, ren):
    # case_path = r"Y:\DYB\data_and_result\doctor tao\data\all_data\Ankle instability\DICOMDIT"
    data_path = os.path.join(case_path, "data.nii")
    # 处理软骨部位的ROI
    ruan_roi_name_list = ["new_ruan_roi_" + str(i) + ".nii" for i in range(1, 9)]
    ruan_roi_path_list = [os.path.join(case_path, i ) for i in ruan_roi_name_list]
    ruan_roi_array_list = [get_array_from_path(i) for i in ruan_roi_path_list]
    data_array = get_array_from_path(data_path)
    # 处理软骨下骨部位的ROI
    xia_roi_name_list = ["roi_"+str(i)+".nii" for i in range(1, 9)]
    xia_roi_path_list = [os.path.join(case_path, i) for i in xia_roi_name_list]
    xia_roi_array_list = [get_array_from_path(i) for i in xia_roi_path_list]
    # 医生标注的软骨与软骨下骨
    yi_roi_name_list = ["ruan_roi_"+str(i)+".nii" for i in range(1, 9)]
    yi_roi_path_list = [os.path.join(case_path, i) for i in yi_roi_name_list]
    yi_roi_array_list = [get_array_from_path(i) for i in yi_roi_path_list]

    def add_sub_plot(roi_index, new_roi_array_list):
        for i in range(data_array.shape[-1]):
            if np.sum(ruan_roi_array_list[roi_index][..., i]) != 0 and np.sum(xia_roi_array_list[roi_index][..., i] != 0):
                if ruan is True:
                    plt.imshow(data_array[..., i], cmap="gray")
                    # 精准软骨的ROI
                    plt.contour(new_roi_array_list[roi_index][..., i], linewidths=0.45, colors="red")
                    # 精准软骨下骨ROI
                    plt.contour(xia_roi_array_list[roi_index][..., i], linewidths=0.45)
                    # 医生标注的ROI
                    # plt.contour(yi_roi_array_list[roi_index][..., i], linewidths=0.45, colors="yellow")
                    plt.axis('off')
                    # plt.show()
                    break
                if xia is True:
                    plt.imshow(data_array[..., i], cmap="gray")
                    # 精准软骨的ROI
                    # plt.contour(new_roi_array_list[roi_index][..., i], linewidths=0.45, colors="red")
                    # 精准软骨下骨ROI
                    plt.contour(xia_roi_array_list[roi_index][..., i], linewidths=0.45)
                    # 医生标注的ROI
                    # plt.contour(yi_roi_array_list[roi_index][..., i], linewidths=0.45, colors="yellow")
                    plt.axis('off')
                    # plt.show()
                    break
                if ren is True:
                    plt.imshow(data_array[..., i], cmap="gray")
                    # 精准软骨的ROI
                    # plt.contour(new_roi_array_list[roi_index][..., i], linewidths=0.45, colors="red")
                    # 精准软骨下骨ROI
                    # plt.contour(xia_roi_array_list[roi_index][..., i], linewidths=0.45)
                    # 医生标注的ROI
                    plt.contour(yi_roi_array_list[roi_index][..., i], linewidths=0.45, colors="yellow")
                    plt.axis('off')
                    # plt.show()
                    break




    for i in range(0, 8):
        plt.subplot(2, 4, i + 1)
        add_sub_plot(roi_index=i, new_roi_array_list=ruan_roi_array_list)
    plt.axis('off')
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.show()


def run_show_ankle(ruan, xia, ren):
    root_path = r"Y:\DYB\data_and_result\doctor tao\data\all_data\Ankle instability"
    for case in os.listdir(root_path):
        case_path = os.path.join(root_path, case)
        show_ankle_figure(case_path, ruan, xia, ren)


# 感觉好像第四个还是第三个都不错，先将这个图定成这样吧
# run_show_ankle()

#######################################
#  直接将8个模型的ROC曲线显示在一张图上
#######################################

def show_test_roc(test_prediction_path):

    pd_test = pd.read_csv(test_prediction_path)
    test_pred = pd_test["Pred"]; test_label = pd_test["Label"]
    fpr_test, tpr_test, auc_test = get_plot(pred_list=test_pred, label_list=test_label)
    plt.plot(fpr_test, tpr_test, linewidth=3)  # 画出test

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')


def show_ruan_roc(train_test_val):
    """显示软骨的ROC曲线"""
    root_path = r"Y:\DYB\data_and_result\doctor tao"
    roi_1_path = os.path.join(root_path, "new_ruan_roi_1", "result", "Mean", "PCC", "Relief_29", "SVM")
    roi_2_path = os.path.join(root_path, "new_ruan_roi_2", "result", "Mean", "PCC", "RFE_11", "SVM")
    roi_3_path = os.path.join(root_path, "new_ruan_roi_3", "result", "MinMax", "PCC", "RFE_16", "SVM")
    roi_4_path = os.path.join(root_path, "new_ruan_roi_4", "result", "Mean", "PCC", "RFE_18", "SVM")
    roi_5_path = os.path.join(root_path, "new_ruan_roi_5", "result", "Zscore", "PCC", "RFE_31", "SVM")
    roi_6_path = os.path.join(root_path, "new_ruan_roi_6", "result", "Mean", "PCC", "RFE_15", "SVM")
    roi_7_path = os.path.join(root_path, "new_ruan_roi_7", "result", "Mean", "PCC", "RFE_16", "SVM")
    roi_8_path = os.path.join(root_path, "new_ruan_roi_8", "result", "Zscore", "PCC", "RFE_37", "SVM")
    roi_path_list = [roi_1_path, roi_2_path, roi_3_path, roi_4_path, roi_5_path, roi_6_path, roi_7_path,
                     roi_8_path]
    roi_path_list = [os.path.join(i, train_test_val + "_prediction.csv") for i in roi_path_list]
    pd_roi = [pd.read_csv(i) for i in roi_path_list]
    test_pred_list = [i["Pred"] for i in pd_roi]
    label_list = [i["Label"] for i in pd_roi]

    f_1, t_1, a_1 = get_plot(pred_list=test_pred_list[0], label_list=label_list[0])
    f_2, t_2, a_2 = get_plot(pred_list=test_pred_list[1], label_list=label_list[1])
    f_3, t_3, a_3 = get_plot(pred_list=test_pred_list[2], label_list=label_list[2])
    f_4, t_4, a_4 = get_plot(pred_list=test_pred_list[3], label_list=label_list[3])
    f_5, t_5, a_5 = get_plot(pred_list=test_pred_list[4], label_list=label_list[4])
    f_6, t_6, a_6 = get_plot(pred_list=test_pred_list[5], label_list=label_list[5])
    f_7, t_7, a_7 = get_plot(pred_list=test_pred_list[6], label_list=label_list[6])
    f_8, t_8, a_8 = get_plot(pred_list=test_pred_list[7], label_list=label_list[7])
    plt.plot(f_1, t_1, linewidth=3)
    plt.plot(f_2, t_2, linewidth=3)
    plt.plot(f_3, t_3, linewidth=3)
    plt.plot(f_4, t_4, linewidth=3)
    plt.plot(f_5, t_5, linewidth=3)
    plt.plot(f_6, t_6, linewidth=3)
    plt.plot(f_7, t_7, linewidth=3)
    plt.plot(f_8, t_8, linewidth=3)

    # plt.legend(["roi 1 " + train_test_val + " (AUC = %0.3f)" % a_1,
    #             "roi 2 " + train_test_val + " (AUC = %0.3f)" % a_2,
    #             "roi 3 " + train_test_val + " (AUC = %0.3f)" % a_3,
    #             "roi 4 " + train_test_val + " (AUC = %0.3f)" % a_4,
    #             "roi 5 " + train_test_val + " (AUC = %0.3f)" % a_5,
    #             "roi 6 " + train_test_val + " (AUC = %0.3f)" % a_6,
    #             "roi 7 " + train_test_val + " (AUC = %0.3f)" % a_7,
    #             "roi 8 " + train_test_val + " (AUC = %0.3f)" % a_8],
    #            loc="lower right")
    plt.legend(["roi 1 (AUC=%0.3f)" % a_1,
                "roi 2 (AUC=%0.3f)" % a_2,
                "roi 3 (AUC=%0.3f)" % a_3,
                "roi 4 (AUC=%0.3f)" % a_4,
                "roi 5 (AUC=%0.3f)" % a_5,
                "roi 6 (AUC=%0.3f)" % a_6,
                "roi 7 (AUC=%0.3f)" % a_7,
                "roi 8 (AUC=%0.3f)" % a_8],
               loc="lower right")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Cartilage ROC curve')
    # plt.show()


def show_xia_roc(train_test_val):
    roi_1_path = r"Y:\DYB\data_and_result\doctor tao\roi_1\result_2\Zscore\PCC\RFE_22\SVM"
    roi_2_path = r"Y:\DYB\data_and_result\doctor tao\roi_2\result_2\Zscore\PCC\ANOVA_7\SVM"
    roi_3_path = r"Y:\DYB\data_and_result\doctor tao\roi_3\result_2\Zscore\PCC\RFE_7\SVM"
    roi_4_path = r"Y:\DYB\data_and_result\doctor tao\roi_4\result_2\Zscore\PCC\RFE_20\SVM"
    roi_5_path = r"Y:\DYB\data_and_result\doctor tao\roi_5\result_2\Zscore\PCC\ANOVA_26\SVM"
    roi_6_path = r"Y:\DYB\data_and_result\doctor tao\roi_6\result_2\Mean\PCC\RFE_18\SVM"
    roi_7_path = r"Y:\DYB\data_and_result\doctor tao\roi_7\result_2\Mean\PCC\RFE_8\SVM"  # 这里是有问题的，现在知道这个问题的影响大不大
    roi_8_path = r"Y:\DYB\data_and_result\doctor tao\roi_8\result_2\Mean\PCC\RFE_12\SVM"
    roi_path_list = [roi_1_path, roi_2_path, roi_3_path, roi_4_path, roi_5_path, roi_6_path, roi_7_path,
                     roi_8_path]
    roi_path_list = [os.path.join(i, train_test_val+"_prediction.csv") for i in roi_path_list]
    pd_roi = [pd.read_csv(i) for i in roi_path_list]
    test_pred_list = [i["Pred"] for i in pd_roi]
    label_list = [i["Label"] for i in pd_roi]

    f_1, t_1, a_1 = get_plot(pred_list=test_pred_list[0], label_list=label_list[0])
    f_2, t_2, a_2 = get_plot(pred_list=test_pred_list[1], label_list=label_list[1])
    f_3, t_3, a_3 = get_plot(pred_list=test_pred_list[2], label_list=label_list[2])
    f_4, t_4, a_4 = get_plot(pred_list=test_pred_list[3], label_list=label_list[3])
    f_5, t_5, a_5 = get_plot(pred_list=test_pred_list[4], label_list=label_list[4])
    f_6, t_6, a_6 = get_plot(pred_list=test_pred_list[5], label_list=label_list[5])
    f_7, t_7, a_7 = get_plot(pred_list=test_pred_list[6], label_list=label_list[6])
    f_8, t_8, a_8 = get_plot(pred_list=test_pred_list[7], label_list=label_list[7])
    plt.plot(f_1, t_1, linewidth=3)
    plt.plot(f_2, t_2, linewidth=3)
    plt.plot(f_3, t_3, linewidth=3)
    plt.plot(f_4, t_4, linewidth=3)
    plt.plot(f_5, t_5, linewidth=3)
    plt.plot(f_6, t_6, linewidth=3)
    plt.plot(f_7, t_7, linewidth=3)
    plt.plot(f_8, t_8, linewidth=3)

    # plt.legend(["roi 1 " + train_test_val + " (AUC = %0.3f)" % a_1,
    #             "roi 2 " + train_test_val + " (AUC = %0.3f)" % a_2,
    #             "roi 3 " + train_test_val + " (AUC = %0.3f)" % a_3,
    #             "roi 4 " + train_test_val + " (AUC = %0.3f)" % a_4,
    #             "roi 5 " + train_test_val + " (AUC = %0.3f)" % a_5,
    #             "roi 6 " + train_test_val + " (AUC = %0.3f)" % a_6,
    #             "roi 7 " + train_test_val + " (AUC = %0.3f)" % a_7,
    #             "roi 8 " + train_test_val + " (AUC = %0.3f)" % a_8],
    #            loc="lower right")
    plt.legend(["roi 1 (AUC=%0.3f)" % a_1,
                "roi 2 (AUC=%0.3f)" % a_2,
                "roi 3 (AUC=%0.3f)" % a_3,
                "roi 4 (AUC=%0.3f)" % a_4,
                "roi 5 (AUC=%0.3f)" % a_5,
                "roi 6 (AUC=%0.3f)" % a_6,
                "roi 7 (AUC=%0.3f)" % a_7,
                "roi 8 (AUC=%0.3f)" % a_8],
               loc="lower right")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Subchondral ROC curve')
    # plt.show()


def show_ruan_and_xia_roc():
    plt.subplot(2, 2, 1)
    show_ruan_roc(train_test_val="test")
    plt.subplot(2, 2, 2)
    show_xia_roc(train_test_val="test")
    plt.show()
    #
    # plt.subplot(2, 2, 3)
    # show_ruan_roc(train_test_val="test")
    # plt.subplot(2, 2, 4)
    # show_xia_roc(train_test_val="test")
    # plt.show()
    # show_ruan_roc(train_test_val="train")
    # show_ruan_roc(train_test_val="test")
    # show_xia_roc(train_test_val="train")
    # show_xia_roc(train_test_val="test")


# show_ruan_and_xia_roc()


def show_3d_roi():
    case_path = r"Y:\DYB\data_and_result\doctor tao\data\all_data\Normal control\10007807"
    ruan_roi_list = ["new_ruan_roi_" + str(i) for i in range(1, 8)]
    xia_roi_list = ["roi_" + str(i) for i in range(1, 8)]
    ruan_roi_path_list = [os.path.join(case_path, i) for i in ruan_roi_list]
    xia_roi_path_list = [os.path.join(case_path, i) for i in xia_roi_list]
    ruan_roi_array_list = [get_array_from_path(i) for i in ruan_roi_path_list]
    xia_roi_array_list = [get_array_from_path(i) for i in xia_roi_path_list]
    data_array = get_array_from_path(os.path.join(case_path, "data.nii"))



#####################
#  特征可视化
#####################
# plt.style.use("ggplot")






class ShowFeature:
    """
    对流程图进行显示
    """
    def __init__(self):
        self.root_path = r"Y:\DYB\data_and_result\doctor tao\sub_model"

    def show_fristorder(self):
        """显示灰阶图"""
        firstorder_path = os.path.join(self.root_path, "firstorder", "firstorder_.csv")
        pd_firstorder = pd.read_csv(firstorder_path)
        feature_name = pd_firstorder.columns
        for single_feature in feature_name[2:]:
            feature = pd_firstorder[single_feature]
            plt.hist(feature, bins=80, edgecolor="black", alpha=0.8)
            plt.show()

    def show_glcm(self):
        """显示热图"""
        glcm_path = os.path.join(self.root_path, "glcm", "glcm_.csv")
        pd_glcm = pd.read_csv(glcm_path)
        print(pd_glcm.shape)
        # from MachineLearning.load_data import LoadData
        # load_data = LoadData()
        # load_data.set_pd_data(pd_data=pd_glcm)
        # case_name, label, feature_name, array = load_data.get_element()

        sns.heatmap(pd_glcm,
                    linewidths=.5, cmap='YlGnBu', vmin=None, vmax=None, center=None, robust=True,
                    annot=None, fmt=".2g", annot_kws=None,
                    linecolor="white",
                    cbar=True, cbar_kws=None, cbar_ax=None,
                    square=False, xticklabels="auto", yticklabels="auto",
                    mask=None, ax=None,
                    )
        plt.show()


    def show_pcc(self):
        """显示相关性系数图PCC"""
        gldm_path = os.path.join(self.root_path, "gldm", "gldm_.csv")
        pd_gldm = pd.read_csv(gldm_path)
        feature_name = pd_gldm.columns[2:]
        for i in range(0, len(feature_name)):
            sns.regplot(x=pd_gldm[feature_name[i]],  # T2值
                        y=pd_gldm[feature_name[i+1]],  # 特征值

                        scatter_kws={'s': 15}, label="feature_1"
                        )
            sns.regplot(x=pd_gldm[feature_name[i+2]],  # T2值
                        y=pd_gldm[feature_name[i+3]],  # 特征值

                        scatter_kws={'s': 15}, label="feature_2")
            plt.show()

    def show_probability(self, youden_index=0.5):
        """显示概率图"""
        csv_path = r"Y:\DYB\data_and_result\doctor tao\ruan_combine_xia\new_result_11-6\result\Mean\PCC\Relief_38\SVM\test_prediction.csv"
        pd_csv = pd.read_csv(csv_path)
        pred = pd_csv["Pred"]
        label = pd_csv["Label"]
        df = pd.DataFrame({"pred":pred, "label":label})
        df = df.sort_values("pred")
        color_list = ["steelblue", "coral"]
        bar_color = [color_list[x] for x in df["label"].values]
        plt.bar(range(len(pred)),
                df["pred"].values-youden_index,
                color=bar_color, width=0.75)

        plt.yticks([df["pred"].values.min() - youden_index,
                    youden_index - youden_index,
                    df["pred"].max() - youden_index],

                   ['{:.2f}'.format(df["pred"].values.min()),
                    "{:.2f}".format(youden_index),
                    "{:.2f}".format(df["pred"].max())])
        plt.xlabel("case")
        plt.ylabel("probability")
        plt.title("R-Score")
        # plt.show()








#########################
# 将ROI进行Crop显示出来
#########################
import os
import cv2
from Tool.DataProcress import get_array_from_path, standard
from MeDIT.Visualization import Imshow3DArray
from MeDIT.SaveAndLoad import SaveArrayToNiiByRef

###################
#  分割前的预处理
###################



def get_roi_index(roi_all):
    roi_sum = np.sum(roi_all, axis=(0, 1))
    roi_index = list(np.where(roi_sum != 0)[0])
    if roi_index[0] != 0:
        roi_index.insert(0, roi_index[0] - 1)
    if roi_index[-1] != roi_sum.shape[0] - 1:
        roi_index.append(roi_index[-1] + 1)
    return roi_index

def get_crop_shape(roi_all):

    h_roi = np.sum(roi_all, axis=0)  # shape(height, channels)
    h_roi = np.sum(h_roi, axis=1)  # 将数据加和到高这个维度上，shape(heights,)
    h_corp = np.where(h_roi != 0)[0]
    h_left, h_right = np.min(h_corp) - 1, np.max(h_corp) + 2

    w_roi = np.sum(roi_all, axis=1)
    w_roi = np.sum(w_roi, axis=1)
    w_crop = np.where(w_roi != 0)[0]
    w_top, w_bottom = np.min(w_crop) - 1, np.max(w_crop) + 2
    return h_left, h_right, w_top, w_bottom


def crop_data_array(roi_all, data_array):
    h_left, h_right, w_top, w_bottom = get_crop_shape(roi_all=roi_all)
    index_roi = get_roi_index(roi_all=roi_all)
    crop_array = data_array[w_top - 15: w_bottom + 15, h_left - 15: h_right + 15, :]
    return crop_array


def get_croped_data_roi(data_path, roi_path, store_path=None):
    data_array = get_array_from_path(data_path)
    # data_array = np.flipud(data_array)  # 注意这里的数据要进行翻转的
    roi_array = get_array_from_path(roi_path)
    data_array = standard(data_array)

    crop_array = crop_data_array(roi_array, data_array)
    def resize_data(data_array, resized_shape):
        resized_data = cv2.resize(data_array, resized_shape, interpolation=cv2.INTER_NEAREST)
        return resized_data

    roi_new = crop_data_array(roi_array, roi_array)
    new_array = resize_data(crop_array, (224, 224))
    roi_new = resize_data(roi_new, (224, 224))

    # 选择只含有roi的层进行保存
    index = []
    for i in range(roi_new.shape[-1]):
        if np.sum(roi_new[..., i]) != 0:
            index.append(i)

    roi_new = roi_new[..., index]
    data_new = new_array[..., index]
    Imshow3DArray(data_new, roi_new)



from sklearn import metrics
from sklearn.calibration import calibration_curve
def CalibrationCurve():
    csv_path = r"Y:\DYB\data_and_result\doctor tao\ruan_combine_xia\new_result_11-6\result\Mean\PCC\Relief_38\SVM\test_prediction.csv"
    pd_csv = pd.read_csv(csv_path)
    pred = pd_csv["Pred"]
    label = pd_csv["Label"]
    prediction = pred
    F, threshold = calibration_curve(label, prediction, n_bins=10)
    clf_score = metrics.brier_score_loss(label, prediction, pos_label=1)
    plt.plot(threshold, F, "s-", label='{:.3f}'.format(clf_score))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.ylim([-0.05, 1.05])
    plt.xlabel("probability")
    plt.legend(loc="lower right")
    plt.title("calibration plot")
    # plt.show()


def show_R_Score_and_CalibrationCurve():
    plt.subplot(2, 2, 1)
    show_feature = ShowFeature()
    show_feature.show_probability()
    plt.subplot(2, 2, 2)
    CalibrationCurve()
    plt.show()


#################
# 显示箱线图
#################

def plot_multi_double_hists(values, values_names=None, row=2, colums=4, fig_size=(18, 8),
                            ysticks=range(1, 10), is_save=False, store_path=''):
    '''
    将多个对比直方图画在一张图里
    :param features: 需要画直方图的数据, 格式为列表 [[value0, value1], [], ...]
    :param feature_names: 需要画的直方图名字列表，
    :param row: 每行放图的个数
    :param colums: 每列放图的个数
    :param fig_size: 图大小
    :param ysticks: y轴的范围
    :return:
    '''
    fig, axes = plt.subplots(row, colums)
    num = len(values)
    for i in range(num):
        # title = values_names[i]
        value0 = values[i][0]
        value1 = values[i][1]
        x = i // colums
        y = i % colums
        axes[x][y].hist(value0, histtype='bar', alpha=0.5, edgecolor='black', color='green', label='cartilage')
        axes[x][y].hist(value1, histtype='bar', alpha=0.5, edgecolor='black', color='red', label='Subchondral')
        # axes[x][y].set_title("xxx")
        axes[x][y].set_yticks(ysticks)
        axes[x][y].legend(loc='upper right')
    if is_save:
        plt.savefig(store_path, format='tif', dpi='300', bbox_inches='tight', pad_inches=0)
    plt.show()


from scipy import stats

def Auc(y_true, y_pred, ci_index=0.95):
    """
    This function can help calculate the AUC value and the confidence intervals. It is note the confidence interval is
    not calculated by the standard deviation. The auc is calculated by sklearn and the auc of the group are bootstraped
    1000 times. the confidence interval are extracted from the bootstrap result.

    Ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2820000515%2919%3A9%3C1141%3A%3AAID-SIM479%3E3.0.CO%3B2-F
    :param y_true: The label, dim should be 1.
    :param y_pred: The prediction, dim should be 1
    :param CI_index: The range of confidence interval. Default is 95%
    """

    single_auc = metrics.roc_auc_score(y_true, y_pred)

    bootstrapped_scores = []

    np.random.seed(42)  # control reproducibility
    seed_index = np.random.randint(0, 65535, 1000)
    for seed in seed_index.tolist():
        np.random.seed(seed)
        pred_one_sample = np.random.choice(y_pred, size=y_pred.size, replace=True)
        np.random.seed(seed)
        label_one_sample = np.random.choice(y_true, size=y_pred.size, replace=True)

        if len(np.unique(label_one_sample)) < 2:
            continue

        score = metrics.roc_auc_score(label_one_sample, pred_one_sample)
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    std_auc, mean_auc = np.std(sorted_scores), np.mean(sorted_scores)

    ci = stats.norm.interval(ci_index, loc=mean_auc, scale=std_auc)
    return single_auc, mean_auc, std_auc, ci


##########################
# 计算软骨与软骨下骨的CI
##########################
def get_ruan_ci():
    ruan_ci_list = []
    root_path = r"Y:\DYB\data_and_result\doctor tao"
    roi_1_path = os.path.join(root_path, "new_ruan_roi_1", "result", "Mean", "PCC", "Relief_29", "SVM")
    roi_2_path = os.path.join(root_path, "new_ruan_roi_2", "result", "Mean", "PCC", "RFE_11", "SVM")
    roi_3_path = os.path.join(root_path, "new_ruan_roi_3", "result", "MinMax", "PCC", "RFE_16", "SVM")
    roi_4_path = os.path.join(root_path, "new_ruan_roi_4", "result", "Mean", "PCC", "RFE_18", "SVM")
    roi_5_path = os.path.join(root_path, "new_ruan_roi_5", "result", "Zscore", "PCC", "RFE_31", "SVM")
    roi_6_path = os.path.join(root_path, "new_ruan_roi_6", "result", "Mean", "PCC", "RFE_15", "SVM")
    roi_7_path = os.path.join(root_path, "new_ruan_roi_7", "result", "Mean", "PCC", "RFE_16", "SVM")
    roi_8_path = os.path.join(root_path, "new_ruan_roi_8", "result", "Zscore", "PCC", "RFE_37", "SVM")
    roi_path_list = [roi_1_path, roi_2_path, roi_3_path, roi_4_path, roi_5_path, roi_6_path, roi_7_path,
                     roi_8_path]
    roi_path_list = [os.path.join(i,   "test_prediction.csv") for i in roi_path_list]
    pd_roi = [pd.read_csv(i) for i in roi_path_list]
    test_pred_list = [i["Pred"] for i in pd_roi]
    label_list = [i["Label"] for i in pd_roi]
    for pred, label in zip(test_pred_list, label_list):
        single_auc, mean_auc, std_auc, ci = Auc(y_true=label, y_pred=pred, ci_index=0.95)
        ruan_ci_list.append(ci)
    return ruan_ci_list

def get_xia_ci():
    xia_ci_list = []
    roi_1_path = r"Y:\DYB\data_and_result\doctor tao\roi_1\result_2\Zscore\PCC\RFE_22\SVM"
    roi_2_path = r"Y:\DYB\data_and_result\doctor tao\roi_2\result_2\Zscore\PCC\ANOVA_7\SVM"
    roi_3_path = r"Y:\DYB\data_and_result\doctor tao\roi_3\result_2\Zscore\PCC\RFE_7\SVM"
    roi_4_path = r"Y:\DYB\data_and_result\doctor tao\roi_4\result_2\Zscore\PCC\RFE_20\SVM"
    roi_5_path = r"Y:\DYB\data_and_result\doctor tao\roi_5\result_2\Zscore\PCC\ANOVA_26\SVM"
    roi_6_path = r"Y:\DYB\data_and_result\doctor tao\roi_6\result_2\Mean\PCC\RFE_18\SVM"
    roi_7_path = r"Y:\DYB\data_and_result\doctor tao\roi_7\result_2\Mean\PCC\RFE_8\SVM"  # 这里是有问题的，现在知道这个问题的影响大不大
    roi_8_path = r"Y:\DYB\data_and_result\doctor tao\roi_8\result_2\Mean\PCC\RFE_12\SVM"
    roi_path_list = [roi_1_path, roi_2_path, roi_3_path, roi_4_path, roi_5_path, roi_6_path, roi_7_path,
                     roi_8_path]
    roi_path_list = [os.path.join(i, "test_prediction.csv") for i in roi_path_list]
    pd_roi = [pd.read_csv(i) for i in roi_path_list]
    test_pred_list = [i["Pred"] for i in pd_roi]
    label_list = [i["Label"] for i in pd_roi]
    for pred, label in zip(test_pred_list, label_list):
        single_auc, mean_auc, std_auc, ci = Auc(y_true=label, y_pred=pred, ci_index=0.95)
        xia_ci_list.append(ci)
    return xia_ci_list


def get_bar_plot():
    """画八个模型的箱线图"""
    plt.subplot(2, 1, 1)
    ruan_list = get_ruan_ci()
    plt.boxplot(ruan_list)
    plt.ylim((0.5, 1))
    plt.ylabel("AUC")
    plt.xlabel("ROI 1~8 of Cartilage")
    plt.subplot(2, 1, 2)
    xia_list = get_xia_ci()
    plt.boxplot(xia_list)
    plt.ylim((0.5, 1))
    plt.ylabel("ACU")
    plt.xlabel("ROI 1~8 of Subchondral")
    plt.show()



if __name__ == '__main__':
    # 显示关节图
    # 显示软骨ROI
    # run_show_ankle(ruan=True, xia=False, ren=False)
    # 显示下骨ROI
    # run_show_ankle(ruan=False, xia=True, ren=False)
    # 显示人工标注的ROI
    # run_show_ankle(ruan=False, xia=False, ren=True)

    # 显示流程图的图
    show_feature = ShowFeature()
    # 显示灰阶图
    # show_feature.show_fristorder()
    # 显示热图
    # show_feature.show_glcm()
    # 显示特征相关性系数PCC
    # show_feature.show_pcc()
    # 显示概率图  R_score图
    # show_feature.show_probability()
    # 显示calibration plot
    # CalibrationCurve()
    # 显示8个模型的bar图
    get_bar_plot()
    # show_R_Score_and_CalibrationCurve()








