'''
@Author: ningtao liu
@Date: 2020-07-15 23:05:01
@LastEditors: ningtao liu
@LastEditTime: 2020-07-16 18:09:30
@FilePath: /ToothAge/ResNet/utils.py
'''
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os
from scipy import interp
from itertools import cycle
import numpy as np


# 多分类
def get_performance(label_list,
                    predict_list,
                    save_path=r'./result/gender/image',
                    classes=(0, 1),
                    task='train',
                    best_auc=-1):
    y_pred = np.array([y.tolist() for y in predict_list])
    y_label = np.zeros((len(label_list), len(classes)))
    for i in range(y_label.shape[0]):
        y_label[i][int(label_list[i])] = 1
    y_label.astype(np.int)

    precision = precision_score(np.argmax(y_label, axis=1), np.argmax(y_pred, axis=1), average='micro')
    recall = recall_score(np.argmax(y_label, axis=1), np.argmax(y_pred, axis=1), average='micro')
    F_1 = f1_score(np.argmax(y_label, axis=1), np.argmax(y_pred, axis=1), average='micro')
    accuracy = accuracy_score(np.argmax(y_label, axis=1), np.argmax(y_pred, axis=1))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    if roc_auc['macro'] > best_auc:
        plt.savefig(os.path.join(save_path, 'roc_gender_'+task+'.png'))

    performance_dict = {'precision': precision,
                        'recall': recall,
                        'F_1': F_1,
                        'accuracy': accuracy,
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc
                        }
    return performance_dict


def get_performance_age(label_list,
                        predict_list,
                        save_path=r'',
                        classes=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                        task='train',
                        best_auc=-10):
    y_pred = np.array([y.tolist() for y in predict_list])
    y_label = np.zeros((len(label_list), len(classes)))
    for i in range(y_label.shape[0]):
        y_label[i][int(label_list[i])] = 1
    y_label.astype(np.int)

    precision = precision_score(np.argmax(y_label, axis=1), np.argmax(y_pred, axis=1), average='micro')
    recall = recall_score(np.argmax(y_label, axis=1), np.argmax(y_pred, axis=1), average='micro')
    F_1 = f1_score(np.argmax(y_label, axis=1), np.argmax(y_pred, axis=1), average='micro')
    accuracy = accuracy_score(np.argmax(y_label, axis=1), np.argmax(y_pred, axis=1))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    if roc_auc['macro'] > best_auc:
        plt.savefig(os.path.join(save_path, 'roc_age_'+task+'.png'))

    performance_dict = {'precision': precision,
                        'recall': recall,
                        'F_1': F_1,
                        'accuracy': accuracy,
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc
                        }
    return performance_dict


def get_performance_age_range(label_list: np.ndarray, predict_list: np.ndarray):
    assert label_list.shape[0] == predict_list.shape[0]
    acc_0 = np.mean((((label_list - 0.5) <= predict_list) & ((label_list + 0.4) >= predict_list))*1.0)
    acc_1 = np.mean((((label_list - 1.5) <= predict_list) & ((label_list + 1.4) >= predict_list))*1.0)
    acc_2 = np.mean((((label_list - 2.5) <= predict_list) & ((label_list + 2.4) >= predict_list))*1.0)
    return round(acc_0, 2), round(acc_1, 2), round(acc_2, 2)


def get_performance_gender(label_list,
                           predict_score,
                           save_path=r'',
                           classes=(0, 1),
                           task='train',
                           best_auc=-1):
    label_name = ['M', 'F']
    label_list = np.array(label_list, dtype=np.int32)
    predict_score = np.array(predict_score)
    predict_result = np.array((predict_score > 0.5) * 1)
    acc = np.mean(predict_result == label_list)

    fpr, tpr, _ = roc_curve(label_list, predict_score)
    roc_auc = auc(fpr, tpr)
    if roc_auc > best_auc:
        cm = confusion_matrix(label_list, predict_result)
        plot_confusion_matrix(cm, label_name, 'M＆F Confusion Matrix', os.path.join(save_path, 'cm_{}.jpg'.format(task)))
        plot_auc_cruve(fpr, tpr, roc_auc, os.path.join(save_path, 'auc_{}.jpg'.format(task)))
    
    report = classification_report(label_list, predict_result, target_names=label_name)
    print(report)
    return report, roc_auc, acc


def plot_auc_cruve(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr,
            label='area = {0:0.2f}'.format(roc_auc),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('M&F ROC curve ')
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, labels_name, title, save_path):
    plt.figure(figsize=(8, 8))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    label_list = np.load(r'')
    predict_score = np.load(r'')
    get_performance_gender(label_list,
                           predict_score,
                           save_path=r'./',
                           classes=(0, 1),
                           task='val',
                           best_auc=-1)