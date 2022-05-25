import os
import Weighted_average
import RF
import CNN

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import matthews_corrcoef
import numpy as np
import tensorflow as tf


def EvaluateMetrics(y_test, label, proba):
    acc = accuracy_score(y_test, label)
    fpr, tpr, thresholdTest = roc_curve(y_test, proba)
    aucv = auc(fpr, tpr)
    mcc = matthews_corrcoef(y_test, label)
    # precision, recall, fscore, support = precision_recall_fscore_support(y_test, label, average='macro')
    print(aucv, acc, mcc)

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData

def nor_train_test(x_train, x_test):
    x = np.concatenate((x_train, x_test), axis=0)
    x = noramlization(x)
    x_train = x[0:len(x_train)]
    x_test = x[len(x_train):]
    return x_train, x_test

global x_train_rf, x_test_rf
# load features of training sets
cell_lines = 'NHEK'
y_train = np.loadtxt('data/' + cell_lines + '/train/y_train.txt')
features = ['cksnap', 'mismatch', 'rckmer', 'psetnc', 'tpcp']
for feature in features:
    feature_path = 'feature/' + cell_lines + '/train/' + feature + '.csv'
    fea = np.loadtxt(feature_path, delimiter=',')[:,1:]
    if feature == 'cksnap':
        x_train_rf = fea
    else:
        x_train_rf = np.concatenate((x_train_rf, fea), axis=1)


x_train_cnn = np.loadtxt('feature/' + cell_lines + '/train/' + 'word2vec.txt')

# load features of test sets
#cell_lines = 'GM12878'
y_test = np.loadtxt('data/' + cell_lines + '/test/y_test.txt')
for feature in features:
    feature_path = 'feature/' + cell_lines + '/test/' + feature + '.csv'
    fea = np.loadtxt(feature_path, delimiter=',')[:,1:]
    if feature == 'cksnap':
        x_test_rf = fea
    else:
        x_test_rf = np.concatenate((x_test_rf, fea), axis=1)


x_test_cnn = np.loadtxt('feature/' + cell_lines + '/test/' + 'word2vec.txt')
x_train_rf, x_test_rf = nor_train_test(x_train_rf, x_test_rf)


x_train_rf, x_weight_rf, x_train_cnn, x_weight_cnn, y_train, y_weight = train_test_split(
    x_train_rf, x_train_cnn, y_train, test_size=1 / 8, random_state=10)
print(x_train_rf.shape)
print(x_test_rf.shape)
# train and test
cnn_lr = 0.001
cnn_KERNEL_SIZE = 11
cnn_KERNEL_NUM = 32
n_trees = 300
rf_weight_proba, rf_test_proba, rf_test_class = RF.pred(x_train_rf, y_train, x_weight_rf, x_test_rf, n_trees)
cnn_weight_proba, cnn_test_proba, cnn_test_class = CNN.pred(x_train_cnn, y_train, x_weight_cnn, x_test_cnn, cnn_lr,
                                                            cnn_KERNEL_NUM, cnn_KERNEL_SIZE)

EvaluateMetrics(y_test, rf_test_class, rf_test_proba)
EvaluateMetrics(y_test, cnn_test_class, cnn_test_proba)

cnn_weight_proba = cnn_weight_proba.reshape(1, -1)
cnn_test_proba = cnn_test_proba.reshape(1, -1)


proba, label = Weighted_average.weight(
    y_weight, rf_weight_proba, cnn_weight_proba, rf_test_proba, cnn_test_proba)

EvaluateMetrics(y_test, label, proba)
