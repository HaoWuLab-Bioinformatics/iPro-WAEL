import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve
import copy
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import os
import sys
from keras.utils import to_categorical
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Bidirectional
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Reshape
from keras.constraints import maxnorm
from keras.callbacks import Callback


def evaluate_metrics(y, final_prediction, final_prediction_label):
    fpr, tpr, thresholdTest = roc_curve(y, final_prediction)
    aucv = auc(fpr, tpr)
    acc = accuracy_score(y, final_prediction_label)
    mcc = matthews_corrcoef(y, final_prediction_label)
    precision, recall, fscore, support = precision_recall_fscore_support(y, final_prediction_label, average='macro')
    return [aucv, acc, mcc, precision, recall, fscore]


def log_loss_func(y_weight, args):
    f = lambda w: log_loss(y_weight, w[0] * args[0] + w[1] * args[1])
    return f


def calculate_final_prediction(weights, y_pred_test, test_num):
    final_prediction = np.zeros(test_num)
    for weight, y_pred in zip(weights, y_pred_test):
        final_prediction += weight * y_pred
    return final_prediction


def calculate_label(final_prediction, test_num):
    final_prediction_label = np.zeros(test_num)
    for i in range(len(final_prediction)):
        if final_prediction[i] >= 0.5:
            final_prediction_label[i] = 1
        else:
            final_prediction_label[i] = 0
    return final_prediction_label

def weight(y_weight, rf_weight_proba, cnn_weight_proba, rf_test_proba, cnn_test_proba):
    iterations_num = 50
    weight_proba = np.zeros((2,len(rf_weight_proba)))
    test_proba = np.zeros((2,len(rf_test_proba)))
    weight_proba[0] = rf_weight_proba
    weight_proba[1] = cnn_weight_proba
    test_proba[0] = rf_test_proba
    test_proba[1] = cnn_test_proba
    print(weight_proba.shape)
    print(test_proba.shape)
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(weight_proba)
    loss = float('inf')
    best_weights = None
    for i in range(iterations_num):
        # print(i)
        prediction_weights = np.random.rand(2)
        prediction_weights = prediction_weights / prediction_weights.sum()  # Initial weight
        result = minimize(log_loss_func(y_weight, weight_proba), prediction_weights, method='SLSQP', bounds=bounds,
                          constraints=constraints)
        # print(result['x'])
        if result['fun'] < loss:
            loss = result['fun']
            best_weights = result['x']  # weights = result['x']

    weights = best_weights
    print(weights)
    final_prediction = calculate_final_prediction(weights, test_proba, len(rf_test_proba))
    final_prediction_label = calculate_label(final_prediction, len(rf_test_proba))
    return final_prediction, final_prediction_label
