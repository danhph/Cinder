import datetime
import logging
import os
import time

import lightgbm as lgb
import matplotlib
import numpy as np
from sklearn import metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_model():
    model_path = os.path.join(os.getcwd(), "model.txt")
    data_dir = os.path.join(os.getcwd(), "ember")

    np.random.seed(2018)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_trees': 400,
        'num_leaves': 64,
        'learning_rate': 0.05,
        'num_threads': 24,
        'min_data': 2000,
    }

    X_train = np.load(os.path.join(data_dir, "X_train.npy"), mmap_mode='r')
    y_train = np.load(os.path.join(data_dir, "y_train.npy"), mmap_mode='r')
    logging.info('Number of training samples: {}'.format(y_train.shape[0]))

    logging.info('Start training the model!')
    start_time = time.time()
    dataset = lgb.Dataset(X_train, y_train)
    model = lgb.train(params, dataset)
    training_time = time.time() - start_time
    logging.info('Training time: {}'.format(datetime.timedelta(seconds=training_time)))
    model.save_model(model_path)
    logging.info('Model saved at: {}'.format(model_path))


def evaluate_model():
    model_path = os.path.join(os.getcwd(), "model.txt")
    roc_curve_path = os.path.join(os.getcwd(), 'roc_curve.png')
    score_dist_path = os.path.join(os.getcwd(), 'score_dist.png')
    dataset_path = os.path.join(os.getcwd(), 'dataset.png')
    data_dir = os.path.join(os.getcwd(), "ember")
    np.random.seed(2018)

    X_test = np.load(os.path.join(data_dir, "X_test.npy"), mmap_mode='r')
    y_test = np.load(os.path.join(data_dir, "y_test.npy"), mmap_mode='r')
    logging.info('Number of testing samples: {}'.format(y_test.shape[0]))

    model = lgb.Booster(model_file=model_path)

    y_pred = model.predict(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

    roc = metrics.auc(fpr, tpr)
    idx_1 = (np.abs(0.001 - fpr[np.where((0.001 - fpr) >= 0)])).argmin()
    idx_2 = (np.abs(0.01 - fpr[np.where((0.01 - fpr) >= 0)])).argmin()
    acc_1 = metrics.accuracy_score(y_test, np.where(y_pred >= threshold[idx_1], 1, 0))
    acc_2 = metrics.accuracy_score(y_test, np.where(y_pred >= threshold[idx_2], 1, 0))

    TN, FP, FN, TP = metrics.confusion_matrix(y_test, np.where(y_pred >= 0.5, 1, 0)).ravel()
    fpr_0 = FP / (FP + TN)
    tpr_0 = TP / (TP + FN)
    acc_0 = (TP + TN) / (TP + FP + FN + TN)

    logging.info("Area Under ROC Curve     : {:.6f}".format(roc))
    logging.info("=====   Threshold at 0.5    =====")
    logging.info("False Alarm Rate         : {:2.4f} %".format(fpr_0 * 100))
    logging.info("Detection Rate           : {:2.4f} %".format(tpr_0 * 100))
    logging.info("Overall Accuracy         : {:2.4f} %".format(acc_0 * 100))
    logging.info("=====  FPR less than 0.1%   =====")
    logging.info("Detection Rate           : {:2.4f} %".format(tpr[idx_1] * 100))
    logging.info("Overall Accuracy         : {:2.4f} %".format(acc_1 * 100))
    logging.info("Threshold                : {:.6f}".format(threshold[idx_1]))
    logging.info("=====  FPR less than 1.0%   =====")
    logging.info("Detection Rate           : {:2.4f} %".format(tpr[idx_2] * 100))
    logging.info("Overall Accuracy         : {:2.4f} %".format(acc_2 * 100))
    logging.info("Threshold                : {:.6f}".format(threshold[idx_2]))

    # Save figures
    plt.title('ROC Curve')
    plt.ylim([0.8, 1])
    plt.xscale('log')
    plt.plot(fpr, tpr, 'b', label='AUC = {:0.6f}'.format(roc))
    tpr_1 = tpr[idx_1]
    plt.plot([10 ** -3, 10 ** -3], [0, tpr_1], 'r')
    plt.plot([0, 10 ** -3], [tpr_1, tpr_1], 'r')
    tpr_2 = tpr[idx_2]
    plt.plot([10 ** -2, 10 ** -2], [0, tpr_2], 'r')
    plt.plot([0, 10 ** -2], [tpr_2, tpr_2], 'r')
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(roc_curve_path)

    plt.clf()
    plt.hist([y_pred[y_test == 0], y_pred[y_test == 1]], bins=20,
             color=['blue', 'orange'],
             histtype='stepfilled',
             label=['benign', 'malicious'])
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend(loc='upper center')
    plt.title('Scores for testing samples')
    plt.savefig(score_dist_path)

    plt.clf()
    maliciousNum = (100000, 300000)
    benignNum = (100000, 300000)
    ind = np.arange(2)  # the x locations for the groups
    width = 0.5  # the width of the bars: can also be len(x) sequence
    plt.figure(figsize=(6, 8))
    p1 = plt.bar(ind, maliciousNum, width)
    p2 = plt.bar(ind, benignNum, width, bottom=maliciousNum)
    plt.xlabel('Subset')
    plt.xticks(ind, ('test', 'train'))
    plt.yticks(np.arange(1, 7) * 100000, ['{}K'.format(i) for i in np.arange(1, 7) * 100])
    plt.legend((p1[0], p2[0]), ('malicious', 'benign'), prop={'size': 16}, loc='upper left')
    plt.savefig(dataset_path)
