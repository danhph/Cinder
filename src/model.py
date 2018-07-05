import os
from datetime import datetime
from datetime import timedelta

import lightgbm as lgb
import matplotlib
import numpy as np
from sklearn import metrics

from features import FeatureExtractor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    force_train = False
    seed = 2018

    np.random.seed(seed)

    data_dir = os.path.join(os.getenv('HOME'), 'ember')
    input_dim = FeatureExtractor.dim
    print('Input dimension: %s' % input_dim)

    X_train = np.load(os.path.join(data_dir, "X_train.npy"), mmap_mode='r')
    y_train = np.load(os.path.join(data_dir, "y_train.npy"), mmap_mode='r')
    print('Number of training samples: %s' % y_train.shape[0])

    X_test = np.load(os.path.join(data_dir, "X_test.npy"), mmap_mode='r')
    y_test = np.load(os.path.join(data_dir, "y_test.npy"), mmap_mode='r')
    print('Number of test samples: %s' % y_test.shape[0])

    model_path = os.path.join(data_dir, "model.txt")
    if force_train or not os.path.exists(model_path):
        start = datetime.utcnow()
        print('\nStarted at   : {}'.format(start + timedelta(hours=7)))

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_trees': 400,
            'num_leaves': 64,
            'learning_rate': 0.05,
            'num_threads': 24,
            'min_data': 2000,
        }
        train_dataset = lgb.Dataset(X_train, y_train)
        model_lgbm = lgb.train(params, train_dataset)
        model_lgbm.save_model(model_path)

        print('Training time: {}\n'.format(datetime.utcnow() - start))
    else:
        model_lgbm = lgb.Booster(model_file=model_path)

    y_pred = model_lgbm.predict(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

    idx_1 = (np.abs(0.001 - fpr[np.where((0.001 - fpr) >= 0)])).argmin()
    idx_2 = (np.abs(0.01 - fpr[np.where((0.01 - fpr) >= 0)])).argmin()
    print("=" * 64)
    print("Fallout   : {:2.4f} %".format(fpr[idx_1] * 100))
    print("Recall    : {:2.4f} %".format(tpr[idx_1] * 100))
    print("Threshold : {:2.4f} %".format(threshold[idx_1] * 100))
    print("=" * 64)
    print("Fallout   : {:2.4f} %".format(fpr[idx_2] * 100))
    print("Recall    : {:2.4f} %".format(tpr[idx_2] * 100))
    print("Threshold : {:2.4f} %".format(threshold[idx_2] * 100))
    print("=" * 64)
    roc = metrics.auc(fpr, tpr)
    plt.title('ROC Curve')
    plt.ylim([0.8, 1])
    plt.xscale('log')
    plt.plot(fpr, tpr, 'b', label='AUC = {:0.6f}'.format(roc))
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(os.path.join(data_dir, 'roc_curve.png'))

    tpr_1 = tpr[idx_1]
    plt.plot([10 ** -3, 10 ** -3], [0, tpr_1], 'r')
    plt.plot([0, 10 ** -3], [tpr_1, tpr_1], 'r')
    tpr_2 = tpr[idx_2]
    plt.plot([10 ** -2, 10 ** -2], [0, tpr_2], 'r')
    plt.plot([0, 10 ** -2], [tpr_2, tpr_2], 'r')
    plt.savefig(os.path.join(data_dir, 'roc_curve_with_highlights.png'))

    plt.clf()

    plt.hist([y_pred[y_test == 0], y_pred[y_test == 1]], bins=20,
             color=['b', 'r'], alpha=0.6, histtype='stepfilled',
             label=['benign', 'malicious'])
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.legend(loc='upper center')
    plt.title('Scores for testing samples')
    plt.savefig(os.path.join(data_dir, 'score_dist.png'))
    # TN, FP, FN, TP = metrics.confusion_matrix(y_test, np.round(y_pred)).ravel()
    # print(TN, FP, FN, TP)
    #
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # # Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    #
    # # Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    #
    # print('Accuracy :  {:2.4f} %'.format(ACC * 100))
    # print('Recall   :  {:2.4f} %'.format(TPR * 100))
    # print('Fallout  :  {:2.4f} %'.format(FPR * 100))
