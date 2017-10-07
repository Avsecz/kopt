"""Global evaluation metrics. Simular to metrics.py but not restricted
to keras backend (TensorFlow, Theano) implementation. Hence they allow for
metrics that are not batch-limited as auc, f1 etc...

See also https://github.com/fchollet/keras/issues/5794
"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from kopt.utils import get_from_module
import sklearn.metrics as skm
from scipy.stats import kendalltau

MASK_VALUE = -1

# Binary classification

def _mask_nan(y_true, y_pred):
    mask_array = ~np.isnan(y_true)
    if np.any(np.isnan(y_pred)):
        print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
              format(np.sum(np.isnan(y_pred)), y_pred.size))
        mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
    return y_true[mask_array], y_pred[mask_array]


def _mask_value(y_true, y_pred, mask=MASK_VALUE):
    mask_array = y_true != mask
    return y_true[mask_array], y_pred[mask_array]


def _mask_value_nan(y_true, y_pred, mask=MASK_VALUE):
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return _mask_value(y_true, y_pred, mask)


def auc(y_true, y_pred, round=True):
    """Area under the ROC curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)

    if round:
        y_true = y_true.round()
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return skm.roc_auc_score(y_true, y_pred)


def auprc(y_true, y_pred):
    """Area under the precision-recall curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    return skm.average_precision_score(y_true, y_pred)


def accuracy(y_true, y_pred, round=True):
    """Classification accuracy
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.accuracy_score(y_true, y_pred)


def tpr(y_true, y_pred, round=True):
    """True positive rate `tp / (tp + fn)`
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.recall_score(y_true, y_pred)


def tnr(y_true, y_pred, round=True):
    """True negative rate `tn / (tn + fp)`
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    c = skm.confusion_matrix(y_true, y_pred)
    return c[0, 0] / c[0].sum()


def mcc(y_true, y_pred, round=True):
    """Matthews correlation coefficient
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.matthews_corrcoef(y_true, y_pred)


def f1(y_true, y_pred, round=True):
    """F1 score: `2 * (p * r) / (p + r)`, where p=precision and r=recall.
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.f1_score(y_true, y_pred)


# Category classification

def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))


# Regression

def cor(y_true, y_pred):
    """Compute Pearson correlation coefficient.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]


def kendall(y_true, y_pred, nb_sample=100000):
    """Kendall's tau coefficient, Kendall rank correlation coefficient
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    if len(y_true) > nb_sample:
        idx = np.arange(len(y_true))
        np.random.shuffle(idx)
        idx = idx[:nb_sample]
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    return kendalltau(y_true, y_pred)[0]


def mad(y_true, y_pred):
    """Median absolute deviation
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root mean-squared error
    """
    return np.sqrt(mse(y_true, y_pred))


def rrmse(y_true, y_pred):
    """1 - rmse
    """
    return 1 - rmse(y_true, y_pred)


def mse(y_true, y_pred):
    """Mean squared error
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return ((y_true - y_pred) ** 2).mean(axis=None)


def ermse(y_true, y_pred):
    """Exponentiated root-mean-squared error
    """
    return 10**np.sqrt(mse(y_true, y_pred))


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    return 1 - var_resid / var_y_true


# available eval metrics --------------------------------------------


BINARY_CLASS = ["auc", "auprc", "accuracy", "tpr", "tnr", "f1", "mcc"]
CATEGORY_CLASS = ["cat_acc"]
REGRESSION = ["mse", "mad", "cor", "ermse", "var_explained"]

AVAILABLE = BINARY_CLASS + CATEGORY_CLASS + REGRESSION


def get(name):
    return get_from_module(name, globals())
