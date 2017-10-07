"""Functions for working with model data from keras
"""
from __future__ import absolute_import
from __future__ import print_function
from sklearn import model_selection
import numpy as np


def test_len(train):
    """Test if all the elements in `train=(x,y)` the same `shape[0]`
    """
    l = train[1].shape[0]
    if isinstance(train[0], dict):
        for x in train[0].keys():
            assert train[0][x].shape[0] == l
    elif isinstance(train[0], (list, tuple)):
        for x in train[0]:
            assert x.shape[0] == l
    elif isinstance(train[0], np.ndarray):
        assert train[0].shape[0] == l
    else:
        raise ValueError("train[0] can only be of type: list, tuple, dict or np.ndarray")


def split_train_test_idx(train, valid_split=.2, stratified=False, random_state=None):
    """Return indicies for train-test split
    """
    test_len(train)
    y = train[1]
    x = np.arange(y.shape[0])
    stratify = y if stratified else None
    return model_selection.train_test_split(x, test_size=valid_split,
                                            random_state=random_state, stratify=stratify)


def split_KFold_idx(train, cv_n_folds=5, stratified=False, random_state=None):
    """Get k-fold indices generator
    """
    test_len(train)
    y = train[1]
    n_rows = y.shape[0]
    if stratified:
        if len(y.shape) > 1:
            if y.shape[1] > 1:
                raise ValueError("Can't use stratified K-fold with multi-column response variable")
            else:
                y = y[:, 0]
            # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.split
        return model_selection.StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=random_state)\
            .split(X=np.zeros((n_rows, 1)), y=y)
    else:
        return model_selection.KFold(n_splits=cv_n_folds, shuffle=True, random_state=random_state)\
            .split(X=np.zeros((n_rows, 1)))


def subset(train, idx, keep_other=True):
    """Subset the `train=(x, y)` data tuple, each of the form:

    - list, np.ndarray
    - tuple, np.ndarray
    - dictionary, np.ndarray
    - np.ndarray, np.ndarray

    # Note
        In case there are other data present in the tuple:
        `(x, y, other1, other2, ...)`, these get passed on as:
        `(x_sub, y_sub, other1, other2)`

    # Arguments
        train: `(x,y, other1, other2, ...)` tuple of data
        idx: indices to subset the data with
        keep_other: bool; If True, the additional tuple elements `(other1, other2, ...)` are passed
    together with `(x, y)` but don't get subsetted.
    """
    test_len(train)
    y = train[1][idx]
    # x split
    if isinstance(train[0], (list, tuple)):
        x = [x[idx] for x in train[0]]
    elif isinstance(train[0], dict):
        x = {k: v[idx] for k, v in train[0].items()}
    elif isinstance(train[0], np.ndarray):
        x = train[0][idx]
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

    if keep_other:
        return (x, y) + train[2:]
    else:
        return (x, y)
