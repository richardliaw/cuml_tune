#!/usr/bin/env python
# coding: utf-8


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

# import ray
# import logging
# logger = logging.getLogger("ray.tune")
# logger.setLevel(logging.ERROR)

import cudf
import numpy as np
import pandas as pd
import pickle
from datasets import prepare_dataset

from cuml.ensemble import RandomForestClassifier as GPURandomForestClassifier

data = prepare_dataset("/data", "airline", None)
X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

QUARTER = len(X_train) // 2
X_train = X_train[QUARTER:]
y_train = y_train[QUARTER:]

X_cudf_train = cudf.DataFrame.from_pandas(X_train)
X_cudf_test = cudf.DataFrame.from_pandas(X_test)
train_mat = X_cudf_train.as_gpu_matrix(order="F")
del X_cudf_train

y_cudf_train = cudf.Series(y_train.values)


cuml_model = GPURandomForestClassifier(
    n_estimators=467,
    max_depth=19,
    max_features=1.0
)

cuml_model.fit(train_mat, y_cudf_train)

fil_preds_orig = cuml_model.predict(X_cudf_test)
fil_acc_orig = accuracy_score(y_test, fil_preds_orig)
