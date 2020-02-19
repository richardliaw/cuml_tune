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
    n_estimators=40,
    max_depth=6,
    max_features=1.0,
    seed=10)

cuml_model.fit(X_cudf_train, y_cudf_train)

fil_preds_orig = cuml_model.predict(X_cudf_test)
fil_acc_orig = accuracy_score(y_test, fil_preds_orig)

# data = prepare_dataset(dataset_folder, dataset, args.nrows)


# from ray import tune
# from ray.tune.utils import pin_in_object_store, get_pinned_object

# data_id = pin_in_object_store([X_train, X_test, y_train, y_test])

# class CUMLTrainable(tune.Trainable):
#     def _setup(self, config):
#         [X_train, X_test, y_train, y_test] = get_pinned_object(data_id)

#         self.cuml_model = curfc(
#             n_estimators=config.get("estimators", 40),
#             max_depth=config.get("depth", 16),
#             max_features=1.0
#         )
#         self.X_cudf_train = cudf.DataFrame.from_pandas(X_train)
#         self.X_cudf_test = cudf.DataFrame.from_pandas(X_test)
#         self.y_cudf_train = cudf.Series(y_train.values)
#         self.y_test = y_test

#     def _train(self):
#         self.cuml_model.fit(
#             self.X_cudf_train,
#             self.y_cudf_train
#         )
#         fil_preds_orig = self.cuml_model.predict(
#             self.X_cudf_test)
#         return {"mean_accuracy": accuracy_score(self.y_test, fil_preds_orig)}

#     def _stop(self):
#         del self.X_cudf_train
#         del self.X_cudf_test
#         del self.y_cudf_train
#         del self.y_test
#         del self.cuml_model


# analysis = tune.run(
#     CUMLTrainable,
#     resources_per_trial={"gpu": 0.3},
#     num_samples=20,
#     config={"depth": tune.choice(list(range(8, 24)))},
#     stop={"training_iteration": 1}, verbose=1)
