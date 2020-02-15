#!/usr/bin/env python
# coding: utf-8

import cudf
import numpy as np
import pandas as pd
import pickle

from cuml.ensemble import RandomForestClassifier as curfc

from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import ray
import logging
logger = logging.getLogger("ray.tune")
logger.setLevel(logging.ERROR)

n_samples = 2**14
n_features = 399
n_info = 300
data_type = np.float32

X,y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_info,
    random_state=123, n_classes=2)

X = pd.DataFrame(X.astype(data_type))

# cuML Random Forest Classifier requires the labels to be integers
y = pd.Series(y.astype(np.int32))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


from ray import tune
from ray.tune.utils import pin_in_object_store, get_pinned_object

data_id = pin_in_object_store([X_train, X_test, y_train, y_test])

class CUMLTrainable(tune.Trainable):
    def _setup(self, config):
        [X_train, X_test, y_train, y_test] = get_pinned_object(data_id)

        self.cuml_model = curfc(
            n_estimators=config.get("estimators", 40),
            max_depth=config.get("depth", 16),
            max_features=1.0
        )
        self.X_cudf_train = cudf.DataFrame.from_pandas(X_train)
        self.X_cudf_test = cudf.DataFrame.from_pandas(X_test)
        self.y_cudf_train = cudf.Series(y_train.values)
        self.y_test = y_test

    def _train(self):
        self.cuml_model.fit(
            self.X_cudf_train,
            self.y_cudf_train
        )
        fil_preds_orig = self.cuml_model.predict(
            self.X_cudf_test)
        return {"mean_accuracy": accuracy_score(self.y_test, fil_preds_orig)}

    def _stop(self):
        del self.X_cudf_train
        del self.X_cudf_test
        del self.y_cudf_train
        del self.y_test
        del self.cuml_model


analysis = tune.run(
    CUMLTrainable,
    resources_per_trial={"gpu": 0.3},
    num_samples=20,
    config={"depth": tune.choice(list(range(8, 24)))},
    stop={"training_iteration": 1}, verbose=1)
