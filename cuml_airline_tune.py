import cudf
import numpy as np
import pandas as pd
import pickle
from datasets import prepare_dataset
from sklearn.metrics import accuracy_score


from cuml.ensemble import RandomForestClassifier as GPURandomForestClassifier

import ray
from ray import tune
from ray.tune.utils import pin_in_object_store, get_pinned_object

data = prepare_dataset("/data", "airline", None)
X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

QUARTER = len(X_train) // 2
X_train = X_train[QUARTER:]
y_train = y_train[QUARTER:]

ray.init()
data_id = pin_in_object_store([X_train, X_test, y_train, y_test])

class CUMLTrainable(tune.Trainable):
    def _setup(self, config):
        [X_train, X_test, y_train, y_test] = get_pinned_object(data_id)

        X_cudf_train = cudf.DataFrame.from_pandas(X_train)
        self.train_mat = X_cudf_train.as_gpu_matrix(order="F")
        del X_cudf_train
        self.X_cudf_test = cudf.DataFrame.from_pandas(X_test)
        self.y_cudf_train = cudf.Series(y_train.values)
        self.y_test = y_test
        self.cuml_model = GPURandomForestClassifier(**self.config)

    def _train(self):
        self.cuml_model.fit(self.train_mat, self.y_cudf_train)
        fil_preds_orig = self.cuml_model.predict(self.X_cudf_test)
        accuracy = accuracy_score(self.y_test, fil_preds_orig)
        return {"mean_accuracy": accuracy}

    def reset_config(self, config):
        del self.cuml_model
        self.cuml_model = GPURandomForestClassifier(config)


analysis = tune.run(
    CUMLTrainable,
    resources_per_trial={"gpu": 1},
    num_samples=10,
    reuse_actors=True,
    config={
        "n_estimators": tune.choice(list(range(80, 500))),
        "max_depth": tune.choice(list(range(8, 24))),
    },
    stop={"training_iteration": 1},
    verbose=1
)
