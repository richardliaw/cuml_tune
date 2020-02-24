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

QUARTER = len(X_train) // 3
X_train = X_train[QUARTER:]
y_train = y_train[QUARTER:]

# ray.init()
# data_id = pin_in_object_store([X_train, X_test, y_train, y_test])

import os
from filelock import FileLock

class CUMLTrainable(tune.Trainable):
    def _setup(self, config):
        # [X_train, X_test, y_train, y_test] = get_pinned_object(data_id)
        self._gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", 0) #ray.get_gpu_ids()[0]
        print("Starting new trainable on {}.".format(self._gpu_id))
        # self._wait_for_gpus()

        with FileLock(os.path.expanduser("~/.tune.gpulock")):
            X_cudf_train = cudf.DataFrame.from_pandas(X_train)
            self.train_mat = X_cudf_train.as_gpu_matrix(order="F")
            del X_cudf_train
            self.X_cudf_test = cudf.DataFrame.from_pandas(X_test)
            self.y_cudf_train = cudf.Series(y_train.values)
            self.y_test = y_test
            config = {k: int(v) for k, v in config.items()}
            self.cuml_model = GPURandomForestClassifier(**config)

    def _train(self):
        self.cuml_model.fit(self.train_mat, self.y_cudf_train)
        fil_preds_orig = self.cuml_model.predict(self.X_cudf_test)
        accuracy = accuracy_score(self.y_test, fil_preds_orig)
        return {"mean_accuracy": accuracy}

    def _stop(self):
        import time
        import GPUtil
        gpu_object = GPUtil.getGPUs()[self._gpu_id]
        print("Deleting the model. Mem: {:0.3f}".format(gpu_object.memoryUsed))
        del self.cuml_model
        print("Deleting the test set. Mem: {:0.3f}".format(gpu_object.memoryUsed))
        del self.X_cudf_test
        print("Deleting the test labels. Mem: {:0.3f}".format(gpu_object.memoryUsed))
        del self.y_test
        print("Deleting the training labels. Mem: {:0.3f}".format(gpu_object.memoryUsed))
        del self.y_cudf_train
        print("Deleting the training matrix. Mem: {:0.3f}".format(gpu_object.memoryUsed))
        del self.train_mat
#         self._wait_for_gpus(retry=1)


    def _wait_for_gpus(self, retry=10):
        import GPUtil
        import time
        gpu_object = GPUtil.getGPUs()[self._gpu_id]
        for i in range(int(retry)):
            if gpu_object.memoryUsed > 0.1:
                print("Waiting for GPU memory to free. Mem: {:0.3f}".format(gpu_object.memoryUsed))
                time.sleep(5)
        time.sleep(5)

    def reset_config(self, config):
        del self.cuml_model
        config = {k: int(v) for k, v in config.items()}
        self.cuml_model = GPURandomForestClassifier(**config)
        return True


def tune_randomsearch():

    analysis = tune.run(
        CUMLTrainable,
        resources_per_trial={"gpu": 1},
        num_samples=10,
        reuse_actors=True,
        config={
            "n_estimators": tune.choice(list(range(80, 300))),
            "max_depth": tune.choice(list(range(8, 24))),
        },
        stop={"training_iteration": 1},
        verbose=1
    )

def tune_hyperopt():
    from hyperopt import hp
    from ray.tune.suggest.hyperopt import HyperOptSearch

    search_alg = HyperOptSearch(
        space={
            "n_estimators": hp.loguniform("n_estimators", 2, 6),
            "max_depth": hp.uniform("max_depth", 2, 20),
            "min_impurity_decrease":  hp.loguniform("min_impurity_decrease", -5, -1)
        },
        metric="mean_accuracy",
        mode="max",
        n_initial_points=5,
        points_to_evaluate=[{"n_estimators": 100, "max_depth": 16, "min_impurity_decrease": 0}]
    )

    analysis = tune.run(
        CUMLTrainable,
        resources_per_trial={"gpu": 1},
        num_samples=100,
    #     reuse_actors=True,
        stop={"training_iteration": 1},
        verbose=1,
        max_failures=0,
        search_alg=search_alg
    )

if __name__ == '__main__':
    for i in range(10):
        config = {
            "n_estimators": random.choice(list(range(80, 300))),
            "max_depth": random.choice(list(range(8, 24))),
        }
        trainable = CUMLTrainable(config=config)
        result = trainable.train()
        trainable.stop()
        print("Accuracy is", result.get("mean_accuracy"))
