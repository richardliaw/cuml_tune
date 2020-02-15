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

from cuml.dask.common import utils as dask_utils
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf

from cuml.dask.ensemble import RandomForestClassifier as cumlDaskRF
from sklearn.ensemble import RandomForestClassifier as sklRF

# This will use all GPUs on the local host by default
cluster = LocalCUDACluster(threads_per_worker=1)
c = Client(cluster)

# Query the client for all connected workers
workers = c.has_what().keys()
n_workers = len(workers)
n_streams = 8 # Performance optimization


X_cudf_train = cudf.DataFrame.from_pandas(X_train)
# X_cudf_test = cudf.DataFrame.from_pandas(X_test)
y_cudf_train = cudf.Series(y_train.values)

n_partitions = n_workers

X_train_dask = dask_cudf.from_cudf(X_cudf_train, npartitions=n_partitions)
y_train_dask = dask_cudf.from_cudf(y_cudf_train, npartitions=n_partitions)


cuml_model = cumlDaskRF(
    n_estimators=10,
    max_depth=6,
    max_features=1.0,
    n_streams=4)


cuml_model.fit(X_train_dask, y_train_dask)
wait(cuml_model.rfs)
