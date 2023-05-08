import pandas as pd
import numpy as np
import os
from cagsgp.util.PicklePersist import PicklePersist
import random
pd.options.display.max_columns = 999


def target(x: np.ndarray):
    return 2.0 - (2.1 * np.cos(9.8 * x[0]) * np.sin(1.3 * x[4]))


if __name__ == "__main__":
    codebase_folder = os.environ["CURRENT_CODEBASE_FOLDER"]
    folder = codebase_folder + 'python_data/CA-GSGP/benchmark/'
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    X_train = np.random.uniform(-50, 50 + 1e-4, (10000, 5))
    X_dev = np.random.uniform(-50, 50 + 1e-4, (10000, 5))
    X_test = np.random.uniform(-50, 50 + 1e-4, (10000, 5))
    print(X_train[0])
    print(X_train[1])
    y_train = np.array([target(X_train[i]) for i in range(X_train.shape[0])])
    y_dev = np.array([target(X_dev[i]) for i in range(X_dev.shape[0])])
    y_test = np.array([target(X_test[i]) for i in range(X_test.shape[0])])
    print(y_train[0])
    print(y_train[1])
    PicklePersist.compress_pickle(folder+"korns-"+str(seed), {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)})
    random.seed(None)
    np.random.seed(None)
