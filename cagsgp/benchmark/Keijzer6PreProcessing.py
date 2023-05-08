import pandas as pd
import numpy as np
import os
from cagsgp.util.PicklePersist import PicklePersist
import random
pd.options.display.max_columns = 999


def target(x: np.ndarray):
    val = x[0]
    s = 0.0
    for i in range(1, val+1):
        s += 1.0/float(i)
    return s


if __name__ == "__main__":
    codebase_folder = os.environ["CURRENT_CODEBASE_FOLDER"]
    folder = codebase_folder + 'python_data/CA-GSGP/benchmark/'
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    X_train = np.array([i for i in range(1, 50 + 1)]).reshape(-1, 1)
    X_dev = np.array([i for i in range(1, 120 + 1)]).reshape(-1, 1)
    X_test = np.array([i for i in range(1, 240 + 1)]).reshape(-1, 1)
    print(X_train[0])
    print(X_train[1])
    y_train = np.array([target(X_train[i]) for i in range(X_train.shape[0])])
    y_dev = np.array([target(X_dev[i]) for i in range(X_dev.shape[0])])
    y_test = np.array([target(X_test[i]) for i in range(X_test.shape[0])])
    print(y_train[0])
    print(y_train[1])
    PicklePersist.compress_pickle(folder+"keijzer-"+str(seed), {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)})
    random.seed(None)
    np.random.seed(None)
