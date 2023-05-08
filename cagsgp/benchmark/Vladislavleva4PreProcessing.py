import os
import pandas as pd
import numpy as np
from cagsgp.util.PicklePersist import PicklePersist
import random
pd.options.display.max_columns = 999


def target(x: np.ndarray):
    s = 0.0
    for i in range(5):
        s += (x[i] - 3.0) ** 2
    return 10.0/float(5.0 + s)


if __name__ == "__main__":
    codebase_folder = os.environ["CURRENT_CODEBASE_FOLDER"]
    folder = codebase_folder + 'python_data/CA-GSGP/benchmark/'
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    X_train = np.random.uniform(0.05, 6.05 + 1e-4, (1024, 5))
    X_dev = np.random.uniform(-0.25, 6.35 + 1e-4, (50, 5))
    X_test = np.random.uniform(-0.25, 6.35 + 1e-4, (30, 5))
    print(X_train[0])
    print(X_train[1])
    y_train = np.array([target(X_train[i]) for i in range(X_train.shape[0])])
    y_dev = np.array([target(X_dev[i]) for i in range(X_dev.shape[0])])
    y_test = np.array([target(X_test[i]) for i in range(X_test.shape[0])])
    print(y_train[0])
    print(y_train[1])
    PicklePersist.compress_pickle(folder + "vladislavleva-"+str(seed), {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)})
    random.seed(None)
    np.random.seed(None)
