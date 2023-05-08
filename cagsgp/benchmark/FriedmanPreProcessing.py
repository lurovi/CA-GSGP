import random
import numpy as np
import os
from sklearn.datasets import make_friedman1

from cagsgp.util.PicklePersist import PicklePersist


if __name__ == "__main__":
    codebase_folder = os.environ["CURRENT_CODEBASE_FOLDER"]
    folder = codebase_folder + 'python_data/CA-GSGP/benchmark/'
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    X_train, y_train = make_friedman1(n_samples=400, n_features=5, noise=0.25, random_state=seed)
    random.seed(seed + 1)
    np.random.seed(seed + 1)
    X_dev, y_dev = make_friedman1(n_samples=200, n_features=5, noise=0.25, random_state=seed + 1)
    random.seed(seed + 2)
    np.random.seed(seed + 2)
    X_test, y_test = make_friedman1(n_samples=100, n_features=5, noise=0.25, random_state=seed + 2)

    PicklePersist.compress_pickle(folder+"friedman1-"+str(seed), {"training": (X_train, y_train),
                                                 "validation": (X_dev, y_dev),
                                                 "test": (X_test, y_test)})
    random.seed(None)
    np.random.seed(None)
