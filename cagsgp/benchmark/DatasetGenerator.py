import random
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler
import pandas as pd
from cagsgp.util.PicklePersist import PicklePersist


class DatasetGenerator:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def convert_text_dataset_to_csv(input_path: str, output_path: str, scale: bool = False) -> pd.DataFrame:
        file = open(input_path, 'r')
        lines: list[str] = file.readlines()
        file.close()

        n_features: int = int(lines[0].split()[0])
        n_samples: int = int(lines[1].split()[0])
        
        lines = lines[2:]
        d: dict[str, list[float]] = {str(k): [] for k in range(n_features)}
        d['target'] = []

        for line in lines:
            l = line.split()
            for i in range(len(l)):
                if i == len(l) - 1:
                    d['target'].append(float(l[i]))
                else:
                    d[str(i)].append(float(l[i]))
        d: pd.DataFrame = pd.DataFrame(d)

        if scale:
            y: np.ndarray = d['target'].to_numpy()
            d.drop('target', axis=1, inplace=True)
            X: np.ndarray = d.to_numpy()

            scaler: StandardScaler = StandardScaler()
            scaler = scaler.fit(X)
            X = scaler.transform(X)
            d = pd.DataFrame(X)
            d.rename({i: str(i) for i in range(n_features)}, inplace=True)
            d["target"] = y
            

        d.to_csv(output_path+'.csv', index=False)
        return d

    @staticmethod
    def read_csv_data(path: str) -> tuple[np.ndarray, np.ndarray]:
        d: pd.DataFrame = pd.read_csv(path)
        y: np.ndarray = d['target'].to_numpy()
        d.drop('target', axis=1, inplace=True)
        X: np.ndarray = d.to_numpy()
        return (X, y)

    @staticmethod
    def generate_dataset(dataset_name: str, seed: int, reset: bool = False, path: str = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if dataset_name == 'friedman1':
            return DatasetGenerator.friedman1(seed=seed, reset=reset, path=path)
        elif dataset_name == 'keijzer6':
            return DatasetGenerator.keijzer6(seed=seed, reset=reset, path=path)
        elif dataset_name == 'korns12':
            return DatasetGenerator.korns12(seed=seed, reset=reset, path=path)
        elif dataset_name == 'vladislavleva4':
            return DatasetGenerator.vladislavleva4(seed=seed, reset=reset, path=path)
        else:
            raise ValueError(f'{dataset_name} is not a valid dataset name.')

    @staticmethod
    def __set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def __set_seed_to_none() -> None:
        random.seed(None)
        np.random.seed(None)

    @staticmethod
    def friedman1(seed: int, reset: bool = False, path: str = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        DatasetGenerator.__set_seed(seed)

        X, y = make_friedman1(n_samples=700, n_features=5, noise=0.25, random_state=seed)

        X_train, y_train = X[:400], y[:400]
        X_dev, y_dev = X[400:600], y[400:600]
        X_test, y_test = X[600:], y[600:]
        
        if reset:
            DatasetGenerator.__set_seed_to_none()
        
        res: dict[str, tuple[np.ndarray, np.ndarray]] = {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)}
        if path is not None:
            PicklePersist.compress_pickle(path+"friedman1-"+str(seed), res)
        
        return res
    
    @staticmethod
    def keijzer6(seed: int, reset: bool = False, path: str = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        DatasetGenerator.__set_seed(seed)

        X_train = np.array([i for i in range(1, 50 + 1)]).reshape(-1, 1)
        X_dev = np.array([i for i in range(1, 120 + 1)]).reshape(-1, 1)
        X_test = np.array([i for i in range(1, 240 + 1)]).reshape(-1, 1)

        y_train = np.array([DatasetGenerator.__keijer6_target_method(X_train[i]) for i in range(X_train.shape[0])])
        y_dev = np.array([DatasetGenerator.__keijer6_target_method(X_dev[i]) for i in range(X_dev.shape[0])])
        y_test = np.array([DatasetGenerator.__keijer6_target_method(X_test[i]) for i in range(X_test.shape[0])])
        
        if reset:
            DatasetGenerator.__set_seed_to_none()
        
        res: dict[str, tuple[np.ndarray, np.ndarray]] = {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)}
        if path is not None:
            PicklePersist.compress_pickle(path+"keijzer6-"+str(seed), res)
        
        return res

    @staticmethod
    def __keijer6_target_method(x: np.ndarray):
        val = x[0]
        s = 0.0
        for i in range(1, val+1):
            s += 1.0/float(i)
        return s

    @staticmethod
    def korns12(seed: int, reset: bool = False, path: str = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        DatasetGenerator.__set_seed(seed)

        X_train = np.random.uniform(-50, 50 + 1e-4, (10000, 5))
        X_dev = np.random.uniform(-50, 50 + 1e-4, (10000, 5))
        X_test = np.random.uniform(-50, 50 + 1e-4, (10000, 5))

        y_train = np.array([DatasetGenerator.__korns12_target_method(X_train[i]) for i in range(X_train.shape[0])])
        y_dev = np.array([DatasetGenerator.__korns12_target_method(X_dev[i]) for i in range(X_dev.shape[0])])
        y_test = np.array([DatasetGenerator.__korns12_target_method(X_test[i]) for i in range(X_test.shape[0])])
        
        if reset:
            DatasetGenerator.__set_seed_to_none()
        
        res: dict[str, tuple[np.ndarray, np.ndarray]] = {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)}
        if path is not None:
            PicklePersist.compress_pickle(path+"korns12-"+str(seed), res)
        
        return res
    
    @staticmethod
    def __korns12_target_method(x: np.ndarray):
        return 2.0 - (2.1 * np.cos(9.8 * x[0]) * np.sin(1.3 * x[4]))

    @staticmethod
    def vladislavleva4(seed: int, reset: bool = False, path: str = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        DatasetGenerator.__set_seed(seed)

        X_train = np.random.uniform(0.05, 6.05 + 1e-4, (1024, 5))
        X_dev = np.random.uniform(-0.25, 6.35 + 1e-4, (50, 5))
        X_test = np.random.uniform(-0.25, 6.35 + 1e-4, (30, 5))

        y_train = np.array([DatasetGenerator.__vladislavleva4_target_method(X_train[i]) for i in range(X_train.shape[0])])
        y_dev = np.array([DatasetGenerator.__vladislavleva4_target_method(X_dev[i]) for i in range(X_dev.shape[0])])
        y_test = np.array([DatasetGenerator.__vladislavleva4_target_method(X_test[i]) for i in range(X_test.shape[0])])
        
        if reset:
            DatasetGenerator.__set_seed_to_none()
        
        res: dict[str, tuple[np.ndarray, np.ndarray]] = {"training": (X_train, y_train),
                                                         "validation": (X_dev, y_dev),
                                                         "test": (X_test, y_test)}
        if path is not None:
            PicklePersist.compress_pickle(path+"vladislavleva4-"+str(seed), res)
        
        return res
    
    @staticmethod
    def __vladislavleva4_target_method(x: np.ndarray):
        s = 0.0
        for i in range(5):
            s += (x[i] - 3.0) ** 2
        return 10.0/float(5.0 + s)
