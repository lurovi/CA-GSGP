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
    def convert_text_dataset_to_csv(input_path: str, output_path: str, idx: int, scale: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        file = open(input_path+'train'+str(idx), 'r')
        lines_train: list[str] = file.readlines()
        file.close()

        file = open(input_path+'test'+str(idx), 'r')
        lines_test: list[str] = file.readlines()
        file.close()

        n_features: int = int(lines_train[0].split()[0])
        
        lines_train = lines_train[2:]
        lines_test = lines_test[2:]

        d_train: dict[str, list[float]] = {str(k): [] for k in range(n_features)}
        d_train['target'] = []

        d_test: dict[str, list[float]] = {str(k): [] for k in range(n_features)}
        d_test['target'] = []

        for line in lines_train:
            l = line.split()
            for i in range(len(l)):
                if i == len(l) - 1:
                    d_train['target'].append(float(l[i]))
                else:
                    d_train[str(i)].append(float(l[i]))
        d_train: pd.DataFrame = pd.DataFrame(d_train)

        for line in lines_test:
            l = line.split()
            for i in range(len(l)):
                if i == len(l) - 1:
                    d_test['target'].append(float(l[i]))
                else:
                    d_test[str(i)].append(float(l[i]))
        d_test: pd.DataFrame = pd.DataFrame(d_test)

        if scale:
            y: np.ndarray = d_train['target'].to_numpy()
            d_train.drop('target', axis=1, inplace=True)
            X: np.ndarray = d_train.to_numpy()

            scaler: StandardScaler = StandardScaler()
            scaler = scaler.fit(X)
            X = scaler.transform(X)
            d_train = pd.DataFrame(X)
            d_train.rename({i: str(i) for i in range(n_features)}, inplace=True)
            d_train["target"] = y

            y = d_test['target'].to_numpy()
            d_test.drop('target', axis=1, inplace=True)
            X = d_test.to_numpy()
            X = scaler.transform(X)
            d_test = pd.DataFrame(X)
            d_test.rename({i: str(i) for i in range(n_features)}, inplace=True)
            d_test["target"] = y

        d_train.to_csv(output_path+'train'+str(idx)+'.csv', index=False)
        d_test.to_csv(output_path+'test'+str(idx)+'.csv', index=False)
        return tuple[d_train, d_test]

    @staticmethod
    def read_csv_data(path: str, idx: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        d: pd.DataFrame = pd.read_csv(path+'train'+str(idx)+'.csv')
        y: np.ndarray = d['target'].to_numpy()
        d.drop('target', axis=1, inplace=True)
        X: np.ndarray = d.to_numpy()
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {'train': (X, y)}
        d = pd.read_csv(path+'test'+str(idx)+'.csv')
        y = d['target'].to_numpy()
        d.drop('target', axis=1, inplace=True)
        X = d.to_numpy()
        result['test'] = (X, y)
        return result

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
