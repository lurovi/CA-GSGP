from typing import Any
import numpy as np


class StatsCollector:
    def __init__(self,
                 objective_names: list[str],
                 revert_sign: list[bool] = None
                 ) -> None:
        self.__data: dict[int, dict[str, np.ndarray]] = {}
        self.__objective_names: list[str] = objective_names
        self.__revert_sign: list[bool] = revert_sign
        if self.__revert_sign is None:
            self.__revert_sign = [False] * len(self.__objective_names)
        if len(self.__objective_names) != len(self.__revert_sign):
            raise AttributeError(f"The length of objective names ({len(self.__objective_names)}) is not equal to the length of revert sign ({len(self.__revert_sign)}).")
        self.__sign_array: np.ndarray = np.array([-1.0 if b else 1.0 for b in self.__revert_sign])

    def update_fitness_stat_dict(self, n_gen: int, data: np.ndarray) -> None:
        da: np.ndarray = data * np.tile(self.__sign_array, (data.shape[0], 1))
        d: dict[str, np.ndarray] = {"mean": np.mean(da, axis=0),
                                    "median": np.median(da, axis=0),
                                    "min": np.min(da, axis=0),
                                    "max": np.max(da, axis=0),
                                    "sum": np.sum(da, axis=0),
                                    "std": np.std(da, axis=0)}
        self.__data[n_gen] = d

    def get_fitness_stat(self, n_gen: int, stat: str) -> np.ndarray:
        return self.__data[n_gen][stat]

    def build_dict(self) -> dict[str, list[Any]]:
        d: dict[str, list[Any]] = {"Generation": [],
                                   "Objective": [],
                                   "Statistic": [],
                                   "Value": []}
        for n_gen in self.__data:
            dd: dict[str, np.ndarray] = self.__data[n_gen]
            for stat in dd:
                val: np.ndarray = dd[stat]
                for i in range(len(val)):
                    objective_name: str = self.__objective_names[i]
                    value: float = val[i]
                    d["Generation"].append(n_gen)
                    d["Objective"].append(objective_name)
                    d["Statistic"].append(stat)
                    d["Value"].append(value)
        return d
