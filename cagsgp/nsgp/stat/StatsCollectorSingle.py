from typing import Any
import numpy as np
import statistics


class StatsCollectorSingle:
    def __init__(self,
                 objective_name: str,
                 revert_sign: bool = False
                 ) -> None:
        self.__data: dict[int, dict[str, float]] = {}
        self.__objective_name: str = objective_name
        self.__revert_sign: bool = revert_sign
        self.__sign_value: float = -1.0 if self.__revert_sign else 1.0

    def update_fitness_stat_dict(self, n_gen: int, data: list[float]) -> None:
        da: list[float] = [-val for val in data] if self.__revert_sign else data
        da: np.ndarray = np.array(da, dtype=np.float32)
        
        #sum_da: float = float(np.sum(da))
        mean_da: float = float(np.mean(da))
        #max_da: float = float(np.amax(da))
        min_da: float = float(np.amin(da))
        median_da: float = float(np.median(da))
        var_da: float = float(np.var(da))
        
        d: dict[str, float] = {
                               "mean": mean_da,
                               "median": median_da,
                               "min": min_da,
                               #"max": max_da,
                               #"sum": sum_da,
                               "var": var_da
                               #"count": da.shape[0]
                              }
        self.__data[n_gen] = d

    def get_fitness_stat(self, n_gen: int, stat: str) -> float:
        return self.__data[n_gen][stat]

    def build_dict(self) -> dict[str, list[Any]]:
        d: dict[str, list[Any]] = {"Generation": [],
                                   "Objective": [],
                                   "Statistic": [],
                                   "Value": []}
        for n_gen in sorted(list(self.__data.keys())):
            dd: dict[str, float] = self.__data[n_gen]
            for stat in dd:
                val: float = dd[stat]
                objective_name: str = self.__objective_name
                value: float = val
                d["Generation"].append(n_gen)
                d["Objective"].append(objective_name)
                d["Statistic"].append(stat)
                d["Value"].append(value)
        return d
    
    def build_list(self) -> list[dict[str, dict[str, float]]]:
        l: list[dict[str, dict[str, float]]] = []
        for n_gen in sorted(list(self.__data.keys())):
            d: dict[str, dict[str, float]] = {self.__objective_name: self.__data[n_gen]}
            l.append(d)
        return l
