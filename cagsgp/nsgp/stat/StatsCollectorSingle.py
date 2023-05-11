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
        length: int = len(da)
        sum_da: float = sum(da)
        mean_da: float = sum_da / float(length)
        max_da: float = max(da)
        min_da: float = min(da)
        median_da: float = statistics.median(da)
        std_da: float = statistics.pstdev(da, mu=mean_da)

        d: dict[str, float] = {"mean": mean_da,
                               "median": median_da,
                               "min": min_da,
                               "max": max_da,
                               "sum": sum_da,
                               "std": std_da}
        self.__data[n_gen] = d

    def get_fitness_stat(self, n_gen: int, stat: str) -> float:
        return self.__data[n_gen][stat]

    def build_dict(self) -> dict[str, list[Any]]:
        d: dict[str, list[Any]] = {"Generation": [],
                                   "Objective": [],
                                   "Statistic": [],
                                   "Value": []}
        for n_gen in self.__data:
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
