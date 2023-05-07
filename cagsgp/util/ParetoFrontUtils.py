from typing import TypeVar

T = TypeVar('T')


class ParetoFrontUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def dominates(point_1: list[float], point_2: list[float]) -> bool:
        if len(point_1) != len(point_2):
            raise AttributeError(f"Lengths must be equal.")
        if point_1 == point_2:
            return False
        n: list[bool] = [point_1[i] <= point_2[i] for i in range(len(point_1))]
        return all(n)

    @staticmethod
    def is_dominated(point: list[float], l: list[tuple[T, list[float]]]) -> bool:
        for _, point_i in l:
            if ParetoFrontUtils.dominates(point_i, point):
                return True
        return False

    @staticmethod
    def filter_non_dominated_points(l: list[tuple[T, list[float]]]) -> list[tuple[T, list[float]]]:
        r: list[tuple[T, list[float]]] = []
        for o, point_i in l:
            if not ParetoFrontUtils.is_dominated(point_i, l):
                r.append((o, point_i))
        return r
