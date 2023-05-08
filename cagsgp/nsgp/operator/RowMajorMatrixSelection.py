import math
import numpy as np
from pymoo.core.selection import Selection
from cagsgp.nsgp.structure.RowMajorMatrix import RowMajorMatrix
from genepro.node import Node


class RowMajorMatrixSelection(Selection):
    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.__n_rows: int = n_rows
        self.__n_cols: int = n_cols

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        individuals: list[tuple[int, Node, float]] = [(i, pop[i].X[0], pop[i].F[0]) for i in range(len(pop))]
        row_major_matrix: RowMajorMatrix = RowMajorMatrix(individuals, n_rows=self.__n_rows, n_cols=self.__n_cols, clone=False)

        S: np.ndarray = np.full((len(pop), 2), fill_value=-1, dtype=np.int32)
        t: int = 0

        for i in range(self.__n_rows):
            for j in range(self.__n_cols):
                competitors: list[tuple[int, Node, float]] = row_major_matrix.neighborhood([i, j], include_current_point=True, clone=False)
                competitors.sort(key=lambda x: x[2], reverse=False)
                first: int = competitors[0][0]
                second: int = competitors[1][0]
                S[t][0] = first
                S[t][1] = second
                t += 1

        return S