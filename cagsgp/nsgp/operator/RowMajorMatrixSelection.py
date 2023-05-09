import math
import numpy as np
from pymoo.core.selection import Selection
from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator
from cagsgp.nsgp.structure.RowMajorMatrix import RowMajorMatrix
from cagsgp.nsgp.structure.TreeStructure import TreeStructure
from genepro.node import Node
from pymoo.core.individual import Individual


class RowMajorMatrixSelection(Selection):
    def __init__(self,
                 structure: TreeStructure,
                 evaluators: list[TreeEvaluator],
                 n_rows: int,
                 n_cols: int,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.__structure: TreeStructure = structure
        self.__evaluators: list[TreeEvaluator] = [e for e in evaluators]
        self.__n_rows: int = n_rows
        self.__n_cols: int = n_cols
        self.__size: int = self.__n_rows * self.__n_cols

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        individuals: list[tuple[int, Node, float]] = [(i, pop[i].X[0], pop[i].F) for i in range(len(pop))]

        curr_length: int = len(individuals)
        target_length: int = self.__size
        diff_length: int = target_length - curr_length
        for i in range(curr_length, curr_length + diff_length):
            tree: Node = self.__structure.generate_tree()
            fitness: list[float] = [e.evaluate(tree) for e in self.__evaluators]
            individuals.append((i, tree, np.array(fitness)))

        row_major_matrix: RowMajorMatrix = RowMajorMatrix(individuals, n_rows=self.__n_rows, n_cols=self.__n_cols, clone=False)

        S: np.ndarray = np.full((self.__size, 2), fill_value=None, dtype=object)
        t: int = 0
    
        for i in range(self.__n_rows):
            for j in range(self.__n_cols):
                competitors: list[tuple[int, Node, np.ndarray]] = row_major_matrix.neighborhood([i, j], include_current_point=True, clone=False)
                competitors.sort(key=lambda x: x[2][0], reverse=False)
                first: tuple[int, Node, np.ndarray] = competitors[0]
                second: tuple[int, Node, np.ndarray] = competitors[1]
                
                if first[0] < curr_length:
                    S[t][0] = pop[first[0]]
                else:
                    ind: Individual = Individual()
                    ind.X = np.array([first[1]], dtype=object)
                    ind.F = first[2]
                    S[t][0] = ind

                if second[0] < curr_length:
                    S[t][1] = pop[second[0]]
                else:
                    ind: Individual = Individual()
                    ind.X = np.array([second[1]], dtype=object)
                    ind.F = second[2]
                    S[t][1] = ind

                t += 1
        
        return S
