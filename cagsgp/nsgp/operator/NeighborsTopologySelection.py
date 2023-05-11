import numpy as np
from pymoo.core.selection import Selection
from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator
from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology
from cagsgp.nsgp.structure.TreeStructure import TreeStructure
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from genepro.node import Node
from pymoo.core.individual import Individual
import itertools
import math
from genepro.node_impl import Pointer


class NeighborsTopologySelection(Selection):
    def __init__(self,
                 structure: TreeStructure,
                 evaluators: list[TreeEvaluator],
                 neighbors_topology_factory: NeighborsTopologyFactory,
                 pop_shape: tuple[int, ...],
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.__structure: TreeStructure = structure
        self.__evaluators: list[TreeEvaluator] = [e for e in evaluators]
        self.__neighbors_topology_factory: NeighborsTopologyFactory = neighbors_topology_factory
        self.__pop_shape: tuple[int, ...] = pop_shape
        self.__size: int = math.prod(self.__pop_shape)
        self.__all_possible_positions: list[tuple[int, ...]] = NeighborsTopologySelection.__get_all_possible_positions(self.__pop_shape)

    @staticmethod
    def __get_all_possible_positions(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
        l: list[list[int]] = [list(range(s)) for s in shape]
        return [elem for elem in itertools.product(*l)]

    def _do(self, problem, pop, n_select, n_parents, **kwargs):
        individuals: list[tuple[int, Node, float]] = [(i, pop[i].X[0], pop[i].F) for i in range(len(pop))]

        curr_length: int = len(individuals)
        target_length: int = self.__size
        diff_length: int = target_length - curr_length
        for i in range(curr_length, curr_length + diff_length):
            tree: Node = self.__structure.generate_tree()
            fitness: list[float] = [e.evaluate(tree) for e in self.__evaluators]
            individuals.append((i, tree, np.array(fitness)))

        neighbors_topology: NeighborsTopology = self.__neighbors_topology_factory.create(individuals, clone=False)

        S: np.ndarray = np.full((self.__size, 2), fill_value=None, dtype=object)
        t: int = 0

        for position in self.__all_possible_positions:
            competitors: list[tuple[int, Node, np.ndarray]] = neighbors_topology.neighborhood(position, include_current_point=True, clone=False)
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
