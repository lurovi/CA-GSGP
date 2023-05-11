from pymoo.core.crossover import Crossover
import numpy as np
from genepro.node import Node
from cagsgp.nsgp.structure.TreeStructure import TreeStructure


class GSGPTreeCrossover(Crossover):
    def __init__(self,
                 structure: TreeStructure,
                 cache: dict[Node, np.ndarray],
                 store_in_cache: bool,
                 fix_properties: bool
                 ) -> None:
        # define the crossover: number of parents and number of offsprings
        super().__init__(n_parents=2, n_offsprings=1, prob=1.0)
        self.__structure: TreeStructure = structure
        self.__cache: dict[Node, np.ndarray] = cache
        self.__store_in_cache: bool = store_in_cache
        self.__fix_properties: bool = fix_properties
        
    def _do(self, problem, x, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        n_parents, n_matings, n_var = x.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        y = np.full((self.n_offsprings, n_matings, problem.n_var), None, dtype=object)
        
        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            p1, p2 = x[0, k, 0], x[1, k, 0]
            # prepare the offsprings
            aa = self.__structure.geometric_semantic_single_tree_crossover(p1, p2, cache=self.__cache, store_in_cache=self.__store_in_cache, fix_properties=self.__fix_properties)
            y[0, k, 0] = aa
        
        return y
