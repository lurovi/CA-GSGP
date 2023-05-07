import numpy as np
from pymoo.core.mutation import Mutation
from genepro.node import Node
from cagsgp.nsgp.structure.TreeStructure import TreeStructure


class GSGPTreeMutation(Mutation):

    def __init__(self,
                 structure: TreeStructure,
                 cache: dict[Node, np.ndarray],
                 store_in_cache: bool,
                 m: float,
                 prob: float = 0.5
                 ) -> None:
        super().__init__(prob=prob)
        self.__structure: TreeStructure = structure
        self.__m: float = m
        self.__cache: dict[Node, np.ndarray] = cache
        self.__store_in_cache: bool = store_in_cache

    def _do(self, problem, x, **kwargs):
        # for each individual
        for i in range(len(x)):
            x[i, 0] = self.__structure.geometric_semantic_tree_mutation(x[i, 0],
                                                                        m=self.__m,
                                                                        cache=self.__cache,
                                                                        store_in_cache=self.__store_in_cache)
        return x
