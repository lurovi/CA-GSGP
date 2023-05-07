import numpy as np
from cagsgp.nsgp.operator.GSGPTreeCrossover import GSGPTreeCrossover
from cagsgp.nsgp.operator.GSGPTreeMutation import GSGPTreeMutation
from cagsgp.nsgp.operator.GSGPTreeSampling import GSGPTreeSampling
from genepro.node import Node
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from cagsgp.nsgp.operator.IndividualSetting import IndividualSetting

from cagsgp.nsgp.structure.TreeStructure import TreeStructure


class GSGPTreeSetting(IndividualSetting):
    def __init__(self,
                 structure: TreeStructure,
                 duplicates_elimination: ElementwiseDuplicateElimination,
                 m: float,
                 cache: dict[Node, np.ndarray],
                 store_in_cache: bool,
                 mutation_prob: float
                 ) -> None:
        super().__init__()
        self.__sampling: GSGPTreeSampling = GSGPTreeSampling(structure=structure)                                                        
        self.__crossover: GSGPTreeCrossover = GSGPTreeCrossover(structure=structure, cache=cache, store_in_cache=store_in_cache)
        self.__mutation: GSGPTreeMutation = GSGPTreeMutation(structure=structure, cache=cache, store_in_cache=store_in_cache, m=m, prob=mutation_prob)
        self.set(sampling=self.__sampling, crossover=self.__crossover, mutation=self.__mutation, duplicates_elimination=duplicates_elimination)
