from pymoo.core.sampling import Sampling
import numpy as np
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure
from genepro.node import Node


class GSGPTreeSampling(Sampling):

    def __init__(self,
                 structure: TreeStructure
                 ) -> None:
        super().__init__()
        self.__structure: TreeStructure = structure

    def _do(self, problem, n_samples, **kwargs):
        x = np.empty((n_samples, 1), dtype=object)

        for i in range(n_samples):
            x[i, 0] = self.__structure.generate_tree()

        return x
