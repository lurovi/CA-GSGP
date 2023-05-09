import numpy as np

from pymoo.core.duplicate import ElementwiseDuplicateElimination
from genepro.node import Node
from genepro.node_impl import Pointer


class DuplicateEliminationSemantic(ElementwiseDuplicateElimination):
    def __init__(self,
                 X: np.ndarray,
                 cache: dict[Node, np.ndarray] = None,
                 store_in_cache: bool = False
                 ) -> None:
        super().__init__()
        self.__X: np.ndarray = X
        if cache is None:
            self.__cache: dict[Node, np.ndarray] = {}
        else:
            self.__cache:  dict[Node, np.ndarray] = cache
        self.__store_in_cache: bool = store_in_cache

    def is_equal(self, a, b) -> bool:
        a: Node = a.X[0]
        b: Node = b.X[0]
        return self.node_semantic_equals(a, b)
    
    def node_semantic_equals(self, a: Node, b: Node) -> bool:
        pointer_a: Node = Pointer(a, cache=self.__cache, store_in_cache=self.__store_in_cache)
        pointer_b: Node = Pointer(b, cache=self.__cache, store_in_cache=self.__store_in_cache)
        return np.array_equal(pointer_a(self.__X), pointer_b(self.__X))
