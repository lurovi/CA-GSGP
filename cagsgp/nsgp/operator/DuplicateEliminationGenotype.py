import numpy as np

from pymoo.core.duplicate import ElementwiseDuplicateElimination
from genepro.node import Node


class DuplicateEliminationGenotype(ElementwiseDuplicateElimination):
    def __init__(self) -> None:
        super().__init__()

    def is_equal(self, a, b) -> bool:
        a: Node = a.X[0]
        b: Node = b.X[0]
        return a == b
