from collections.abc import MutableSequence
from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology
from cagsgp.nsgp.structure.RowMajorLine import RowMajorLine
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar

T = TypeVar('T')


class RowMajorLineFactory(NeighborsTopologyFactory):
    def __init__(self,
                 radius: int = 1
                 ) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError(f'Radius must be at least 1, found {radius} instead.')
        self.__radius: int = radius

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return RowMajorLine(collection=collection, clone=clone, radius=self.__radius)
    