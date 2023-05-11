from collections.abc import MutableSequence
from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology
from cagsgp.nsgp.structure.RowMajorMatrix import RowMajorMatrix
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar

T = TypeVar('T')


class RowMajorMatrixFactory(NeighborsTopologyFactory):
    def __init__(self,
                 n_rows: int,
                 n_cols: int
                 ) -> None:
        super().__init__()
        self.__n_rows: int = n_rows
        self.__n_cols: int = n_cols

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return RowMajorMatrix(collection=collection, n_rows=self.__n_rows, n_cols=self.__n_cols, clone=clone)
    