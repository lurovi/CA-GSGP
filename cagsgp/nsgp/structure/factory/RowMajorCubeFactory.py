from collections.abc import MutableSequence
from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology
from cagsgp.nsgp.structure.RowMajorCube import RowMajorCube
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar

T = TypeVar('T')


class RowMajorCubeFactory(NeighborsTopologyFactory):
    def __init__(self,
                 n_channels: int,
                 n_rows: int,
                 n_cols: int,
                 radius: int = 1
                 ) -> None:
        super().__init__()
        self.__n_channels: int = n_channels
        self.__n_rows: int = n_rows
        self.__n_cols: int = n_cols
        self.__radius: int = radius

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return RowMajorCube(collection=collection, n_channels=self.__n_channels, n_rows=self.__n_rows, n_cols=self.__n_cols, clone=clone, radius=self.__radius)
    