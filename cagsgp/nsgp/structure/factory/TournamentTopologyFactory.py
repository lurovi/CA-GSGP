from collections.abc import MutableSequence
from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology
from cagsgp.nsgp.structure.TournamentTopology import TournamentTopology
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar

T = TypeVar('T')


class TournamentTopologyFactory(NeighborsTopologyFactory):
    def __init__(self,
                 pressure: int = 3
                 ) -> None:
        super().__init__()
        if pressure < 1:
            raise ValueError(f'Pressure must be at least 1, found {pressure} instead.')
        self.__pressure: int = pressure

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return TournamentTopology(collection=collection, clone=clone, pressure=self.__pressure)
    