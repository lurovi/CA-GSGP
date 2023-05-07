from abc import abstractmethod, ABC
from collections.abc import MutableSequence
from typing import TypeVar

T = TypeVar('T')


class NeighborsTopology(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def class_name(self) -> str:
        self.__class__.__name__

    @abstractmethod
    def neighborhood(self, indices: list[int], include_current_point: bool = True, clone: bool = False) -> MutableSequence[T]:
        pass
