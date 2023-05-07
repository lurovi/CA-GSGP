from abc import ABC, abstractmethod
from genepro.node import Node


class TreeEvaluator(ABC):
    def __init__(self) -> None:
        super().__init__()

    def class_name(self) -> str:
        self.__class__.__name__

    @abstractmethod
    def evaluate(self, tree: Node) -> float:
        pass
