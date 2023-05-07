from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")


class Parallelizer(ABC):
    """
    Abstract class that represents an abstract parallelizer. A parallelizer is a class that enables to
    parallelize a given method, i.e., it provides a way to execute a method with different sets of parameters in parallel.
    A concrete implementation of this class defines the way a method is parallelized to the provided sequence of inputs,
    where each input is a given set of parameters.
    """
    def __init__(self,
                 num_workers: int = 0
                 ) -> None:
        """
        Parallelizer constructor. It creates a Parallelizer instance with the specification of the number of workers (default 0, meaning no parallelization).
        :param num_workers: Number of workers to use within the parallelization process (default 0).
        :type num_workers: int
        """
        super().__init__()
        self.__num_workers: int = num_workers

    def class_name(self) -> str:
        """
        Gets the name of the particular class.
        :returns: The name of the particular class.
        :rtype: str
        """
        return self.__class__.__name__

    def num_workers(self) -> int:
        """
        Get the number of workers stored in this Parallelizer.
        :returns: The number of workers that has been set in this Parallelizer.
        :rtype: int
        """
        return self.__num_workers

    @abstractmethod
    def parallelize(self, target_method: Callable, parameters: list[dict[str, Any]]) -> list[T]:
        """
        Abstract method that gets a Python method (target_method) as input and applies the method to each set of parameters in the provided
        list (parameters). Each set of parameters in the list is a Python dictionary containing all <attribute, parameter> pairs related to the
        arguments accepted by the target method. It returns a list of results, depending on the return type of the provided method.
        A concrete implementation of this method defines a specific way of performing a parallelization of a given method a given sequence of sets of parameters.
        :param target_method: Method that should be applied to different inputs.
        :type target_method: Callable
        :param parameters: List of inputs to be used for the provided method. Each input in the list is a dictionary, i.e., a set of <attribute, parameter> pairs that defines the values to be used when calling the method.
        :type parameters: list(dict(str, Any))
        :returns: List of the results obtained by applying the provided method to each input.
        :rtype: list(T)
        """
        pass
