from typing import Any, TypeVar
import os
from collections.abc import Callable
import ray

from cagsgp.util.parallel.Parallelizer import Parallelizer

T = TypeVar('T')
ray.init()


class RayParallelizer(Parallelizer):
    """
    Subclass of Parallelizer. It performs a parallelization of a given method to a sequence of possible inputs by leveraging
    the ray module of Python.
    """
    def __init__(self,
                 num_workers: int = 0
                 ) -> None:
        """
        RayParallelizer constructor. It creates a RayParallelizer instance with the specification of the number of workers (default 0, meaning no parallelization).
        The parameter num_workers in this case is an int that must be in the range [-2, cpu_count]:
        - -2 means that number of workers is set to be equal to the total number of cores in your machine;
        - -1 means that number of workers is set to be equal to the total number of cores in your machine minus 1 (a single core remains free of work, so that the system is less likely to get frozen during the execution of the method);
        - 0 means that no parallelization is performed;
        - a strictly positive value means that the number of workers is set to be equal to the exact specified number which, of course, must not be higher than the available cores.
        :param num_workers: Number of workers to use within the parallelization process (default 0).
        :type num_workers: int
        """
        super().__init__(num_workers=num_workers)
        if self.num_workers() < -2:
            raise AttributeError(
                f"Specified an invalid number of cores {self.num_workers()}: this is a negative number lower than -2.")
        if self.num_workers() > os.cpu_count():
            raise AttributeError(
                f"Specified a number of cores ({self.num_workers()}) that is greater than the number of cores supported by your computer ({os.cpu_count()}).")


    def parallelize(self, target_method: Callable, parameters: list[dict[str, Any]]) -> list[T]:
        """
        Method that gets a Python method (target_method) as input and applies the method to each set of parameters in the provided
        list (parameters). Each set of parameters in the list is a Python dictionary containing all <attribute, parameter> pairs related to the
        arguments accepted by the target method. It returns a list of results, depending on the return type of the provided method.
        This method performs a parallelization of the provided method on the provided inputs by using map built-in method.
        :param target_method: Method that should be applied to different inputs.
        :type target_method: Callable
        :param parameters: List of inputs to be used for the provided method. Each input in the list is a dictionary, i.e., a set of <attribute, parameter> pairs that defines the values to be used when calling the method.
        :type parameters: list(dict(str, Any))
        :returns: List of the results obtained by applying the provided method to each input.
        :rtype: list(T)
        """
        if self.num_workers() == 0:
            return [target_method(**t) for t in parameters]

        number_of_processes: int = {-2: (os.cpu_count()), -1: (os.cpu_count() - 1)}.get(self.num_workers(), self.num_workers())

        res = ray.get([target_method_wrapper.remote(parameter=parameter, target_method=target_method) for parameter in parameters])

        return res


@ray.remote
def target_method_wrapper(parameter: dict[str, Any], target_method: Callable) -> T:
    # This method is simply a wrapper that unpacks the provided input and calls the provided function with the unpacked input.
    # Since this method must be parallelized with map, it must be declared and implemented in the global scope of a Python script.
    return target_method(**parameter)
