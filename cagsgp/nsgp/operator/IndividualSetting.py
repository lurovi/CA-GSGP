from abc import ABC
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.duplicate import ElementwiseDuplicateElimination


class IndividualSetting(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.__sampling: Sampling = None
        self.__crossover: Crossover = None
        self.__mutation: Mutation = None
        self.__duplicates_elimination: ElementwiseDuplicateElimination = None

    def set(self,
            sampling: Sampling,
            crossover: Crossover,
            mutation: Mutation,
            duplicates_elimination: ElementwiseDuplicateElimination
            ) -> None:
        if self.__sampling is None:
            self.__sampling = sampling
        if self.__crossover is None:
            self.__crossover = crossover
        if self.__mutation is None:
            self.__mutation = mutation
        if self.__duplicates_elimination is None:
            self.__duplicates_elimination = duplicates_elimination

    def get_sampling(self) -> Sampling:
        return self.__sampling

    def get_crossover(self) -> Crossover:
        return self.__crossover

    def get_mutation(self) -> Mutation:
        return self.__mutation

    def get_duplicates_elimination(self) -> ElementwiseDuplicateElimination:
        return self.__duplicates_elimination
