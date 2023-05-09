from pymoo.core.problem import Problem
import numpy as np

from genepro.node import Node
from collections.abc import Callable
from functools import partial

from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator
from cagsgp.util.parallel.Parallelizer import Parallelizer
from cagsgp.util.parallel.FakeParallelizer import FakeParallelizer
from cagsgp.nsgp.stat.StatsCollector import StatsCollector


class MultiObjectiveMinimizationProblem(Problem):
    def __init__(self,
                 evaluators: list[TreeEvaluator],
                 revert_sign: list[bool] = None,
                 parallelizer: Parallelizer = None
                 ) -> None:
        
        if len(evaluators) < 1:
            raise ValueError(f'The evaluators must be at least 1, found {len(evaluators)} instead.')
        
        super().__init__(n_var=1, n_obj=len(evaluators), n_ieq_constr=0, n_eq_constr=0)
        if revert_sign is not None and len(revert_sign) != len(evaluators):
            raise ValueError(f'The length of revert_sign (found {len(revert_sign)}) must match the length of evaluators (found {len(evaluators)}).')
        self.__evaluators: list[TreeEvaluator] = [e for e in evaluators]
        self.__n_objectives: int = len(self.__evaluators)
        self.__is_multi_objective: bool = self.__n_objectives > 1
        self.__n_gen: int = -1
        self.__parallelizer: Parallelizer = parallelizer if parallelizer is not None else FakeParallelizer()

        if revert_sign is None:
            self.__revert_sign: list[bool] = [False] * self.__n_objectives
        else:
            self.__revert_sign: list[bool] = revert_sign

        self.__stats_collector: StatsCollector = StatsCollector(objective_names=[e.class_name() for e in evaluators], revert_sign=self.__revert_sign)

    def stats_collector(self) -> StatsCollector:
        return self.__stats_collector

    def _evaluate(self, x, out, *args, **kwargs):
        self._eval(x, out, *args, **kwargs)

    def _eval(self, x, out, *args, **kwargs):
        self.__n_gen += 1

        pp: Callable = partial(single_evaluation, evaluators=self.__evaluators)
        result: list[list[float]] = self.__parallelizer.parallelize(pp, [{'individual': x[i, 0]} for i in range(len(x))])
        result_np: np.ndarray = np.array(result, dtype=np.float32)
        out["F"] = result_np.reshape(1, -1)[0] if not self.__is_multi_objective else result_np

        self.__stats_collector.update_fitness_stat_dict(n_gen=self.__n_gen, data=result_np)


def single_evaluation(individual: Node, evaluators: list[TreeEvaluator]) -> list[float]:
    return [e.evaluate(individual) for e in evaluators]
