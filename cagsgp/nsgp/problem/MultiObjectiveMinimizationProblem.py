from pymoo.core.problem import Problem
import numpy as np

from genepro.node import Node
from collections.abc import Callable
from functools import partial

from cagsgp.nsgp.operator.DuplicateEliminationSemantic import DuplicateEliminationSemantic
from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator
from cagsgp.util.parallel.Parallelizer import Parallelizer
from cagsgp.util.parallel.FakeParallelizer import FakeParallelizer
from cagsgp.nsgp.stat.StatsCollector import StatsCollector


class MultiObjectiveMinimizationProblem(Problem):
    def __init__(self,
                 evaluators: list[TreeEvaluator],
                 semantic_dupl_elim: DuplicateEliminationSemantic,
                 revert_sign: list[bool] = None,
                 parallelizer: Parallelizer = None,
                 compute_biodiversity: bool = False
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
        self.__semantic_dupl_elim: DuplicateEliminationSemantic = semantic_dupl_elim

        if revert_sign is None:
            self.__revert_sign: list[bool] = [False] * self.__n_objectives
        else:
            self.__revert_sign: list[bool] = revert_sign

        self.__stats_collector: StatsCollector = StatsCollector(objective_names=[e.class_name() for e in evaluators], revert_sign=self.__revert_sign)

        self.__biodiversity: dict[str, list[float]] = {"structural": [], "semantic": []}
        self.__compute_biodiversity: bool = compute_biodiversity

    def stats_collector(self) -> StatsCollector:
        return self.__stats_collector
    
    def biodiversity(self) -> dict[str, list[float]]:
        return self.__biodiversity

    def _evaluate(self, x, out, *args, **kwargs):
        self._eval(x, out, *args, **kwargs)

    def _eval(self, x, out, *args, **kwargs):
        self.__n_gen += 1

        pp: Callable = partial(single_evaluation, evaluators=self.__evaluators)
        all_inds: list[dict[str, Node]] = [{'individual': x[i, 0]} for i in range(len(x))]
        
        result: list[list[float]] = self.__parallelizer.parallelize(pp, all_inds)
        result_np: np.ndarray = np.array(result, dtype=np.float32)
        out["F"] = result_np.reshape(1, -1)[0] if not self.__is_multi_objective else result_np

        self.__stats_collector.update_fitness_stat_dict(n_gen=self.__n_gen, data=result_np)

        if self.__compute_biodiversity:
            current_biodiversity: dict[str, float] = self.__count_duplicates(all_inds=all_inds)
            self.__biodiversity['structural'].append(current_biodiversity['structural'])
            self.__biodiversity['semantic'].append(current_biodiversity['semantic'])

    def __count_duplicates(self, all_inds: list[dict[str, Node]]) -> dict[str, float]:
        count_structural: float = 0
        count_semantic: float = 0
        length: int = len(all_inds)

        for i in range(length):
            is_duplicate_structural: bool = False
            is_duplicate_semantic: bool = False
            j: int = i + 1
            while (not is_duplicate_structural or not is_duplicate_semantic) and j < length:
                node_i: Node = all_inds[i]['individual']
                node_j: Node = all_inds[j]['individual']
                if node_i == node_j:
                    is_duplicate_structural = True
                if self.__semantic_dupl_elim.node_semantic_equals(node_i, node_j):
                    is_duplicate_semantic = True
                j += 1
            if is_duplicate_structural:
                count_structural += 1
            if is_duplicate_semantic:
                count_semantic += 1

        return {"structural": 1.0 - count_structural/float(len(all_inds) - 1), "semantic": 1.0 - count_semantic/float(len(all_inds) - 1)}


def single_evaluation(individual: Node, evaluators: list[TreeEvaluator]) -> list[float]:
    return [e.evaluate(individual) for e in evaluators]
