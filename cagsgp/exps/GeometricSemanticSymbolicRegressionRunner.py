from collections.abc import Callable
import time
from typing import Any
from pymoo.core.result import Result
from pymoo.core.population import Population
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from numpy.random import Generator
from cagsgp.nsgp.evaluator.MSE import MSE
from cagsgp.nsgp.operator.DuplicateEliminationGenotype import DuplicateEliminationGenotype
from cagsgp.nsgp.operator.DuplicateEliminationSemantic import DuplicateEliminationSemantic
from cagsgp.nsgp.operator.GSGPTreeSetting import GSGPTreeSetting
from cagsgp.nsgp.operator.RowMajorCubeSelection import RowMajorCubeSelection
from cagsgp.nsgp.operator.RowMajorMatrixSelection import RowMajorMatrixSelection
from cagsgp.nsgp.problem.MultiObjectiveMinimizationProblem import MultiObjectiveMinimizationProblem
from cagsgp.nsgp.stat.StatsCollector import StatsCollector
from cagsgp.nsgp.structure.TreeStructure import TreeStructure
from cagsgp.util.PicklePersist import PicklePersist
from cagsgp.util.ResultUtils import ResultUtils
from cagsgp.util.parallel.FakeParallelizer import FakeParallelizer
from cagsgp.util.parallel.MultiProcessingParallelizer import MultiProcessingParallelizer
from cagsgp.util.parallel.Parallelizer import Parallelizer
from genepro.node import Node
from pymoo.core.selection import Selection
from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
import random


class GeometricSemanticSymbolicRegression:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def run_symbolic_regression_with_cellular_automata_gsgp(
        dataset_path: str,
        dataset_name: str,
        pop_size: int,
        pop_shape: tuple[int, ...],
        num_gen: int,
        max_depth: int,
        operators: list[Node],
        low_erc: float,
        high_erc: float,
        seed: int = None,
        multiprocess: bool = False,
        verbose: bool = False,
        mutation_probability: float = 0.3,
        m: float = 0.5,
        store_in_cache: bool = True,
        duplicates_elimination: str = 'semantic',
        neighbors_topology: str = 'matrix'
    ) -> tuple[dict[str, Any], dict[str, list[Any]], str]:
        dataset: dict[str, tuple[np.ndarray, np.ndarray]] = PicklePersist.decompress_pickle(dataset_path)
        X: np.ndarray = dataset['training'][0]
        y: np.ndarray = dataset['training'][1]
        cache: dict[Node, np.ndarray] = {}

        if multiprocess:
            parallelizer: Parallelizer = MultiProcessingParallelizer(-1)
        else:
            parallelizer: Parallelizer = FakeParallelizer()

        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        problem: MultiObjectiveMinimizationProblem = MultiObjectiveMinimizationProblem(evaluators=[MSE(X, y)], parallelizer=parallelizer)

        if low_erc > high_erc:
            raise AttributeError(f"low erc is higher than high erc.")
        elif low_erc < high_erc:
            ephemeral_func: Callable = lambda: generator.uniform(low=low_erc, high=high_erc)
        else:
            ephemeral_func: Callable = None

        structure: TreeStructure = TreeStructure(operators=operators,
                                                 ephemeral_func=ephemeral_func,
                                                 n_features=X.shape[1],
                                                 max_depth=max_depth)
        

        if duplicates_elimination == 'semantic':
            dupl_el: ElementwiseDuplicateElimination = DuplicateEliminationSemantic(X, cache=cache, store_in_cache=store_in_cache)
        elif duplicates_elimination == 'structural':
            dupl_el: ElementwiseDuplicateElimination = DuplicateEliminationGenotype()
        else:
            raise ValueError(f'{duplicates_elimination} is an invalid duplicates elimination method.')

        setting: GSGPTreeSetting = GSGPTreeSetting(structure=structure,
                                                   m=m,
                                                   cache=cache,
                                                   store_in_cache=store_in_cache,
                                                   mutation_prob=mutation_probability,
                                                   duplicates_elimination=dupl_el
                                                   )
        
        if neighbors_topology == 'matrix':
            pressure: int = 9
            selector: Selection = RowMajorMatrixSelection(n_rows=pop_shape[0], n_cols=pop_shape[1])
        elif neighbors_topology == 'cube':
            pressure: int = 27
            selector: Selection = RowMajorCubeSelection(n_channels=pop_shape[0], n_rows=pop_shape[1], n_cols=pop_shape[2])
        else:
            raise ValueError(f'{neighbors_topology} is not a valid neighbors topology.')
        
        algorithm: NSGA2 = NSGA2(pop_size=pop_size,
                                 selection=selector,
                                 sampling=setting.get_sampling(),
                                 crossover=setting.get_crossover(),
                                 mutation=setting.get_mutation(),
                                 eliminate_duplicates=setting.get_duplicates_elimination())
        
        start_time: float = time.time()
        res: Result = minimize(problem,
                               algorithm,
                               termination=('n_gen', num_gen),
                               seed=seed,
                               verbose=verbose,
                               return_least_infeasible=False,
                               save_history=True
                               )
        end_time: float = time.time()
        execution_time_in_minutes: float = (end_time - start_time)*(1/60)
        problem: MultiObjectiveMinimizationProblem = res.problem

        stats: StatsCollector = problem.stats_collector()
        all_stats: dict[str, list[Any]] = stats.build_dict()
        
        opt: Population = res.opt
        history: list[Algorithm] = res.history

        pareto_front_df: dict[str, Any] = ResultUtils.parse_result(opt=opt,
                                                                   history=history,
                                                                   objective_names=['MSE'],
                                                                   seed=seed,
                                                                   pop_size=pop_size,
                                                                   num_gen=num_gen,
                                                                   num_offsprings=pop_size,
                                                                   max_depth=max_depth,
                                                                   pressure=pressure,
                                                                   pop_shape=pop_shape,
                                                                   crossover_probability=1.0,
                                                                   mutation_probability=mutation_probability,
                                                                   m=m,
                                                                   execution_time_in_minutes=execution_time_in_minutes,
                                                                   neighbors_topology=neighbors_topology,
                                                                   dataset=dataset_name,
                                                                   duplicates_elimination=duplicates_elimination)
        
        run_id: str = f"symbolictreesMSECAGSGPNSGA2-popsize_{pop_size}-numgen_{num_gen}-maxdepth_{max_depth}-neighbors_topology_{neighbors_topology}-dataset_{dataset_name}-duplicates_elimination_{duplicates_elimination}-pop_shape_{str([str(n) for n in pop_shape].join('|'))}-SEED{seed}"
        if verbose:
            print(f"\nSYMBOLIC TREES MSE CA-GSGP NSGA2: Completed with seed {seed}, PopSize {pop_size}, NumGen {num_gen}, MaxDepth {max_depth}, Neighbors Topology {neighbors_topology}, Dataset {dataset_name}, Duplicates Elimination {duplicates_elimination}, Pop Shape {str(pop_shape)}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        return pareto_front_df, all_stats, run_id
