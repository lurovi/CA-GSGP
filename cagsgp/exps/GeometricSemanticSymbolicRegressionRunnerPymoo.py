from collections.abc import Callable
import time
from typing import Any
from pymoo.core.result import Result
from pymoo.core.population import Population
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from numpy.random import Generator
from cagsgp.benchmark.DatasetGenerator import DatasetGenerator
from cagsgp.nsgp.evaluator.RMSE import RMSE
from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator
from cagsgp.nsgp.operator.DuplicateEliminationGenotype import DuplicateEliminationGenotype
from cagsgp.nsgp.operator.DuplicateEliminationSemantic import DuplicateEliminationSemantic
from cagsgp.nsgp.operator.GSGPTreeSetting import GSGPTreeSetting
from cagsgp.nsgp.operator.NeighborsTopologySelection import NeighborsTopologySelection
from cagsgp.nsgp.problem.MultiObjectiveMinimizationProblem import MultiObjectiveMinimizationProblem
from cagsgp.nsgp.stat.StatsCollector import StatsCollector
from cagsgp.nsgp.structure.TreeStructure import TreeStructure
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from cagsgp.nsgp.structure.factory.RowMajorCubeFactory import RowMajorCubeFactory
from cagsgp.nsgp.structure.factory.RowMajorMatrixFactory import RowMajorMatrixFactory
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

from genepro.node_impl import Constant


class GeometricSemanticSymbolicRegressionRunnerPymoo:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def run_symbolic_regression_with_cellular_automata_gsgp(
        pop_size: int,
        pop_shape: tuple[int, ...],
        num_gen: int,
        max_depth: int,
        operators: list[Node],
        low_erc: float,
        high_erc: float,
        n_constants: int,
        dataset_name: str,
        dataset_path: str = None,
        seed: int = None,
        multiprocess: bool = False,
        verbose: bool = False,
        mutation_probability: float = 0.5,
        m: float = 0.5,
        store_in_cache: bool = True,
        fix_properties: bool = True,
        duplicates_elimination: str = 'nothing',
        neighbors_topology: str = 'matrix'
    ) -> tuple[dict[str, Any], str]:
        
        if dataset_path is not None:        
            dataset: dict[str, tuple[np.ndarray, np.ndarray]] = PicklePersist.decompress_pickle(dataset_path)
        else:
            dataset: dict[str, tuple[np.ndarray, np.ndarray]] = DatasetGenerator.generate_dataset(dataset_name=dataset_name, seed=seed, reset=False, path=None)
        
        X_train: np.ndarray = dataset['training'][0]
        y_train: np.ndarray = dataset['training'][1]
        X_dev: np.ndarray = dataset['validation'][0]
        y_dev: np.ndarray = dataset['validation'][1]
        X_test: np.ndarray = dataset['test'][0]
        y_test: np.ndarray = dataset['test'][1]

        cache: dict[Node, np.ndarray] = {}

        if multiprocess:
            parallelizer: Parallelizer = MultiProcessingParallelizer(-1)
        else:
            parallelizer: Parallelizer = FakeParallelizer()

        evaluators: list[TreeEvaluator] = [RMSE(X_train, y_train, cache=cache, store_in_cache=store_in_cache, fix_properties=fix_properties)]

        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        semantic_dupl_elim: DuplicateEliminationSemantic = DuplicateEliminationSemantic(X_train, cache=cache, store_in_cache=store_in_cache, fix_properties=fix_properties)
        problem: MultiObjectiveMinimizationProblem = MultiObjectiveMinimizationProblem(evaluators=evaluators, parallelizer=parallelizer, semantic_dupl_elim=semantic_dupl_elim, compute_biodiversity=False)

        if low_erc > high_erc:
            raise AttributeError(f"low erc is higher than high erc.")
        elif low_erc < high_erc:
            ephemeral_func: Callable = lambda: generator.uniform(low=low_erc, high=high_erc)
        else:
            ephemeral_func: Callable = None

        if n_constants > 0:
            constants: list[Constant] = [Constant(ephemeral_func()) for _ in range(n_constants)]

        structure: TreeStructure = TreeStructure(operators=operators,
                                                 constants=constants if n_constants > 0 else None,
                                                 ephemeral_func=ephemeral_func if n_constants == 0 else None,
                                                 n_features=X_train.shape[1],
                                                 max_depth=max_depth)
        
        if duplicates_elimination == 'semantic':
            dupl_el: ElementwiseDuplicateElimination = semantic_dupl_elim
        elif duplicates_elimination == 'structural':
            dupl_el: ElementwiseDuplicateElimination = DuplicateEliminationGenotype()
        elif duplicates_elimination == 'nothing':
            dupl_el: ElementwiseDuplicateElimination = None
        else:
            raise ValueError(f'{duplicates_elimination} is an invalid duplicates elimination method.')

        setting: GSGPTreeSetting = GSGPTreeSetting(structure=structure,
                                                   m=m,
                                                   cache=cache,
                                                   store_in_cache=store_in_cache,
                                                   fix_properties=fix_properties,
                                                   mutation_prob=mutation_probability,
                                                   duplicates_elimination=dupl_el
                                                   )
        
        if neighbors_topology == 'matrix':
            pressure: int = 9
            neighbors_topology_factory: NeighborsTopologyFactory = RowMajorMatrixFactory(n_rows=pop_shape[0], n_cols=pop_shape[1])
        elif neighbors_topology == 'cube':
            pressure: int = 27
            neighbors_topology_factory: NeighborsTopologyFactory = RowMajorCubeFactory(n_channels=pop_shape[0], n_rows=pop_shape[1], n_cols=pop_shape[2])
        else:
            raise ValueError(f'{neighbors_topology} is not a valid neighbors topology.')

        selector: Selection = NeighborsTopologySelection(structure=structure, evaluators=evaluators, pop_shape=pop_shape, neighbors_topology_factory=neighbors_topology_factory)

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

        stats_collector: StatsCollector = problem.stats_collector()
        stats: dict[str, list[Any]] = stats_collector.build_dict()

        biodiversity: dict[str, list[float]] = problem.biodiversity()
        
        opt: Population = res.opt
        history: list[Algorithm] = res.history

        pareto_front_df: dict[str, Any] = ResultUtils.parse_result(opt=opt,
                                                                   history=history,
                                                                   stats=stats,
                                                                   biodiversity=biodiversity,
                                                                   objective_names=[e.class_name() for e in evaluators],
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
        
        run_id: str = f"symbolictreesRMSECAGSGPNSGA2-popsize_{pop_size}-numgen_{num_gen}-maxdepth_{max_depth}-neighbors_topology_{neighbors_topology}-dataset_{dataset_name}-duplicates_elimination_{duplicates_elimination}-pop_shape_{'x'.join([str(n) for n in pop_shape])}-crossprob_{str(round(1.0, 2))}-mutprob_{str(round(mutation_probability, 2))}-m_{str(round(m, 2))}-SEED{seed}"
        if verbose:
            print(f"\nSYMBOLIC TREES RMSE CA-GSGP NSGA2: Completed with seed {seed}, PopSize {pop_size}, NumGen {num_gen}, MaxDepth {max_depth}, Neighbors Topology {neighbors_topology}, Dataset {dataset_name}, Duplicates Elimination {duplicates_elimination}, Pop Shape {str(pop_shape)}, Crossover Probability {str(round(1.0, 2))}, Mutation Probability {str(round(mutation_probability, 2))}, M {str(round(m, 2))}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        return pareto_front_df, run_id
