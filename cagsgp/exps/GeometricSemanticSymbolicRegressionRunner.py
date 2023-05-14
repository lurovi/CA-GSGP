from collections.abc import Callable
from copy import deepcopy
from functools import partial
import itertools
import time
from typing import Any
from weakref import WeakKeyDictionary

from numpy.random import Generator
from prettytable import PrettyTable
from cagsgp.benchmark.DatasetGenerator import DatasetGenerator
from cagsgp.nsgp.evaluator.RMSE import RMSE
from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator
from cagsgp.nsgp.stat.StatsCollectorSingle import StatsCollectorSingle
from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology
from cagsgp.nsgp.structure.TreeStructure import TreeStructure
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from cagsgp.nsgp.structure.factory.RowMajorCubeFactory import RowMajorCubeFactory
from cagsgp.nsgp.structure.factory.RowMajorMatrixFactory import RowMajorMatrixFactory
from cagsgp.util.EvaluationMetrics import EvaluationMetrics
from cagsgp.util.PicklePersist import PicklePersist
from cagsgp.util.ResultUtils import ResultUtils
from cagsgp.util.parallel.FakeParallelizer import FakeParallelizer
from cagsgp.util.parallel.MultiProcessingParallelizer import MultiProcessingParallelizer
from cagsgp.util.parallel.Parallelizer import Parallelizer
from genepro.node import Node

import numpy as np
import random

from genepro.node_impl import Constant, SemanticVector
from genepro.storage import Cache
from genepro.util import get_subtree_as_full_list


class GeometricSemanticSymbolicRegressionRunner:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def run_symbolic_regression_with_cellular_automata_gsgp(
        pop_size: int,
        pop_shape: tuple[int, ...],
        num_gen: int,
        max_depth: int,
        generation_strategy: str,
        operators: list[Node],
        low_erc: float,
        high_erc: float,
        n_constants: int,
        dataset_name: str,
        dataset_path: str = None,
        seed: int = None,
        multiprocess: bool = False,
        verbose: bool = False,
        gen_verbosity_level: int = 1,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.6,
        m: float = 2.0,
        duplicates_elimination: str = 'nothing',
        neighbors_topology: str = 'matrix',
        radius: int = 1
    ) -> tuple[dict[str, Any], str]:
        
        # ===========================
        # LOADING DATASET
        # ===========================
        
        if dataset_path is not None:        
            #dataset: dict[str, tuple[np.ndarray, np.ndarray]] = PicklePersist.decompress_pickle(dataset_path)
            dataset: tuple[np.ndarray, np.ndarray] = DatasetGenerator.read_csv_data(path=dataset_path)
            X_train: np.ndarray = dataset[0]
            y_train: np.ndarray = dataset[1]
            dataset = None
        else:
            dataset: dict[str, tuple[np.ndarray, np.ndarray]] = DatasetGenerator.generate_dataset(dataset_name=dataset_name, seed=seed, reset=False, path=None)
            X_train: np.ndarray = dataset['training'][0]
            y_train: np.ndarray = dataset['training'][1]
            dataset = None

        if multiprocess:
            parallelizer: Parallelizer = MultiProcessingParallelizer(-1)
        else:
            parallelizer: Parallelizer = FakeParallelizer()


        # ===========================
        # EVALUATOR AND CACHE
        # ===========================

        #cache: Cache = Cache(WeakKeyDictionary())
        #cache.enable()

        
        #evaluators: list[TreeEvaluator] = [RMSE(X_train, y_train, cache=cache, store_in_cache=True, fix_properties=True)]

        # ===========================
        # ERC CREATION
        # ===========================

        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        if low_erc > high_erc:
            raise AttributeError(f"low erc is higher than high erc.")
        elif low_erc < high_erc:
            ephemeral_func: Callable = lambda: generator.uniform(low=low_erc, high=high_erc)
        else:
            ephemeral_func: Callable = None

        if n_constants > 0:
            constants: list[Constant] = [Constant(ephemeral_func()) for _ in range(n_constants)]

        # ===========================
        # TREE STRUCTURE
        # ===========================

        structure: TreeStructure = TreeStructure(operators=operators,
                                                 fixed_constants=constants if n_constants > 0 else None,
                                                 ephemeral_func=ephemeral_func if n_constants == 0 else None,
                                                 n_features=X_train.shape[1],
                                                 max_depth=max_depth,
                                                 generation_strategy=generation_strategy)

        # ===========================
        # NEIGHBORS TOPOLOGY FACTORY
        # ===========================

        pressure: int = (2 * radius + 1) ** len(pop_shape)
        if neighbors_topology == 'matrix':
            neighbors_topology_factory: NeighborsTopologyFactory = RowMajorMatrixFactory(n_rows=pop_shape[0], n_cols=pop_shape[1], radius=radius)
        elif neighbors_topology == 'cube':
            neighbors_topology_factory: NeighborsTopologyFactory = RowMajorCubeFactory(n_channels=pop_shape[0], n_rows=pop_shape[1], n_cols=pop_shape[2], radius=radius)
        else:
            raise ValueError(f'{neighbors_topology} is not a valid neighbors topology.')

        # ===========================
        # GSGP RUN
        # ===========================

        start_time: float = time.time()

        res: dict[str, Any] = GeometricSemanticSymbolicRegressionRunner.__ca_inspired_gsgp(
            pop_size=pop_size,
            pop_shape=pop_shape,
            num_gen=num_gen,
            seed=seed,
            structure=structure,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            m=m,
            verbose=verbose,
            gen_verbosity_level=gen_verbosity_level,
            parallelizer=parallelizer,
            neighbors_topology_factory=neighbors_topology_factory,
            train_set=(X_train, y_train)
        )

        end_time: float = time.time()
        execution_time_in_minutes: float = (end_time - start_time)*(1/60)

        # ===========================
        # COLLECT RESULTS
        # ===========================

        pareto_front_df: dict[str, Any] = ResultUtils.parse_result_soo(
            result=res,
            objective_names=['RMSE'], #[e.class_name() for e in evaluators],
            seed=seed,
            pop_size=pop_size,
            num_gen=num_gen,
            num_offsprings=pop_size,
            max_depth=max_depth,
            generation_strategy=generation_strategy,
            pressure=pressure,
            pop_shape=pop_shape,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            m=m,
            execution_time_in_minutes=execution_time_in_minutes,
            neighbors_topology=neighbors_topology,
            radius=radius,
            dataset_name=dataset_name,
            duplicates_elimination=duplicates_elimination
        )
        
        run_id: str = f"symbolictreesRMSECAGSGPSOO-popsize_{pop_size}-numgen_{num_gen}-maxdepth_{max_depth}-neighbors_topology_{neighbors_topology}-dataset_{dataset_name}-duplicates_elimination_{duplicates_elimination}-pop_shape_{'x'.join([str(n) for n in pop_shape])}-crossprob_{str(round(crossover_probability, 2))}-mutprob_{str(round(mutation_probability, 2))}-m_{str(round(m, 2))}-radius_{str(radius)}-genstrategy_{generation_strategy}-SEED{seed}"
        if verbose:
            print(f"\nSYMBOLIC TREES RMSE CA-GSGP SOO: Completed with seed {seed}, PopSize {pop_size}, NumGen {num_gen}, MaxDepth {max_depth}, Neighbors Topology {neighbors_topology}, Dataset {dataset_name}, Duplicates Elimination {duplicates_elimination}, Pop Shape {str(pop_shape)}, Crossover Probability {str(round(crossover_probability, 2))}, Mutation Probability {str(round(mutation_probability, 2))}, M {str(round(m, 2))}, Radius {str(radius)}, Generation Strategy {generation_strategy}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        
        return pareto_front_df, run_id
    
    @staticmethod
    def __ca_inspired_gsgp(
        pop_size: int,
        pop_shape: int,
        num_gen: int,
        seed: int,
        structure: TreeStructure,
        crossover_probability: float,
        mutation_probability: float,
        m: float,
        verbose: bool,
        gen_verbosity_level: int,
        parallelizer: Parallelizer,
        neighbors_topology_factory: NeighborsTopologyFactory,
        train_set: tuple[np.ndarray, np.ndarray]
    ) -> dict[str, Any]:
        
        # ===========================
        # SEED SET
        # ===========================

        random.seed(seed)
        np.random.seed(seed)

        # ===========================
        # BASE STRUCTURES
        # ===========================

        # == RESULT, STATISTICS, EVALUATOR, FITNESS, TOPOLOGY COORDINATES ==
        #rmse: TreeEvaluator = evaluators[0]
        all_possible_coordinates: list[tuple[int, ...]] = [elem for elem in itertools.product(*[list(range(s)) for s in pop_shape])]
        result: dict[str, Any] = {'best': {}, 'history': []}
        stats_collector: StatsCollectorSingle = StatsCollectorSingle(objective_name='RMSE', revert_sign=False)
        
        # == ALL POSSIBLE NEIGHBORHOODS ==
        neigh_top_indices: NeighborsTopology = neighbors_topology_factory.create(all_possible_coordinates, clone=False)
        all_neighborhoods_indices: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        for coordinate in all_possible_coordinates:
            curr_neighs: list[tuple[int, ...]] = neigh_top_indices.neighborhood(coordinate, include_current_point=True, clone=False)
            all_neighborhoods_indices[coordinate] = curr_neighs
        curr_neighs = None
        neigh_top_indices = None

        # ===========================
        # INITIALIZATION
        # ===========================
        
        #previous_pop: list[tuple[Node, float]] = []
        pop: list[tuple[Node, float]] = [(structure.generate_tree(), None) for _ in range(pop_size)]
        
        # ===========================
        # ITERATIONS
        # ===========================
        
        for current_gen in range(num_gen):

            # ===========================
            # FITNESS EVALUATION AND UPDATE
            # ===========================

            fit_values: list[float] = GeometricSemanticSymbolicRegressionRunner.__fitness_evaluation_and_update_statistics_and_result(
                parallelizer=parallelizer,
                pop=pop,
                pop_size=pop_size,
                stats_collector=stats_collector,
                current_gen=current_gen,
                verbose=verbose,
                gen_verbosity_level=gen_verbosity_level,
                result=result,
                train_set=train_set
            )

            # == FREE SPACE IN CACHE ==
            #for old_individual, _ in set(previous_pop).difference(set(pop)):
            #    cache.remove(old_individual)

            # ===========================
            # SELECTION
            # ===========================

            evaluated_individuals: list[tuple[int, Node, float]] = [(i, pop[i][0], fit_values[i]) for i in range(pop_size)]
            parents: list[tuple[tuple[Node, float], tuple[Node, float]]] = []
            neighbors_topology: NeighborsTopology = neighbors_topology_factory.create(evaluated_individuals, clone=False)

            for coordinate in all_possible_coordinates:
                competitors: list[tuple[int, Node, float]] = [neighbors_topology.get(idx_tuple, clone=False) for idx_tuple in all_neighborhoods_indices[coordinate]]
                competitors.sort(key=lambda x: x[2], reverse=False)
                first: tuple[int, Node, float] = competitors[0]
                second: tuple[int, Node, float] = competitors[1]
                parents.append(((first[1], first[2]), (second[1], second[2])))

            evaluated_individuals = None
            neighbors_topology = None
            competitors = None
            first = None
            second = None

            # ===========================
            # CROSSOVER AND MUTATION
            # ===========================

            offsprings: list[tuple[Node, float]] = []
            for i, both_trees in enumerate(parents, 0):
                # == CROSSOVER ==
                if random.random() < crossover_probability:
                    first_parent: tuple[Node, float] = both_trees[0]
                    second_parent: tuple[Node, float] = both_trees[1]
                    new_tree: tuple[Node, float] = (structure.geometric_semantic_single_tree_crossover(first_parent[0], second_parent[0], cache=None, store_in_cache=False, fix_properties=True), None)
                else:
                    new_tree: tuple[Node, float] = pop[i]

                # == MUTATION ==
                if random.random() < mutation_probability:
                    new_tree = (structure.geometric_semantic_tree_mutation(new_tree[0], m=m, cache=None, store_in_cache=False, fix_properties=True), None)
                
                offsprings.append(new_tree)

            # == CHANGE POPULATION ==

            #previous_pop = pop
            pop = offsprings
            parents = None
            offsprings = None
            new_tree = None
            
            #print(cache.cache_size())
            #print(cache.hit_ratio())

            # == NEXT GENERATION ==

        # == END OF EVOLUTION ==

        # ===========================
        # LAST FITNESS EVALUATION AND UPDATE
        # ===========================

        fit_values = GeometricSemanticSymbolicRegressionRunner.__fitness_evaluation_and_update_statistics_and_result(
                parallelizer=parallelizer,
                pop=pop,
                pop_size=pop_size,
                stats_collector=stats_collector,
                current_gen=num_gen,
                verbose=verbose,
                gen_verbosity_level=gen_verbosity_level,
                result=result,
                train_set=train_set
            )
        
        #cache.empty_cache()
        #cache.disable()

        result['statistics'] = stats_collector.build_dict()

        #best_tree_overall: Node = result['best']['tree']
        #del result['best']['tree']

        #slope, intercept = EvaluationMetrics.compute_linear_scaling(train_set[1], np.core.umath.clip(best_tree_overall(train_set[0]), -1e+10, 1e+10))
        #final_pred: np.ndarray = EvaluationMetrics.linear_scale_predictions(np.core.umath.clip(best_tree_overall(test_set[0]), -1e+10, 1e+10), slope, intercept)
        #result['best']['Test RMSE'] = EvaluationMetrics.root_mean_squared_error(test_set[1], final_pred)
        #result['best']['Slope'] = slope
        #result['best']['Intercept'] = intercept

        return result

    @staticmethod
    def __fitness_evaluation_and_update_statistics_and_result(
        parallelizer: Parallelizer,
        pop: list[tuple[Node, float]],
        pop_size: int,
        stats_collector: StatsCollectorSingle,
        current_gen: int,
        verbose: bool,
        gen_verbosity_level: int,
        result: dict[str, Any],
        train_set: tuple[np.ndarray, np.ndarray],
    ) -> list[float]:
        
        # ===========================
        # FITNESS EVALUATION
        # ===========================

        #if parallelizer.class_name() == 'FakeParallelizer':
        #fit_values: list[float] = [GeometricSemanticSymbolicRegressionRunner.__compute_single_fitness_value(pop=pop, idx=i, evaluator=rmse)
        #                           for i in range(pop_size)]
        
        fit_values: list[float] = [GeometricSemanticSymbolicRegressionRunner.__compute_single_RMSE_value_and_replace(pop=pop, idx=i, X=train_set[0], y=train_set[1])
                                   for i in range(pop_size)]

        #else:
        #    all_inds: list[dict[str, Node]] = [{'individual': pop[i]} for i in range(pop_size)]
        #    pp: Callable = partial(single_evaluation_single_objective_no_fit_update, evaluator=rmse, fitness=fitness)
        #    fit_values: list[float] = parallelizer.parallelize(pp, all_inds)
        #    for i in range(len(fit_values)):
        #        fitness.set(pop[i], fit_values[i])
        
        # ===========================
        # UPDATE STATISTICS
        # ===========================

        stats_collector.update_fitness_stat_dict(n_gen=current_gen, data=fit_values)
        table: PrettyTable = PrettyTable(["Generation", "Min", "Max", "Mean", "Std"])
        table.add_row([str(current_gen),
                        stats_collector.get_fitness_stat(current_gen, 'min'),
                        stats_collector.get_fitness_stat(current_gen, 'max'),
                        stats_collector.get_fitness_stat(current_gen, 'mean'),
                        stats_collector.get_fitness_stat(current_gen, 'std')])
        
        if verbose and current_gen > 0 and current_gen % gen_verbosity_level == 0:
            print(table)

        # ===========================
        # UPDATE BEST AND HISTORY
        # ===========================

        min_value: float = min(fit_values)
        #index_of_min_value: int = fit_values.index(min_value)
        best_ind_here_totally: dict[str, Any] = {
            'ParsableTree': '', #str(get_subtree_as_full_list(best_ind_here)),
            'LatexTree': '', #ResultUtils.safe_latex_format(best_ind_here),
            'Fitness': {'RMSE': min_value}
        }

        if len(result['best']) == 0:
            result['best'] = best_ind_here_totally
            #result['best']['tree'] = pop[index_of_min_value][0]
        else:
            if best_ind_here_totally['Fitness']['RMSE'] < result['best']['Fitness']['RMSE']:
                result['best'] = best_ind_here_totally
                #result['best']['tree'] = pop[index_of_min_value][0]
        
        result['history'].append({kk: result['best'][kk] for kk in result['best']})

        return fit_values

    @staticmethod
    def __compute_single_fitness_value(pop: list[tuple[Node, float]], idx: int, evaluator: TreeEvaluator) -> float:
        current_individual: tuple[Node, float] = pop[idx]
        if current_individual[1] is not None:
            # This individual has already been evaluated before, no need to recompute its fitness again
            current_fitness: float = current_individual[1]
        else:
            # This individual has never been evaluated, need to compute its fitness
            current_fitness: float = evaluator.evaluate(current_individual[0])
            pop[idx] = (current_individual[0], current_fitness)
        return current_fitness
    
    @staticmethod
    def __compute_single_RMSE_value_and_replace(pop: list[tuple[Node, float]], idx: int, X: np.ndarray, y: np.ndarray) -> float:
        current_individual: tuple[Node, float] = pop[idx]
        if current_individual[1] is not None:
            # This individual has already been evaluated before, no need to recompute its fitness again
            current_fitness: float = current_individual[1]
        else:
            # This individual has never been evaluated, need to compute its fitness
            current_tree: Node = current_individual[0]
            p: np.ndarray = np.core.umath.clip(current_tree(X), -1e+10, 1e+10)
            new_replaced_tree: Node = SemanticVector(p=p, fix_properties=True)
            current_fitness: float = EvaluationMetrics.root_mean_squared_error(y=y, p=p, linear_scaling=True, slope=None, intercept=None)
            pop[idx] = (new_replaced_tree, current_fitness)
        return current_fitness
    
        
'''
def single_evaluation_single_objective_no_fit_update(individual: Node, evaluator: TreeEvaluator, fitness: Cache) -> float:
    r: float = fitness.get(individual)
    if r is not None:
        return r
    f: float = evaluator.evaluate(individual)
    return f
'''
