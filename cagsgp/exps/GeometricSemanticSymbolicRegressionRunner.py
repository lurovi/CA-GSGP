from collections.abc import Callable
from copy import deepcopy
from functools import partial
import itertools
import time
from typing import Any

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
from cagsgp.nsgp.structure.factory.RowMajorLineFactory import RowMajorLineFactory
from cagsgp.nsgp.structure.factory.RowMajorMatrixFactory import RowMajorMatrixFactory
from cagsgp.util.EvaluationMetrics import EvaluationMetrics
from cagsgp.util.ResultUtils import ResultUtils
from genepro.node import Node

import numpy as np
import random

from genepro.node_impl import Constant
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
        dataset_path: str,
        seed: int = None,
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
        
        dataset: dict[str, tuple[np.ndarray, np.ndarray]] = DatasetGenerator.read_csv_data(path=dataset_path, idx=seed)
        X_train: np.ndarray = dataset['train'][0]
        y_train: np.ndarray = dataset['train'][1]
        X_test: np.ndarray = dataset['test'][0]
        y_test: np.ndarray = dataset['test'][1]
        dataset = None

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
            constants: list[Constant] = [Constant(round(ephemeral_func(), 2)) for _ in range(n_constants)]

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
        elif neighbors_topology == 'line':
            neighbors_topology_factory: NeighborsTopologyFactory = RowMajorLineFactory(radius=radius)
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
            neighbors_topology_factory=neighbors_topology_factory,
            train_set=(X_train, y_train),
            test_set=(X_test, y_test)
        )

        end_time: float = time.time()
        execution_time_in_minutes: float = (end_time - start_time)*(1/60)

        # ===========================
        # COLLECT RESULTS
        # ===========================

        pareto_front_df: dict[str, Any] = ResultUtils.parse_result_soo(
            result=res,
            objective_names=['RMSE'],
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
        neighbors_topology_factory: NeighborsTopologyFactory,
        train_set: tuple[np.ndarray, np.ndarray],
        test_set: tuple[np.ndarray, np.ndarray]
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
        
        pop: list[tuple[Node, float]] = [(structure.generate_tree(), None) for _ in range(pop_size)]
        
        # ===========================
        # ITERATIONS
        # ===========================
        
        for current_gen in range(num_gen):

            # ===========================
            # FITNESS EVALUATION AND UPDATE
            # ===========================
            
            fit_values: list[float] = GeometricSemanticSymbolicRegressionRunner.__fitness_evaluation_and_update_statistics_and_result(
                pop=pop,
                pop_size=pop_size,
                stats_collector=stats_collector,
                current_gen=current_gen,
                verbose=verbose,
                gen_verbosity_level=gen_verbosity_level,
                result=result,
                train_set=train_set,
                test_set=test_set
            )

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
                    cx_tree: Node = structure.geometric_semantic_single_tree_crossover(first_parent[0], second_parent[0], enable_caching=True, fix_properties=True)
                    new_tree: tuple[Node, float] = (cx_tree, None)
                else:
                    new_tree: tuple[Node, float] = pop[i]

                # == MUTATION ==
                if random.random() < mutation_probability:
                    mut_tree: Node = structure.geometric_semantic_tree_mutation(new_tree[0], m=m, fix_properties=True)
                    new_tree = (mut_tree, None)
                
                offsprings.append(new_tree)

            # ===========================
            # CHANGE POPULATION
            # ===========================

            pop = offsprings
            parents = None
            offsprings = None
            new_tree = None
            

            # == NEXT GENERATION ==

        # == END OF EVOLUTION ==

        # ===========================
        # LAST FITNESS EVALUATION AND UPDATE
        # ===========================

        fit_values = GeometricSemanticSymbolicRegressionRunner.__fitness_evaluation_and_update_statistics_and_result(
                pop=pop,
                pop_size=pop_size,
                stats_collector=stats_collector,
                current_gen=num_gen,
                verbose=verbose,
                gen_verbosity_level=gen_verbosity_level,
                result=result,
                train_set=train_set,
                test_set=test_set
            )
        

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
        pop: list[tuple[Node, float]],
        pop_size: int,
        stats_collector: StatsCollectorSingle,
        current_gen: int,
        verbose: bool,
        gen_verbosity_level: int,
        result: dict[str, Any],
        train_set: tuple[np.ndarray, np.ndarray],
        test_set: tuple[np.ndarray, np.ndarray],
    ) -> list[float]:
        
        # ===========================
        # FITNESS EVALUATION
        # ===========================

        fit_values: list[float] = [GeometricSemanticSymbolicRegressionRunner.__compute_single_RMSE_value_and_replace(pop=pop, idx=i, X=train_set[0], y=train_set[1])
                                   for i in range(pop_size)]

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
    def __compute_single_RMSE_value_and_replace(pop: list[tuple[Node, float]], idx: int, X: np.ndarray, y: np.ndarray) -> float:
        current_individual: tuple[Node, float] = pop[idx]
        if current_individual[1] is not None:
            # This individual has already been evaluated before, no need to recompute its fitness again
            current_fitness: float = current_individual[1]
        else:
            # This individual has never been evaluated, need to compute its fitness
            current_tree: Node = current_individual[0]
            p: np.ndarray = np.core.umath.clip(current_tree(X), -1e+10, 1e+10)
            current_fitness: float = EvaluationMetrics.root_mean_squared_error(y=y, p=p, linear_scaling=True, slope=None, intercept=None)
            pop[idx] = (current_tree, current_fitness)
        return current_fitness
    
    '''
    @staticmethod
    def __clean_pred_from_tree(tree: Node, curr_depth: int, target_depth: int, max_depth: int) -> None:
        if curr_depth < target_depth:
            for i in range(tree.arity):
                GeometricSemanticSymbolicRegressionRunner.__clean_pred_from_tree(tree.get_child(i), curr_depth=curr_depth+1, target_depth=target_depth, max_depth=max_depth)
            return
        if target_depth <= curr_depth < max_depth:
            if isinstance(tree, GSGPCrossover):
                tree.clean_pred()
                first: Node = tree.get_child(0)
                second: Node = tree.get_child(1)
                GeometricSemanticSymbolicRegressionRunner.__clean_pred_from_tree(first, curr_depth=curr_depth+1, target_depth=target_depth, max_depth=max_depth)
                GeometricSemanticSymbolicRegressionRunner.__clean_pred_from_tree(second, curr_depth=curr_depth+1, target_depth=target_depth, max_depth=max_depth)
                return
            for i in range(tree.arity):
                GeometricSemanticSymbolicRegressionRunner.__clean_pred_from_tree(tree.get_child(i), curr_depth=curr_depth+1, target_depth=target_depth, max_depth=max_depth)
            return
        if curr_depth == max_depth:
            if isinstance(tree, GSGPCrossover):
                tree.clean_pred()
        return
    '''
