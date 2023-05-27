from collections.abc import Callable
import itertools
import time
from typing import Any

from numpy.random import Generator
from prettytable import PrettyTable
from cagsgp.benchmark.DatasetGenerator import DatasetGenerator

from cagsgp.nsgp.stat.StatsCollectorSingle import StatsCollectorSingle
from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology
from cagsgp.nsgp.structure.TreeStructure import TreeStructure
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from cagsgp.nsgp.structure.factory.RowMajorCubeFactory import RowMajorCubeFactory
from cagsgp.nsgp.structure.factory.RowMajorLineFactory import RowMajorLineFactory
from cagsgp.nsgp.structure.factory.RowMajorMatrixFactory import RowMajorMatrixFactory
from cagsgp.nsgp.structure.factory.TournamentTopologyFactory import TournamentTopologyFactory
from cagsgp.util.EvaluationMetrics import EvaluationMetrics
from cagsgp.util.ResultUtils import ResultUtils
from genepro.node import Node

import numpy as np
import random

from genepro.node_impl import Constant

    
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
    seed: int,
    verbose: bool,
    gen_verbosity_level: int,
    crossover_probability: float,
    mutation_probability: float,
    m: float,
    competitor_rate: float,
    duplicates_elimination: str,
    neighbors_topology: str,
    radius: int,
    elitism: bool
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
    constants = None
    ephemeral_func = None
    generator = None

    # ===========================
    # NEIGHBORS TOPOLOGY FACTORY
    # ===========================

    is_tournament_selection: bool = False
    pressure: int = (2 * radius + 1) ** len(pop_shape)
    if neighbors_topology == 'matrix':
        neighbors_topology_factory: NeighborsTopologyFactory = RowMajorMatrixFactory(n_rows=pop_shape[0], n_cols=pop_shape[1], radius=radius)
    elif neighbors_topology == 'cube':
        neighbors_topology_factory: NeighborsTopologyFactory = RowMajorCubeFactory(n_channels=pop_shape[0], n_rows=pop_shape[1], n_cols=pop_shape[2], radius=radius)
    elif neighbors_topology == 'line':
        neighbors_topology_factory: NeighborsTopologyFactory = RowMajorLineFactory(radius=radius)
    elif neighbors_topology == 'tournament':
        pressure = radius
        radius = 0
        is_tournament_selection = True
        neighbors_topology_factory: NeighborsTopologyFactory = TournamentTopologyFactory(pressure=pressure)
    else:
        raise ValueError(f'{neighbors_topology} is not a valid neighbors topology.')

    # ===========================
    # GSGP RUN
    # ===========================

    start_time: float = time.time()

    res: dict[str, Any] = __ca_inspired_gsgp(
        pop_size=pop_size,
        pop_shape=pop_shape,
        num_gen=num_gen,
        seed=seed,
        structure=structure,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        m=m,
        competitor_rate=competitor_rate,
        verbose=verbose,
        gen_verbosity_level=gen_verbosity_level,
        neighbors_topology_factory=neighbors_topology_factory,
        elitism=elitism,
        is_tournament_selection=is_tournament_selection,
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
        competitor_rate=competitor_rate,
        execution_time_in_minutes=execution_time_in_minutes,
        neighbors_topology=neighbors_topology,
        radius=radius,
        elitism=elitism,
        dataset_name=dataset_name,
        duplicates_elimination=duplicates_elimination
    )
    
    run_id: str = f"cgsgp-popsize_{pop_size}-numgen_{num_gen}-maxdepth_{max_depth}-neighbors_topology_{neighbors_topology}-dataset_{dataset_name}-duplicates_elimination_{duplicates_elimination}-pop_shape_{'x'.join([str(n) for n in pop_shape])}-crossprob_{str(round(crossover_probability, 2))}-mutprob_{str(round(mutation_probability, 2))}-m_{str(round(m, 2))}-radius_{str(radius)}-pressure_{str(pressure)}-genstrategy_{generation_strategy}-elitism_{str(int(elitism))}-SEED{seed}"
    if verbose:
        print(f"\nSYMBOLIC TREES RMSE CA-GSGP SOO: Completed with seed {seed}, PopSize {pop_size}, NumGen {num_gen}, MaxDepth {max_depth}, Neighbors Topology {neighbors_topology}, Dataset {dataset_name}, Duplicates Elimination {duplicates_elimination}, Pop Shape {str(pop_shape)}, Crossover Probability {str(round(crossover_probability, 2))}, Mutation Probability {str(round(mutation_probability, 2))}, M {str(round(m, 2))}, CompetitorRate {str(round(competitor_rate, 2))}, Radius {str(radius)}, Pressure {str(pressure)}, Generation Strategy {generation_strategy}, Elitism {str(int(elitism))}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
    
    return pareto_front_df, run_id


def __ca_inspired_gsgp(
    pop_size: int,
    pop_shape: int,
    num_gen: int,
    seed: int,
    structure: TreeStructure,
    crossover_probability: float,
    mutation_probability: float,
    m: float,
    competitor_rate: float,
    verbose: bool,
    gen_verbosity_level: int,
    neighbors_topology_factory: NeighborsTopologyFactory,
    elitism: bool,
    is_tournament_selection: bool,
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
    result: dict[str, Any] = {'best': {}, 'history': []}
    stats_collector: dict[str, StatsCollectorSingle] = {'train': StatsCollectorSingle(objective_name='RMSE', revert_sign=False),
                                                        'test': StatsCollectorSingle(objective_name='RMSE', revert_sign=False)}
    
    # == ALL POSSIBLE NEIGHBORHOODS ==
    if len(pop_shape) > 1 and not is_tournament_selection:
        all_possible_coordinates: list[tuple[int, ...]] = [elem for elem in itertools.product(*[list(range(s)) for s in pop_shape])]
    else:
        all_possible_coordinates: list[tuple[int, ...]] = [(i,) for i in range(pop_size)]

    neigh_top_indices: NeighborsTopology = neighbors_topology_factory.create(all_possible_coordinates, clone=False)
    all_neighborhoods_indices: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    
    if not is_tournament_selection:
        for coordinate in all_possible_coordinates:
            curr_neighs: list[tuple[int, ...]] = neigh_top_indices.neighborhood(coordinate, include_current_point=True, clone=False, distinct_coordinates=True)
            all_neighborhoods_indices[coordinate] = curr_neighs
    curr_neighs = None
    neigh_top_indices = None

    # ===========================
    # INITIALIZATION
    # ===========================
    
    pop: list[tuple[Node, dict[str, float]]] = [(structure.generate_tree(), {}) for _ in range(pop_size)]
    
    # ===========================
    # ITERATIONS
    # ===========================
    
    for current_gen in range(num_gen):

        # ===========================
        # FITNESS EVALUATION AND UPDATE
        # ===========================
        
        tt: tuple[dict[str, list[float]], int] = __fitness_evaluation_and_update_statistics_and_result(
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
        fit_values_train: list[float] = tt[0]['train']
        fit_values_test: list[float] = tt[0]['test']
        index_of_min_value: int = tt[1]

        # ===========================
        # SELECTION
        # ===========================

        evaluated_individuals: list[tuple[int, Node, float, float]] = [(i, pop[i][0], fit_values_train[i], fit_values_test[i]) for i in range(pop_size)]
        parents: list[tuple[tuple[Node, float, float], tuple[Node, float, float]]] = []
        neighbors_topology: NeighborsTopology = neighbors_topology_factory.create(evaluated_individuals, clone=False)

        for coordinate in all_possible_coordinates:
            if not is_tournament_selection:
                competitors: list[tuple[int, Node, float, float]] = [neighbors_topology.get(idx_tuple, clone=False) for idx_tuple in all_neighborhoods_indices[coordinate]]
                competitors.sort(key=lambda x: x[0], reverse=False)
                sampled_competitors: list[tuple[int, Node, float, float]] = [competitor for competitor in competitors if random.random() < competitor_rate]
                while len(sampled_competitors) < 2:
                    sampled_competitors.append(competitors[int(random.random()*len(competitors))])
                sampled_competitors.sort(key=lambda x: x[2], reverse=False)
                first: tuple[int, Node, float, float] = sampled_competitors[0]
                second: tuple[int, Node, float, float] = sampled_competitors[1]
            else:
                first_tournament: list[tuple[int, Node, float, float]] = neighbors_topology.neighborhood(coordinate, include_current_point=True, clone=False, distinct_coordinates=False)
                first_tournament.sort(key=lambda x: x[2], reverse=False)
                first: tuple[int, Node, float, float] = first_tournament[0]
                second_tournament: list[tuple[int, Node, float, float]] = neighbors_topology.neighborhood(coordinate, include_current_point=True, clone=False, distinct_coordinates=False)
                second_tournament.sort(key=lambda x: x[2], reverse=False)
                second: tuple[int, Node, float, float] = second_tournament[0]
            
            parents.append(((first[1], first[2], first[3]), (second[1], second[2], second[3])))

        evaluated_individuals = None
        neighbors_topology = None
        competitors = None
        first = None
        second = None

        # ===========================
        # CROSSOVER AND MUTATION
        # ===========================

        offsprings: list[tuple[Node, dict[str, float]]] = []
        for i, both_trees in enumerate(parents, 0):
            first_parent: tuple[Node, float, float] = both_trees[0]
            second_parent: tuple[Node, float, float] = both_trees[1]
            
            if elitism and i == index_of_min_value:
                # If elitism, preserve the best individual to the next generation
                offsprings.append(pop[i])
            else:
                # == CROSSOVER ==
                if random.random() < crossover_probability:
                    cx_tree: Node = structure.geometric_semantic_single_tree_crossover(first_parent[0], second_parent[0], enable_caching=True, fix_properties=True)
                    new_tree: tuple[Node, dict[str, float]] = (cx_tree, {})
                else:
                    new_tree: tuple[Node, dict[str, float]] = (first_parent[0], {'train': first_parent[1], 'test': first_parent[2]})

                # == MUTATION ==
                if random.random() < mutation_probability:
                    mutation_step: float = m if m != 0.0 else random.uniform(0.0, 1.0 + 1e-8)
                    mut_tree: Node = structure.geometric_semantic_tree_mutation(new_tree[0], m=mutation_step, fix_properties=True)
                    new_tree = (mut_tree, {})
                
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

    _ = __fitness_evaluation_and_update_statistics_and_result(
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
    

    result['train_statistics'] = stats_collector['train'].build_list()
    result['test_statistics'] = stats_collector['test'].build_list()

    return result


def __fitness_evaluation_and_update_statistics_and_result(
    pop: list[tuple[Node, dict[str, float]]],
    pop_size: int,
    stats_collector: dict[str, StatsCollectorSingle],
    current_gen: int,
    verbose: bool,
    gen_verbosity_level: int,
    result: dict[str, Any],
    train_set: tuple[np.ndarray, np.ndarray],
    test_set: tuple[np.ndarray, np.ndarray],
) -> tuple[dict[str, list[float]], int]:
    
    # ===========================
    # FITNESS EVALUATION
    # ===========================

    fit_values_dict: dict[str, list[float]] = __compute_single_RMSE_value_and_replace(
            pop=pop,
            pop_size=pop_size,
            X_train=train_set[0],
            y_train=train_set[1],
            X_test=test_set[0],
            y_test=test_set[1]
    )   

    # ===========================
    # UPDATE STATISTICS
    # ===========================
    
    for dataset_type in ['train', 'test']:
        stats_collector[dataset_type].update_fitness_stat_dict(n_gen=current_gen, data=fit_values_dict[dataset_type])
    
    table: PrettyTable = PrettyTable(["Generation", "Min", "Max", "Median", "Std"])
    table.add_row([str(current_gen),
                   stats_collector['train'].get_fitness_stat(current_gen, 'min'),
                   stats_collector['train'].get_fitness_stat(current_gen, 'max'),
                   stats_collector['train'].get_fitness_stat(current_gen, 'median'),
                   stats_collector['train'].get_fitness_stat(current_gen, 'std')])
    
    if verbose and current_gen > 0 and current_gen % gen_verbosity_level == 0:
        print(table)

    # ===========================
    # UPDATE BEST AND HISTORY
    # ===========================

    min_value: float = min(fit_values_dict['train'])
    index_of_min_value: int = fit_values_dict['train'].index(min_value)
    best_ind_here_totally: dict[str, Any] = {
        'Fitness': {'Train RMSE': min_value, 'Test RMSE': fit_values_dict['test'][index_of_min_value]},
        'PopIndex': index_of_min_value,
        'Generation': current_gen
    }

    if len(result['best']) == 0:
        result['best'] = best_ind_here_totally
    else:
        if best_ind_here_totally['Fitness']['Train RMSE'] < result['best']['Fitness']['Train RMSE']:
            result['best'] = best_ind_here_totally
    
    result['history'].append({kk: result['best'][kk] for kk in result['best']})

    return (fit_values_dict, index_of_min_value)


def __compute_single_RMSE_value_and_replace(pop: list[tuple[Node, dict[str, float]]], pop_size: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, list[float]]:
    fit_values_dict: dict[str, list[float]] = {'train': [], 'test': []}
    
    for i in range(pop_size):
        current_individual: tuple[Node, dict[str, float]] = pop[i]
        current_tree: Node = current_individual[0]
        current_fitness: dict[str, float] = current_individual[1]
        if len(current_fitness) != 0:
            # This individual has already been evaluated before, no need to recompute its fitness again
            new_fitness: dict[str, float] = current_fitness
        else:
            # This individual has never been evaluated, need to compute its fitness
            p_train: np.ndarray = np.core.umath.clip(current_tree(X_train, dataset_type='train'), -1e+10, 1e+10)
            p_test: np.ndarray = np.core.umath.clip(current_tree(X_test, dataset_type='test'), -1e+10, 1e+10)
            train_fitness: float = EvaluationMetrics.root_mean_squared_error(y=y_train, p=p_train, linear_scaling=False, slope=None, intercept=None)
            test_fitness: float = EvaluationMetrics.root_mean_squared_error(y=y_test, p=p_test, linear_scaling=False, slope=None, intercept=None)
            new_fitness: dict[str, float] = {'train': train_fitness, 'test': test_fitness}
            pop[i] = (current_tree, new_fitness)
        fit_values_dict['train'].append(new_fitness['train'])
        fit_values_dict['test'].append(new_fitness['test'])

    return fit_values_dict
