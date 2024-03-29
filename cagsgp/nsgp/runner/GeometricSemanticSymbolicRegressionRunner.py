from collections.abc import Callable
import itertools
import math
import time
from typing import Any

from numpy.random import Generator
from prettytable import PrettyTable
from cagsgp.benchmark.DatasetGenerator import DatasetGenerator
from cagsgp.nsgp.stat.SemanticDistance import SemanticDistance

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
    mode: str,
    linear_scaling: bool,
    pop_shape: tuple[int, ...],
    pop_size: int,
    num_gen: int,
    num_gen_post: int,
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
    expl_pipe: str,
    duplicates_elimination: str,
    torus_dim: int,
    radius: int,
    elitism: bool
) -> tuple[dict[str, Any], str, str]:

    # ===========================
    # SETTING EXPL_PIPE STUFF
    # ===========================

    if expl_pipe == 'crossmut':
        execute_crossover: bool = True
        execute_mutation: bool = True
    elif expl_pipe == 'crossonly':
        execute_crossover: bool = True
        execute_mutation: bool = False
    elif expl_pipe == 'mutonly':
        execute_crossover: bool = False
        execute_mutation: bool = True
    else:
        raise ValueError(f'{expl_pipe} is not a valid exploration pipeline.')

    if mode != 'gsgpgp':
        num_gen_post = 0

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
    
    if mode == 'gsgp' or mode == 'gsgpgp':
        structure.set_fix_properties(True)
    elif mode == 'gp':
        structure.set_fix_properties(False)
    else:
        raise AttributeError(f'Invalid mode ({mode}).')
    
    constants = None
    ephemeral_func = None
    generator = None

    # ===========================
    # NEIGHBORS TOPOLOGY FACTORY
    # ===========================

    is_tournament_selection: bool = False
    pressure: int = (2 * radius + 1) ** len(pop_shape)
    if torus_dim == 2:
        neighbors_topology_factory: NeighborsTopologyFactory = RowMajorMatrixFactory(n_rows=pop_shape[0], n_cols=pop_shape[1], radius=radius)
    elif torus_dim == 3:
        neighbors_topology_factory: NeighborsTopologyFactory = RowMajorCubeFactory(n_channels=pop_shape[0], n_rows=pop_shape[1], n_cols=pop_shape[2], radius=radius)
    elif torus_dim == 1:
        neighbors_topology_factory: NeighborsTopologyFactory = RowMajorLineFactory(radius=radius)
    elif torus_dim == 0:
        pressure = radius
        radius = 0
        competitor_rate = 0.0
        is_tournament_selection = True
        neighbors_topology_factory: NeighborsTopologyFactory = TournamentTopologyFactory(pressure=pressure)
    else:
        raise ValueError(f'{torus_dim} is not a valid torus dimension.')

    # ===========================
    # GSGP RUN
    # ===========================

    start_time: float = time.time()

    res: dict[str, Any] = __ca_inspired_gsgp(
        mode=mode,
        linear_scaling=linear_scaling,
        pop_size=pop_size,
        pop_shape=pop_shape,
        num_gen=num_gen,
        num_gen_post=num_gen_post,
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
        execute_crossover=execute_crossover,
        execute_mutation=execute_mutation,
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
        mode=mode,
        linear_scaling=linear_scaling,
        pop_size=pop_size,
        num_gen=num_gen,
        num_gen_post=num_gen_post,
        num_offsprings=pop_size,
        max_depth=max_depth,
        generation_strategy=generation_strategy,
        pressure=pressure,
        pop_shape=pop_shape,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        m=m,
        competitor_rate=competitor_rate,
        expl_pipe=expl_pipe,
        execution_time_in_minutes=execution_time_in_minutes,
        torus_dim=torus_dim,
        radius=radius,
        elitism=elitism,
        dataset_name=dataset_name,
        duplicates_elimination=duplicates_elimination
    )
    
    path_run_id: str = f'linearscaling{int(linear_scaling)}/numgenpost{num_gen_post}/cmprate{str(round(competitor_rate, 2))}/{expl_pipe}/'
    run_id: str = f"c{mode}-popsize_{pop_size}-numgen_{num_gen}-maxdepth_{max_depth}-torus_dim_{torus_dim}-dataset_{dataset_name}-dupl_elim_{duplicates_elimination}-pop_shape_{'x'.join([str(n) for n in pop_shape])}-crossprob_{str(round(crossover_probability, 2))}-mutprob_{str(round(mutation_probability, 2))}-m_{str(round(m, 2))}-radius_{str(radius)}-pressure_{str(pressure)}-genstrat_{generation_strategy}-elitism_{str(int(elitism))}-SEED{seed}"
    if verbose:
        print(f"\nSYMBOLIC TREES RMSE CA-{mode.upper()} SOO: Completed with seed {seed}, PopSize {pop_size}, NumGen {num_gen}, MaxDepth {max_depth}, Torus Dim {torus_dim}, Dataset {dataset_name}, Duplicates Elimination {duplicates_elimination}, Pop Shape {str(pop_shape)}, Crossover Probability {str(round(crossover_probability, 2))}, Mutation Probability {str(round(mutation_probability, 2))}, M {str(round(m, 2))}, CompetitorRate {str(round(competitor_rate, 2))}, Radius {str(radius)}, Pressure {str(pressure)}, Generation Strategy {generation_strategy}, Elitism {str(int(elitism))}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
    
    return pareto_front_df, path_run_id, run_id


def __ca_inspired_gsgp(
    mode: str,
    linear_scaling: bool,
    pop_size: int,
    pop_shape: int,
    num_gen: int,
    num_gen_post: int,
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
    execute_crossover: bool,
    execute_mutation: bool,
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

    # == WEIGHTS MATRIX FOR MORAN's I ==

    weights_matrix_moran: list[list[float]] = SemanticDistance.zero_matrix(pop_size)
    if not is_tournament_selection:
        for i in range(pop_size):
            coordinate_i: tuple[int, ...] = all_possible_coordinates[i]
            neigh_indices_of_i: list[tuple[int, ...]] = all_neighborhoods_indices[coordinate_i]
            for j in range(pop_size):
                coordinate_j: tuple[int, ...] = all_possible_coordinates[j]
                if i != j and coordinate_j in neigh_indices_of_i:
                    weights_matrix_moran[i][j] = 1.0
    else:
        weights_matrix_moran = SemanticDistance.one_matrix_zero_diagonal(pop_size)
    moran_formula_coef: float = float(pop_size) / SemanticDistance.sum_of_all_elem_in_matrix(weights_matrix_moran)

    # ===========================
    # INITIALIZATION
    # ===========================
    
    pop: list[tuple[Node, dict[str, float]]] = [(structure.generate_tree(), {}) for _ in range(pop_size)]
    use_gsgp_semantic_vector_as_train_y: bool = False
    perform_gp_operators: bool = False

    # ===========================
    # ITERATIONS
    # ===========================
    
    for current_gen in range(num_gen):

        # ===========================
        # FITNESS EVALUATION AND UPDATE
        # ===========================
        
        tt: tuple[dict[str, list[float]], int, np.ndarray] = __fitness_evaluation_and_update_statistics_and_result(
            mode=mode,
            linear_scaling=linear_scaling,
            pop=pop,
            pop_size=pop_size,
            stats_collector=stats_collector,
            current_gen=current_gen,
            verbose=verbose,
            gen_verbosity_level=gen_verbosity_level,
            result=result,
            train_set=train_set,
            test_set=test_set,
            weights_matrix_moran=weights_matrix_moran,
            moran_formula_coef=moran_formula_coef,
            use_gsgp_semantic_vector_as_train_y=use_gsgp_semantic_vector_as_train_y,
            best_tree_semantic_vector=None
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

            first, second = __simple_selection_process(
                is_tournament_selection=is_tournament_selection,
                competitor_rate=competitor_rate,
                neighbors_topology=neighbors_topology,
                coordinate=coordinate,
                all_neighborhoods_indices=all_neighborhoods_indices
            )
            
            parents.append(((first[1], first[2], first[3]), (second[1], second[2], second[3])))

        evaluated_individuals = None
        neighbors_topology = None
        first = None
        second = None

        # ===========================
        # CROSSOVER AND MUTATION
        # ===========================

        offsprings: list[tuple[Node, dict[str, float]]] = []
        for i, both_trees in enumerate(parents, 0):
            first_parent: tuple[Node, float, float] = both_trees[0]
            second_parent: tuple[Node, float, float] = both_trees[1]
            
            new_tree: tuple[Node, dict[str, float]] = __simple_crossmut_process(
                elitism=elitism,
                i=i,
                index_of_min_value=index_of_min_value,
                pop=pop,
                execute_crossover=execute_crossover,
                execute_mutation=execute_mutation,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                mode=mode,
                structure=structure,
                m=m,
                first_parent=first_parent,
                second_parent=second_parent,
                perform_gp_operators=perform_gp_operators
            )
                        
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

    _, _, best_tree_semantic_vector = __fitness_evaluation_and_update_statistics_and_result(
            mode=mode,
            linear_scaling=linear_scaling,
            pop=pop,
            pop_size=pop_size,
            stats_collector=stats_collector,
            current_gen=num_gen,
            verbose=verbose,
            gen_verbosity_level=gen_verbosity_level,
            result=result,
            train_set=train_set,
            test_set=test_set,
            weights_matrix_moran=weights_matrix_moran,
            moran_formula_coef=moran_formula_coef,
            use_gsgp_semantic_vector_as_train_y=use_gsgp_semantic_vector_as_train_y,
            best_tree_semantic_vector=None
        )

    # ===========================
    # POST EVOLUTION WITH GP FOR GSGPGP
    # ===========================
    
    if mode == 'gsgpgp':

        structure.set_fix_properties(False)

        # ====
        # INITIALIZATION
        # ====
        
        pop = [(structure.generate_tree(), {}) for _ in range(pop_size)]
        use_gsgp_semantic_vector_as_train_y = True
        perform_gp_operators = True

        # ====
        # ITERATIONS
        # ====
        
        for current_gen in range(num_gen_post):

            # ====
            # FITNESS EVALUATION AND UPDATE
            # ====
            
            tt: tuple[dict[str, list[float]], int, np.ndarray] = __fitness_evaluation_and_update_statistics_and_result(
                mode=mode,
                linear_scaling=linear_scaling,
                pop=pop,
                pop_size=pop_size,
                stats_collector=stats_collector,
                current_gen=current_gen,
                verbose=verbose,
                gen_verbosity_level=gen_verbosity_level,
                result=result,
                train_set=train_set,
                test_set=test_set,
                weights_matrix_moran=weights_matrix_moran,
                moran_formula_coef=moran_formula_coef,
                use_gsgp_semantic_vector_as_train_y=use_gsgp_semantic_vector_as_train_y,
                best_tree_semantic_vector=best_tree_semantic_vector
            )
            fit_values_train: list[float] = tt[0]['train']
            fit_values_test: list[float] = tt[0]['test']
            index_of_min_value: int = tt[1]

            # ====
            # SELECTION
            # ====

            evaluated_individuals: list[tuple[int, Node, float, float]] = [(i, pop[i][0], fit_values_train[i], fit_values_test[i]) for i in range(pop_size)]
            parents: list[tuple[tuple[Node, float, float], tuple[Node, float, float]]] = []
            neighbors_topology: NeighborsTopology = neighbors_topology_factory.create(evaluated_individuals, clone=False)

            for coordinate in all_possible_coordinates:

                first, second = __simple_selection_process(
                    is_tournament_selection=is_tournament_selection,
                    competitor_rate=competitor_rate,
                    neighbors_topology=neighbors_topology,
                    coordinate=coordinate,
                    all_neighborhoods_indices=all_neighborhoods_indices
                )
                
                parents.append(((first[1], first[2], first[3]), (second[1], second[2], second[3])))

            evaluated_individuals = None
            neighbors_topology = None
            first = None
            second = None

            # ====
            # CROSSOVER AND MUTATION
            # ====

            offsprings: list[tuple[Node, dict[str, float]]] = []
            for i, both_trees in enumerate(parents, 0):
                first_parent: tuple[Node, float, float] = both_trees[0]
                second_parent: tuple[Node, float, float] = both_trees[1]
                
                new_tree: tuple[Node, dict[str, float]] = __simple_crossmut_process(
                    elitism=elitism,
                    i=i,
                    index_of_min_value=index_of_min_value,
                    pop=pop,
                    execute_crossover=execute_crossover,
                    execute_mutation=execute_mutation,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    mode=mode,
                    structure=structure,
                    m=m,
                    first_parent=first_parent,
                    second_parent=second_parent,
                    perform_gp_operators=perform_gp_operators
                )
                            
                offsprings.append(new_tree)

            # ====
            # CHANGE POPULATION
            # ====

            pop = offsprings
            parents = None
            offsprings = None
            new_tree = None
            

            # == NEXT GENERATION ==

        # == END OF EVOLUTION ==

        # ====
        # LAST FITNESS EVALUATION AND UPDATE
        # ====

        _ = __fitness_evaluation_and_update_statistics_and_result(
                mode=mode,
                linear_scaling=linear_scaling,
                pop=pop,
                pop_size=pop_size,
                stats_collector=stats_collector,
                current_gen=num_gen_post,
                verbose=verbose,
                gen_verbosity_level=gen_verbosity_level,
                result=result,
                train_set=train_set,
                test_set=test_set,
                weights_matrix_moran=weights_matrix_moran,
                moran_formula_coef=moran_formula_coef,
                use_gsgp_semantic_vector_as_train_y=use_gsgp_semantic_vector_as_train_y,
                best_tree_semantic_vector=best_tree_semantic_vector
            )

    # ===========================
    # RETURNING RESULTS
    # ===========================

    result['train_statistics'] = stats_collector['train'].build_list()
    result['test_statistics'] = stats_collector['test'].build_list()

    return result


def __fitness_evaluation_and_update_statistics_and_result(
    mode: str,
    linear_scaling: bool,
    pop: list[tuple[Node, dict[str, float]]],
    pop_size: int,
    stats_collector: dict[str, StatsCollectorSingle],
    current_gen: int,
    verbose: bool,
    gen_verbosity_level: int,
    result: dict[str, Any],
    train_set: tuple[np.ndarray, np.ndarray],
    test_set: tuple[np.ndarray, np.ndarray],
    weights_matrix_moran: list[list[float]],
    moran_formula_coef: float,
    use_gsgp_semantic_vector_as_train_y: bool,
    best_tree_semantic_vector: np.ndarray
) -> tuple[dict[str, list[float]], int, np.ndarray]:
    
    # ===========================
    # FITNESS EVALUATION
    # ===========================

    tt: tuple[dict[str, list[float]], list[np.ndarray]] = __compute_single_RMSE_value_and_replace(
            linear_scaling=linear_scaling,
            pop=pop,
            pop_size=pop_size,
            X_train=train_set[0],
            y_train=train_set[1],
            X_test=test_set[0],
            y_test=test_set[1],
            use_gsgp_semantic_vector_as_train_y=use_gsgp_semantic_vector_as_train_y,
            best_tree_semantic_vector=best_tree_semantic_vector
    )

    fit_values_dict: dict[str, list[float]] = tt[0]
    semantic_vectors: list[np.ndarray] = tt[1]

    min_value: float = min(fit_values_dict['train'])
    index_of_min_value: int = fit_values_dict['train'].index(min_value)
    best_tree_in_this_gen: Node = pop[index_of_min_value][0]
    best_tree_semantic_vector_C: np.ndarray = semantic_vectors[index_of_min_value]

    # ===========================
    # UPDATE STATISTICS
    # ===========================
    
    for dataset_type in ['train', 'test']:
        stats_collector[dataset_type].update_fitness_stat_dict(n_gen=current_gen, data=fit_values_dict[dataset_type])
    
    table: PrettyTable = PrettyTable(["Generation", "TrMin", "TeMin", "Median", "Var"])
    table.add_row([str(current_gen),
                   stats_collector['train'].get_fitness_stat(current_gen, 'min'),
                   fit_values_dict['test'][index_of_min_value],
                   stats_collector['train'].get_fitness_stat(current_gen, 'median'),
                   stats_collector['train'].get_fitness_stat(current_gen, 'var')])
    
    if verbose and current_gen > 0 and current_gen % gen_verbosity_level == 0:
        print(table)

    # ===========================
    # UPDATE BEST AND HISTORY
    # ===========================
    
    best_ind_here_totally: dict[str, Any] = {
        'Fitness': {'Train RMSE': min_value, 'Test RMSE': fit_values_dict['test'][index_of_min_value]},
        'PopIndex': index_of_min_value,
        'Generation': current_gen,
        'LogNNodes': math.log(best_tree_in_this_gen.get_n_nodes(), 10),
        'Height': best_tree_in_this_gen.get_height()
        #'NNodesHeight': float(best_tree_in_this_gen.get_n_nodes()) / float(best_tree_in_this_gen.get_height() + 1),
        #'HeightNNodes': float(best_tree_in_this_gen.get_height() + 1) / float(best_tree_in_this_gen.get_n_nodes())
    }

    if mode == 'gp' or use_gsgp_semantic_vector_as_train_y:
        best_ind_here_totally['Tree'] = TreeStructure.get_subtree_as_full_string(best_tree_in_this_gen)
    else:
        best_ind_here_totally['Tree'] = ''

    if len(result['best']) == 0:
        result['best'] = best_ind_here_totally
    else:
        if best_ind_here_totally['Fitness']['Train RMSE'] < result['best']['Fitness']['Train RMSE']:
            result['best'] = best_ind_here_totally
    
    all_n_nodes_in_this_gen: list[float] = [math.log(pop[i][0].get_n_nodes(), 10) for i in range(pop_size)]
    all_height_in_this_gen: list[float] = [float(pop[i][0].get_height()) for i in range(pop_size)]
    #all_n_nodes_height_in_this_gen: list[float] = [float(pop[i][0].get_n_nodes()) / float(pop[i][0].get_height() + 1) for i in range(pop_size)]
    #all_height_n_nodes_in_this_gen: list[float] = [float(pop[i][0].get_height() + 1) / float(pop[i][0].get_n_nodes()) for i in range(pop_size)]

    result['history'].append(
        {kk: result['best'][kk] for kk in result['best']}
        |
        {   
            'EuclideanDistanceStats': SemanticDistance.compute_stats_all_distinct_distances(semantic_vectors),
            'GlobalMoranI': SemanticDistance.global_moran_I_coef(semantic_vectors, weights_matrix_moran, moran_formula_coef),
            'LogNNodesStats': SemanticDistance.compute_stats(all_n_nodes_in_this_gen),
            'HeightStats': SemanticDistance.compute_stats(all_height_in_this_gen)
            #'NNodesHeightStats': SemanticDistance.compute_stats(all_n_nodes_height_in_this_gen),
            #'HeightNNodesStats': SemanticDistance.compute_stats(all_height_n_nodes_in_this_gen)
        }
    )

    return fit_values_dict, index_of_min_value, best_tree_semantic_vector_C


def __compute_single_RMSE_value_and_replace(
        linear_scaling: bool,
        pop: list[tuple[Node, dict[str, float]]],
        pop_size: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_gsgp_semantic_vector_as_train_y: bool,
        best_tree_semantic_vector: np.ndarray
) -> tuple[dict[str, list[float]], list[np.ndarray]]:
    fit_values_dict: dict[str, list[float]] = {'train': [], 'test': []}
    semantic_vectors: list[np.ndarray] = []

    for i in range(pop_size):
        current_individual: tuple[Node, dict[str, float]] = pop[i]
        current_tree: Node = current_individual[0]
        current_fitness: dict[str, float] = current_individual[1]
        p_train: np.ndarray = np.core.umath.clip(current_tree(X_train, dataset_type='train'), -1e+10, 1e+10)
        semantic_vectors.append(p_train)
        if len(current_fitness) != 0:
            # This individual has already been evaluated before, no need to recompute its fitness again
            new_fitness: dict[str, float] = current_fitness
        else:
            # This individual has never been evaluated, need to compute its fitness
            p_test: np.ndarray = np.core.umath.clip(current_tree(X_test, dataset_type='test'), -1e+10, 1e+10)

            if linear_scaling:
                slope, intercept = EvaluationMetrics.compute_linear_scaling(y=best_tree_semantic_vector if use_gsgp_semantic_vector_as_train_y else y_train, p=p_train)
                train_fitness: float = EvaluationMetrics.root_mean_squared_error(y=best_tree_semantic_vector if use_gsgp_semantic_vector_as_train_y else y_train, p=p_train, linear_scaling=False, slope=slope, intercept=intercept)
                test_fitness: float = EvaluationMetrics.root_mean_squared_error(y=y_test, p=p_test, linear_scaling=False, slope=slope, intercept=intercept)
            else:
                train_fitness: float = EvaluationMetrics.root_mean_squared_error(y=best_tree_semantic_vector if use_gsgp_semantic_vector_as_train_y else y_train, p=p_train, linear_scaling=False, slope=None, intercept=None)
                test_fitness: float = EvaluationMetrics.root_mean_squared_error(y=y_test, p=p_test, linear_scaling=False, slope=None, intercept=None)
            
            new_fitness: dict[str, float] = {'train': train_fitness, 'test': test_fitness}
            pop[i] = (current_tree, new_fitness)
        fit_values_dict['train'].append(new_fitness['train'])
        fit_values_dict['test'].append(new_fitness['test'])

    return fit_values_dict, semantic_vectors

def __simple_selection_process(
        is_tournament_selection: bool,
        competitor_rate: float,
        neighbors_topology: NeighborsTopology,
        coordinate: tuple[int, ...],
        all_neighborhoods_indices: dict[tuple[int, ...], list[tuple[int, ...]]]
) -> tuple[tuple[int, Node, float, float], tuple[int, Node, float, float]]:
    if not is_tournament_selection:
        competitors: list[tuple[int, Node, float, float]] = [neighbors_topology.get(idx_tuple, clone=False) for idx_tuple in all_neighborhoods_indices[coordinate]]
        competitors.sort(key=lambda x: x[0], reverse=False)
        
        if competitor_rate == 1.0:
            sampled_competitors: list[tuple[int, Node, float, float]] = competitors
        else:
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
    
    return first, second

def __simple_crossmut_process(
        elitism: bool,
        i: int,
        index_of_min_value: int,
        pop: list[tuple[Node, dict[str, float]]],
        execute_crossover: bool,
        execute_mutation: bool,
        crossover_probability: float,
        mutation_probability: float,
        mode: str,
        structure: TreeStructure,
        m: float,
        first_parent: tuple[Node, float, float],
        second_parent: tuple[Node, float, float],
        perform_gp_operators: bool
) -> tuple[Node, dict[str, float]]:
    if elitism and i == index_of_min_value:
        # If elitism, preserve the best individual to the next generation
        new_tree: tuple[Node, dict[str, float]] = pop[i]
    else:
        # == CROSSOVER ==
        if execute_crossover and random.random() < crossover_probability:
            if not perform_gp_operators and (mode == 'gsgp' or mode == 'gsgpgp'):
                cx_tree: Node = structure.geometric_semantic_single_tree_crossover(first_parent[0], second_parent[0], enable_caching=True, fix_properties=True)
            elif mode == 'gp' or perform_gp_operators:
                cx_tree: Node = structure.safe_subtree_crossover_two_children(first_parent[0], second_parent[0])[0]
            else:
                raise AttributeError(f'Invalid mode ({mode}).')
            new_tree: tuple[Node, dict[str, float]] = (cx_tree, {})
        else:
            new_tree: tuple[Node, dict[str, float]] = (first_parent[0], {'train': first_parent[1], 'test': first_parent[2]})

        # == MUTATION ==
        if execute_mutation and random.random() < mutation_probability:
            mutation_step: float = m if m != 0.0 else random.uniform(0.0, 1.0 + 1e-8)
            if not perform_gp_operators and (mode == 'gsgp' or mode == 'gsgpgp'):
                mut_tree: Node = structure.geometric_semantic_tree_mutation(new_tree[0], m=mutation_step, enable_caching=True, fix_properties=True)
            elif mode == 'gp' or perform_gp_operators:
                mut_tree: Node = structure.safe_subtree_mutation(new_tree[0])
            else:
                raise AttributeError(f'Invalid mode ({mode}).')
            new_tree = (mut_tree, {})
    
    return new_tree
