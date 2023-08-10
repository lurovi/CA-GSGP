import argparse
import os
from collections.abc import Callable
from cagsgp.util.parallel.FakeParallelizer import FakeParallelizer
from cagsgp.util.parallel.MultiProcessingParallelizer import MultiProcessingParallelizer
from cagsgp.util.parallel.Parallelizer import Parallelizer
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
import time
import cProfile
from typing import Any
from cagsgp.nsgp.runner.GeometricSemanticSymbolicRegressionRunner import run_symbolic_regression_with_cellular_automata_gsgp
from cagsgp.util.ResultUtils import ResultUtils
from genepro.node import Node
from genepro.node_impl import Plus, Minus, Times, Div
from functools import partial


def run_single_experiment(
        folder_name: str,
        dataset_name: str,
        dataset_path_folder: str,
        neighbors_topology: str,
        radius: int,
        pop_size: int,
        pop_shape: int,
        num_gen: int,
        max_depth: int,
        generation_strategy: str,
        operators: list[Node],
        low_erc: float,
        high_erc: float,
        n_constants: int,
        crossover_probability: float,
        mutation_probability: float,
        m: float,
        competitor_rate: float,
        duplicates_elimination: str,
        elitism: bool,
        start_seed: int,
        end_seed: int,
        gen_verbosity_level: int,
        verbose: bool
) -> None:
    
    dataset_path = dataset_path_folder + dataset_name + '/'
    for seed in range(start_seed, end_seed + 1):
        t: tuple[dict[str, Any], str] = run_symbolic_regression_with_cellular_automata_gsgp(
            pop_size=pop_size,
            pop_shape=pop_shape,
            num_gen=num_gen,
            max_depth=max_depth,
            generation_strategy=generation_strategy,
            operators=operators,
            low_erc=low_erc,
            high_erc=high_erc,
            n_constants=n_constants,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            seed=seed,
            verbose=verbose,
            gen_verbosity_level=gen_verbosity_level,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            m=m,
            competitor_rate=competitor_rate,
            duplicates_elimination=duplicates_elimination,
            neighbors_topology=neighbors_topology,
            radius=radius,
            elitism=elitism
        )
        ResultUtils.write_result_to_json(path=folder_name, run_id=t[1], pareto_front_dict=t[0])
    
    print(f'Dataset {dataset_name} Radius {radius} NeighborsTopology {neighbors_topology} COMPLETED')


def run_experiment_all_datasets(
        folder_name: str,
        parameters: list[dict[str, Any]],
        dataset_path_folder: str,
        pop_size: int,
        num_gen: int,
        max_depth: int,
        generation_strategy: str,
        operators: list[Node],
        low_erc: float,
        high_erc: float,
        n_constants: int,
        crossover_probability: float,
        mutation_probability: float,
        m: float,
        competitor_rate: float,
        duplicates_elimination: str,
        elitism: bool,
        start_seed: int,
        end_seed: int,
        gen_verbosity_level: int,
        multiprocess: bool,
        verbose: bool
) -> None:
    if not multiprocess:
        parallelizer: Parallelizer = FakeParallelizer()
    else:
        parallelizer: Parallelizer = MultiProcessingParallelizer(len(parameters) if os.cpu_count() >= len(parameters) else os.cpu_count())
    parallel_func: Callable = partial(run_single_experiment,
                                        folder_name=folder_name,
                                        dataset_path_folder=dataset_path_folder,
                                        pop_size=pop_size,
                                        num_gen=num_gen,
                                        max_depth=max_depth,
                                        generation_strategy=generation_strategy,
                                        operators=operators,
                                        low_erc=low_erc,
                                        high_erc=high_erc,
                                        n_constants=n_constants,
                                        crossover_probability=crossover_probability,
                                        mutation_probability=mutation_probability,
                                        m=m,
                                        competitor_rate=competitor_rate,
                                        duplicates_elimination=duplicates_elimination,
                                        elitism=elitism,
                                        start_seed=start_seed,
                                        end_seed=end_seed,
                                        gen_verbosity_level=gen_verbosity_level,
                                        verbose=verbose
                                    )
    _ = parallelizer.parallelize(parallel_func, parameters=parameters)


if __name__ == '__main__':
    # Datasets: ['airfoil', 'bioav', 'concrete', 'parkinson', 'ppb', 'slump', 'toxicity', 'vladislavleva-14', 'yacht']
    # Datasets: ['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht']
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1.5_0.4' + '/'
    #folder_name: str = '/home/luigirovito/python_data/' + 'CA-GSGP/' + 'results_1.5_0.4' + '/'
    dataset_path_folder: str = codebase_folder + 'python_data/CA-GSGP/datasets_csv/'
    #dataset_path_folder: str = '/home/luigirovito/python_data/' + 'CA-GSGP/datasets_csv/'

    pop_size: int = 100
    num_gen: int = 1000
    m: float = 0.0
    competitor_rate: float = 0.40
    max_depth: int = 6
    elitism: bool = True
    generation_strategy: str = 'half'
    crossover_probability: float = 0.90
    mutation_probability: float = 0.50
    duplicates_elimination: str = 'nothing'

    operators: list[Node] = [Plus(fix_properties=True), Minus(fix_properties=True), Times(fix_properties=True), Div(fix_properties=True)]
    low_erc: float = -100.0
    high_erc: float = 100.0 + 1e-8
    n_constants: int = 100

    parameters: list[dict[str, Any]] = []
    for dataset_name in ['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht']:
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 4,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 8,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 12,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 16,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 20,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 5,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 10,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 15,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 20,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 25,
        #                   'pop_shape': (pop_size,)})
        #parameters.append({'dataset_name': dataset_name,
        #                   'neighbors_topology': 'tournament',
        #                   'radius': 30,
        #                   'pop_shape': (pop_size,)})
        # parameters.append({'dataset_name': dataset_name,
        #                     'neighbors_topology': 'line',
        #                     'radius': 1,
        #                     'pop_shape': (100,)})
        # parameters.append({'dataset_name': dataset_name,
        #                    'neighbors_topology': 'line',
        #                    'radius': 2,
        #                    'pop_shape': (100,)})
        # parameters.append({'dataset_name': dataset_name,
        #                     'neighbors_topology': 'line',
        #                     'radius': 3,
        #                     'pop_shape': (100,)})
        # parameters.append({'dataset_name': dataset_name,
        #                     'neighbors_topology': 'line',
        #                     'radius': 4,
        #                     'pop_shape': (100,)})
        # parameters.append({'dataset_name': dataset_name,
        #                     'neighbors_topology': 'matrix',
        #                     'radius': 1,
        #                     'pop_shape': (10,10)})
        parameters.append({'dataset_name': dataset_name,
                           'neighbors_topology': 'matrix',
                           'radius': 2,
                           'pop_shape': (10,10)})
        # parameters.append({'dataset_name': dataset_name,
        #                     'neighbors_topology': 'matrix',
        #                     'radius': 3,
        #                     'pop_shape': (10,10)})
        # parameters.append({'dataset_name': dataset_name,
        #                     'neighbors_topology': 'matrix',
        #                     'radius': 4,
        #                     'pop_shape': (10,10)})
        # parameters.append({'dataset_name': dataset_name,
        #                     'neighbors_topology': 'cube',
        #                     'radius': 1,
        #                     'pop_shape': (4,5,5)})
        # parameters.append({'dataset_name': dataset_name,
        #                    'neighbors_topology': 'cube',
        #                    'radius': 2,
        #                    'pop_shape': (4,5,5)})
    

    start_time: float = time.time()

    run_experiment_all_datasets(
        folder_name=folder_name,
        parameters=parameters,
        dataset_path_folder=dataset_path_folder,
        pop_size=pop_size,
        num_gen=num_gen,
        max_depth=max_depth,
        generation_strategy=generation_strategy,
        operators=operators,
        low_erc=low_erc,
        high_erc=high_erc,
        n_constants=n_constants,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        m=m,
        competitor_rate=competitor_rate,
        duplicates_elimination=duplicates_elimination,
        elitism=elitism,
        start_seed=1,
        end_seed=1,
        gen_verbosity_level=num_gen,
        multiprocess=False,
        verbose=True
    )

    end_time: float = time.time()

    execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)
    print("TOTAL TIME (minutes): " + str(execution_time_in_minutes))
