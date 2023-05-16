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
from cagsgp.exps.GeometricSemanticSymbolicRegressionRunner import *
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
        duplicates_elimination: str,
        elitism: bool
) -> None:
    
    dataset_path = dataset_path_folder + dataset_name + '/'
    for seed in range(1, 1 + 1):
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
            verbose=True,
            gen_verbosity_level=1,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            m=m,
            duplicates_elimination=duplicates_elimination,
            neighbors_topology=neighbors_topology,
            radius=radius,
            elitism=elitism
        )
        ResultUtils.write_result_to_json(path=folder_name, run_id=t[1], pareto_front_dict=t[0])
    
    print(f'Dataset {dataset_name} Radius {radius} NeighborsTopology {neighbors_topology} COMPLETED')


def run_experiment_all_datasets(
        folder_name: str,
        dataset_names: list[str],
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
        duplicates_elimination: str,
        elitism: bool
) -> None:
    parallelizer: Parallelizer = FakeParallelizer()
    #parallelizer: Parallelizer = MultiProcessingParallelizer(len(dataset_names))
    parallel_func: Callable = partial(run_single_experiment,
                                        folder_name=folder_name,
                                        dataset_path_folder=dataset_path_folder,
                                        neighbors_topology=neighbors_topology,
                                        radius=radius,
                                        pop_size=pop_size,
                                        pop_shape=pop_shape,
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
                                        duplicates_elimination=duplicates_elimination,
                                        elitism=elitism
                                    )
    _ = parallelizer.parallelize(parallel_func, [{'dataset_name': dataset_name} for dataset_name in dataset_names])


if __name__ == '__main__':
    # Datasets: ['airfoil', 'bioav', 'concrete', 'parkinson', 'ppb', 'slump', 'toxicity', 'vladislavleva-14', 'yacht']
    dataset_names: list[str] = ['airfoil']
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1' + '/'
    #folder_name: str = '/mnt/data/lrovito/' + 'CA-GSGP/' + 'results_1' + '/'
    dataset_path_folder: str = codebase_folder + 'python_data/CA-GSGP/datasets_csv/'
    #dataset_path_folder: str = '/mnt/data/lrovito/' + 'CA-GSGP/datasets_csv/'

    pop_size: int = 100
    pop_shape: tuple[int, ...] = (100,)
    num_gen: int = 100
    m: float = 0.0
    max_depth: int = 6
    elitism: bool = True
    generation_strategy: str = 'half'
    crossover_probability: float = 0.9
    mutation_probability: float = 0.5
    duplicates_elimination: str = 'nothing'

    operators: list[Node] = [Plus(fix_properties=True), Minus(fix_properties=True), Times(fix_properties=True), Div(fix_properties=True)]
    low_erc: float = -100.0
    high_erc: float = 100.0 + 1e-8
    n_constants: int = 100

    start_time: float = time.time()

    for neighbors_topology in ['tournament']:
        for radius in [4]:    
            run_experiment_all_datasets(
                folder_name=folder_name,
                dataset_names=dataset_names,
                dataset_path_folder=dataset_path_folder,
                neighbors_topology=neighbors_topology,
                radius=radius,
                pop_size=pop_size,
                pop_shape=pop_shape,
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
                duplicates_elimination=duplicates_elimination,
                elitism=elitism
            )

    end_time: float = time.time()
    execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)
    print("TOTAL TIME (minutes): " + str(execution_time_in_minutes))
