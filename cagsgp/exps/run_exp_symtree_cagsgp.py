import argparse
import datetime
import os
from collections.abc import Callable
from cagsgp.util.parallel.FakeParallelizer import FakeParallelizer
from cagsgp.util.parallel.MultiProcessingParallelizer import MultiProcessingParallelizer
from cagsgp.util.parallel.Parallelizer import Parallelizer
from cagsgp.util.parallel.ProcessPoolExecutorParallelizer import ProcessPoolExecutorParallelizer
from cagsgp.util.parallel.ThreadPoolExecutorParallelizer import ThreadPoolExecutorParallelizer
from cagsgp.util.parallel.RayParallelizer import RayParallelizer
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
        torus_dim: int,
        radius: int,
        pop_shape: tuple[int, ...],
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
        expl_pipe: str,
        duplicates_elimination: str,
        elitism: bool,
        start_seed: int,
        end_seed: int,
        gen_verbosity_level: int,
        verbose: bool
) -> None:
    
    dataset_path: str = dataset_path_folder + dataset_name + '/'
    for seed in range(start_seed, end_seed + 1):
        t: tuple[dict[str, Any], str, str] = run_symbolic_regression_with_cellular_automata_gsgp(
            pop_shape=pop_shape,
            pop_size=pop_size,
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
            expl_pipe=expl_pipe,
            duplicates_elimination=duplicates_elimination,
            torus_dim=torus_dim,
            radius=radius,
            elitism=elitism
        )
        ResultUtils.write_result_to_json(path=folder_name, path_run_id=t[1], run_id=t[2], pareto_front_dict=t[0])
    
    verbose_output: str = f'PopSize {pop_size} NumGen {num_gen} ExplPipe {expl_pipe} Dataset {dataset_name} Radius {radius} TorusDim {torus_dim} CompetitorRate {competitor_rate} COMPLETED'
    print(verbose_output)
    with open(folder_name + 'terminal_std_out.txt', 'a+') as terminal_std_out:
        terminal_std_out.write(verbose_output)
        terminal_std_out.write('\n')


if __name__ == '__main__':
    # Datasets: ['vladislavleva14', 'airfoil', 'keijzer6', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson']
    # Datasets: ['vladislavleva14', 'airfoil', 'keijzer6', 'concrete', 'slump', 'toxicity', 'yacht']
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1' + '/'
    #folder_name: str = '/home/luigirovito/python_data/' + 'CA-GSGP/' + 'results_1' + '/'
    dataset_path_folder: str = codebase_folder + 'python_data/CA-GSGP/datasets_csv/'
    #dataset_path_folder: str = '/home/luigirovito/python_data/' + 'CA-GSGP/datasets_csv/'

    # ===========================
    # COMMON AND FIXED PARAMETERS
    # ===========================

    m: float = 0.0
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

    # ===========================
    # VARIABLE PARAMETERS
    # ===========================

    parameters: list[dict[str, Any]] = []

    for dataset_name in ['vladislavleva14', 'airfoil', 'keijzer6', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson']:
        for pop_size, num_gen in [(100, 1000), (400, 250), (900, 111)]:
            for expl_pipe in ['crossmut', 'crossonly', 'mutonly']:
                for torus_dim in [0, 2]:
                    if torus_dim == 0:
                        parameters.append({'dataset_name': dataset_name,
                                           'torus_dim': torus_dim,
                                           'radius': 4,
                                           'pop_shape': (pop_size,),
                                           'pop_size': pop_size,
                                           'num_gen': num_gen,
                                           'competitor_rate': 0.0,
                                           'expl_pipe': expl_pipe})
                    elif torus_dim == 2:
                        for competitor_rate in [1.0, 0.6]:
                            for radius in [1, 2, 3]:
                                parameters.append({'dataset_name': dataset_name,
                                                   'torus_dim': torus_dim,
                                                   'radius': radius,
                                                   'pop_shape': (int(pop_size ** (1/torus_dim)), int(pop_size ** (1/torus_dim))),
                                                   'pop_size': pop_size,
                                                   'num_gen': num_gen,
                                                   'competitor_rate': competitor_rate,
                                                   'expl_pipe': expl_pipe})
    
    # ===========================
    # RUN EXPERIMENT
    # ===========================

    start_time: float = time.time()

    # = EXPERIMENT MULTIPROCESSING AND VERBOSE PARAMETERS =

    start_seed: int = 1
    end_seed: int = 100
    gen_verbosity_level: int = 50
    verbose: bool = False
    multiprocess: bool = True
    num_cores: int = os.cpu_count()
    if len(parameters) <= num_cores:
        total_num_of_param_blocks: int = 1
    else:
        if len(parameters) % num_cores == 0:
            total_num_of_param_blocks: int = int(len(parameters)/num_cores)
        else:
            total_num_of_param_blocks: int = int(len(parameters)/num_cores) + 1

    with open(folder_name + 'terminal_std_out.txt', 'a+') as terminal_std_out:
        terminal_std_out.write(str(datetime.datetime.now()))
        terminal_std_out.write('\n\n\n')
    
    # = EXPERIMENTS =

    for curr_ind_num_cores in range(total_num_of_param_blocks):
        
        parameters_start_ind: int = curr_ind_num_cores * num_cores
        parameters_end_ind: int = parameters_start_ind + num_cores if curr_ind_num_cores != total_num_of_param_blocks - 1 else len(parameters)
        parameters_temp: list[dict[str, Any]] = parameters[parameters_start_ind:parameters_end_ind]

        if not multiprocess:
            parallelizer: Parallelizer = FakeParallelizer()
        else:
            #parallelizer: Parallelizer = MultiProcessingParallelizer(len(parameters_temp))
            #parallelizer: Parallelizer = ProcessPoolExecutorParallelizer(len(parameters_temp))
            #parallelizer: Parallelizer = ThreadPoolExecutorParallelizer(len(parameters_temp))
            parallelizer: Parallelizer = RayParallelizer(len(parameters_temp))
        
        # = PARALLEL EXECUTION =

        parallel_func: Callable = partial(run_single_experiment,
                                            folder_name=folder_name,
                                            dataset_path_folder=dataset_path_folder,
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
                                            elitism=elitism,
                                            start_seed=start_seed,
                                            end_seed=end_seed,
                                            gen_verbosity_level=gen_verbosity_level,
                                            verbose=verbose
                                        )
        _ = parallelizer.parallelize(parallel_func, parameters=parameters_temp)


    end_time: float = time.time()

    execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)
    print("TOTAL TIME (minutes): " + str(execution_time_in_minutes))
