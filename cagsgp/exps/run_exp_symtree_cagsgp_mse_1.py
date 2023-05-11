import argparse
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
import time
import cProfile
from typing import Any
from cagsgp.exps.GeometricSemanticSymbolicRegressionRunner import GeometricSemanticSymbolicRegressionRunner
from cagsgp.util.ResultUtils import ResultUtils
from genepro.node import Node
from genepro.node_impl import Plus, Minus, Times, Div


if __name__ == '__main__':
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1' + '/'

    pop_size: int = 100
    num_gen: int = 10
    m: float = 2.0
    radius: int = 1
    crossover_probability: float = 0.9
    mutation_probability: float = 0.6

    dataset_name: str = 'korns12'
    neighbors_topology: str = 'matrix'
    duplicates_elimination: str = 'nothing'
    pop_shape: tuple[int, ...] = (10, 10)

    operators: list[Node] = [Plus(fix_properties=True), Minus(fix_properties=True), Times(fix_properties=True), Div(fix_properties=True)]
    
    low_erc: float = -100.0
    high_erc: float = 100.0 + 1e-4

    n_constants: int = 100

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=str, help='Seed to be adopted for the experiment run.', required=False)

    args: argparse.Namespace = parser.parse_args()
    seed: str = args.seed

    if seed is None:
        seed_list: list[int] = list(range(1, 1 + 1))
    else:
        seed_list: list[int] = [int(i) for i in seed.split(",")]

    start_time: float = time.time()
        
    for seed in seed_list:
        for max_depth in [6]:
            pr = cProfile.Profile()
            #pr.enable()
            t: tuple[dict[str, Any], str] = GeometricSemanticSymbolicRegressionRunner.run_symbolic_regression_with_cellular_automata_gsgp(
                pop_size=pop_size,
                pop_shape=pop_shape,
                num_gen=num_gen,
                max_depth=max_depth,
                operators=operators,
                low_erc=low_erc,
                high_erc=high_erc,
                n_constants=n_constants,
                dataset_name=dataset_name,
                dataset_path=None,
                seed=seed,
                multiprocess=False,
                verbose=True,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                m=m,
                store_in_cache=True,
                fix_properties=True,
                duplicates_elimination=duplicates_elimination,
                neighbors_topology=neighbors_topology,
                radius=radius
            )
            #pr.disable()
            #pr.print_stats(sort='tottime')
            ResultUtils.write_result_to_json(path=folder_name, run_id=t[1], pareto_front_dict=t[0])
            print("NEXT")

    end_time: float = time.time()
    execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)
    print("TOTAL TIME (minutes): " + str(execution_time_in_minutes))
