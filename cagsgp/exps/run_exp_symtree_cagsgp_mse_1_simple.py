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
    dataset_names: list[str] = ['airfoil', 'bioav', 'concrete', 'parkinson', 'ppb', 'slump', 'toxicity', 'vladislavleva-14', 'yacht']
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1' + '/'

    pop_size: int = 100
    num_gen: int = 100
    m: float = 2.0
    max_depth: int = 6
    generation_strategy: str = 'half'
    crossover_probability: float = 0.9
    mutation_probability: float = 0.6

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

    for neighbors_topology in ['matrix']:
        for radius in [1]:
            for dataset_name in ['parkinson']:    
                for seed in [1]:
                    pr = cProfile.Profile()
                    #pr.enable()
                    dataset_path = codebase_folder + 'python_data/CA-GSGP/' + 'datasets_csv/' + dataset_name + '/'
                    t: tuple[dict[str, Any], str] = GeometricSemanticSymbolicRegressionRunner.run_symbolic_regression_with_cellular_automata_gsgp(
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
                        multiprocess=False,
                        verbose=True,
                        gen_verbosity_level=1,
                        crossover_probability=crossover_probability,
                        mutation_probability=mutation_probability,
                        m=m,
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
