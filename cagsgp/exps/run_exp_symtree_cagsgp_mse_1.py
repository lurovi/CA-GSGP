import argparse
import os
import time
from typing import Any
from cagsgp.exps.GeometricSemanticSymbolicRegressionRunner import GeometricSemanticSymbolicRegression
from cagsgp.util.ResultUtils import ResultUtils
from genepro.node import Node
from genepro.node_impl import Plus, Minus, Times, Div, Square, Max


if __name__ == '__main__':
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_2' + '/'

    pop_size: int = 100
    num_gen: int = 4
    m: float = 0.5
    mutation_probability: float = 0.6

    dataset_name: str = 'vladislavleva4'
    neighbors_topology: str = 'matrix'
    duplicates_elimination: str = 'semantic'
    pop_shape: tuple[int, ...] = (10, 10)

    operators: list[Node] = [Plus(), Minus(), Times(), Div(), Square(), Max()]
    
    low_erc: float = -1.0
    high_erc: float = 1.0 + 1e-4

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=str, help='Seed to be adopted for the experiment run.', required=False)

    args: argparse.Namespace = parser.parse_args()
    seed: str = args.seed

    if seed is None:
        seed_list: list[int] = list(range(1, 2 + 1))
    else:
        seed_list: list[int] = [int(i) for i in seed.split(",")]

    start_time: float = time.time()
        
    for seed in seed_list:
        for max_depth in [3]:

            t: tuple[dict[str, Any], str] = GeometricSemanticSymbolicRegression.run_symbolic_regression_with_cellular_automata_gsgp(
                pop_size=pop_size,
                pop_shape=pop_shape,
                num_gen=num_gen,
                max_depth=max_depth,
                operators=operators,
                low_erc=low_erc,
                high_erc=high_erc,
                dataset_name=dataset_name,
                dataset_path=None,
                seed=seed,
                multiprocess=False,
                verbose=True,
                mutation_probability=mutation_probability,
                m=m,
                store_in_cache=True,
                duplicates_elimination=duplicates_elimination,
                neighbors_topology=neighbors_topology
            )

            ResultUtils.write_result_to_json(path=folder_name, run_id=t[1], pareto_front_dict=t[0])
            print("NEXT")

    end_time: float = time.time()
    execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)
    print("TOTAL TIME (minutes): " + str(execution_time_in_minutes))
