import os
from typing import Any
import scipy.stats as stats
import statistics
from cagsgp.util.ResultUtils import ResultUtils
from cagsgp.util.StringUtils import StringUtils


class TableUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_table_wilcoxon_datasets_cellular_vs_tournament(
            folder_name: str,
            seed_list: list[int],
            topologies_radius_shapes: list[tuple[str, int, tuple[int, ...]]],
            dataset_names: list[str],
            pop_size: int,
            num_gen: int,
            last_gen: int,
            max_depth: int,
            duplicates_elimination: str,
            crossover_probability: float,
            mutation_probability: float,
            m: float,
            generation_strategy: str,
            elitism: bool
    ) -> None:
            
        tab_str: str = ''

        # K1: Method - K2: DatasetName - K3: SplitType - V: list of best RMSE across all seeds
        fitness: dict[str, dict[str, dict[str, list[float]]]] = {}

        # ===================
        # Load tournament-4 baseline result
        # ===================

        fitness['Tournament-4'] = {}
        for dataset_name in dataset_names:
            fitness['Tournament-4'][dataset_name] = {'Train': [], 'Test': []}
            for seed in seed_list:
                d: dict[str, Any] = ResultUtils.read_single_json_file(
                    folder_name=folder_name,
                    result_file_type='b',
                    pop_size=pop_size,
                    num_gen=num_gen,
                    max_depth=max_depth,
                    neighbors_topology='tournament',
                    dataset_name=dataset_name,
                    duplicates_elimination=duplicates_elimination,
                    pop_shape=(pop_size,),
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    m=m,
                    radius=4,
                    generation_strategy=generation_strategy,
                    elitism=elitism,
                    seed=seed
                )
                best: dict[str, Any] = d['history'][last_gen]
                fitness['Tournament-4'][dataset_name]['Train'].append(best['Fitness']['Train RMSE'])
                fitness['Tournament-4'][dataset_name]['Test'].append(best['Fitness']['Test RMSE'])
    
        # ===================
        # Load other methods
        # ===================

        for topology, radius, shape in topologies_radius_shapes:
            current_method: str = StringUtils.only_first_char_upper(topology)+'-'+str(radius)
            fitness[current_method] = {}
            for dataset_name in dataset_names:
                fitness[current_method][dataset_name] = {'Train': [], 'Test': []}
                for seed in seed_list:
                    d: dict[str, Any] = ResultUtils.read_single_json_file(
                        folder_name=folder_name,
                        result_file_type='b',
                        pop_size=pop_size,
                        num_gen=num_gen,
                        max_depth=max_depth,
                        neighbors_topology=topology,
                        dataset_name=dataset_name,
                        duplicates_elimination=duplicates_elimination,
                        pop_shape=shape,
                        crossover_probability=crossover_probability,
                        mutation_probability=mutation_probability,
                        m=m,
                        radius=radius,
                        generation_strategy=generation_strategy,
                        elitism=elitism,
                        seed=seed
                    )
                    best: dict[str, Any] = d['history'][last_gen]
                    fitness[current_method][dataset_name]['Train'].append(best['Fitness']['Train RMSE'])
                    fitness[current_method][dataset_name]['Test'].append(best['Fitness']['Test RMSE'])

        # ===================
        # Print table content
        # ===================

        for method in ['Line-1', 'Line-2', 'Line-3', 'Line-4',
                        'Matrix-1', 'Matrix-2', 'Matrix-3', 'Matrix-4',
                        'Cube-1', 'Cube-2', 'Cube-3', 'Cube-4']:
            tab_str += '{' + method + '}' + '\n'
            for dataset_name in ['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht']:
                for split_type in ['Train', 'Test']:
                    a: list[float] = fitness[method][dataset_name][split_type]
                    b: list[float] = fitness['Tournament-4'][dataset_name][split_type]
                    p: float = round(stats.wilcoxon(a, b, alternative="less").pvalue, 2) if a != b else 1.0
                    tab_str += '& ' + str(p) + ' '
            tab_str += '\n'
            tab_str += '\\\\'
            tab_str += '\n'
        print(tab_str)


    @staticmethod
    def print_table_wilcoxon_medianrmse_datasets_cellular_vs_tournament_for_single_split_type(
            folder_name: str,
            split_type: str,
            seed_list: list[int],
            topologies_radius_shapes: list[tuple[str, int, tuple[int, ...]]],
            dataset_names: list[str],
            pop_size: int,
            num_gen: int,
            last_gen: int,
            max_depth: int,
            duplicates_elimination: str,
            crossover_probability: float,
            mutation_probability: float,
            m: float,
            generation_strategy: str,
            elitism: bool
    ) -> None:
        
        print(split_type)
        tab_str: str = ''
        split_type: str = StringUtils.only_first_char_upper(split_type)

        # K1: Method - K2: DatasetName - V: list of best RMSE across all seeds
        fitness: dict[str, dict[str, list[float]]] = {}
        tournament_pressure: str = str(5)

        # ===================
        # Load tournament baseline result
        # ===================

        fitness['Tournament-'+tournament_pressure] = {}
        for dataset_name in dataset_names:
            fitness['Tournament-'+tournament_pressure][dataset_name] = []
            for seed in seed_list:
                d: dict[str, Any] = ResultUtils.read_single_json_file(
                    folder_name=folder_name,
                    result_file_type='b',
                    pop_size=pop_size,
                    num_gen=num_gen,
                    max_depth=max_depth,
                    neighbors_topology='tournament',
                    dataset_name=dataset_name,
                    duplicates_elimination=duplicates_elimination,
                    pop_shape=(pop_size,),
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    m=m,
                    radius=int(tournament_pressure),
                    generation_strategy=generation_strategy,
                    elitism=elitism,
                    seed=seed
                )
                best: dict[str, Any] = d['history'][last_gen]
                fitness['Tournament-'+tournament_pressure][dataset_name].append(best['Fitness'][split_type+' RMSE'])
    
        # ===================
        # Load other methods
        # ===================

        for topology, radius, shape in topologies_radius_shapes:
            current_method: str = StringUtils.only_first_char_upper(topology)+'-'+str(radius)
            fitness[current_method] = {}
            for dataset_name in dataset_names:
                fitness[current_method][dataset_name] = []
                for seed in seed_list:
                    d: dict[str, Any] = ResultUtils.read_single_json_file(
                        folder_name=folder_name,
                        result_file_type='b',
                        pop_size=pop_size,
                        num_gen=num_gen,
                        max_depth=max_depth,
                        neighbors_topology=topology,
                        dataset_name=dataset_name,
                        duplicates_elimination=duplicates_elimination,
                        pop_shape=shape,
                        crossover_probability=crossover_probability,
                        mutation_probability=mutation_probability,
                        m=m,
                        radius=radius,
                        generation_strategy=generation_strategy,
                        elitism=elitism,
                        seed=seed
                    )
                    best: dict[str, Any] = d['history'][last_gen]
                    fitness[current_method][dataset_name].append(best['Fitness'][split_type+' RMSE'])

        # ===================
        # Print table content
        # ===================

        for method in ['Tournament-'+tournament_pressure, 'Line-1', 'Line-2', 'Line-3', 'Line-4',
                        'Matrix-1', 'Matrix-2', 'Matrix-3', 'Matrix-4',
                        'Cube-1', 'Cube-2']:
            tab_str += '{' + method + '}' + '\n'
            for dataset_name in ['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht']:
                a: list[float] = fitness[method][dataset_name]
                b: list[float] = fitness['Tournament-'+tournament_pressure][dataset_name]
                p: float = round(stats.wilcoxon(a, b, alternative="less").pvalue, 2) if a != b else 1.0
                is_meaningful: bool = p < 0.05
                tab_str += '& ' + str(round(statistics.median(a), 2)) + ('{\\textbf{*}}' if is_meaningful else '') + ' '
            tab_str += '\n'
            tab_str += '\\\\'
            tab_str += '\n'
        
        print(tab_str)


if __name__ == '__main__':
    # Datasets: ['airfoil', 'bioav', 'concrete', 'parkinson', 'ppb', 'slump', 'toxicity', 'vladislavleva-14', 'yacht']
    # Datasets: ['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht']
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1' + '/'

    TableUtils.print_table_wilcoxon_medianrmse_datasets_cellular_vs_tournament_for_single_split_type(folder_name=folder_name,
                                              split_type='Test',
                                              seed_list=list(range(1, 100 + 1)), 
                                              topologies_radius_shapes=[('line',1,(100,)),
                                                                        ('line',2,(100,)),
                                                                        ('line',3,(100,)),
                                                                        ('line',4,(100,)),
                                                                        ('matrix',1,(10,10)),
                                                                        ('matrix',2,(10,10)),
                                                                        ('matrix',3,(10,10)),
                                                                        ('matrix',4,(10,10)),
                                                                        ('cube',1,(4,5,5)),
                                                                        ('cube',2,(4,5,5))],
                                              dataset_names=['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht'],
                                              pop_size=100,
                                              num_gen=1000,
                                              last_gen=1000,
                                              max_depth=6,
                                              duplicates_elimination='nothing',
                                              crossover_probability=0.90,
                                              mutation_probability=0.50,
                                              m=0.0,
                                              generation_strategy='half',
                                              elitism=True
                                              )
    