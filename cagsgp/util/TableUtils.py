import os
from typing import Any
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
import statistics
import numpy as np
from cagsgp.util.ResultUtils import ResultUtils
from cagsgp.util.StringUtils import StringUtils


class TableUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_table_wilcoxon_datasets_cellular_vs_tournament(
            folder_name: str,
            seed_list: list[int],
            mode: str,
            linear_scaling: bool,
            torusdim_radius_shapes: list[tuple[int, int, tuple[int, ...]]],
            dataset_names: list[str],
            pop_size: int,
            num_gen: int,
            last_gen: int,
            max_depth: int,
            expl_pipe: str,
            competitor_rate: float,
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

        fitness[StringUtils.concat('TD', str(0))+'-'+str(4)] = {}
        for dataset_name in dataset_names:
            fitness[StringUtils.concat('TD', str(0))+'-'+str(4)][dataset_name] = {'Train': [], 'Test': []}
            for seed in seed_list:
                d: dict[str, Any] = ResultUtils.read_single_json_file(
                    folder_name=folder_name,
                    mode=mode,
                    linear_scaling=linear_scaling,
                    result_file_type='b',
                    pop_size=pop_size,
                    num_gen=num_gen,
                    max_depth=max_depth,
                    torus_dim=0,
                    dataset_name=dataset_name,
                    expl_pipe=expl_pipe,
                    competitor_rate=competitor_rate,
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
                fitness[StringUtils.concat('TD', str(0))+'-'+str(4)][dataset_name]['Train'].append(best['Fitness']['Train RMSE'])
                fitness[StringUtils.concat('TD', str(0))+'-'+str(4)][dataset_name]['Test'].append(best['Fitness']['Test RMSE'])
    
        # ===================
        # Load other methods
        # ===================

        for torus_dim, radius, shape in torusdim_radius_shapes:
            current_method: str = StringUtils.concat('TD', str(torus_dim))+'-'+str(radius)
            fitness[current_method] = {}
            for dataset_name in dataset_names:
                fitness[current_method][dataset_name] = {'Train': [], 'Test': []}
                for seed in seed_list:
                    d: dict[str, Any] = ResultUtils.read_single_json_file(
                        folder_name=folder_name,
                        mode=mode,
                        linear_scaling=linear_scaling,
                        result_file_type='b',
                        pop_size=pop_size,
                        num_gen=num_gen,
                        max_depth=max_depth,
                        torus_dim=torus_dim,
                        dataset_name=dataset_name,
                        expl_pipe=expl_pipe,
                        competitor_rate=competitor_rate,
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

        cgsgp_sel_method: str = 'w/\\fullN' if competitor_rate == 1.0 else 'w/\\partialN{' + str(round(competitor_rate, 2)) +'}'
        map_method_command: dict[str, str] = {'TD0-4': 'GSGP',
                                              'TD2-1': '\\toroid{'+'2'+'}{'+'1'+'} '+cgsgp_sel_method,
                                              'TD2-2': '\\toroid{'+'2'+'}{'+'2'+'} '+cgsgp_sel_method,
                                              'TD2-3': '\\toroid{'+'2'+'}{'+'3'+'} '+cgsgp_sel_method}
        map_expl_pipe_command: dict[str, str] = {'crossmut': '\\CxMut', 'crossonly': '\\Cx', 'mutonly': '\\Mut'}
        
        for method in ['TD2-1', 'TD2-2', 'TD2-3']:
            tab_str += '{' + map_method_command[method] + '}' + '\n'
            for dataset_name in ['vladislavleva14', 'keijzer6', 'airfoil', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson']:
                for split_type in ['Train', 'Test']:
                    a: list[float] = fitness[method][dataset_name][split_type]
                    b: list[float] = fitness[StringUtils.concat('TD', str(0))+'-'+str(4)][dataset_name][split_type]
                    p: float = round(stats.wilcoxon(a, b, alternative="less").pvalue, 2) if a != b else 1.0
                    tab_str += '& ' + str(p) + ' '
            tab_str += '\n'
            tab_str += '\\\\'
            tab_str += '\n'
        print(tab_str)


    @staticmethod
    def print_table_wilcoxon_medianrmse_datasets_cellular_vs_tournament_for_single_split_type(
            folder_name: str,
            mode: str,
            linear_scaling: bool,
            split_type: str,
            seed_list: list[int],
            tournament_pressures: list[int],
            bonferroni_correction: bool,
            wilcoxon_only_with_baseline: bool,
            torusdim_radius_shapes: list[tuple[int, int, tuple[int, ...]]],
            dataset_names: list[str],
            pop_size: int,
            num_gen: int,
            last_gen: int,
            max_depth: int,
            expl_pipe: str,
            competitor_rate: float,
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

        # ===================
        # Load tournament baseline result
        # ===================

        for tournament_pressure in tournament_pressures:
            fitness['TD0-'+str(tournament_pressure)] = {}
            for dataset_name in dataset_names:
                fitness['TD0-'+str(tournament_pressure)][dataset_name] = []
                for seed in seed_list:
                    d: dict[str, Any] = ResultUtils.read_single_json_file(
                        folder_name=folder_name,
                        mode=mode,
                        linear_scaling=linear_scaling,
                        result_file_type='b',
                        pop_size=pop_size,
                        num_gen=num_gen,
                        max_depth=max_depth,
                        torus_dim=0,
                        dataset_name=dataset_name,
                        expl_pipe=expl_pipe,
                        competitor_rate=competitor_rate,
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
                    fitness['TD0-'+str(tournament_pressure)][dataset_name].append(best['Fitness'][split_type+' RMSE'])
    
        # ===================
        # Load other methods
        # ===================

        for torus_dim, radius, shape in torusdim_radius_shapes:
            current_method: str = StringUtils.concat('TD', str(torus_dim))+'-'+str(radius)
            fitness[current_method] = {}
            for dataset_name in dataset_names:
                fitness[current_method][dataset_name] = []
                for seed in seed_list:
                    d: dict[str, Any] = ResultUtils.read_single_json_file(
                        folder_name=folder_name,
                        result_file_type='b',
                        mode=mode,
                        linear_scaling=linear_scaling,
                        pop_size=pop_size,
                        num_gen=num_gen,
                        max_depth=max_depth,
                        torus_dim=torus_dim,
                        dataset_name=dataset_name,
                        expl_pipe=expl_pipe,
                        competitor_rate=competitor_rate,
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

        cgsgp_sel_method: str = 'w/\\fullN' if competitor_rate == 1.0 else 'w/\\partialN{' + str(round(competitor_rate, 2)) +'}'
        map_method_command: dict[str, str] = {'TD0-4': 'GSGP',
                                              'TD2-1': '\\toroid{'+'2'+'}{'+'1'+'} '+cgsgp_sel_method,
                                              'TD2-2': '\\toroid{'+'2'+'}{'+'2'+'} '+cgsgp_sel_method,
                                              'TD2-3': '\\toroid{'+'2'+'}{'+'3'+'} '+cgsgp_sel_method}
        map_expl_pipe_command: dict[str, str] = {'crossmut': '\\CxMut', 'crossonly': '\\Cx', 'mutonly': '\\Mut'}

        all_methods: list[str] = [] #['TD0-'+str(tournament_pressure) for tournament_pressure in tournament_pressures]
        if len(torusdim_radius_shapes) > 0:
            all_methods = all_methods + ['TD0-4', 'TD2-1', 'TD2-2', 'TD2-3']
        
        for method in all_methods:
            tab_str += '{' + map_method_command[method] + '}' + '\n'
            
            for dataset_name in ['vladislavleva14', 'keijzer6', 'airfoil', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson']:
                a: list[float] = fitness[method][dataset_name]
                outperformed_methods: list[str] = []
                all_p_values: list[float] = []
                is_the_wilcoxon_test_meaningful: bool = False
                all_results_from_competitor_methods: list[list[float]] = []
                all_results_from_competitor_methods.append(a)

                for method_2 in [mmm for mmm in all_methods if mmm != method]:
                    b: list[float] = fitness[method_2][dataset_name]
                    all_results_from_competitor_methods.append(b)
                    p: float = round(stats.wilcoxon(a, b, alternative="less").pvalue, 2) if a != b else 1.0

                    all_p_values.append(p)
                    is_meaningful: bool = p < 0.05
                    if is_meaningful:
                        outperformed_methods.append(method_2)
                        if method_2 == StringUtils.concat('TD', str(0))+'-'+str(4):
                            is_the_wilcoxon_test_meaningful = True
                
                kruskal_statistic, kruskal_pvalue = stats.kruskal(*all_results_from_competitor_methods)
                is_the_kruskal_test_meaningful: bool = kruskal_pvalue < 0.05

                if wilcoxon_only_with_baseline:
                    tab_str += '& ' + str(round(statistics.median(a), 2))
                    if is_the_wilcoxon_test_meaningful:
                        tab_str += '{$^{\\scalebox{0.90}{'
                        tab_str += '\\textbf{'+ '*' +  '},'
                        tab_str = tab_str[:-1]
                        tab_str += '}'+'}$}'
                else:
                    if bonferroni_correction:
                        reject_bonferroni, pvals_corrected_bonferroni, _, _ = multipletests(all_p_values, alpha=0.05, method='holm')
                        is_the_bonferroni_test_meaningful: bool = np.sum(reject_bonferroni) == len(all_p_values)
                        
                        tab_str += '& ' + ('\\bfseries ' if is_the_wilcoxon_test_meaningful else '') + str(round(statistics.median(a), 2))
                        if is_the_kruskal_test_meaningful and is_the_bonferroni_test_meaningful:
                            tab_str += '{$^{\\scalebox{0.90}{'
                            tab_str += '\\textbf{'+ '*' +  '},'
                            tab_str = tab_str[:-1]
                            tab_str += '}'+'}$}'
                    
                    else:
                        if is_the_kruskal_test_meaningful and len(outperformed_methods) == len(all_methods) - 1:
                            tab_str += '& ' + str(round(statistics.median(a), 2))
                            tab_str += '{$^{\\scalebox{0.90}{'
                            tab_str += '\\textbf{'+ '*' +  '},'
                            tab_str = tab_str[:-1]
                            tab_str += '}'+'}$}'
                        else:
                            tab_str += '& ' + str(round(statistics.median(a), 2))
                
                tab_str += ' '
            
            tab_str += '\n'
            tab_str += '\\\\'
            tab_str += '\n'
        
        print(tab_str)



if __name__ == '__main__':
    # Datasets: ['vladislavleva14', 'keijzer6', 'airfoil', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson']
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1' + '/'

    linear_scaling: bool = False
    mode: str = 'gsgp'
    pop_size: int = 100
    num_gen: int = 1000
    competitor_rate: float = 0.6
    expl_pipe: str = 'mutonly'
    torus_dim: int = 2
    pop_shape: tuple[int, ...] = (int(pop_size ** (1/torus_dim)), int(pop_size ** (1/torus_dim)))
    

    TableUtils.print_table_wilcoxon_medianrmse_datasets_cellular_vs_tournament_for_single_split_type(folder_name=folder_name,
                                              split_type='Test',
                                              mode=mode,
                                              linear_scaling=linear_scaling,
                                              seed_list=list(range(1, 100 + 1)),
                                              tournament_pressures=[],
                                              bonferroni_correction=True,
                                              wilcoxon_only_with_baseline=False,
                                              #torusdim_radius_shapes=[],
                                              torusdim_radius_shapes=[(0,4,(pop_size,)),
                                                                        (torus_dim,1,pop_shape),
                                                                        (torus_dim,2,pop_shape),
                                                                        (torus_dim,3,pop_shape)
                                                                        ],
                                              dataset_names=['vladislavleva14', 'keijzer6', 'airfoil', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson'],
                                              pop_size=pop_size,
                                              num_gen=num_gen,
                                              last_gen=num_gen,
                                              max_depth=6,
                                              expl_pipe=expl_pipe,
                                              competitor_rate=competitor_rate,
                                              duplicates_elimination='nothing',
                                              crossover_probability=0.90,
                                              mutation_probability=0.50,
                                              m=0.0,
                                              generation_strategy='half',
                                              elitism=True
                                              )
    