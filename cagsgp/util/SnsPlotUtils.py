import os
from typing import Any
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from cagsgp.util.ResultUtils import ResultUtils


class SnsPlotUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def only_first_char_upper(s: str) -> str:
        return s[0].upper() + s[1:]

    @staticmethod
    def simple_line_plot_topology_split(
            folder_name: str,
            output_path: str,
            seed_list: list[int],
            topologies_radius_shapes: list[tuple[str, int, tuple[int, ...]]],
            main_topology_name: str,
            dataset_names: list[str],
            log_scaled_datasets: list[str],
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
        
        for dataset_name in dataset_names:
            log_scale_y: bool = True if dataset_name in log_scaled_datasets else False
            data: dict[str, list[Any]] = {'Generation': [],
                                          'Split type': [],
                                          'Topology': [],
                                          'Best RMSE': []}
            
            title: str = SnsPlotUtils.only_first_char_upper(dataset_name) + ' Median Best RMSE'
            file_name: str = 'lineplot'+'-'+dataset_name+'-'+main_topology_name+'.png'

            for topology, radius, shape in topologies_radius_shapes:
                for seed in seed_list:
                    d: dict[str, Any] = ResultUtils.read_single_json_file(
                        folder_name=folder_name,
                        result_file_type='res',
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
                    history: list[dict[str, Any]] = d['history'][:(last_gen + 1)]
                    d = None
                    for i, d in enumerate(history, 0):
                        data['Generation'].append(i)
                        data['Topology'].append(SnsPlotUtils.only_first_char_upper(topology)+'-'+str(radius))
                        data['Split type'].append('Train')
                        data['Best RMSE'].append(d['Fitness']['Train RMSE'])

                        data['Generation'].append(i)
                        data['Topology'].append(SnsPlotUtils.only_first_char_upper(topology)+'-'+str(radius))
                        data['Split type'].append('Test')
                        data['Best RMSE'].append(d['Fitness']['Test RMSE'])
                    history = None

            data: pd.DataFrame = pd.DataFrame(data)
            sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=1.6,
                      rc={'figure.figsize': (13, 10), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
            g = sns.lineplot(data=data, x="Generation", y="Best RMSE", hue="Topology", style="Split type", estimator=np.median, errorbar=None, palette="colorblind")
            plt.title(title)
            plt.legend(fontsize='small', title_fontsize='small')
            if log_scale_y:
                plt.yscale('log')
            plt.savefig(output_path+file_name)
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()

    @staticmethod
    def simple_box_plot_topology_split(
            folder_name: str,
            output_path: str,
            seed_list: list[int],
            topologies_radius_shapes: list[tuple[str, int, tuple[int, ...]]],
            main_topology_name: str,
            dataset_names: list[str],
            log_scaled_datasets: list[str],
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
        
        for dataset_name in dataset_names:
            log_scale_y: bool = True if dataset_name in log_scaled_datasets else False
            data: dict[str, list[Any]] = {'Topology': [],
                                          'Split type': [],
                                          'Best RMSE': []}
            
            title: str = SnsPlotUtils.only_first_char_upper(dataset_name) + ' Best RMSE'
            file_name: str = 'boxplot'+'-'+dataset_name+'-'+main_topology_name+'.png'

            for topology, radius, shape in topologies_radius_shapes:
                for seed in seed_list:
                    d: dict[str, Any] = ResultUtils.read_single_json_file(
                        folder_name=folder_name,
                        result_file_type='res',
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
                    d = None
                    data['Topology'].append(SnsPlotUtils.only_first_char_upper(topology)+'-'+str(radius))
                    data['Split type'].append('Train')
                    data['Best RMSE'].append(best['Fitness']['Train RMSE'])

                    data['Topology'].append(SnsPlotUtils.only_first_char_upper(topology)+'-'+str(radius))
                    data['Split type'].append('Test')
                    data['Best RMSE'].append(best['Fitness']['Test RMSE'])
                    best = None

            data: pd.DataFrame = pd.DataFrame(data)
            sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=1.6,
                      rc={'figure.figsize': (13, 10), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
            g = sns.boxplot(data=data, x="Split type", y="Best RMSE", hue="Topology", palette="colorblind")
            plt.title(title)
            plt.legend(fontsize='small', title_fontsize='small')
            if log_scale_y:
                plt.yscale('log')
            plt.savefig(output_path+file_name)
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()        


    @staticmethod
    def simple_heat_plot_topology_split_wilcoxon(
            folder_name: str,
            output_path: str,
            seed_list: list[int],
            topologies_radius_shapes: list[tuple[str, int, tuple[int, ...]]],
            main_topology_name: str,
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
        
        for dataset_name in dataset_names:
            for split_type in ['Train', 'Test']:
                data: dict[str, list[Any]] = {'Source topology': [],
                                            'Target topology': [],
                                            'p-value': []}
                fitness: dict[str, list[float]] = {}
                
                title: str = SnsPlotUtils.only_first_char_upper(dataset_name) + ' Wilcoxon Test on Best RMSE (' + split_type + ')'
                file_name: str = 'heatmap-wilcoxon'+'-'+split_type+'-'+dataset_name+'-'+main_topology_name+'.png'

                for topology, radius, shape in topologies_radius_shapes:
                    name: str = SnsPlotUtils.only_first_char_upper(topology)+'-'+str(radius)
                    fitness[name] = []
                    for seed in seed_list:
                        d: dict[str, Any] = ResultUtils.read_single_json_file(
                            folder_name=folder_name,
                            result_file_type='res',
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
                        d = None
                        fitness[name].append(best['Fitness'][split_type+' RMSE'])
                        best = None
                
                for k1 in sorted(list(fitness.keys())):
                    for k2 in sorted(list(fitness.keys())):
                        val1: list[float] = fitness[k1]
                        val2: list[float] = fitness[k2]
                        p: float = round(stats.wilcoxon(val1, val2, alternative="less").pvalue, 2) if val1 != val2 else 1.0
                        data['Source topology'].append(k1)
                        data['Target topology'].append(k2)
                        data['p-value'].append(p)

                data: pd.DataFrame = pd.DataFrame(data).pivot(index='Target topology', columns='Source topology', values='p-value')

                sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=1.6,
                        rc={'figure.figsize': (13, 10), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                            'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
                g = sns.heatmap(data=data, annot=True, fmt='.2f', cmap='flare', cbar_kws={'label': 'p-value'})
                plt.title(title)
                #plt.legend(fontsize='small', title_fontsize='small')
                plt.savefig(output_path+file_name)
                # plt.show()
                plt.clf()
                plt.cla()
                plt.close()  


if __name__ == '__main__':
    # Datasets: ['airfoil', 'bioav', 'concrete', 'parkinson', 'ppb', 'slump', 'toxicity', 'vladislavleva-14', 'yacht']
    # Datasets: ['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht']
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1' + '/'
    output_path: str = codebase_folder + 'python_data/CA-GSGP/' + 'images_1' + '/'

    SnsPlotUtils.simple_line_plot_topology_split(folder_name=folder_name,
                                              output_path=output_path,
                                              seed_list=list(range(1, 100 + 1)), 
                                              topologies_radius_shapes=[('tournament',4,(100,)),
                                                                        ('cube',1,(4,5,5)),
                                                                        ('cube',2,(4,5,5)),
                                                                        ('cube',3,(4,5,5)),
                                                                        ('cube',4,(4,5,5))],
                                              main_topology_name='cube',
                                              dataset_names=['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht'],
                                              log_scaled_datasets=[],
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

    
    SnsPlotUtils.simple_box_plot_topology_split(folder_name=folder_name,
                                              output_path=output_path,
                                              seed_list=list(range(1, 100 + 1)), 
                                              topologies_radius_shapes=[('tournament',4,(100,)),
                                                                        ('cube',1,(4,5,5)),
                                                                        ('cube',2,(4,5,5)),
                                                                        ('cube',3,(4,5,5)),
                                                                        ('cube',4,(4,5,5))],
                                              main_topology_name='cube',
                                              dataset_names=['airfoil', 'bioav', 'concrete', 'ppb', 'slump', 'toxicity', 'yacht'],
                                              log_scaled_datasets=['bioav', 'concrete', 'ppb', 'slump', 'toxicity'],
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
    
    SnsPlotUtils.simple_heat_plot_topology_split_wilcoxon(folder_name=folder_name,
                                              output_path=output_path,
                                              seed_list=list(range(1, 100 + 1)), 
                                              topologies_radius_shapes=[('tournament',4,(100,)),
                                                                        ('cube',1,(4,5,5)),
                                                                        ('cube',2,(4,5,5)),
                                                                        ('cube',3,(4,5,5)),
                                                                        ('cube',4,(4,5,5))],
                                              main_topology_name='cube',
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
    


    






        
