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
from cagsgp.util.StringUtils import StringUtils


class SnsPlotUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def simple_line_plot_topology_split(
            folder_name: str,
            output_path: str,
            mode: str,
            linear_scaling: bool,
            seed_list: list[int],
            torusdim_radius_shapes: list[tuple[int, int, tuple[int, ...]]],
            main_topology_name: str,
            dataset_names: list[str],
            log_scaled_datasets: list[str],
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
        
        for dataset_name in dataset_names:
            log_scale_y: bool = True if dataset_name in log_scaled_datasets else False
            data: dict[str, list[Any]] = {'Generation': [],
                                          'Split type': [],
                                          'Topology': [],
                                          'Best RMSE': []}
            
            title: str = StringUtils.only_first_char_upper(dataset_name) + ' Median Best RMSE'
            file_name: str = 'lineplot'+'-'+dataset_name+'-'+main_topology_name+'.svg'

            for torus_dim, radius, shape in torusdim_radius_shapes:
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
                    history: list[dict[str, Any]] = d['history'][:(last_gen + 1)]
                    d = None
                    for i, d in enumerate(history, 0):
                        data['Generation'].append(i)
                        data['Topology'].append(StringUtils.concat('TD', str(torus_dim))+'-'+str(radius))
                        data['Split type'].append('Train')
                        data['Best RMSE'].append(d['Fitness']['Train RMSE'])

                        data['Generation'].append(i)
                        data['Topology'].append(StringUtils.concat('TD', str(torus_dim))+'-'+str(radius))
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
            plt.savefig(output_path+file_name, format='svg')
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()

    @staticmethod
    def simple_box_plot_topology_split(
            folder_name: str,
            output_path: str,
            mode: str,
            linear_scaling: bool,
            seed_list: list[int],
            torusdim_radius_shapes: list[tuple[int, int, tuple[int, ...]]],
            main_topology_name: str,
            dataset_names: list[str],
            log_scaled_datasets: list[str],
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
        
        for dataset_name in dataset_names:
            log_scale_y: bool = True if dataset_name in log_scaled_datasets else False
            data: dict[str, list[Any]] = {'Topology': [],
                                          'Split type': [],
                                          'Best RMSE': []}
            
            title: str = StringUtils.only_first_char_upper(dataset_name) + ' Best RMSE'
            file_name: str = 'boxplot'+'-'+dataset_name+'-'+main_topology_name+'.svg'

            for torus_dim, radius, shape in torusdim_radius_shapes:
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
                    d = None
                    data['Topology'].append(StringUtils.concat('TD', str(torus_dim))+'-'+str(radius))
                    data['Split type'].append('Train')
                    data['Best RMSE'].append(best['Fitness']['Train RMSE'])

                    data['Topology'].append(StringUtils.concat('TD', str(torus_dim))+'-'+str(radius))
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
            plt.savefig(output_path+file_name, format='svg')
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()        


    @staticmethod
    def simple_heat_plot_topology_split_wilcoxon(
            folder_name: str,
            output_path: str,
            mode: str,
            linear_scaling: bool,
            seed_list: list[int],
            torusdim_radius_shapes: list[tuple[int, int, tuple[int, ...]]],
            main_topology_name: str,
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
        
        for dataset_name in dataset_names:
            for split_type in ['Train', 'Test']:
                data: dict[str, list[Any]] = {'Source topology': [],
                                            'Target topology': [],
                                            'p-value': []}
                fitness: dict[str, list[float]] = {}
                
                title: str = StringUtils.only_first_char_upper(dataset_name) + ' Wilcoxon Test on Best RMSE (' + split_type + ')'
                file_name: str = 'heatmap-wilcoxon'+'-'+split_type+'-'+dataset_name+'-'+main_topology_name+'.svg'

                for torus_dim, radius, shape in torusdim_radius_shapes:
                    name: str = StringUtils.concat('TD', str(torus_dim))+'-'+str(radius)
                    fitness[name] = []
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
                plt.savefig(output_path+file_name, format='svg')
                # plt.show()
                plt.clf()
                plt.cla()
                plt.close()  

    @staticmethod
    def simple_line_plot_topology_single_split_type(
            folder_name: str,
            output_path: str,
            mode: str,
            linear_scaling: bool,
            split_type: str,
            seed_list: list[int],
            torusdim_radius_shapes: list[tuple[int, int, tuple[int, ...]]],
            main_topology_name: str,
            dataset_names: list[str],
            log_scaled_datasets: list[str],
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
        
        split_type: str = StringUtils.only_first_char_upper(split_type)
        for dataset_name in dataset_names:
            log_scale_y: bool = True if dataset_name in log_scaled_datasets else False
            data: dict[str, list[Any]] = {'Generation': [],
                                          'Topology': [],
                                          'Best RMSE': []}
            
            title: str = StringUtils.only_first_char_upper(dataset_name) + ' ' + split_type + ' Median Best RMSE'
            file_name: str = 'lineplot'+'-'+split_type+'-'+dataset_name+'-'+main_topology_name+'.svg'
            #file_name: str = 'lineplot'+'-'+split_type+'-'+dataset_name+'-'+main_topology_name+'.png'
            
            result_file_type: str = 'b'
            if result_file_type == 'b':
                name_of_the_first_key: str = 'history'
            elif result_file_type == 'tr':
                name_of_the_first_key: str = 'train_statistics'
            elif result_file_type == 'te':
                name_of_the_first_key: str = 'test_statistics'
            else:
                raise ValueError(f'{result_file_type} is not a valid result file type')

            for torus_dim, radius, shape in torusdim_radius_shapes:
                for seed in seed_list:
                    d: dict[str, Any] = ResultUtils.read_single_json_file(
                        folder_name=folder_name,
                        result_file_type=result_file_type,
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
                    
                    history: list[dict[str, Any]] = d[name_of_the_first_key][:(last_gen + 1)]
                    d = None
                    for i, d in enumerate(history, 0):
                        data['Generation'].append(i)
                        data['Topology'].append(StringUtils.concat('TD', str(torus_dim))+'-'+str(radius))
                        #data['Best RMSE'].append(d['Fitness'][split_type+' RMSE']) # Test RMSE preso dal file che inizia con b
                        #data['Best RMSE'].append(d['EuclideanDistanceStats']['median']) # distance euclidea mediana presa dal file che inizia con b
                        #data['Best RMSE'].append(d['Height']) # height preso dal file che inizia con b
                        #data['Best RMSE'].append(d['LogNNodes']) # log nnodes preso dal file che inizia con b
                        #data['Best RMSE'].append(d['HeightStats']['median']) # height pop stats median preso dal file che inizia con b
                        data['Best RMSE'].append(d['LogNNodesStats']['median']) # log nnodes pop stats median preso dal file che inizia con b
                        #data['Best RMSE'].append(d['GlobalMoranI']) # global moran I preso dal file che inizia con b
                        #data['Best RMSE'].append(d['RMSE']['median']) # la fitness mediana sul train di tutti gli individui nella popolazione presa dal file che inizia con tr 
                    history = None

            data: pd.DataFrame = pd.DataFrame(data)
            sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=1.6,
                      rc={'figure.figsize': (13, 10), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
            g = sns.lineplot(data=data, x="Generation", y="Best RMSE", hue="Topology", estimator=np.median, errorbar=None, palette="colorblind")
            plt.title(title)
            plt.legend(fontsize='small', title_fontsize='small')
            if log_scale_y:
                plt.yscale('log')
            plt.savefig(output_path+file_name, format='svg')
            #plt.savefig(output_path+file_name, format='png')
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()


    @staticmethod
    def simple_box_plot_topology_single_split_type(
            folder_name: str,
            output_path: str,
            mode: str,
            linear_scaling: bool,
            split_type: str,
            seed_list: list[int],
            torusdim_radius_shapes: list[tuple[int, int, tuple[int, ...]]],
            main_topology_name: str,
            dataset_names: list[str],
            log_scaled_datasets: list[str],
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
        
        split_type: str = StringUtils.only_first_char_upper(split_type)
        for dataset_name in dataset_names:
            log_scale_y: bool = True if dataset_name in log_scaled_datasets else False
            data: dict[str, list[Any]] = {'Topology': [],
                                          'Best RMSE': []}

            title: str = StringUtils.only_first_char_upper(dataset_name) + ' ' + split_type + ' Best RMSE'
            file_name: str = 'boxplot'+'-'+split_type+'-'+dataset_name+'-'+main_topology_name+'.svg'

            for torus_dim, radius, shape in torusdim_radius_shapes:
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
                    d = None

                    data['Topology'].append(StringUtils.concat('TD', str(torus_dim))+'-'+str(radius))
                    data['Best RMSE'].append(best['Fitness'][split_type+' RMSE'])
                    best = None

            data: pd.DataFrame = pd.DataFrame(data)
            sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=1.6,
                      rc={'figure.figsize': (13, 10), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
            g = sns.boxplot(data=data, x="Topology", y="Best RMSE", palette="colorblind")
            plt.title(title)
            #plt.legend(fontsize='small', title_fontsize='small')
            if log_scale_y:
                plt.yscale('log')
            plt.savefig(output_path+file_name, format='svg')
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()   


if __name__ == '__main__':
    # Datasets: ['vladislavleva14', 'keijzer6', 'airfoil', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson']
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_data/CA-GSGP/' + 'results_1' + '/'
    output_path: str = codebase_folder + 'python_data/CA-GSGP/' + 'images_1' + '/'

    mode: str = 'gsgp'
    linear_scaling: bool = False
    pop_size: int = 900
    num_gen: int = 111
    competitor_rate: float = 0.6
    expl_pipe: str = 'crossmut'
    torus_dim: int = 2
    pop_shape: tuple[int, ...] = (int(pop_size ** (1/torus_dim)), int(pop_size ** (1/torus_dim)))

    SnsPlotUtils.simple_line_plot_topology_single_split_type(folder_name=folder_name,
                                              output_path=output_path,
                                              split_type='Test',
                                              mode=mode,
                                              linear_scaling=linear_scaling,
                                              seed_list=list(range(1, 100 + 1)), 
                                              torusdim_radius_shapes=[(0,4,(pop_size,)),
                                                                        (torus_dim,1,pop_shape),
                                                                        (torus_dim,2,pop_shape),
                                                                        (torus_dim,3,pop_shape)
                                                                        ],
                                              main_topology_name='matrix',
                                              dataset_names=['vladislavleva14', 'keijzer6', 'airfoil', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson'],
                                              log_scaled_datasets=['vladislavleva14', 'keijzer6', 'airfoil', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson'],
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

    '''
    SnsPlotUtils.simple_box_plot_topology_single_split_type(folder_name=folder_name,
                                              output_path=output_path,
                                              mode=mode,
                                              linear_scaling=linear_scaling,
                                              split_type='Test',
                                              seed_list=list(range(1, 100 + 1)), 
                                              torusdim_radius_shapes=[(0,4,(100,)),
                                                                        (2,1,(10,10)),
                                                                        (2,2,(10,10)),
                                                                        (2,3,(10,10))
                                                                        ],
                                              main_topology_name='all',
                                              dataset_names=['vladislavleva14', 'airfoil', 'keijzer6', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson'],
                                              log_scaled_datasets=['concrete', 'slump', 'toxicity'],
                                              pop_size=100,
                                              num_gen=1000,
                                              last_gen=1000,
                                              max_depth=6,
                                              expl_pipe='crossmut',
                                              competitor_rate=0.6,
                                              duplicates_elimination='nothing',
                                              crossover_probability=0.90,
                                              mutation_probability=0.50,
                                              m=0.0,
                                              generation_strategy='half',
                                              elitism=True
                                              )
    '''
    
    '''
    SnsPlotUtils.simple_line_plot_topology_split(folder_name=folder_name,
                                              output_path=output_path,
                                              mode=mode,
                                              linear_scaling=linear_scaling,
                                              seed_list=list(range(1, 100 + 1)), 
                                              torusdim_radius_shapes=[(0,4,(100,)),
                                                                        (2,1,(10,10)),
                                                                        (2,2,(10,10)),
                                                                        (2,3,(10,10))
                                                                        ],
                                              main_topology_name='matrix',
                                              dataset_names=['vladislavleva14', 'keijzer6', 'airfoil', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson'],
                                              log_scaled_datasets=[],
                                              pop_size=100,
                                              num_gen=1000,
                                              last_gen=1000,
                                              max_depth=6,
                                              expl_pipe='crossmut',
                                              competitor_rate=1.0,
                                              duplicates_elimination='nothing',
                                              crossover_probability=0.90,
                                              mutation_probability=0.50,
                                              m=0.0,
                                              generation_strategy='half',
                                              elitism=True
                                              )

    
    SnsPlotUtils.simple_box_plot_topology_split(folder_name=folder_name,
                                              output_path=output_path,
                                              mode=mode,
                                              linear_scaling=linear_scaling,
                                              seed_list=list(range(1, 100 + 1)), 
                                              torusdim_radius_shapes=[(0,4,(100,)),
                                                                        (2,1,(10,10)),
                                                                        (2,2,(10,10)),
                                                                        (2,3,(10,10))
                                                                        ],
                                              main_topology_name='matrix',
                                              dataset_names=['vladislavleva14', 'airfoil', 'keijzer6', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson'],
                                              log_scaled_datasets=['concrete', 'slump', 'toxicity'],
                                              pop_size=100,
                                              num_gen=1000,
                                              last_gen=1000,
                                              max_depth=6,
                                              expl_pipe='crossmut',
                                              competitor_rate=0.6,
                                              duplicates_elimination='nothing',
                                              crossover_probability=0.90,
                                              mutation_probability=0.50,
                                              m=0.0,
                                              generation_strategy='half',
                                              elitism=True
                                              )
    
    SnsPlotUtils.simple_heat_plot_topology_split_wilcoxon(folder_name=folder_name,
                                              output_path=output_path,
                                              mode=mode,
                                              linear_scaling=linear_scaling,
                                              seed_list=list(range(1, 100 + 1)), 
                                              torusdim_radius_shapes=[(0,4,(100,)),
                                                                        (2,1,(10,10)),
                                                                        (2,2,(10,10)),
                                                                        (2,3,(10,10))
                                                                        ],
                                              main_topology_name='matrix',
                                              dataset_names=['vladislavleva14', 'airfoil', 'keijzer6', 'concrete', 'slump', 'toxicity', 'yacht', 'parkinson'],
                                              pop_size=100,
                                              num_gen=1000,
                                              last_gen=1000,
                                              max_depth=6,
                                              expl_pipe='crossmut',
                                              competitor_rate=0.6,
                                              duplicates_elimination='nothing',
                                              crossover_probability=0.90,
                                              mutation_probability=0.50,
                                              m=0.0,
                                              generation_strategy='half',
                                              elitism=True
                                              )
    '''


    






        
