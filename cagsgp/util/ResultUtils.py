import simplejson as json
import re
from typing import Any

from genepro.node import Node
from pytexit import py2tex

from genepro.util import get_subtree_as_full_list


class ResultUtils:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def safe_latex_format(tree: Node) -> str:
        readable_repr = tree.get_readable_repr().replace("u-", "-")
        try:
            latex_repr = ResultUtils.GetLatexExpression(tree)
        except (RuntimeError, TypeError, ZeroDivisionError, Exception) as e:
            latex_repr = readable_repr
        return re.sub(r"(\.[0-9][0-9])(\d+)", r"\1", latex_repr)

    @staticmethod
    def format_tree(tree: Node) -> dict[str, str]:
        latex_repr = ResultUtils.safe_latex_format(tree)
        parsable_repr = str(tree.get_subtree())
        return {"latex": latex_repr, "parsable": parsable_repr}

    @staticmethod
    def GetHumanExpression(tree: Node):
        result = ['']  # trick to pass string by reference
        ResultUtils._GetHumanExpressionRecursive(tree, result)
        return result[0]

    @staticmethod
    def GetLatexExpression(tree: Node):
        human_expression = ResultUtils.GetHumanExpression(tree)
        # add linear scaling coefficients
        latex_render = py2tex(human_expression.replace("^", "**"),
                              print_latex=False,
                              print_formula=False,
                              simplify_output=False,
                              verbose=False,
                              simplify_fractions=False,
                              simplify_ints=False,
                              simplify_multipliers=False,
                              ).replace('$$', '').replace('--', '+')
        # fix {x11} and company and change into x_{11}
        latex_render = re.sub(
            r"x(\d+)",
            r"x_{\1}",
            latex_render
        )
        latex_render = latex_render.replace('\\timesx', '\\times x').replace('--', '+').replace('+-', '-').replace('-+',
                                                                                                                   '-')
        return latex_render

    @staticmethod
    def _GetHumanExpressionRecursive(tree: Node, result):
        args = []
        for i in range(tree.arity):
            ResultUtils._GetHumanExpressionRecursive(tree.get_child(i), result)
            args.append(result[0])
        result[0] = ResultUtils._GetHumanExpressionSpecificNode(tree, args)
        return result

    @staticmethod
    def _GetHumanExpressionSpecificNode(tree: Node, args):
        return tree._get_args_repr(args)

    @staticmethod
    def parse_result_soo(
        result: dict[str, Any],
        objective_names: list[str],
        seed: int,
        mode: str,
        linear_scaling: bool,
        pop_size: int,
        num_gen: int,
        num_gen_post: int,
        num_offsprings: int,
        max_depth: int,
        generation_strategy: str,
        pressure: int,
        pop_shape: tuple[int, ...],
        crossover_probability: float,
        mutation_probability: float,
        m: float,
        competitor_rate: float,
        expl_pipe: str,
        execution_time_in_minutes: float,
        torus_dim: int,
        radius: int,
        elitism: bool,
        dataset_name: str,
        duplicates_elimination: str
    ) -> dict[str, Any]:
        n_objectives: int = len(objective_names)

        pareto_front_dict: dict[str, Any] = {"parameters": {},
                                             "best": result['best'],
                                             "history": result['history'],
                                             "train_statistics": result['train_statistics'],
                                             "test_statistics": result['test_statistics']
                                             }
        
        pareto_front_dict["parameters"]["LinearScaling"] = int(linear_scaling)
        pareto_front_dict["parameters"]["Mode"] = mode
        pareto_front_dict["parameters"]["PopSize"] = pop_size
        pareto_front_dict["parameters"]["NumGen"] = num_gen
        pareto_front_dict["parameters"]["NumGenPost"] = num_gen_post
        pareto_front_dict["parameters"]["NumOffsprings"] = num_offsprings
        pareto_front_dict["parameters"]["MaxDepth"] = max_depth
        pareto_front_dict["parameters"]["GenerationStrategy"] = generation_strategy
        pareto_front_dict["parameters"]["Pressure"] = pressure
        pareto_front_dict["parameters"]["CrossoverProbability"] = crossover_probability
        pareto_front_dict["parameters"]["MutationProbability"] = mutation_probability
        pareto_front_dict["parameters"]["m"] = m
        pareto_front_dict["parameters"]["CompetitorRate"] = competitor_rate
        pareto_front_dict["parameters"]["ExplPipe"] = expl_pipe
        pareto_front_dict["parameters"]["ExecutionTimeInMinutes"] = execution_time_in_minutes
        pareto_front_dict["parameters"]["TorusDim"] = torus_dim
        pareto_front_dict["parameters"]["Radius"] = radius
        pareto_front_dict["parameters"]["Elitism"] = int(elitism)
        pareto_front_dict["parameters"]["Dataset"] = dataset_name
        pareto_front_dict["parameters"]["DuplicatesElimination"] = duplicates_elimination
        pareto_front_dict["parameters"]["Seed"] = seed
        pareto_front_dict["parameters"]["NumObjectives"] = n_objectives
        pareto_front_dict["parameters"]["ObjectiveNames"] = objective_names
        pareto_front_dict["parameters"]["PopShape"] = [n for n in pop_shape]
    
        return pareto_front_dict

    @staticmethod
    def write_result_to_json(path: str, path_run_id: str, run_id: str, pareto_front_dict: dict[str, Any]) -> None:
        d: dict[str, Any] = {k: pareto_front_dict[k] for k in pareto_front_dict}
        with open(path + path_run_id + "tr" + run_id + ".json", "w") as outfile:
            json.dump({"train_statistics": d['train_statistics']}, outfile)
        with open(path + path_run_id + "te" + run_id + ".json", "w") as outfile:
            json.dump({"test_statistics": d['test_statistics']}, outfile)
        del d['train_statistics']
        del d['test_statistics']
        with open(path + path_run_id + "b" + run_id + ".json", "w") as outfile:
            json.dump(d, outfile)

    @staticmethod
    def read_single_json_file(
        folder_name: str,
        result_file_type: str,
        mode: str,
        linear_scaling: bool,
        pop_size: int,
        num_gen: int,
        num_gen_post: int,
        max_depth: int,
        torus_dim: int,
        dataset_name: str,
        competitor_rate: float,
        expl_pipe: str,
        duplicates_elimination: str,
        pop_shape: tuple[int, ...],
        crossover_probability: float,
        mutation_probability: float,
        m: float,
        radius: int,
        generation_strategy: str,
        elitism: bool,
        seed: int
    ) -> dict[str, Any]:
        
        if mode != 'gsgpgp':
            num_gen_post = 0

        pop_shape_str: str = 'x'.join([str(n) for n in pop_shape])
        crossprob: str = str(round(crossover_probability, 2))
        mutprob: str = str(round(mutation_probability, 2))
        cmprate: str = str(round(competitor_rate, 2)) if torus_dim != 0 else str(0.0)
        m_str: str = str(round(m, 2))
        pressure_str: str = str(radius) if torus_dim == 0 else str((2*radius + 1)**len(pop_shape))
        radius_str: str = str(radius) if torus_dim != 0 else str(0)
        elitism_str: str = str(int(elitism))
        seed_str: str = str(seed)
        
        path_run_id: str = f'linearscaling{int(linear_scaling)}/numgenpost{num_gen_post}/cmprate{cmprate}/{expl_pipe}/'
        run_id: str = f"c{mode}-popsize_{str(pop_size)}-numgen_{str(num_gen)}-maxdepth_{str(max_depth)}-torus_dim_{str(torus_dim)}-dataset_{dataset_name}-dupl_elim_{duplicates_elimination}-pop_shape_{pop_shape_str}-crossprob_{crossprob}-mutprob_{mutprob}-m_{m_str}-radius_{radius_str}-pressure_{pressure_str}-genstrat_{generation_strategy}-elitism_{elitism_str}-SEED{seed_str}"

        with open(folder_name + path_run_id + result_file_type + run_id + '.json', 'r') as f:
            data: dict[str, Any] = json.load(f)

        return data
