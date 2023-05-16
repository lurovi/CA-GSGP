import json
import re
from typing import Any

from genepro.node import Node
from pymoo.core.population import Population
from pymoo.core.algorithm import Algorithm
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
    def parse_result(opt: Population,
                     history: list[Algorithm],
                     stats: dict[str, list[Any]],
                     biodiversity: dict[str, list[float]],
                     objective_names: list[str],
                     seed: int,
                     pop_size: int,
                     num_gen: int,
                     num_offsprings: int,
                     max_depth: int,
                     generation_strategy: str,
                     pressure: int,
                     pop_shape: tuple[int, ...],
                     crossover_probability: float,
                     mutation_probability: float,
                     m: float,
                     execution_time_in_minutes: float,
                     neighbors_topology: str,
                     radius: int,
                     dataset_name: str,
                     duplicates_elimination: str
                     ) -> dict[str, Any]:
        n_objectives: int = len(objective_names)

        pareto_front_dict: dict[str, Any] = {"parameters": {},
                                             "optimal": [],
                                             "history": [],
                                             "n_evals": [],
                                             "statistics": stats,
                                             "biodiversity": biodiversity}
        
        pareto_front_dict["parameters"]["PopSize"] = pop_size
        pareto_front_dict["parameters"]["NumGen"] = num_gen
        pareto_front_dict["parameters"]["NumOffsprings"] = num_offsprings
        pareto_front_dict["parameters"]["MaxDepth"] = max_depth
        pareto_front_dict["parameters"]["GenerationStrategy"] = generation_strategy
        pareto_front_dict["parameters"]["Pressure"] = pressure
        pareto_front_dict["parameters"]["CrossoverProbability"] = crossover_probability
        pareto_front_dict["parameters"]["MutationProbability"] = mutation_probability
        pareto_front_dict["parameters"]["m"] = m
        pareto_front_dict["parameters"]["ExecutionTimeInMinutes"] = execution_time_in_minutes
        pareto_front_dict["parameters"]["NeighborsTopology"] = neighbors_topology
        pareto_front_dict["parameters"]["Radius"] = radius
        pareto_front_dict["parameters"]["Dataset"] = dataset_name
        pareto_front_dict["parameters"]["DuplicatesElimination"] = duplicates_elimination
        pareto_front_dict["parameters"]["Seed"] = seed
        pareto_front_dict["parameters"]["NumObjectives"] = n_objectives
        pareto_front_dict["parameters"]["ObjectiveNames"] = objective_names
        pareto_front_dict["parameters"]["PopShape"] = [n for n in pop_shape]

        for individual in opt:
            tree: Node = individual.X[0]
            fitness: list[float] = [individual.F[i] for i in range(n_objectives)]
            
            current_point: dict[str, Any] = {}
            current_point["ParsableTree"] = str(get_subtree_as_full_list(tree))
            current_point["LatexTree"] = ResultUtils.safe_latex_format(tree)
            current_point["Fitness"] = {objective_names[i]: fitness[i] for i in range(n_objectives)}

            pareto_front_dict["optimal"].append(current_point)

        for alg in history:
            current_gen: list[dict[str, Any]] = []

            for individual in alg.opt:
                tree: Node = individual.X[0]
                fitness: list[float] = [individual.F[i] for i in range(n_objectives)]
                
                current_point: dict[str, Any] = {}
                current_point["ParsableTree"] = str(get_subtree_as_full_list(tree))
                current_point["LatexTree"] = ResultUtils.safe_latex_format(tree)
                current_point["Fitness"] = {objective_names[i]: fitness[i] for i in range(n_objectives)}

                current_gen.append(current_point)

            pareto_front_dict["history"].append(current_gen)
            pareto_front_dict["n_evals"].append(alg.evaluator.n_eval)
    
        return pareto_front_dict

    @staticmethod
    def parse_result_soo(
        result: dict[str, Any],
        objective_names: list[str],
        seed: int,
        pop_size: int,
        num_gen: int,
        num_offsprings: int,
        max_depth: int,
        generation_strategy: str,
        pressure: int,
        pop_shape: tuple[int, ...],
        crossover_probability: float,
        mutation_probability: float,
        m: float,
        execution_time_in_minutes: float,
        neighbors_topology: str,
        radius: int,
        elitism: bool,
        dataset_name: str,
        duplicates_elimination: str
    ) -> dict[str, Any]:
        n_objectives: int = len(objective_names)

        pareto_front_dict: dict[str, Any] = {"parameters": {},
                                             "best": result['best'],
                                             "history": result['history'],
                                             "pop_fitness_per_gen": result['pop_fitness_per_gen'],
                                             "train_statistics": result['train_statistics'],
                                             "test_statistics": result['test_statistics']
                                             }
        
        pareto_front_dict["parameters"]["PopSize"] = pop_size
        pareto_front_dict["parameters"]["NumGen"] = num_gen
        pareto_front_dict["parameters"]["NumOffsprings"] = num_offsprings
        pareto_front_dict["parameters"]["MaxDepth"] = max_depth
        pareto_front_dict["parameters"]["GenerationStrategy"] = generation_strategy
        pareto_front_dict["parameters"]["Pressure"] = pressure
        pareto_front_dict["parameters"]["CrossoverProbability"] = crossover_probability
        pareto_front_dict["parameters"]["MutationProbability"] = mutation_probability
        pareto_front_dict["parameters"]["m"] = m
        pareto_front_dict["parameters"]["ExecutionTimeInMinutes"] = execution_time_in_minutes
        pareto_front_dict["parameters"]["NeighborsTopology"] = neighbors_topology
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
    def write_result_to_json(path: str, run_id: str, pareto_front_dict: dict[str, Any]) -> None:
        d: dict[str, Any] = {k: pareto_front_dict[k] for k in pareto_front_dict}
        with open(path + "res-" + run_id + ".json", "w") as outfile:
            json.dump(d, outfile)
