from cagsgp.nsgp.structure.RowMajorMatrix import RowMajorMatrix
from cagsgp.nsgp.structure.RowMajorCube import RowMajorCube
from cagsgp.nsgp.structure.RowMajorLine import RowMajorLine
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np

from genepro.node import Node
from genepro.node_impl import Plus, Minus, Times, Div

from collections.abc import Callable
import itertools
import time
from typing import Any

from numpy.random import Generator
from prettytable import PrettyTable
from cagsgp.benchmark.DatasetGenerator import DatasetGenerator
from cagsgp.nsgp.stat.SemanticDistance import SemanticDistance

from cagsgp.nsgp.stat.StatsCollectorSingle import StatsCollectorSingle
from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology
from cagsgp.nsgp.structure.TreeStructure import TreeStructure
from cagsgp.nsgp.structure.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from cagsgp.nsgp.structure.factory.RowMajorCubeFactory import RowMajorCubeFactory
from cagsgp.nsgp.structure.factory.RowMajorLineFactory import RowMajorLineFactory
from cagsgp.nsgp.structure.factory.RowMajorMatrixFactory import RowMajorMatrixFactory
from cagsgp.nsgp.structure.factory.TournamentTopologyFactory import TournamentTopologyFactory
from cagsgp.util.EvaluationMetrics import EvaluationMetrics
from cagsgp.util.ResultUtils import ResultUtils
from genepro.node import Node
import pandas as pd
import numpy as np
import random

from genepro.node_impl import Constant

def add_generation_column(n: int, path: str, output_file:str ) -> None:
    df: pd.DataFrame = pd.read_csv(path, sep=" ")
    df['Generation'] = df.reset_index().index
    df.to_csv(output_file, sep=' ', index=False)


if __name__ == '__main__':
    l: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #lll: RowMajorLine = RowMajorLine(l, radius=2)
    #print(lll.neighborhood((15,)))
    r: RowMajorMatrix = RowMajorMatrix(l, 4, 4)
    r1: RowMajorMatrix = RowMajorMatrix([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 33, 14, 15], 4, 4)
    print(r.neighborhood((0, 1), True))
    print(r.get_matrix_as_string())
    l: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    r: RowMajorCube = RowMajorCube(l, 2, 3, 3)
    r1: RowMajorCube = RowMajorCube([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 33, 14, 15, 16, 17], 2, 3, 3)
    print(r.neighborhood((0, 0, 0), True))
    print(r.get_cube_as_string())
    
    pvals = [.01, 0, 0, 0.02]

    # Create a list of the adjusted p-values
    reject_bonferroni, pvals_corrected_bonferroni, _, _  = multipletests(pvals, alpha=.05, method='holm')
    
    # Print the resulting conclusions
    print(np.sum(reject_bonferroni))
    
    # Print the adjusted p-values themselves 
    print(pvals_corrected_bonferroni)

    # ==================

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

    generator: Generator = np.random.default_rng(1)
    random.seed(1)
    np.random.seed(1)
    
    if low_erc > high_erc:
        raise AttributeError(f"low erc is higher than high erc.")
    elif low_erc < high_erc:
        ephemeral_func: Callable = lambda: generator.uniform(low=low_erc, high=high_erc)
    else:
        ephemeral_func: Callable = None

    if n_constants > 0:
        constants: list[Constant] = [Constant(round(ephemeral_func(), 2)) for _ in range(n_constants)]

    structure: TreeStructure = TreeStructure(operators=operators,
                                                fixed_constants=constants if n_constants > 0 else None,
                                                ephemeral_func=ephemeral_func if n_constants == 0 else None,
                                                n_features=4,
                                                max_depth=max_depth,
                                                generation_strategy=generation_strategy)

    for _ in range(4):
        a: Node = structure.generate_tree()
    
    print(a.get_readable_repr())
    print(a.get_n_nodes())
    print(a.get_height())
    print(a.get_n_nodes() / float(a.get_height() + 1))
    print((a.get_height() + 1) / float(a.get_n_nodes()))
