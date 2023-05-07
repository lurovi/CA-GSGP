from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator
from genepro.node import Node


class Size(TreeEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, tree: Node) -> float:
        return tree.get_n_nodes()
    