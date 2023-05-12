import numpy as np
from cagsgp.util.EvaluationMetrics import EvaluationMetrics
from genepro.node_impl import Pointer
from genepro.node import Node
from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator
from genepro.storage import WeakCache


class RMSE(TreeEvaluator):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray = None,
                 cache: WeakCache = None,
                 store_in_cache: bool = False,
                 fix_properties: bool = False
                 ) -> None:
        super().__init__()
        if y is None:
            raise AttributeError("Labels must be set.")
        if y.shape[0] != X.shape[0]:
            raise AttributeError(
                f"The number of observations in X is {X.shape[0]} and it is different from the number of observations in y, i.e., {y.shape[0]}.")
        if len(y.shape) != 1:
            raise AttributeError(
                f"y must be one-dimensional. The number of dimensions that have been detected in y are, on the contrary, {len(y.shape)}.")
        self.__X: np.ndarray = X
        self.__y: np.ndarray = y
        self.__cache: WeakCache = cache
        self.__store_in_cache: bool = store_in_cache
        self.__fix_properties: bool = fix_properties

    def evaluate(self, tree: Node) -> float:
        p: np.ndarray = np.core.umath.clip(Pointer(tree, cache=self.__cache, store_in_cache=self.__store_in_cache, fix_properties=self.__fix_properties)(self.__X), -1e+10, 1e+10)
        return EvaluationMetrics.root_mean_squared_error(y=self.__y, p=p, linear_scaling=True, slope=None, intercept=None)
