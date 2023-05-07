import numpy as np
from genepro.util import compute_linear_scaling

from genepro.node import Node
from cagsgp.nsgp.evaluator.TreeEvaluator import TreeEvaluator


class MSE(TreeEvaluator):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray = None
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

    def evaluate(self, tree: Node) -> float:
        res: np.ndarray = np.core.umath.clip(tree(self.__X), -1e+10, 1e+10)
        slope, intercept = compute_linear_scaling(self.__y, res)
        slope = np.core.umath.clip(slope, -1e+10, 1e+10)
        intercept = np.core.umath.clip(intercept, -1e+10, 1e+10)
        res = intercept + np.core.umath.clip(slope * res, -1e+10, 1e+10)
        res = np.core.umath.clip(res, -1e+10, 1e+10)
        mse: float = np.square(np.core.umath.clip(res - self.__y, -1e+20, 1e+20)).sum() / float(len(self.__y))
        if mse > 1e+20:
            mse = 1e+20
        return mse
