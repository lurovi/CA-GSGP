from cagsgp.nsgp.structure.RowMajorMatrix import RowMajorMatrix
from cagsgp.nsgp.structure.RowMajorCube import RowMajorCube
from cagsgp.nsgp.structure.RowMajorLine import RowMajorLine
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np


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
