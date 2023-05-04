from cagsgp.utils.RowMajorMatrix import RowMajorMatrix


if __name__ == '__main__':
    l: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    r: RowMajorMatrix = RowMajorMatrix(l, 4, 4)
    r1: RowMajorMatrix = RowMajorMatrix([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 33, 14, 15], 4, 4)
    print(r.neighborhood(2, 2, True))
    print(r.get_matrix_as_string())
