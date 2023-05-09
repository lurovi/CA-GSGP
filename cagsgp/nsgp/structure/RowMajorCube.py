from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import MutableSequence
from copy import deepcopy

from cagsgp.nsgp.structure.NeighborsTopology import NeighborsTopology

T = TypeVar('T')


class RowMajorCube(NeighborsTopology):
    def __init__(self,
                 collection: MutableSequence[T],
                 n_channels: int,
                 n_rows: int,
                 n_cols: int,
                 clone: bool = False
                 ) -> None:
        super().__init__()
        self.__n_channels: int = n_channels
        self.__n_rows: int = n_rows
        self.__n_cols: int = n_cols
        if len(collection) != self.__n_channels * self.__n_rows * self.__n_cols:
            raise ValueError(f'The length of the collection (found {len(collection)}) must match the product between number of channels ({self.__n_channels}), number of rows ({self.__n_rows}), and number of columns ({self.__n_cols}).')
        self.__collection: MutableSequence[T] = deepcopy(collection) if clone else collection
        self.__size_of_a_single_channel: int = self.n_rows() * self.n_cols()

    def __hash__(self) -> int:
        molt: int = 31
        h: int = 0
        for s in self.__collection:
            h = h * molt + hash(s)
        return h
    
    def __str__(self) -> str:
        return str(self.__collection)
    
    def __repr__(self) -> str:
        return 'RowMajorCube(' + str(self) + ')'
    
    def __eq__(self, value: RowMajorCube) -> bool:
        if self.n_channels() != value.n_channels():
            return False
        if self.n_rows() != value.n_rows():
            return False
        if self.n_cols() != value.n_cols():
            return False
        for l in range(self.n_channels()):
            for i in range(self.n_rows()):
                for j in range(self.n_cols()):
                    if self.get(l, i, j) != value.get(l, i, j):
                        return False
        return True
    
    def __len__(self) -> int:
        return self.n_channels() * self.n_rows() * self.n_cols()
    
    def n_channels(self) -> int:
        return self.__n_channels
    
    def n_rows(self) -> int:
        return self.__n_rows
    
    def n_cols(self) -> int:
        return self.__n_cols
    
    def get_whole_collection(self, clone: bool = False) -> MutableSequence[T]:
        return deepcopy(self.__collection) if clone else self.__collection
    
    def get(self, l: int, i: int, j: int, clone: bool = False) -> T:
        self.__check_channel_index(l)
        self.__check_row_index(i)
        self.__check_col_index(j)
        val: T = self.__collection[l * self.__size_of_a_single_channel + i * self.n_cols() + j]
        return deepcopy(val) if clone else val
    
    def set(self, l: int, i: int, j: int, val: T, clone: bool = False) -> T:
        self.__check_channel_index(l)
        self.__check_row_index(i)
        self.__check_col_index(j)
        offset: int = l * self.__size_of_a_single_channel + i * self.n_cols() + j
        old_val: T = self.__collection[offset]
        old_val = deepcopy(old_val) if clone else old_val
        self.__collection[offset] = val
        return old_val

    def neighborhood(self, indices: list[int], include_current_point: bool = True, clone: bool = False) -> MutableSequence[T]:
        if len(indices) != 3:
            raise ValueError(f'The length of indices must be 3, found {len(indices)} instead.')
        l: int = indices[0]
        i: int = indices[1]
        j: int = indices[2]
        self.__check_channel_index(l)
        self.__check_row_index(i)
        self.__check_col_index(j)
        result: MutableSequence[T] = []
        for ll in range(l - 1, l + 2):
            for ii in range(i - 1, i + 2):
                for jj in range(j - 1, j + 2):
                    if ll == l and ii == i and jj == j:
                        if include_current_point:
                            result.append(self.get(ll,ii,jj,clone=clone))
                    else:
                        if 0 <= ll < self.n_channels() and 0 <= ii < self.n_rows() and 0 <= jj < self.n_cols():
                            result.append(self.get(ll,ii,jj,clone=clone))
        return result

    def get_cube_as_string(self) -> str:
        s: str = '[\n'

        for l in range(self.n_channels()):
            s += '['
            s += '\n'
            for i in range(self.n_rows()):
                s += '['
                s += '\t'
                for j in range(self.n_cols()):
                    s += str(self.get(l, i, j))
                    s += '\t'
                s += ']\n'
            s += ']\n'
        s += ']\n'
        return s

    def __check_channel_index(self, l: int) -> None:
        if not 0 <= l < self.n_channels():
            raise IndexError(f'Index {l} is out of range with declared number of channels ({self.n_channels()})')
    
    def __check_row_index(self, i: int) -> None:
        if not 0 <= i < self.n_rows():
            raise IndexError(f'Index {i} is out of range with declared number of rows ({self.n_rows()})')
        
    def __check_col_index(self, j: int) -> None:
        if not 0 <= j < self.n_cols():
            raise IndexError(f'Index {j} is out of range with declared number of cols ({self.n_cols()})')
