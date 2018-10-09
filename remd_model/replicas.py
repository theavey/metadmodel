"""
This defines the replicas class.

Copyright (C) 2018 Thomas John Heavey IV

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If
not, see http://www.gnu.org/licenses/.
"""

import numpy as np

from .walker import Walker


class Replicas(object):

    def __init__(self, size: int,
                 start_temp: float=300., scaling_exponent: float=0.05,
                 width_param: float=5.):
        self.size = size
        self.walkers = np.zeros(size, dtype=Walker)
        self.temps = np.zeros(size, dtype=float)
        for i in range(size):
            temp = start_temp * np.exp(i * scaling_exponent)
            self.temps[i] = temp
            self.walkers[i] = Walker(i, temp, width_param=width_param)
        self.indexes = np.arange(0, size)

    def __iter__(self):
        return self.replicas.__iter__()

    @property
    def replicas(self) -> np.ndarray:
        return np.array(self.walkers[self.indexes])

    def exchange(self, lower_ind: int):
        indexer: list[int] = [lower_ind, lower_ind + 1]
        temps: np.ndarray(dtype=float, shape=(2,)) = self.temps[indexer]
        inds: np.ndarray(dtype=int, shape=(2,)) = self.indexes[indexer]
        self.indexes[indexer] = inds[::-1]
        for walker, temp, index in zip(self.replicas[indexer], temps, indexer):
            walker.temp = temp
            walker.r_index = index
