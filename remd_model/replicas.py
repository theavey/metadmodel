"""
This defines the replicas class.

Copyright (C) 2017 Thomas John Heavey IV

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
                 start_temp: float=300., scaling_exponent: float=0.05):
        self.size = size
        self.walkers = np.zeros(size, dtype=Walker)
        self.temps = np.zeros(size, dtype=float)
        for i in range(size):
            temp = np.exp(start_temp * i * scaling_exponent)
            self.temps[i] = temp
            self.walkers[i] = Walker(i, temp)
        self.indexes = np.arange(0, size)

    def __iter__(self):
        return self.replicas.__iter__()

    @property
    def replicas(self):
        return np.array(self.walkers[self.indexes])

    def exchange(self, lower_ind):
        indexer = [lower_ind, lower_ind + 1]
        temps = self.temps[indexer]
        inds = self.indexes[indexer]
        self.indexes[indexer] = inds[::-1]
        for walker, temp in zip(self.replicas[indexer], temps):
            walker.temp = temp
