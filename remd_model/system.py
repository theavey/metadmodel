"""
This defines the system class.

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
from .replicas import Replicas


class System(object):

    def __init__(self, size: int,
                 start_temp: float=300., scaling_exponent: float=0.05,
                 width_param=5):
        self.size = size
        self.replicas = Replicas(size,
                                 start_temp=start_temp,
                                 scaling_exponent=scaling_exponent,
                                 width_param=width_param)
        self._last_exchange_even = False

    @property
    def state(self) -> np.ndarray:
        return self.replicas.indexes

    @property
    def energies(self) -> np.ndarray:
        return np.array(list((r.energy for r in self.replicas)))

    def exchange(self) -> None:
        energies = self.energies
        temps = self.replicas.temps
        offset = 1 if self._last_exchange_even else 0
        n = self.size + offset % 2
        probs = np.zeros(n, dtype=float)
        rands = np.random.rand(n)
        for i in range(n):
            ind = 2 * i + offset
            expo = ((energies[ind] - energies[ind+1]) *
                    (1 / temps[ind] - 1 / temps[ind+1]))
            probs[i] = min(1, np.exp(expo))
            if probs[i] > rands[i]:
                self.replicas.exchange(ind)
        self._last_exchange_even = not self._last_exchange_even
