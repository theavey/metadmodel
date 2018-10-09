"""
This defines the simulation class.

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

from .system import System


class Simulation(object):

    def __init__(self, size: int, n_steps: int, interval: int,
                 start_temp: float=300., scaling_exponent: float=0.05,
                 width_param: float=5.):
        self.size = size
        self.n_steps = n_steps
        self.interval = interval
        self.system = System(size, start_temp=start_temp,
                             scaling_exponent=scaling_exponent,
                             width_param=width_param)
        self.energies = np.zeros((n_steps, size), dtype=float)
        self.states = np.zeros((n_steps, size), dtype=int)

    def run(self):
        for i in range(self.n_steps):
            self.energies[i] = self.system.energies
            self.states[i] = self.system.state
            if ((i+1) % self.interval) == 0:
                self.system.exchange()
