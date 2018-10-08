"""
This defines the walker class.

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

from numpy import exp
from numpy.random import normal


class Walker(object):
    """
    Class for a single walker to be in a replica exchange set of simulations

    """

    def __init__(self, index: int, temp: float):
        self._index = index
        self.temp = temp

    @property
    def energy(self) -> float:
        return normal(self.temp, exp(self.temp))

    @property
    def index(self) -> int:
        return self._index
