"""
This defines the walker class.

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

from numpy.random import normal


class Walker(object):
    """
    Class for a single walker to be in a replica exchange set of simulations

    """

    def __init__(self, index: int, temp: float, width_param: float=5.):
        self._w_index: int = index
        self._r_index: int = index
        self.temp: float = temp
        self.width_param: float = width_param

    @property
    def energy(self) -> float:
        return normal(self.temp, self.temp/self.width_param)

    @property
    def w_index(self) -> int:
        return self._w_index

    @property
    def r_index(self) -> int:
        return self._r_index

    @r_index.setter
    def r_index(self, val: int):
        if abs(val - self._r_index) > 1 or val == self._r_index:
            raise ValueError(f'New replica index value must be +/- 1. '
                             f'Old: {self._r_index}. New: {val}')
        self._r_index = val

    def __repr__(self) -> str:
        return (f'{self.__class__} with w_index {self.w_index} '
                f'and r_index {self.r_index}')
