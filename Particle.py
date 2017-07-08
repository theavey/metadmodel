"""
This defines the Particle class for an object that will 'travel' on an FES.

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


class Particle(object):
    """"""

    def __init__(self, fes, mass=1, x0, v0):
        """

        :param fes:
        :param mass:
        :param x0:
        :param v0:
        """
        self.FES = fes
        self.mass = mass
        self.velocity = v0
        self.position = x0
        self.trajectory = np.array([[x0, v0]])

    @property
    def calc_force(self):
        """
        Calculates the force on the Particle at the current position

        :return: the force on the particle
        :rtype: np.array
        """
        return -self.FES.deriv(self.position)

    def calc_acceleration(self):
        """
        Calculates the acceleration of the Particle at the current location

        :return:
        """
