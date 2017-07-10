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
from . import FES


class Particle(object):
    """"""

    def __init__(self, fes, x0, v0, mass=1., time_step_size=1.):
        """

        :param FES.FES fes: FES on which the particle moves
        :param np.array x0: initial position of the particle
        :param np.array v0: initial velocity of the particle
        :param float mass: mass of the particle
        :param float time_step_size: size of time steps to take
        """
        self._FES = fes
        self._mass = mass
        self._velocity = v0
        self._position = x0
        self._trajectory = np.array([[x0, v0]])
        self._time_step_size = time_step_size

    @property
    def position(self):
        """
        Position of the particle currently

        :return: the position
        :rtype: np.array
        """
        return self._position

    @position.setter
    def position(self, value):
        print('Overriding current position.')
        self._position = value

    @property
    def velocity(self):
        """
        Velocity of the particle currently

        :return: the velocity
        :rtype: np.array
        """
        return self._velocity

    @velocity.setter
    def velocity(self, value: np.array):
        print('Overriding current velocity')
        self._velocity = value

    @property
    def force(self):
        """
        The force on the Particle at the current position

        :return: the force on the particle
        :rtype: np.array
        """
        return -self._FES.deriv(self._position)

    @force.setter
    def force(self, value):
        raise AttributeError('force is not currently settable')

    @property
    def acceleration(self):
        """
        The acceleration of the Particle at the current location

        :return: the acceleration
        :rtype: np.array
        """
        return self.force / self._mass

    @acceleration.setter
    def acceleration(self, value):
        raise AttributeError('acceleration not currently settable')

    @property
    def dimensionality(self) -> int:
        """
        Dimensionality of the FES on which the particle travels

        :return: dimensionality of FES
        """
        return self._FES.dimensionality

    @dimensionality.setter
    def dimensionality(self, value):
        raise AttributeError('Cannot change (or set) the dimensionality!')

    def move(self, time: float=1., return_prev: bool=False) -> tuple:
        """
        Move particle using Velocity Verlet algorithm

        :param float time: number of time steps to move
        :param bool return_prev: Also return starting location and velocity (before
        movement)
        :return:
        """
        time_step = self._time_step_size * time
        prev_position = self._position
        prev_velocity = self._velocity
        prev_acceleration = self.acceleration
        self._position = prev_position + prev_velocity * time_step + \
            0.5 * prev_acceleration * time_step**2
        self._velocity = prev_velocity + 0.5 * time_step * \
            (prev_acceleration + self.acceleration)
        if return_prev:
            ret_values = self._position, self._velocity, prev_position, prev_velocity
        else:
            ret_values = self._position, self._velocity
        return ret_values
