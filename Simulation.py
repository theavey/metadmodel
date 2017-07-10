"""
Defines a Simulation class for simulation of metadynamics.

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

from . import Particle
from . import FES
import numpy as np
import matplotlib.pyplot as plt


class Simulation(object):
    """"""

    def __init__(self, dimension=None, particle=None, fes=None, metad_freq: int=5):
        """

        :param int dimension: Dimensionality of the simulation. Currently either 1 or 2.
        Default is 1.

        This will be overridden by the dimensionality of any passed in FES or Particle.
        :param Particle.Particle particle: Particle to use in the simulation. Can be
        set later.
        :param FES.FES fes: FES to use for the simulation as passed to Particle.

        If a Particle is provided, this argument is ignored.
        :param bool metad: Do simulation with metadynamics. Default is False.
        """
        self._dimension: int = None
        self._particle: Particle.Particle = None
        self._FES: FES.FES = None
        self._metad: bool = None
        self._metad_freq = metad_freq
        self._trajectory: np.array = None

        if dimension is not None:
            self._dimension = dimension
        else:
            self._dimension = 1
        if particle is not None:
            dim_from_part = particle.dimensionality
            if dim_from_part != self._dimension:
                print("Dimensionality from Particle doesn't match default or passed "
                      "value.\n"
                      "Using the value from the given Particle.")
                self._dimension = dim_from_part
            self.particle = particle
        else:
            if fes is not None:
                dim_from_fes = fes.dimensionality
                if dim_from_fes != self._dimension:
                    print("Dimensionality from FES doesn't match default or passed "
                          "value.\n"
                          "Using the value from the given FES.")
                self._dimension = dim_from_fes
                self._FES = fes

    # Properties ########################

    @property
    def particle(self) -> Particle.Particle:
        """
        Particle in use for this simulation

        :return: the Particle
        :rtype: Particle.Particle
        """
        return self._particle

    @particle.setter
    def particle(self, particle):
        self._particle = particle

    @property
    def trajectory(self) -> np.array:
        """
        Trajectory of the particle

        :return: trajectory (n x 2)
        :rtype: np.array
        """
        if self._trajectory is None:
            print('No trajectory data yet! Have you run yet?')
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value):
        # alternatively, could just make this append, but not sure if that's desirable
        raise AttributeError('Cannot directly set the trajectory')

    @property
    def positions(self) -> np.array:
        """
        Positions from the trajectory

        :return: the positions
        """
        if self._trajectory is None:
            print('No trajectory data yet! Have you run yet?')
        return self.trajectory[:, 0:self._dimension]

    @positions.setter
    def positions(self, value):
        raise AttributeError('Cannot directly set positions or trajectory')

    @property
    def velocities(self) -> np.array:
        """
        Positions from the trajectory

        :return: the velocities
        """
        if self._trajectory is None:
            print('No trajectory data yet! Have you run yet?')
        dim = self._dimension
        return self.trajectory[:, dim:2*dim]

    @velocities.setter
    def velocities(self, value):
        raise AttributeError('Cannot directly set velocities or trajectory')

    # Running Simulation #####################

    def _time_step(self, step_num):
        """
        Move the particle and append position to trajectory
        :return: nothing
        """
        new_position, new_velocity = self.particle.move(0.5)
        self.trajectory[step_num] = new_position, new_velocity

    def run(self, steps: int =1000):
        """
        Run the simulation for a number of steps

        If no self.particle is yet defined, a default will be used
        :param steps:
        :return:
        """
        if self.particle is None:
            print('using default particle')
            # todo put in default particle here
            pass
        self._metad = self.particle.metad
        self._trajectory = np.zeros((steps+1, 2*self._dimension), float)
        self._trajectory[0] = self.particle.position, self.particle.velocity
        if not self._metad:
            for i in range(1, steps+1):
                self._time_step(i)
        else:
            for i in range(1, steps+1):
                if i % self._metad_freq == 0:
                    self.particle.add_hill()
                self._time_step(i)
        print(f'Done running {steps}!')

    # Analysis and Plotting #####################

    def plot_trajectory(self, **kwargs) -> plt.figure:
        """
        Plot the trajectory on a scatter plot

        If it's a 2D trajectory, the axes will be the two dimensions.
        If it's a 1D trajectory, it will be the position as a function of time.

        :param kwargs: arguments to be passed to the plot function
        :return: figure object of the scatter plot
        """
        fig, ax = plt.subplots()
        if self._dimension == 1:
            ax.plot(self.positions)
            ax.set_xlabel('time step')
            ax.set_ylabel('$x$')
            fig.tight_layout()
            return fig
        else:
            raise NotImplementedError
