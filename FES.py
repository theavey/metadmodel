"""
This defines the FES classes.

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

import autograd as ag
import autograd.numpy as anp
import autograd.scipy as asp
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable


class FES(object):
    """
    This is the base class for FES objects of various dimensions.

    Nothing is implemented here, but this should be used to enforce implementation of
    methods and such in derived classes.
    """

    def __init__(self, func):
        """Initialize a FES object

        :param func: A function that defines the FES. It must work with autograd which
        means that it must contain basic python operations or autograd wrapped numpy
        functions. There are some special other things it can accept. See the autograd
        documentation: https://github.com/HIPS/autograd

        The function must return a scalar value, and the dimensionality of the input
        must match the dimensionality of the desired FES.
        """
        self._dimensionality = None
        self._func = func
        self._metad = None
        pass

    def value(self, *args):
        """
        Return the value of the FES at this location

        :param args:
        :return:
        """
        raise NotImplementedError

    def deriv(self, *args):
        """
        Return the derivative of the FES at this location

        The ad package (https://pypi.python.org/pypi/ad) seems interesting for taking
        derivatives of arbitrary functions. Might be excessive here, but still nice to
        be able to handle pretty much anything. Hasn't been updated in a couple years.

        This seems newer: https://github.com/HIPS/autograd
        :param args:
        :return:
        """
        raise NotImplementedError

    @property
    def dimensionality(self):
        """Dimensionality of the FES"""
        return self._dimensionality

    @dimensionality.setter
    def dimensionality(self, value):
        raise AttributeError('The dimensionality is not settable. If you want different '
                             'dimensionality, use the class for that dimension.')

    @property
    def metad(self) -> bool:
        """
        Whether or not this is a metadynamics FES

        :return: metad or not
        """
        return self._metad

    @metad.setter
    def metad(self, value):
        raise AttributeError('The metadynamics state is not settable. \nUse a specific'
                             'metad FES if that is what you want.')

    def add_hill(self, *args):
        raise AttributeError('Cannot add a hill to this non-metadynamics FES!')

    def plot_hills(self, **kwargs):
        raise AttributeError('Cannot plot hills from this non-metadynamics FES')

    def plot_eff_fes(self, **kwargs):
        raise AttributeError('Cannot plot effective FES from this non-metadynamics FES')

    def plot_fes(self, **kwargs):
        pass


class FES1D(FES):
    """"""

    def __init__(self, func, *args):
        """
        Initialize a FES object

        :param args:
        """
        super().__init__(func)
        self._dimensionality = 1
        self._metad = False
        self._grad_func = ag.grad(self._func)

    def value(self, x) -> float:
        """
        Return the value of the FES at this location

        :param x: location
        :return: value of the FES at this location
        """
        return self._func(x)

    def deriv(self, x) -> float:
        """
        Return the derivative of the FES at this location

        :param x:
        :return:
        """
        return self._grad_func(x)


class FES2D(FES):
    """"""

    def __init__(self, func, *args):
        """

        :param args:
        """
        super().__init__(func)
        self._dimensionality = 2
        self._metad = False
        raise NotImplementedError

    def value(self, x, y):
        """
        Return the value of the FES at this location

        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError

    def deriv(self, x, y):
        """
        Return the derivative of the FES at this location

        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError


class MetadFES1D(FES1D):
    """


    There is some way to enforce 'multiple inheritance' in the sense of forcing this to
    implement certain things (as would be appropriate for a metadynamics object).
    Not sure how to do that right now, and I'm not sure it's worth figuring out at this
    point.
    """

    def __init__(self, func, width: float, height: float, *args):
        """

        :param args:
        """
        super().__init__(func, *args)
        self._width = width
        self._height = height
        self._metad = True
        self._hill_list = []  # might be better to do an array? pre-allocated?
        self._fes_with_hills = self._func
        self._grad_func = ag.grad(self._fes_with_hills)
        self._hills: Callable = None

    def _gaussian(self, center: float):
        """
        Gaussian function centered at center

        :param center: middle of the Gaussian
        :return: Gaussian function
        :rtype: lambda
        """
        # return lambda x: (self._height / (self._width * (2. * math.pi))) * \
        #     anp.exp(-0.5 * ((x - center) / self._width)**2)
        return self._height * asp.norm(loc=center, scale=self._width)

    def _make_hills(self):
        """"""
        if not self._hill_list:
            return lambda x: 0.
        # Could parallelize (and save memory) by saving this as a list of Gaussians
        # and then summing (derivatives) afterwards.
        return lambda x: sum(self._gaussian(c)(x) for c in self._hill_list)

    def add_hill(self, x: float) -> None:
        """
        Add a hill to the FES centered here

        :param x: location of the particle
        :return:
        """
        self._hill_list.append(x)
        # I feel like this style below is terrible, but I am not sure how else to do it
        # todo implement this on a grid (much faster for long simulations)
        self._fes_with_hills = lambda var: self._make_hills()(var) + self._func(var)
        self._grad_func = ag.grad(self._fes_with_hills)
        # Might also need to redefine self.deriv

    def value(self, x) -> float:
        """
        Return the value of the FES and hills at this location

        :param x: location
        :return: value of the FES and hills at this location
        """
        return self._fes_with_hills(x)

    def plot_hills(self, points: int=300, minmax: Tuple[float, float]=None,
                   expand: float=0.1, mintozero: bool=True, **kwargs) -> plt.figure:
        """
        Plot the metadynamics hills and return the figure

        :param points: number of points to plot
        :param minmax: minimum and maximum of the plot range
        :param expand: factor to plot beyond min and max of added hills

        This is ignored if minmax is given
        :param mintozero: If True, sets the minimum of the calculated FES to zero
        :param kwargs: arguments to be passed to ax.plot
        :return: figure of the plot
        """
        if not self._hill_list:
            print('No hills listed. Are you sure this has been run already?')
            return None
        if self._hills is None:
            self._hills: Callable = self._make_hills()
        if minmax:
            min_hill, max_hill = minmax
        else:
            min_hill, max_hill = min(self._hill_list), max(self._hill_list)
            span = abs(max_hill - min_hill)
            min_hill, max_hill = min_hill - expand * span, max_hill + expand * span
        x = np.linspace(min_hill, max_hill, points)
        fig, ax = plt.subplots()
        hills = - self._hills(x)
        if mintozero:
            hills = hills - min(hills)
        ax.plot(x, hills, **kwargs)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$V$')
        fig.tight_layout()
        return fig

    def plot_eff_fes(self, points: int=300, minmax: Tuple[float, float]=None,
                     expand: float=0.1, drawboth: bool=True, **kwargs) -> plt.figure:
        """
        Plot the FES on which the particle actually travels (including hills)

        :param points: number of points to plot
        :param minmax: minimum and maximum of the plot range
        :param expand: factor to plot beyond min and max of added hills

        This is ignored if minmax is given
        :param drawboth: If True, draw both original FES function and the one including
        the hills.
        :param kwargs: arguments to be passed to ax.plot
        :return: figure of the plot
        """
        if not self._hill_list:
            print('No hills listed. Are you sure this has been run already?')
            return None
        if minmax:
            min_hill, max_hill = minmax
        else:
            min_hill, max_hill = min(self._hill_list), max(self._hill_list)
            span = abs(max_hill - min_hill)
            min_hill, max_hill = min_hill - expand * span, max_hill + expand * span
        x = np.linspace(min_hill, max_hill, points)
        fig, ax = plt.subplots()
        ax.plot(x, self._fes_with_hills(x))
        if drawboth:
            ax.plot(x, self._func(x))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$V$')
        fig.tight_layout()
        return fig

    def plot_fes(self, points: int=300, minmax: Tuple[float, float]=None,
                 expand: float=0.1, **kwargs) -> plt.figure:
        """
        Plot the FES on which the particle actually travels (including hills)

        :param points: number of points to plot
        :param minmax: minimum and maximum of the plot range
        :param expand: factor to plot beyond min and max of added hills

        This is ignored if minmax is given
        :param kwargs: arguments to be passed to ax.plot
        :return: figure of the plot
        """
        if not self._hill_list:
            print('No hills listed. Are you sure this has been run already?')
            return None
        if minmax:
            min_hill, max_hill = minmax
        else:
            min_hill, max_hill = min(self._hill_list), max(self._hill_list)
            span = abs(max_hill - min_hill)
            min_hill, max_hill = min_hill - expand * span, max_hill + expand * span
        x = np.linspace(min_hill, max_hill, points)
        fig, ax = plt.subplots()
        ax.plot(x, self._func(x))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$V$')
        fig.tight_layout()
        return fig




class MetadFES2D(FES2D):
    """"""

    def __init__(self, func, width, height, *args):
        """

        :param args:
        """
        super().__init__(func, *args)
        self._width = width
        self._height = height
        self._metad = True

    def add_hill(self, x, y):
        """
        Add a hill to the FES centered here

        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError
