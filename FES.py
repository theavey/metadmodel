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


class FES(object):
    """
    This is the base class for FES objects of various dimensions.

    Nothing is implemented here, but this should be used to enforce implementation of
    methods and such in derived classes.
    """

    # def __init__(self):
    #     """Initialize a FES object
    #
    #     """
    #     raise NotImplementedError

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
        be able to handle pretty much anything.
        :param args:
        :return:
        """
        raise NotImplementedError


class FES1D(FES):
    """"""

    def __init__(self, *args):
        """
        Initialize a FES object

        :param args:
        """
        self.dimensionality = 1
        raise NotImplementedError

    def value(self, x):
        """
        Return the value of the FES at this location

        :param x:
        :return:
        """
        raise NotImplementedError

    def deriv(self, x):
        """
        Return the derivative of the FES at this location

        :param x:
        :return:
        """
        raise NotImplementedError


class FES2D(FES):
    """"""

    def __init__(self, *args):
        """

        :param args:
        """
        self.dimensionality = 2
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

    def __init__(self, width, height, *args):
        """

        :param args:
        """
        super().__init__(*args)
        self._width = width
        self._height = height

    def add_hill(self, x):
        """
        Add a hill to the FES centered here

        :param x:
        :return:
        """
        raise NotImplementedError


class MetadFES2D(FES2D):
    """"""

    def __init__(self, width, height, *args):
        """

        :param args:
        """
        super().__init__(*args)
        self._width = width
        self._height = height

    def add_hill(self, x, y):
        """
        Add a hill to the FES centered here

        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError
