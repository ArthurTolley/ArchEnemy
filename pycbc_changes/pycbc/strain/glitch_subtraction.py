# Copyright (C) 2022 Arthur Tolley
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Generals
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
This module contains functions for subtracting glitches from strain data.
"""
import os
import copy
import h5py
import logging
import numpy
from pycbc.types import float64, float32, TimeSeries
from pycbc.filter import highpass
import pycbc.io
import scipy.signal as sig

class GlitchSubtractionSet(object):
    """Manages sets of subtractions and subtract them from time series.

    Subtractions are read from HDF files.

    Parameters
    ----------
    sim_file : string
        Path to an hdf file that contains a GlitchTable.
    \**kwds :
        The rest of the keyword arguments are passed to the artefact generation
        function when generating subtractions.

    Attributes
    ----------
    table
    """

    def __init__(self, sim_file, **kwds):
        ext = os.path.basename(sim_file)
        self._subhandler = GlitchHDFSubtractionSet(sim_file)
        self.table = self._subhandler.table
        self.apply = self._subhandler.apply
        self.make_strain_from_sub_object = \
            self._subhandler.make_strain_from_sub_object
        self.times = self._subhandler.times

    @staticmethod
    def from_cli(opt):
        """Return an instance of GlitchSubtractionSet configured as specified
        on the command line.
        """
        if opt.glitch_subtraction_file is None:
            return None

        kwa = {}
        return GlitchSubtractionSet(opt.glitch_subtraction_file, **kwa)

class GlitchHDFSubtractionSet():
    """Manages glitch subtractions.

    Parameters
    ----------
    sim_file : string
        Path to an hdf file containing subtractions

    Attributes
    ----------
    table
    required_params : tuple
        Parameter names that must exist for each glitch class in the
        subtraction HDF file to create subtractions of that type

    """

    _tableclass = pycbc.io.FieldArray
    glitch_types = ['scattered_light']
    required_params = ()

    def __init__(self, sim_file, hdf_group=None, **kwds):

        # TODO: This needs to filter the hdf file by detector and provide a warning
        #  for mismatched frame and channel names.

        # open the file
        fp = h5py.File(sim_file, 'r')

        # Grab all dataset names (glitch types)
        dataset_names = list(fp.keys())

        # Grab all datasets
        datasets = {dataset: fp[dataset] for dataset in dataset_names}

        # Required parameters for those glitch types
        parameters = {dataset: self.required_params(dataset) for dataset in dataset_names}

        # Grab parameter values for each glitch type to subtraction
        for glitch in self.glitch_types:
            if glitch in datasets:
                subvals = {param: datasets[glitch][param] for param in parameters[glitch]}
                subvals['glitch_type'] = numpy.array([glitch for i in range(len(datasets[glitch]))], dtype=str).reshape((len(datasets[glitch]), 1))

            # Include a warning when the dataset doesn't contain the required parameters

        # initialize the table
        self.table = self._tableclass.from_kwargs(**subvals)


    def apply(self, strain, subtraction_sample_rate=None):
        """Add subtractions to a time series.

        Parameters
        ----------
        strain : TimeSeries
            Time series to subtract artefacts from, of type float32 or float64.
        subtraction_sample_rate: float, optional
            The sample rate to generate the artefact before subtraction

        Returns
        -------
        None

        Raises
        ------
        TypeError
            For invalid types of `strain`.
        """
        if strain.dtype not in (float32, float64):
            raise TypeError("Strain dtype must be float32 or float64, not " \
                    + str(strain.dtype))

        t0 = float(strain.start_time)
        t1 = float(strain.end_time)

        delta_t = strain.delta_t
        if subtraction_sample_rate is not None:
            delta_t = 1.0 / subtraction_sample_rate

        subtractions = self.table

        subtracted_ids = []
        for ii, sub in enumerate(subtractions):

            start_time, end_time = self.get_start_end_times(sub)
            centre_time = (end_time + start_time) * 0.5
            
            if start_time < t0 or end_time > t1:
                continue

            logging.info('Subtracting %s artefact at %.3f', sub['glitch_type'][0], float(centre_time))

            # Create the time series object containing the artefact
            signal = self.make_strain_from_sub_object(sub, 1.0/delta_t)

            if sub['glitch_type'] == 'scattered_light':

                length = strain.duration * (1.0/delta_t)

                # Get start, centre and, end times relative to strain not gps time
                artefact_time = float(sub['centre_time']) - float(strain.start_time)
                artefact_start_time = artefact_time - (sub['time_period'] * 0.5)
                artefact_end_time = artefact_time + (sub['time_period'] * 0.5)

                idxstart = numpy.round(artefact_start_time * (1.0/delta_t))
                idxend = numpy.round(artefact_end_time * (1.0/delta_t))

                # Subtract the relevant slice of signal from the raw strain
                signal = TimeSeries(signal, delta_t=delta_t, dtype=strain.dtype,
                                    epoch=strain[int(idxstart):int(idxend)].start_time)
                strain[int(idxstart):int(idxend)] -= signal[:int(idxend-idxstart)]

            # Append ids to track later
            subtracted_ids.append(ii)

        subtracted = copy.copy(self)
        subtracted.table = subtractions[numpy.array(subtracted_ids).astype(int)]
        return subtracted

    def make_strain_from_sub_object(self, sub, subtraction_sample_rate):
        """Make a h(t) strain time-series from a subtraction object.

        Parameters
        -----------
        sub : subtraction object
            The subtraction object to turn into a strain h(t). Can be any
            object which has artefact parameters as attributes, such as an
            element in a ``FieldArray``.
        delta_t : float
            Sample rate to make subtraction at.

        Returns
        --------
        signal : float
            h(t) corresponding to the subtraction.
        """

        generator = glitch_generator_dict[sub['glitch_type'][0]](sub, subtraction_sample_rate)
        signal = generator.generate_template()

        return signal

    def times(self):
        """Return the times of all subtractions"""
        return self.table.centre_time

    def required_params(self, glitch_type):
        """Define and return the required parameters for each glitch type"""

        required_params = {'scattered_light':
                           ['fringe_frequency', 'time_period',
                           'amplitude', 'phase', 'centre_time']}

        return required_params[glitch_type]

    def get_start_end_times(self, sub):
        """Calculate and return the start and end times for each glitch type"""

        if sub['glitch_type'] == 'scattered_light':
            start_time = float(sub['centre_time']) - 0.5 * float(sub['time_period'])
            end_time = float(sub['centre_time']) + 0.5 * float(sub['time_period'])


        return start_time, end_time


class ScatteredLightGenerator:
    """A class containing the artefact generation methods for generating
    scattered light glitches.

    Scattered light glitches are generated with the centre of the arch
    half way through the time series.

    Required scattered light artefact parameters, found in sub:
    fringe_frequency : float
        The fringe frequency of the artefact to be generated.
    timeperiod : float
        The time period of the artefact to be generated.
    amplitude : float
        The amplitude of the artefact to be generated.
    phase : float
        The phase of the artefact to be generated.
    time_of_artefact : float
        The time of the artefact to be generated.
        This refers to the time of the centre of the arch for scattered
          light artefacts.

    Inputs
    ------
    sub : (?)
        An object containing the scattered light artefact parameters:
    subtraction_sample_rate : float
        The sample rate of the artefact time series.
        Typically the same as that of the data.

    Outputs
    -------

    """

    def __init__(self, sub, subtraction_sample_rate):

        self.fringe_frequency = sub['fringe_frequency']
        self.timeperiod = sub['time_period']
        self.amplitude = sub['amplitude']
        self.phase = sub['phase']
        self.time_of_artefact = sub['centre_time']
        self.sample_rate = subtraction_sample_rate

    def generate_template(self):
        """Comprehensive method for template generation

        Inputs
        ------

        Outputs
        -------
        self.template_timeseries : pycbc.TimeSeries
            A TimeSeries object containing the desired artefact.

        """

        t_initial = -self.timeperiod/2.0
        t_end = self.timeperiod/2.0
        dt = 1./(self.sample_rate)
        t = numpy.arange(t_initial, t_end, dt)
        f_rep = 1./(2.*self.timeperiod)

        self.template_array = self.amplitude * numpy.sin((self.fringe_frequency/f_rep)*numpy.sin(2*numpy.pi*f_rep*t) + self.phase)

        self.template_timeseries = TimeSeries(self.template_array,
                                              delta_t=1./self.sample_rate,
                                              epoch=0)

        self.template_timeseries  = highpass(self.template_timeseries , 15) * sig.tukey(len(self.template_timeseries), 0.2)

        return self.template_timeseries

glitch_generator_dict = {
    'scattered_light': ScatteredLightGenerator,
}
