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
This module contains functions for generating scattered light templates.
"""
import logging
import numpy

from scipy import signal

from pycbc.types import TimeSeries, FrequencySeries
from pycbc.filter import make_frequency_series

class ArtefactGeneration:
    """Contains the method for artefact generation.

    Inputs
    ------
    fringe_frequency : float
        The fringe frequency of the artefact to be generated.
    timeperiod : float
        The time period of the artefact to be generated.
    amplitude : float
        The amplitude of the artefact to be generated
    phase : float
        The phase of the artefact to be generated
    centre_time : float
        The centre time of the artefact.
    sample_rate : float
        The sample rate of the template.
        Typically the same as that of the data.
        Default = 2048.0

    Outputs
    -------

    """
    def __init__(self,
                 fringe_frequency: float,
                 timeperiod: float,
                 amplitude: float,
                 phase: float,
                 centre_time: float = None,
                 sample_rate: float = 2048.0) -> None:

        self.fringe_frequency = fringe_frequency
        self.timeperiod = timeperiod
        self.amplitude = amplitude
        self.phase = phase
        self.centre_time = centre_time
        self.sample_rate = sample_rate

    # Scattered Light Model
    
    def generate_template(self, duration=None, start_time=None,
                          roll=False, pad=False, make_timeseries=False,
                          make_frequency_series=False) -> None:
        """Generates a scattered light artefact template.
        
        Inputs
        ------
        duration : float | None
            The desired duration of the output object in seconds.
            Used in conjunction with padding.    
        start_time : float | None
            The start time of the output.
            Used along with the centre_time to place the artefact at
              the right time.
        roll : bool
            Roll the template so the centre point is at the beginning.
            Reqiored pad = True.
        pad : bool
            Pad the template making the output object the same length.
            Padding is required for creating non-numpy objects.
        make_timeseries : bool
            Create a PyCBC TimeSeries instead of a numpy array
            Requires pad = True.
        make_frequency_series : bool
            Create a PyCBC FrequencySeries instead of a numpy array
            Requires pad = True.
        
        Outputs
        -------
        self.template : numpy.array | TimeSeries | FrequencySeries
            The object containing the scattered light artefact template.
        self.fs_template : FrequencySeries
            The FrequencySeries representation of the object.
        """
        
        if roll is True:
            self.centre_time is None
        
        t_initial = -self.timeperiod/2.0
        t_end = self.timeperiod/2.0
        dt = 1./(self.sample_rate)
        t = numpy.arange(t_initial, t_end, dt)
        f_rep = 1./(2.*self.timeperiod)
        
        template = self.amplitude * numpy.sin((self.fringe_frequency/f_rep)*numpy.sin(2*numpy.pi*f_rep*t) + self.phase)
        self.template = template * signal.tukey(len(self.template_array), 0.2)
        
        if pad is True:
            array_length = duration * self.sample_rate
            template_length = len(self.template)
            empty_array = numpy.zeros(int(array_length))
            
            if self.centre_time is not None:
                if start_time is None:
                    print('Defaulting start time to 0.')
                artefact_start = (self.centre_time - (self.timeperiod/2)) - start_time
                start_index = numpy.round(artefact_start * self.sample_rate)
                end_index = numpy.round((artefact_start + self.timeperiod) * self.sample_rate)
            
            else:
                start_index = numpy.round(array_length/2 - int(template_length/2.))
                end_index = numpy.round(start_index + template_length)
                
            empty_array[int(start_index):int(start_index)] = self.template
                        
            if roll is True:
                empty_array = numpy.roll(empty_array, (int(len(empty_array)/2.0)))
                
            self.template = empty_array

            if make_timeseries is True:
                self.template = TimeSeries(empty_array,
                                           delta_t=1./self.sample_rate,
                                           epoch=start_time)
                                           
            if make_frequency_series is True:
                self.template = TimeSeries(empty_array,
                                           delta_t=1./self.sample_rate,
                                           epoch=start_time)
                self.fs_template = make_frequency_series(self.template)

    def whiten_template(self,
                        psd: FrequencySeries) -> TimeSeries:
        """Whiten the template using a provided psd.

        Inputs
        ------
        psd : pycbc.FrequencySeries
            The power spectral density to whiten the template.

        Outputs
        -------
        white_template : pycbc.TimeSeries
            A TimeSeries containing the whitened template.

        """

        if self.psd is None:
            logging.info('Please provide a psd to whiten a template.')

        white_template = (self.fs_temp / psd**0.5).to_timeseries()

        return white_template