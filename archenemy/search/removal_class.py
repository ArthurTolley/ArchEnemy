# Imports
import logging
from typing import Union
import numpy as np
import scipy.signal as sig

import pycbc.strain
from pycbc.events.coinc import cluster_over_time
from pycbc.filter import highpass, matched_filter, matched_filter_core, sigma, \
                         resample_to_delta_t, sigmasq, make_frequency_series
from pycbc.frame import query_and_read_frame, read_frame
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.fft import IFFT
from pycbc.types.timeseries import TimeSeries, FrequencySeries, zeros
from pycbc.types import complex_same_precision_as
from ligotimegps import LIGOTimeGPS
from pycbc.filter.matchedfilter import _BaseCorrelator


class ArtefactGeneration:
    """A class containing the artefact generation methods.

    Inputs
    ------
    start_time : float
        The desired start time of the artefact time series.
    data_time_length : float
        The length of the data in time.
    fringe_frequency : float
        The fringe frequency of the artefact to be generated.
    timeperiod : float
        The time period of the artefact to be generated.
    amplitude : float
        The amplitude of the artefact to be generated
    phase : float
        The phase of the artefact to be generated
    pad : bool
        Determines whether padding is added to the artefact time series.
        Default = True
    sample_rate : float
        The sample rate of the artefact time series.
        Typically the same as that of the data.
        Default = 2048.0
    psd : pycbc.FrequencySeries
        The power spectral density to weight the template by when
        performing the whitening.
        If no whitening is needed, doesn't need to be provided.
        Default = None

    Outputs
    -------
    self objects containing the inputs.

    """
    def __init__(self,
                 start_time: float,
                 data_time_length: Union[float, None],
                 fringe_frequency: float,
                 timeperiod: float,
                 amplitude: float,
                 phase: float,
                 pad: bool = True,
                 sample_rate: float = 2048,
                 psd: FrequencySeries = None,
                 roll: bool = True) -> None:

        self.start_time = start_time
        self.data_time_length = data_time_length
        self.fringe_frequency = fringe_frequency
        self.timeperiod = timeperiod
        self.amplitude = amplitude
        self.phase = phase
        self.pad = pad
        self.sample_rate = sample_rate
        self.psd = psd
        self.roll = roll

        self.fs_temp = None

    # Scattered Light Model

    def generate_template_array(self,
                                tukey_length: float = 0.2) -> None:
        """Generate a numpy array containing the artefact.

        Inputs
        ------
        tukey_length : float
            The tukey window input to the scipy function.

        Outputs
        -------
        self.template_array : numpy.array
            A numpy array object containing the generated artefact.

        """

        t_initial = -self.timeperiod/2.0
        t_end = self.timeperiod/2.0
        dt = 1./(self.sample_rate)
        t = np.arange(t_initial, t_end, dt)
        f_rep = 1./(2.*self.timeperiod)

        self.template_array = self.amplitude * np.sin((self.fringe_frequency/f_rep)*np.sin(2*np.pi*f_rep*t) + self.phase)
        self.template_array = self.template_array * sig.tukey(len(self.template_array), tukey_length)

    def generate_template_timeseries(self) -> TimeSeries:
        """Generate a pycbc TimeSeries from a numpy array.

        Inputs
        ------

        Outputs
        -------
        self.template_timeseries : pycbc.TimeSeries
            A TimeSeries object containing the artefact.

        """

        if self.pad is True:
            length = self.data_time_length * self.sample_rate
            template_length = len(self.template_array)

            ts_array = np.zeros(int(length))
            idxstart = length/2 - int(template_length/2.)
            idxend = idxstart + template_length
            ts_array[int(idxstart):int(idxend)] = self.template_array
            if self.roll is True:
                ts_array = np.roll(ts_array, (int(len(ts_array)/2.0)))

            self.template_timeseries = TimeSeries(ts_array,
                                                  delta_t=1./self.sample_rate,
                                                  epoch=self.start_time)

        else:
            self.template_timeseries = TimeSeries(self.template_array,
                                                  delta_t=1./self.sample_rate,
                                                  epoch=self.start_time)

    def generate_template(self) -> TimeSeries:
        """Comprehensive method for template generation

        Inputs
        ------

        Outputs
        -------
        self.template_timeseries : pycbc.TimeSeries
            A TimeSeries object containing the desired artefact.

        """

        self.generate_template_array()
        self.generate_template_timeseries()

        return self.template_timeseries

    def generate_frequency_template(self) -> FrequencySeries:

        self.fs_temp = make_frequency_series(self.template_timeseries)

        return self.fs_temp

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

class ChiSquared:
    """This class contains the methods for performing a chi squared test
    on a provided template and data.
    """
    def __init__(self,
                 template,
                 fs_temp,
                 white_template,
                 data,
                 white_data,
                 weighted_data,
                 psd,
                 fringe_frequency,
                 timeperiod,
                 snr_ts,
                 sample_rate,
                 start_time,
                 temp_mem,
                 snr_mem,
                 corr_mem):
        """
        Inputs
        ------
        template : pycbc.TimeSeries
            A TimeSeries with the template used in the chi squared test.
        fs_temp : pycbc.FrequencySeries
            A FrequencySeries with the template used in the chi squared test.
        white_template : pycbc.TimeSeries
            The whitened template to use in the chi squared test.
        white_data : pycbc.TimeSeries
            The whitened data to use in the chi squared test.
        psd : pycbc.FrequencySeries
            The power spectral density to use in the chi squared test.
        data : pycbc.TimeSeries
            The data to use in the chi squared test.
        fringe_frequency : float
            The fringe frequency of the template used in the test.
        timeperiod : float
            The time period of the template used in the test.
        snr_ts : pycbc.TimeSeries
            The signal-to-noise ratio time series, created by matched
            filtering the template and the data.
        sample_rate : float
            The sample rate used throughout the workflow.
            This should be consistent across data and template TimeSeries.
        start_time : float
            The start time of the data.

        """

        self.template = template
        self.fs_temp = fs_temp
        self.white_template = white_template
        self.white_data = white_data
        self.weighted_data = weighted_data
        self.psd = psd
        self.data = data
        self.fringe_frequency = fringe_frequency
        self.timeperiod = timeperiod
        self.snr_ts = snr_ts
        self.sample_rate = sample_rate
        self.dt = 1/self.sample_rate
        self.start_time = start_time
        self.temp_mem = temp_mem
        self.snr_mem = snr_mem
        self.corr_mem = corr_mem
        self.frac_1 = None
        self.frac_2 = None
        self.frac_3 = None
        self.frac_4 = None

    # Chi Squared

    def find_segment_widths(self,
                            pregenerated_splits: list = None) -> None:
        """Find the fractional widths of each segment for the test.

        Inputs
        ------

        Outputs
        -------
        self.frac_1 : float
            Related to the fraction of the total width of the template
            that segment 1 should contain.
            This is a float and can be converted to a percentage by
            multiplying by 100.
        self.frac_2 : float
            Related to the fraction of the total width of the template
            that segment 2 should contain.
        self.frac_3 : float
            Related to the fraction of the total width of the template
            that segment 3 should contain.
        self.frac_4 : float
            Related to the fraction of the total width of the template
            that segment 4 should contain.

        """

        if pregenerated_splits is not None:
            self.frac_1, self.frac_2, self.frac_3, self.frac_4 = pregenerated_splits
            return

        gen_template = ArtefactGeneration(start_time = self.start_time,
                                           data_time_length = None,
                                           fringe_frequency = self.fringe_frequency,
                                           timeperiod = self.timeperiod,
                                           amplitude = 1e-21,
                                           phase = 0.0,
                                           pad = False,
                                           sample_rate = self.sample_rate,
                                           psd = self.psd)
        template = gen_template.generate_template()
        fs_temp = gen_template.generate_frequency_template()

        # Interpolate PSD
        interpolated_psd = interpolate(self.psd, template.delta_f)

        if len(fs_temp) != len(interpolated_psd):
            interpolated_psd.resize(int(len(fs_temp)))

        # Whiten the template
        white_temp = gen_template.whiten_template(interpolated_psd)

        ex_frac = np.cumsum((white_temp*white_temp))[-1] / 4

        cumsum = np.cumsum((white_temp*white_temp))
        idx = (np.abs(cumsum - ex_frac)).argmin()
        idx_2 = idx + (np.abs(cumsum[idx:] - ex_frac*2)).argmin()
        idx_3 = idx_2 + (np.abs(cumsum[idx_2:] - ex_frac*3)).argmin()
        idx_4 = len(cumsum) - idx_3

        self.frac_1 = (idx)/(self.timeperiod * self.sample_rate)
        self.frac_2 = (idx_2)/(self.timeperiod * self.sample_rate) - self.frac_1
        self.frac_3 = (idx_3)/(self.timeperiod * self.sample_rate) - self.frac_2 - self.frac_1
        self.frac_4 = (idx_4)/(self.timeperiod * self.sample_rate)

    def split_template(self):
        """Split the template into 4 segments for the test.

        Inputs
        ------

        Outputs
        ------
        self.segs : list
            A list of TimeSeries objects, the TimeSeries' correspond to
            the segments used in the chi squared test.

        """

        # We need to calculate the right lens for the segment boundaries
        len_data = int(len(self.data))
        len_temp = int(self.timeperiod * self.sample_rate)

        # Define segment boundary indices
        seg_3 = [int(0), int(len_temp*self.frac_3)]
        seg_4 = [int(seg_3[1]), int(seg_3[1] + len_temp*self.frac_4)]
        seg_2 = [int(len_data - len_temp * self.frac_2), int(len_data)]
        seg_1 = [int(seg_2[0] - len_temp * self.frac_1), int(seg_2[0])]
        power_segs = [seg_1, seg_2, seg_3, seg_4]

        self.segs=[]
        for seg in power_segs:
            seg_array = np.zeros_like(self.white_template)
            start_idx = int(seg[0])
            end_idx = int(seg[1])

            seg_array[start_idx:end_idx] = self.white_template[start_idx:end_idx]
            seg_ts = TimeSeries(seg_array, delta_t=self.dt, epoch=self.start_time)
            self.segs.append(seg_ts)

    def calculate_chi_sq(self,
                         pycbc_mf: bool = False,
                         chi_mf_instance = None):
        """Calculate the chi squared for a given template and data, then
        re-weight the signal-to-noise ratio time series of the matched
        filter between template and data.

        Inputs
        ------

        Outputs
        -------
        self.snr_ts : pycbc.TimeSeries
            The original signal-to-noise ratio time series of the
            matched filter between the template and data.
        chi_sq : numpy.array
            An array containing the chi squared values for each point
            used for re-weighting the original snr time series.
        rw_snr : pycbc.TimeSeries
            The re-weighted time series.

        """

        # If not given the snr timeseries to reweight, calculate it first.
        if self.snr_ts is None:
            logging.info('Give an snr timeseries to reweight.')
            
        if pycbc_mf is True:
            mfs = []
            for seg in self.segs:
                
                snr_ts = matched_filter(seg, self.data, self.psd)
                snr_ts = snr_ts.crop(2,2)
                mfs.append(snr_ts)

        # Matched filter our segments
        else:
            if chi_mf_instance is None:
                chi_mf_instance = MatchedFilter(self.data,
                                            self.white_data,
                                            self.psd,
                                            self.temp_mem,
                                            self.snr_mem,
                                            self.corr_mem)
                chi_mf_instance.instantiate()
            mfs = []
            for seg in self.segs:
                fs_seg = make_frequency_series(seg)
                chi_mf_instance.white_matched_filter(seg,
                                                     fs_seg)

                # TODO: crop variables input
                crop_start = 3.0
                crop_end = 3.0
                crop_start_idx = int(crop_start * self.data.sample_rate)
                crop_end_idx = int(-(crop_end * self.data.sample_rate))
                crop_slice =  slice(crop_start_idx, crop_end_idx)
                snr_ts = self.snr_mem[crop_slice]
                snr_ts = TimeSeries(snr_ts,
                                    delta_t = 1/self.data.sample_rate,
                                    epoch = self.data.start_time + crop_start)
                mfs.append(snr_ts)

        # Calculate the chi sq
        chi_sq = sum(((abs(self.snr_ts/2 - mf))**2) for mf in mfs)/6

        # Reweighting
        delta = 1
        sigma_val = 1
        beta = 3
        alpha = 6

        # Re-weighting snr
        rw_snr = np.where(abs(chi_sq) < 1, abs(self.snr_ts), abs(self.snr_ts) * ( (delta + (abs(chi_sq)/sigma_val)**beta) / (delta+1) )**(-1/alpha))
        rw_snr = TimeSeries(rw_snr, delta_t = 1/self.data.sample_rate, epoch=self.snr_ts.start_time)

        return self.snr_ts, chi_sq, rw_snr

class MatchedFilter():
    def __init__(self,
                 data,
                 weighted_data,
                 psd,
                 temp_mem,
                 snr_mem,
                 corr_mem):

        self.data = data
        self.weighted_data = weighted_data
        self.psd = psd
        self.temp_mem = temp_mem
        self.snr_mem = snr_mem
        self.corr_mem = corr_mem
        self.correlator = None
        self.iffter = None

    def instantiate(self) -> None:

        data_time_length = self.data.duration

        # Low and high frequency cutoffs
        kmin = int(15 * data_time_length)
        kmax = int(1024 * data_time_length)
        corr_slice = slice(kmin, kmax)

        # Correlate setup
        self.correlator = CPUCorrelator(self.temp_mem[corr_slice],
                                        self.weighted_data[corr_slice],
                                        self.corr_mem[corr_slice])

        # IFFT setup
        self.iffter = IFFT(self.corr_mem,
                           self.snr_mem)

    def matched_filter_template(self,
                                template,
                                fs_template) -> None:

        interpolated_psd = interpolate(self.psd,
                                       template.delta_f)
        temp_norm = sigmasq(fs_template,
                            interpolated_psd,
                            low_frequency_cutoff=15,
                            high_frequency_cutoff=1024)

        self.temp_mem._data[:len(fs_template)] = fs_template._data[:]
        norm = (4.0 * self.data.delta_f) / np.sqrt(temp_norm)

        self.correlator.correlate()
        self.iffter.execute()

        self.snr_mem *= norm

    def white_matched_filter(self,
                             template,
                             fs_template):

        temp_norm = sigmasq(fs_template,
                            low_frequency_cutoff=15,
                            high_frequency_cutoff=1024)

        self.temp_mem._data[:len(fs_template)] = fs_template._data[:]
        norm = (4.0 * self.data.delta_f) / np.sqrt(temp_norm)

        self.correlator.correlate()
        self.iffter.execute()

        self.snr_mem *= norm

class CPUCorrelator(_BaseCorrelator):
    def __init__(self, x, y, z):
        self.x = np.array(x.data, copy=False)
        self.y = np.array(y.data, copy=False)
        self.z = np.array(z.data, copy=False)

    def correlate(self):
        self.z[:] = np.conjugate(self.x.data)
        self.z *= self.y


class GlitchSearch:
    """This class contains the methods for searching for artefacts
    in data.
    """

    def __init__(self):
        """
        Inputs
        ------
        self.data : pycbc.TimeSeries
            The data to be search over.
            Fetch or loaded in methods.
        self.psd : pycbc.FrequencySeries
            The power spectral density to use in the search.
            Calculated using the data.
        self.white_data : pycbc.TimeSeries
            The whitened original data.
            Whitenend in a method.
        self.template_bank : list[list[fringe_frequency, timeperiod]]
            The template bank containing all templates to be matched
            filtered with the data.
            Imported from an external file in a method.
        self.all_triggers : [[fringe_frequency, timeperiod, snr, time]]
              or [[fringe_frequency, timeperiod, snr, time, chi_value]]
            A list containing all of the triggers found by clustered
            signal-to-noise ratio time series of each template.
            This is found when matched filtering and clustering.
        self.lowest_timeperiod : float
            The lowest timeperiod in the template bank.
            Used for windowing lengths in clustering.
        self.peaks : [[fringe_frequency, timeperiod, snr, time]
               or [[fringe_frequency, timeperiod, snr, time, chi_value]]
            A list containing the highest signal-to-noise ratio triggers
            found for each period of cluster of artefacts.
            This is generated in the second stage of clustering.
        self.harmonics : [[fringe_frequency, timeperiod, snr, time]
               or [[fringe_frequency, timeperiod, snr, time, chi_value]]
            A list containing all of the triggers found in the data,
            including the harmonic artefacts in an artefact period.
            This is generated in the clustering in frequency method.
        """

        self.data = None
        self.psd = None
        self.white_data = None
        self.template_bank = None
        self.all_triggers = None
        self.lowest_timeperiod = None
        self.peaks = None
        self.harmonics = None

        self.perform_chi_sq = None
        self.pregenerated_splits = None
        self.weighted_data = None
        self.temp_mem = None
        self.snr_mem = None
        self.corr_mem = None

    # Preparation

    def get_LIGO_data(self,
                      frame_type: str,
                      channel_name: str,
                      start_time: float,
                      end_time: float) -> None:
        """Fetch gravitational wave data from LIGO sources.

        Inputs
        ------
        frame_type : str
            The frame to retrieve data from.
            e.g. 'H1_HOFT_C00'
        channel_name : str
            The name of the channel within the frame file.
            e.g. 'H1:GDS-CALIB_STRAIN'
        start_time : float
            The start time of the data.
        end_time : float
            The end time of the data.

        Outputs
        -------
        self.data : pycbc.TimeSeries
            A TimeSeries containing the data.

        """

        self.data = query_and_read_frame(frame_type,
                                         channel_name,
                                         np.floor(start_time),
                                         np.ceil(end_time))

    def get_local_data(self,
                       frame_loc: str,
                       frame_channel: str) -> None:
        """Fetch gravitational wave data from a local source.
            e.g. A frame file

        Inputs
        ------
        frame_loc : str
            The location of the frame file in the file system.
        frame_channel : str
            The name of the channel within the frame file.

        Outputs
        -------
        self.data : pycbc.TimeSeries
            A TimeSeries containing the data.

        """

        self.data = read_frame(frame_loc,
                               frame_channel)
        
    def get_from_cli(self,
                     args) -> None:
        
        self.data = pycbc.strain.from_cli(args,
                                          precision='double')
        
    def condition_data(self,
                       perform_highpass: bool = True,
                       highpass_limit: float = 15.0,
                       perform_resample: bool = True,
                       delta_t: float = 1/2048.0,
                       perform_crop: bool = True,
                       crop_start: float = 2,
                       crop_end: float = 2) -> None:
        """Condition the data for searching, with multiple options.

        Also create the self.fs_data object, which is used in to
        whiten and weight the data in other methods.

        Inputs
        ------
        perform_highpass : bool
            True/False | Peform a highpass, cutting frequencies below
                         a certain frequency.
            Default = True
        highpass_limit : float
            The frequency at which to cut below.
            Default = 15.0
        perform_resample : bool
            True/False | Peform resampling to a different sample rate.
            Default = True
        delta_t : float
            The delta_t to resample the data to.
            Default = 1/2048.0 (2048 Hz)
        perform_crop : bool
            True/False | Peform cropping of the data in time.
            Default = True
        crop_start : float
            The number of seconds to cut from the beginning of the data
            Default = 2 (2 seconds)
        crop_end : float
            The number of seconds to cut from the end of the data
            Default = 2 (2 seconds)

        Outputs
        -------
        self.fs_data : pycbc.FrequencySeries
            The frequency series of the data.

        """

        if perform_highpass is True:
            self.data = highpass(self.data, highpass_limit)

        if perform_resample is True:
            self.data = resample_to_delta_t(self.data, delta_t)

        if perform_crop is True:
            self.data = self.data.crop(crop_start, crop_end)

        self.fs_data = make_frequency_series(self.data)

    def generate_psd(self,
                     segment_duration: float = 4,
                     low_freq_cutoff: float = 15.0) -> None:
        """Generate a power spectral density of the data.

        Inputs
        ------
        segment_duration : float
            The segment duration used in the generation of the psd.
            Default = 4
        low_freq_cutoff : float
            The low frequency cutoff to apply to the psd when truncating
            Default = 15.0

        Outputs
        -------
        self.psd : pycbc.FrequencySeries
            The psd object, used through the workflow.

        """
        self.psd = self.data.psd(segment_duration)
        self.psd = interpolate(self.psd, self.data.delta_f)
        self.psd = inverse_spectrum_truncation(self.psd,
                                               int(4*self.data.sample_rate),
                                               low_frequency_cutoff = low_freq_cutoff)

    def whiten_data(self) -> None:
        """Whiten the data.

        Inputs
        ------

        Outputs
        -------
        self.white_data : pycbc.FrequencySeries
            The whitened data.

        """

        self.white_data = (self.fs_data / self.psd**0.5)

    def weight_data(self) -> None:
        """Weight the data by the PSD.

        Inputs
        ------

        Outputs
        -------
        self.weighted_data : pycbc.FrequencySeries
            The weighted data.

        """

        self.weighted_data = self.fs_data / self.psd

    def preallocate_memory(self) -> None:
        """Preallocate objects for memory purposes.

        Inputs
        ------

        Outputs
        -------
        self.temp_mem : zeros
            An array to store the template information.

        self.snr_mem : zeros
            An array to store the signal-to-noise ratio time series.

        self.corr_mem : zeros
            An array to store the correlation time series information.
        """

        tlen = len(self.data)
        data_time_length = self.data.duration
        self.temp_mem = zeros(tlen, dtype=np.complex128)
        self.snr_mem = zeros(tlen, dtype=np.complex128)
        self.corr_mem = zeros(tlen, dtype=np.complex128)

    def import_template_bank(self,
                             file_location: str,
                             pregenerated_splits: bool = False) -> None:
        """Import the template bank from a local file location

        Inputs
        ------
        file_location : str
            The location of the file containing the template bank on the
            local file storage system.

        Outputs
        -------
        self.template_bank : list
            The list of templates to matched filter with the data.

        """

        self.pregenerated_splits = pregenerated_splits
        self.template_bank = np.loadtxt(file_location, skiprows = 1)
        self.find_lowest_timeperiod()

    def find_lowest_timeperiod(self):
        """Find the lowest time period of any template within the
        template bank.

        Inputs
        ------

        Outputs
        -------
        self.lowest_timeperiod : float
            The lowest timeperiod found in the template bank.

        """
        self.lowest_timeperiod = min([template[1] for template in self.template_bank])

    # Matched Filtering and Clustering

    def matched_filter_template(self,
                                template: TimeSeries,
                                low_freq_cutoff: float = 15,
                                crop_start: float = 3,
                                crop_end: float = 3) -> TimeSeries:
        """Matched filter a template and the data.

        Inputs
        ------
        template : TimeSeries
            The template to match filter with the data.
        low_freq_cutoff : float
            The low frequency cutoff.
            Default = 15
        crop_start : float
            The number of seconds to crop from the beginning of the
            signal-to-noise ratio time series.
            Default =  2
        crop_end : float
            The number of seconds to crop from the end of the
            signal-to-noise ratio time series.
            Default = 2

        Outputs
        -------
        snr_ts : pycbc.TimeSeries
            The signal-to-noise ratio time series resulting from the
            matched filter of the template and data.

        """

        snr_ts = matched_filter(template,
                                self.data,
                                psd=self.psd,
                                low_frequency_cutoff = low_freq_cutoff,
                                sigmasq=None)
        snr_ts = snr_ts.crop(crop_start, crop_end)

        return snr_ts

    def class_matched_filter(self,
                             template,
                             fs_temp,
                             crop_start = 3.0,
                             crop_end = 3.0):

        self.weight_data()
        self.preallocate_memory()

        # Matched Filter the template
        mf = MatchedFilter(self.data,
                           self.weighted_data,
                           self.psd,
                           self.temp_mem,
                           self.snr_mem,
                           self.corr_mem)
        mf.instantiate()
        mf.matched_filter_template(template,
                                   fs_temp)

        crop_start_idx = int(crop_start * self.data.sample_rate)
        crop_end_idx = int(-(crop_end * self.data.sample_rate))
        crop_slice =  slice(crop_start_idx, crop_end_idx)
        snr_ts = self.snr_mem[crop_slice]
        snr_ts = TimeSeries(snr_ts,
                    delta_t = 1/self.data.sample_rate,
                    epoch = self.data.start_time + crop_start)

        return snr_ts

    def cluster_snr_timeseries(self,
                               snr_timeseries: TimeSeries,
                               window: float,
                               snr_limit: float = 8.0) -> list[complex, complex]:
        """Cluster the signal-to-noise ratio time series to generate a
        list of triggers.

        Inputs
        ------
        snr_timeseries : pycbc.TimeSeries
            The signal-to-noise ratio time series you want to cluster in
            time.
        window : float
            The clustering window.
        snr_limit : float
            The signal-to-noise ratio limit, any triggers found below
            this limit are discarded.
            Default = 8.0

        Outputs
        -------
        triggers : list[[snr, time]]
            The list of triggers, represented by the signal-to-noise
            ratios and time of the trigger. The clustering can produce
            multiple triggers.

        """

        snr_ts = np.column_stack((snr_timeseries, snr_timeseries.sample_times))
        snr_ts = snr_ts[abs(snr_ts[:,0]) > snr_limit]
        snr = abs(snr_ts[:,0])
        times = snr_ts[:,1]
        trigger_idxs = cluster_over_time(snr, times, window=window)
        triggers = snr_ts[trigger_idxs]
        triggers = abs(triggers[abs(triggers[:,0]) > snr_limit])

        return triggers, trigger_idxs

    def matched_filter_and_cluster(self,
                                   ffringe: float,
                                   timeperiod: float,
                                   snr_limit: float = 8.0,
                                   pregenerated_splits: list = None,
                                   window: float = None,
                                   perform_chi_sq: bool = True,
                                   mf_instance = None,
                                   chi_mf_instance = None) -> list[list[complex, complex]]:
        """Generate a template, matched filter with the data, re-weight
        with a chi squared test, cluster in time. Producing a list of
        triggers for further analysis in the workflow.

        Inputs
        ------
        ffringe : float
            The fringe frequency of the template.
        timeperiod : float
            The time period of the template.
        window : float
            The clustering window.
            Default = None (Calculated within the method)
        perform_chi_sq : bool
            True/False | Perform the chi squared test
            Default = True

        Outputs
        -------
        when perform_chi_sq is False:
            triggers : list
                The list of triggers [ffringe, timeperiod, snr, time_of]
            snr_ts : pycbc.TimeSeries
                The signal-to-noise-ratio time series of the template
                matched filtered with the data.

        when perform_chi_sq is True:
            full_triggers : list
                The list of triggers
                          [ffringe, timeperiod, snr, time_of, chi_value]
            rw_snr_ts : pycbc.TimeSeries
                The chi squared re-weighted snr time series.

        """
        
        # Generate the template
        gen_template = ArtefactGeneration(start_time = self.data.start_time,
                                           data_time_length =  self.data.duration,
                                           fringe_frequency = ffringe,
                                           timeperiod = timeperiod,
                                           amplitude = 1,
                                           phase = 0.0,
                                           pad = True,
                                           sample_rate = self.data.sample_rate,
                                           psd = self.psd)
        template = gen_template.generate_template()
        fs_temp = gen_template.generate_frequency_template()

        # Matched Filter the template
        if mf_instance is None:
            mf_instance = MatchedFilter(self.data,
                                        self.weighted_data,
                                        self.psd,
                                        self.temp_mem,
                                        self.snr_mem,
                                        self.corr_mem)
            mf_instance.instantiate()
        mf_instance.matched_filter_template(template,
                                            fs_temp)

        # TODO: crop variables input
        crop_start = 3.0
        crop_end = 3.0
        crop_start_idx = int(crop_start * self.data.sample_rate)
        crop_end_idx = int(-(crop_end * self.data.sample_rate))
        crop_slice =  slice(crop_start_idx, crop_end_idx)
        snr_ts = self.snr_mem[crop_slice]
        snr_ts = TimeSeries(snr_ts,
                            delta_t = 1/self.data.sample_rate,
                            epoch = self.data.start_time + crop_start)

        if len(snr_ts[abs(snr_ts) > snr_limit]) == 0:
            return

        if perform_chi_sq is False:
            if window is None:
                window = timeperiod/2
            triggers, trigger_idxs = self.cluster_snr_timeseries(snr_ts,
                                                   window = window,
                                                   snr_limit = snr_limit)
            return triggers, snr_ts

        white_template = gen_template.whiten_template(self.psd)
        
        if chi_mf_instance is None:
            chi_mf_instance = MatchedFilter(self.data,
                                            self.white_data,
                                            self.psd,
                                            self.temp_mem,
                                            self.snr_mem,
                                            self.corr_mem)
            chi_mf_instance.instantiate()

        chi_squared = ChiSquared(template,
                                 fs_temp,
                                 white_template,
                                 self.data,
                                 self.white_data,
                                 self.weighted_data,
                                 self.psd,
                                 ffringe,
                                 timeperiod,
                                 snr_ts,
                                 self.data.sample_rate,
                                 self.data.start_time,
                                 self.temp_mem,
                                 self.snr_mem,
                                 self.corr_mem)

        chi_squared.find_segment_widths(pregenerated_splits)
        chi_squared.split_template()
        og_snr_ts, chi_sq, rw_snr_ts = chi_squared.calculate_chi_sq(chi_mf_instance=chi_mf_instance)

        if len(rw_snr_ts[abs(rw_snr_ts) > snr_limit]) == 0:
            return

        if window is None:
            window = timeperiod/2
        triggers, trigger_idxs = self.cluster_snr_timeseries(rw_snr_ts,
                                                             window = window,
                                                             snr_limit = snr_limit)
        
        full_triggers = []
        for trigger in range(len(triggers)):
            
            rw_snr = triggers[trigger][0]
            og_snr_idx = list(abs(rw_snr_ts)).index(rw_snr)
            og_snr = og_snr_ts[og_snr_idx]
            time = triggers[trigger][1]
            
            dt = time - self.data.start_time
            aligned = template.cyclic_time_shift(dt)
            amplitude = abs(og_snr/sigma(aligned, psd=self.psd, low_frequency_cutoff = 15.0))
            phase = np.arctan2(og_snr.imag,og_snr.real)
            
            chi_sq_idx = int(np.round(float((time - rw_snr_ts.start_time) * rw_snr_ts.sample_rate)))
            chi_sq_value = chi_sq[chi_sq_idx]
            
#             print('Fringe Frequency: ', ffringe)
#             print('Timeperiod: ', timeperiod)
#             print('Original SNR: ', abs(og_snr))
#             print('Reweighted SNR: ', rw_snr)
#             print('Amplitude: ', amplitude)
#             print('Phase: ', phase)
#             print('Time of: ', time)
#             print('Chi Squared Value: ', chi_sq_value)

            # TODO: Output original SNR as well
            full_triggers.append([abs(og_snr), rw_snr, amplitude, phase, time, chi_sq_value])

        return full_triggers, rw_snr_ts

    def matched_filter_bank_and_cluster(self,
                                        window: float = None,
                                        snr_limit: float = 8.0,
                                        perform_chi_sq: bool = True) -> list[float]:
        """Matched filter the whole bank and cluster each template.

        Inputs
        ------
        window : float
            The clustering window.
            Default = None
        perform_chi_sq : bool
            True/False | Perform the chi squared test.
            Default = True

        Outputs
        -------
        self.all_triggers : list
            A list containing all of the triggers found by clustering
            all of the templates.

        """
        self.perform_chi_sq = perform_chi_sq
        self.whiten_data()
        self.weight_data()
        self.preallocate_memory()
        self.all_triggers = []
        tb_len = len(self.template_bank)

        j = 0
        
        mf_instance = MatchedFilter(self.data,
                                    self.weighted_data,
                                    self.psd,
                                    self.temp_mem,
                                    self.snr_mem,
                                    self.corr_mem)
        mf_instance.instantiate()
        
        chi_mf_instance = MatchedFilter(self.data,
                                        self.white_data,
                                        self.psd,
                                        self.temp_mem,
                                        self.snr_mem,
                                        self.corr_mem)
        chi_mf_instance.instantiate()
        
        for template in self.template_bank:
            j += 1
            if self.pregenerated_splits is False:
                ffringe, timeperiod = template
                splits = None
            if self.pregenerated_splits is True:
                ffringe, timeperiod, frac1, frac2, frac3, frac4 = template
                splits = [frac1, frac2, frac3, frac4]
            output = self.matched_filter_and_cluster(ffringe,
                                                     timeperiod,
                                                     snr_limit = snr_limit,
                                                     pregenerated_splits = splits,
                                                     window = window,
                                                     perform_chi_sq = self.perform_chi_sq,
                                                     mf_instance = mf_instance,
                                                     chi_mf_instance = chi_mf_instance)

            if output is None:
                #logging.info('Template: %s / %s - No Triggers', j, tb_len)
                continue
            triggers = output[0]
            snr_ts = output[1]
            logging.info('Template: %s / %s', j, tb_len)

            full_output = np.copy(triggers)
            full_output = np.insert(full_output, 0, [[ffringe], [timeperiod],  ], axis=1)

            for i in range(len(full_output)):
                self.all_triggers.append(full_output[i])

        self.all_triggers = np.array(self.all_triggers)


    def cluster_snr_peaks(self):
        """Cluster all triggers found by all templates.

        Inputs
        ------

        Outputs
        -------
        self.peaks : list
            A list containing the highest signal-to-noise ratio trigger
            found by the clustering within each cluster.

        Additional explanation:
        All triggers found by clustering all templates provides us with
        a picture of where artefacts lie within our data.

        By clustering, we find the highest snr trigger in each cluster
        of triggers which gives us the location and approximate
        time period of the artefacts within that cluster, especially
        if harmonics exist.

        """

        if self.perform_chi_sq is True:
            snrs = abs(self.all_triggers[:,3])
            times = self.all_triggers[:,6]
        else:
            snrs = abs(self.all_triggers[:,2])
            times = self.all_triggers[:,3]
            

        clusters = cluster_over_time(snrs,
                                     times,
                                     window=self.lowest_timeperiod)

        self.peaks = self.all_triggers[clusters]


    def calculate_artefact_periods(self):
        """Calculate the periods of time which contain artefacts.

        Inputs
        ------

        Outputs
        -------
        self.artefact_periods : list
            This list contains the start and end times of each period
            we have found to contain artefacts.

        """

        artefact_times = abs(self.peaks[:,6])
        self.artefact_periods = []
        for time_of in artefact_times:
            start_period = time_of - 0.05
            end_period = time_of + 0.05
            if start_period < self.data.start_time + 8.0:
                start_period = self.data.start_time + 8.0
                end_period  = self.data.start_time + 8.1
            self.artefact_periods.append([start_period, end_period])


    def generate_timeperiod_bounds(self):
        """Generate boundaries for the time period of each template for
        our artefact periods.

        Inputs
        ------

        Outputs
        -------
        self.timeperiod_bounds : list
            A list containing the bounds on the time periods of the
            templates we are looking at within our periods of artefacts.

        """

        timeperiods = abs(self.peaks[:,1])
        self.timeperiod_bounds = []

        for timeperiod in timeperiods:
            lower_timeperiod = timeperiod * 0.90
            upper_timeperiod = timeperiod * 1.10
            self.timeperiod_bounds.append([lower_timeperiod, upper_timeperiod])

    def cluster_in_frequency(self,
                             window: float = 4.0,
                             snr_limit:float = 8.0):
        """We can cluster our triggers by template fringe frequency,
        thereby locating the harmonic artefacts at the same time.

        Inputs
        ------
        window : float
            The clustering window (in Hz)

        Outputs
        -------
        self.harmonics : list
            A list containing all of the artefacts within our data,
            including harmonics
            no chi squared: [ffringe, timeperiod, snr, time_of]
            chi squared: [ffringe, timeperiod, snr, time_of, chi_value]

        """

        self.harmonics = []
        self.generate_timeperiod_bounds()

        for period, timeperiod in zip(self.artefact_periods, self.timeperiod_bounds):

            period_start = period[0]
            period_end = period[1]

            lower_T = timeperiod[0]
            upper_T = timeperiod[1]

            # Filter templates by boundaries surrounding the original best timeperiod found
            triggers_in_T = self.all_triggers[(self.all_triggers[:,1]>=lower_T) & (self.all_triggers[:,1]<=upper_T)]

            triggers_in_period = triggers_in_T[(abs(triggers_in_T[:,6])>=period_start) & (abs(triggers_in_T[:,6])<=period_end)]

            frequency_triggers = cluster_over_time(abs(triggers_in_period[:,3]), triggers_in_period[:,0], window = window)
            harmonics = triggers_in_period[frequency_triggers]
            for harmonic in harmonics:
                # TODO: Not sure I need this, the triggers should already be pre-filtered
                if abs(harmonic[2]) < snr_limit:
                    continue
                else:
                    self.harmonics.append(harmonic)

        self.harmonics = np.asarray(self.harmonics)
        
class GlitchSubtraction:
    def __init__(self,
                 artefacts,
                 performed_chi_sq=True,
                 artefact_periods=None,
                 cut_bank=None,
                 subtracted_data = None,
                 subtracted_artefacts = None):
    
        self.artefacts = artefacts
        self.performed_chi_sq = performed_chi_sq
        self.artefact_periods = artefact_periods
        self.removal_instance = None
        self.cut_bank = None
        self.subtracted_data = None
        self.subtracted_artefacts = []
        self.pregenerated_splits = False
        self.mf_instance = None
        self.chi_mf_instance = None
        
    def load_triggers_from_file(self, file_loc):

        self.artefacts = np.loadtxt(file_loc)

    def load_triggers_from_results(self, folder_loc):

        all_artefacts = []

        for folder in folder_names:
            folder_name = str(folder).strip(".0")
            folder_location = (folder_loc + folder_name)
            file_loc = folder_location + '/harmonics.txt'
            my_file = Path(file_loc)

            if my_file.is_file():
                artefacts = np.loadtxt(file_loc, skiprows=0, delimiter=' ')

                if len(artefacts) == 0:
                    continue

                if len(np.shape(artefacts)) == 1:
                    all_artefacts.append(artefacts)

                else:
                    for artefact in artefacts:
                        all_artefacts.append(artefact)

        all_artefacts = np.asarray(all_artefacts)
        print('Number of Artefact Triggers: ', len(all_artefacts))

        self.artefacts = all_artefacts
        
    def find_artefact_periods(self):

        if self.performed_chi_sq is True:
            artefact_times = self.artefacts[self.artefacts[:, 6].argsort()][:,6]
        if self.performed_chi_sq is False:
            artefact_times = self.artefacts[self.artefacts[:, 5].argsort()][:,5]

        previous_period = None
        artefact_periods = []

        for i in range(len(artefact_times)):

            # Calculate the 16 second period surrounding this trigger
            current_period = [artefact_times[i] - 8.0, artefact_times[i] + 8.0]

            # For the first trigger, set this period as the first period
            if previous_period == None:
                previous_period = current_period

            # If the start of the current period is before the end of the previous period then
            #  set the end of the previous period to be the end of the current period
            #  effectively extending the period by the current trigger period
            if (current_period[0] < previous_period[1]):
                previous_period[1] = current_period[1]

            # If these periods don't overlap, append the previous period as a complete period
            #  and set the previous period equal to the current period to calculate the next
            #  artefact period
            else:
                artefact_periods.append(previous_period)
                previous_period = current_period

        artefact_periods.append(previous_period)        
        self.artefact_periods = np.asarray(artefact_periods)
        
    def get_data(self,
                 source,
                 start_time,
                 end_time,
                 frame_loc=None,
                 frame_type=None,
                 channel_name=None,
                 args=None):
        
        # Grab a bit more data than needed
        start_time -= 6.0
        end_time += 6.0
        
        if source == 'local':
            self.removal_instance.get_local_data(frame_loc,
                                                 channel_name)
            data_start_time = self.removal_instance.data.start_time
            data_end_time = self.removal_instance.data.end_time
            data_duration = self.removal_instance.data.duration
            
            start_diff = start_time - data_start_time
            end_diff = data_end_time - end_time
            
            self.removal_instance.data = self.removal_instance.data.crop(start_diff, end_diff)
            
            if self.removal_instance.data.duration != (end_time - start_time):
                logging.info('Data Cropping Unsuccessful')
            
        elif source == 'LIGO':
            self.removal_instance.get_LIGO_data(frame_type,
                                                channel_name,
                                                start_time,
                                                end_time)
            
        elif source == 'from_cli':
            self.removal_instance.get_from_cli(self,
                                               args)
            
        self.removal_instance.condition_data(perform_highpass = True,
                       highpass_limit = 15.0,
                       perform_resample = True,
                       delta_t = 1/2048.0,
                       perform_crop = True,
                       crop_start = 2,
                       crop_end = 2)
            
            
        self.data = self.removal_instance.data
            
    def get_psd(self):
        self.removal_instance.generate_psd(segment_duration = 4,
                                           low_freq_cutoff = 15.0)
        self.psd = self.removal_instance.psd
        
    def get_template_bank(self,
                          file_location,
                          pregenerated_splits=True):
        self.removal_instance.import_template_bank(file_location,
                                                   pregenerated_splits = pregenerated_splits)
        self.template_bank = self.removal_instance.template_bank
        self.pregenerated_splits = pregenerated_splits
        
    def get_search_objects(self,
                           source,
                           start_time,
                           end_time,
                           frame_loc=None,
                           frame_type=None,
                           channel_name=None,
                           args=None,
                           template_bank_loc=None,
                           pregenerated_splits=True):
        
        self.removal_instance = GlitchSearch()
        
        self.get_data(source,
                      start_time,
                      end_time,
                      frame_loc,
                      frame_type,
                      channel_name,
                      args)
        
        self.get_psd()
        self.get_template_bank(template_bank_loc,
                               pregenerated_splits)
        
    def cut_template_bank(self):

        template_bank = self.template_bank
        print('Length of initial bank: ', len(template_bank))
        cut_bank = template_bank[(template_bank[:,1] > (self.artefacts[:,1].min()-0.25)) & (template_bank[:,1] < (self.artefacts[:,1].max()+0.25))]

        cut_bank = cut_bank[(cut_bank[:,0] > (self.artefacts[:,0].min()-1.0)) & (cut_bank[:,0] < (self.artefacts[:,0].max()+1.0))]

        print('Length of cut bank: ', len(cut_bank))
        print('')
        
    def cut_period_template_bank(self):

        if self.performed_chi_sq is True:
            artefacts_in_period = self.artefacts[(self.data.start_time < self.artefacts[:,6]) & (self.artefacts[:,6] < self.data.end_time)]
        if self.performed_chi_sq is False:
            artefacts_in_period = self.artefacts[(self.data.start_time < self.artefacts[:,5]) & (self.artefacts[:,5] < self.data.end_time)]

        template_bank = self.template_bank
        print('Length of initial bank: ', len(template_bank))
        cut_bank = template_bank[(template_bank[:,1] > (artefacts_in_period[:,1].min()-0.5)) & (template_bank[:,1] < (artefacts_in_period[:,1].max()+0.5))]
        cut_bank = cut_bank[(cut_bank[:,0] > (artefacts_in_period[:,0].min()-1.0)) & (cut_bank[:,0] < (artefacts_in_period[:,0].max()+artefacts_in_period[:,0].min()))]

        print('Length of cut bank: ', len(cut_bank))
        print('')

        self.cut_bank = cut_bank
              
    def subtract_artefact(self,
                          artefact,
                          data=None):
              
        # Generate a template
        if self.performed_chi_sq is True: 
            fringe_frequency = artefact[0]
            timeperiod = artefact[1]
            OG_SNR = artefact[2]
            RW_SNR = artefact[3]
            amplitude = artefact[4]
            phase = artefact[5]
            time_of = artefact[6]
            chi_sq_value = artefact[7]
        
        if self.performed_chi_sq is False:
            fringe_frequency = artefact[0]
            timeperiod = artefact[1]
            OG_SNR = artefact[2]
            amplitude = artefact[3]
            phase = artefact[4]
            time_of = artefact[5]
              
        gen_template = ArtefactGeneration(start_time = self.data.start_time,
                                          data_time_length = self.data.duration,
                                          fringe_frequency = fringe_frequency,
                                          timeperiod = timeperiod,
                                          amplitude = 1,
                                          phase = 0,
                                          pad = True,
                                          sample_rate = self.data.sample_rate,
                                          psd = self.psd)
        
        template = gen_template.generate_template()

        # Align the template with the right time and phase
        dt = time_of - self.data.start_time
        aligned = template.cyclic_time_shift(dt)
        
        real_comp = OG_SNR * np.cos(phase)
        imag_comp = OG_SNR * np.sin(phase)
        complex_snr = complex(real_comp, imag_comp)
        
        aligned /= sigma(aligned, psd=self.psd, low_frequency_cutoff = 15.0)
        aligned = (aligned.to_frequencyseries() * complex_snr).to_timeseries()
        aligned.start_time = self.data.start_time

        # Subtract the data
        if data is None:
            data = self.data
        subtracted_data = data - aligned
        
        self.subtracted_data = subtracted_data
        self.subtracted_artefacts.append(artefact)

        print('Template Subtracted: ', artefact)
              
    def find_best_template(self,
                           data):
        
        if data is None:
            data = self.data

        # Data related processes
        fs_data = make_frequency_series(data)
        white_data = (fs_data / self.psd**0.5)
        overwhitened_data = fs_data / self.psd
        self.removal_instance.preallocate_memory()
        
        self.mf_instance = MatchedFilter(data,
                                         overwhitened_data,
                                         self.psd,
                                         self.removal_instance.temp_mem,
                                         self.removal_instance.snr_mem,
                                         self.removal_instance.corr_mem)
        self.mf_instance.instantiate()
            
        self.chi_mf_instance = MatchedFilter(data,
                                             white_data,
                                             self.psd,
                                             self.removal_instance.temp_mem,
                                             self.removal_instance.snr_mem,
                                             self.removal_instance.corr_mem)
        self.chi_mf_instance.instantiate()

        snrs = []
        
        i = 1
        for template in self.cut_bank:
            print('Template number: ', i)
            i += 1
            # Import template parameters
            fringe_frequency = template[0]
            timeperiod = template[1]
            if self.pregenerated_splits is True:
                pregenerated_splits = [template[2], template[3], template[4], template[5]]
            elif self.pregenerated_splits is False:
                pregenerated_splits = None

            # Generate all forms of the template
            gen_template = ArtefactGeneration(start_time = data.start_time,
                                              data_time_length = data.duration,
                                              fringe_frequency = fringe_frequency,
                                              timeperiod = timeperiod,
                                              amplitude = 1e-21,
                                              phase = 0.0,
                                              pad = True,
                                              sample_rate = data.sample_rate,
                                              psd = self.psd)
            ts_template = gen_template.generate_template()
            fs_temp = gen_template.generate_frequency_template()
            white_template = gen_template.whiten_template(self.psd)

            # Matched filter the template and the data
            snr_ts = self.mf_instance.matched_filter_template(ts_template,
                                                              fs_temp)
            
            # TODO: crop variables input
            crop_start = 3.0
            crop_end = 3.0
            crop_start_idx = int(crop_start * data.sample_rate)
            crop_end_idx = int(-(crop_end * data.sample_rate))
            crop_slice =  slice(crop_start_idx, crop_end_idx)
            snr_ts = self.removal_instance.snr_mem[crop_slice]
            snr_ts = TimeSeries(snr_ts,
                                delta_t = 1/data.sample_rate,
                                epoch = data.start_time + crop_start)

            # Calculate the re-weighted snr of the template
            chi_squared = ChiSquared(ts_template,
                                    fs_temp,
                                    white_template,
                                    data,
                                    white_data,
                                    overwhitened_data,
                                    self.psd,
                                    fringe_frequency,
                                    timeperiod,
                                    snr_ts,
                                    data.sample_rate,
                                    data.start_time,
                                    self.removal_instance.temp_mem,
                                    self.removal_instance.snr_mem,
                                    self.removal_instance.corr_mem)

            chi_squared.find_segment_widths(pregenerated_splits)
            chi_squared.split_template()
            og_snr_ts, chi_sq, rw_snr_ts = chi_squared.calculate_chi_sq(chi_mf_instance = self.chi_mf_instance)

            max_idx = np.argmax(abs(rw_snr_ts))
            max_rw_snr = abs(rw_snr_ts[max_idx])
            og_snr = og_snr_ts[max_idx]
            max_og_snr = abs(og_snr_ts[max_idx])
            time_of = rw_snr_ts.sample_times[max_idx]
            chi_val = chi_sq[max_idx]
            
            # Find amplitude and phase
            dt = time_of - data.start_time
            aligned = ts_template.cyclic_time_shift(dt)
            amplitude = abs(og_snr/sigma(aligned, psd=self.psd, low_frequency_cutoff = 15.0))
            phase = np.arctan2(og_snr.imag,og_snr.real)
            
#             if max_og_snr < 6.0:
#                 continue
            
            if max_rw_snr < 6.0:
                continue
            snrs.append([fringe_frequency, timeperiod, max_og_snr, max_rw_snr, amplitude, phase, time_of, chi_val])

        if len(snrs) <= 1:
            return [0, 0, 0, 0, 0, 0, 0 ,0]
        bank_results = np.asarray(snrs)
        max_template_idx = np.argmax(bank_results[:,2])
        max_template = bank_results[max_template_idx]

        return max_template
              
    def subtract_multiple_artefacts(self,
                                    artefacts,
                                    data=None):
      
        if data is None:
            data = self.data

        for artefact in artefacts:
            self.subtract_artefact(artefact,
                                   data=data)
            data = self.subtracted_data

              
    def subtract_period_hierarchically(self):
        
        # We are assuming the period has already been loaded into data
        
        # Subtract the loudest signal first
        highest_snr = self.artefacts[np.argmax(self.artefacts[:,2])]
        
        self.subtract_artefact(highest_snr, data=self.data)
        
        snr_val = np.inf
        while snr_val > 8.0:
            best_template = self.find_best_template(self.subtracted_data)
            snr_val = best_template[2]

            if snr_val < 8.0:
                print('SNR Value: ', snr_val)
                print('Breaking: Less than limit')
                break
                
            self.subtract_artefact(best_template, data=self.subtracted_data)
            
            
    def save_subtracted_artefacts(self,
                                  file_name):
        
        np.savetxt(f'{file_name}.txt',
                   self.subtracted_artefacts,
                   header='Ffringe | Timeperiod | OG SNR | RW SNR | Amplitude | Phase | Artefact Time | Chi Sq Value',
                   fmt='%1.3f')
            
        
        
        
        
        