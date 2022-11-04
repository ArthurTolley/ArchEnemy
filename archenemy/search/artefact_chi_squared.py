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
This module contains functions for performing the scattered light chi squared test.
"""


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
