from pathlib import Path
import sys
import os
p = str(Path(os.path.abspath('')).parents[0])
sys.path.insert(1, p)

import pycbc
import pycbc.psd
import logging
from typing import Union
import math
from removal.removal_class import ArtefactGeneration
from pycbc.psd import interpolate
import numpy as np
from pycbc.filter import make_frequency_series, match
from pycbc.types.timeseries import TimeSeries, FrequencySeries

class GenerateTemplateBank:
    def __init__(self,
                 fringe_frequency_lower: float = None,
                 fringe_frequency_upper: float = None,
                 timeperiod_lower: float = None,
                 timeperiod_upper: float = None,
                 match_limit: float = None,
                 maximum_templates: float = None,
                 psd: Union[FrequencySeries, str]  = None,
                 sample_rate: float = 2048.0,
                 f_low: float = 15.0,
                 find_template_splits: bool = False):

        self.fringe_frequency_lower = fringe_frequency_lower
        self.fringe_frequency_upper = fringe_frequency_upper

        self.timeperiod_lower = timeperiod_lower
        self.timeperiod_upper = timeperiod_upper

        self.match_limit = match_limit
        self.maximum_templates = maximum_templates

        self.sample_rate = sample_rate

        self.template_bank = None

        self.white_temp = None
        self.find_template_splits = find_template_splits
        self.frac_1 = None
        self.frac_2 = None
        self.frac_3 = None
        self.frac_4 = None

        self.f_low = f_low
        self.psd = psd
        if not isinstance(self.psd, FrequencySeries):
            self.generate_psd()

    def generate_psd(self):
        try:
            tlen = int(self.timeperiod_upper * self.sample_rate)
            f_low = 15.0
            delta_f = 1.0 / self.timeperiod_upper
            flen = int(tlen//2 + 1)
            self.psd = pycbc.psd.from_string(self.psd, flen, delta_f, f_low)
        except:
            print('PSD not generated correctly.')

    def generate_template(self,
                          fringe_frequency: float,
                          timeperiod: float):
        data_time_length = self.timeperiod_upper
        gen_template = ArtefactGeneration(start_time = 0,
                                    data_time_length = data_time_length,
                                    fringe_frequency = fringe_frequency,
                                    timeperiod = timeperiod,
                                    amplitude = 1e-21,
                                    phase = 0.0,
                                    pad = True,
                                    sample_rate = self.sample_rate,
                                    psd = self.psd,
                                    roll = False)
        template = gen_template.generate_template()

        return template

    def calculate_template_splits(self,
                                  fringe_frequency: float,
                                  timeperiod: float):

        gen_template = ArtefactGeneration(start_time = 0,
                                          data_time_length = None,
                                          fringe_frequency = fringe_frequency,
                                          timeperiod = timeperiod,
                                          amplitude = 1e-21,
                                          phase = 0.0,
                                          pad = False,
                                          sample_rate = self.sample_rate,
                                          psd = self.psd)
        self.template = gen_template.generate_template()
        fs_template = make_frequency_series(self.template)

        interpolated_psd = interpolate(self.psd, self.template.delta_f)

        if len(fs_template) != len(interpolated_psd):
            interpolated_psd.resize(int(len(fs_template)))

        cut_idx = self.f_low * self.template.duration
        interpolated_psd = interpolated_psd[int(cut_idx+1):-1]
        fs_template = fs_template[int(cut_idx+1):-1]
        self.white_temp = (fs_template / interpolated_psd**0.5).to_timeseries()

        cumsum = np.cumsum((self.white_temp*self.white_temp))
        ex_frac = cumsum[-1] / 4

        idx = (np.abs(cumsum - ex_frac)).argmin()
        idx_2 = idx + (np.abs(cumsum[idx:] - ex_frac*2)).argmin()
        idx_3 = idx_2 + (np.abs(cumsum[idx_2:] - ex_frac*3)).argmin()
        idx_4 = len(cumsum) - idx_3

        self.frac_1 = (idx)/(timeperiod * self.sample_rate)
        self.frac_2 = (idx_2)/(timeperiod * self.sample_rate) - self.frac_1
        self.frac_3 = (idx_3)/(timeperiod * self.sample_rate) - self.frac_2 - self.frac_1
        self.frac_4 = (idx_4)/(timeperiod * self.sample_rate)

    def match_template_with_bank(self,
                                 template):
    # Sort the template bank by distance first
        self.template_bank = sorted(self.template_bank, key=lambda x: math.hypot(x[1] - template[1], (x[0]/10.0) - (template[0]/10.0)))

        # Below 0.5 counter
        counter = 0

        for i in range(len(self.template_bank)):
            match_number, _ = match(template[2], self.template_bank[i][2])

            if match_number <= 0.5:
                counter += 1
                if counter > 200:
                    # assume there aren't going to be any further-away templates which match better
                    return True

            if match_number > 0.5:
                counter = 0

            if match_number > self.match_limit:
                return False
        return True

    def generate_template_bank(self):
        first_template = (np.random.uniform(low=self.fringe_frequency_lower, high=self.fringe_frequency_upper),
                          np.random.uniform(low=self.timeperiod_lower, high=self.timeperiod_upper))
        first_ts = self.generate_template(first_template[0], first_template[1])
        first_fs = make_frequency_series(first_ts)
        first_template = (first_template[0],
                          first_template[1],
                          first_fs)

        if self.find_template_splits is True:
            self.calculate_template_splits(first_template[0],
                                           first_template[1])
            first_template = (first_template[0],
                              first_template[1],
                              first_fs,
                              self.frac_1,
                              self.frac_2,
                              self.frac_3,
                              self.frac_4)
            self.template_bank = [first_template]
        else:
            self.template_bank = [first_template]


        N_templates = 1
        since_template_added = 0
        total_tried = 1
        while N_templates < self.maximum_templates:
            print('Number of templates in bank: ', len(self.template_bank))

            new_template = (np.random.uniform(low=self.fringe_frequency_lower, high=self.fringe_frequency_upper),
                            np.random.uniform(low=self.timeperiod_lower, high=self.timeperiod_upper))
            new_ts = self.generate_template(new_template[0], new_template[1])
            new_fs = make_frequency_series(new_ts)
            new_template = (new_template[0], new_template[1], new_fs)
            total_tried += 1

            if self.match_template_with_bank(new_template):
                since_template_added = 0
                if self.find_template_splits is True:
                    self.calculate_template_splits(new_template[0],
                                                   new_template[1])
                    new_template = (new_template[0],
                                    new_template[1],
                                    new_fs,
                                    self.frac_1,
                                    self.frac_2,
                                    self.frac_3,
                                    self.frac_4)
                    self.template_bank.append(new_template)
                else:
                    self.template_bank.append(new_template)
                N_templates += 1
            else:
                since_template_added += 1
                if since_template_added > 250:
                    break

        return self.template_bank, total_tried
