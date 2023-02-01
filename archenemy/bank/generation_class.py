import numpy as np
import logging
from typing import Union, List
import math

import pycbc
import pycbc.psd
from pycbc.psd import interpolate
from pycbc.filter import make_frequency_series, match
from pycbc.types.timeseries import TimeSeries, FrequencySeries

from search.artefact_template import ArtefactGeneration

class GenerateTemplateBank:
    def __init__(self,
                 glitch_frequency_range: List[float, float] = None,
                 timeperiod_range: List[float, float] = None,
                 match_limit: float = None,
                 maximum_templates: float = None,
                 psd: FrequencySeries = None,
                 sample_rate: float = 2048.0,
                 f_low: float = 15.0,
                 find_template_splits: bool = False):

        self.glitch_frequency_lower = glitch_frequency_range[0]
        self.glitch_frequency_upper = glitch_frequency_range[0]

        self.timeperiod_lower = timeperiod_range[0]
        self.timeperiod_upper = timeperiod_range[1]

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

    def generate_template(self):
        duration = self.timeperiod_upper
        gen_template = ArtefactGeneration(fringe_frequency = self.glitch_frequency,
                                          timeperiod = self.timeperiod,
                                          amplitude = 1e-21,
                                          phase = 0.0,
                                          sample_rate = self.sample_rate)

        gen_template.generate_template(duration = duration,
                                       start_time = 0.0,
                                       pad = True,
                                       make_timeseries=True,
                                       make_frequency_series=True)

        self.ts_template = gen_template.template
        self.fs_template = gen_template.fs_template

    def calculate_template_splits(self):

        interpolated_psd = interpolate(self.psd, self.template.delta_f)

        if len(fs_template) != len(interpolated_psd):
            interpolated_psd.resize(int(len(fs_template)))

        cut_idx = self.f_low * self.template.duration
        interpolated_psd = interpolated_psd[int(cut_idx+1):-1]
        fs_template = self.fs_template[int(cut_idx+1):-1]
        self.white_temp = (fs_template / interpolated_psd**0.5).to_timeseries()

        cumsum = np.cumsum((self.white_temp*self.white_temp))
        ex_frac = cumsum[-1] / 4

        idx = (np.abs(cumsum - ex_frac)).argmin()
        idx_2 = idx + (np.abs(cumsum[idx:] - ex_frac*2)).argmin()
        idx_3 = idx_2 + (np.abs(cumsum[idx_2:] - ex_frac*3)).argmin()
        idx_4 = len(cumsum) - idx_3

        self.frac_1 = (idx)/(self.timeperiod * self.sample_rate)
        self.frac_2 = (idx_2)/(self.timeperiod * self.sample_rate) - self.frac_1
        self.frac_3 = (idx_3)/(self.timeperiod * self.sample_rate) - self.frac_2 - self.frac_1
        self.frac_4 = (idx_4)/(self.timeperiod * self.sample_rate)

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
        self.glitch_frequency = np.random.uniform(low=self.glitch_frequency_lower,
                                                  high=self.glitch_frequency_upper)
        self.timeperiod = np.random.uniform(low=self.timeperiod_lower,
                                            high=self.timeperiod_upper)
        self.generate_template()
        first_template = (self.glitch_frequency,
                          self.timeperiod,
                          self.fs_template)

        if self.find_template_splits is True:
            self.calculate_template_splits()
            first_template = (self.glitch_frequency,
                              self.timeperiod,
                              self.fs_template,
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

            self.glitch_frequency = np.random.uniform(low=self.glitch_frequency_lower,
                                                    high=self.glitch_frequency_upper)
            self.timeperiod = np.random.uniform(low=self.timeperiod_lower,
                                                high=self.timeperiod_upper)
            self.generate_template()
            new_template = (self.glitch_frequency,
                            self.timeperiod,
                            self.fs_template)
            total_tried += 1

            if self.match_template_with_bank(new_template):
                since_template_added = 0
                if self.find_template_splits is True:
                    self.calculate_template_splits()
                    new_template = (self.glitch_frequency,
                                    self.timeperiod,
                                    self.fs_template,
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
