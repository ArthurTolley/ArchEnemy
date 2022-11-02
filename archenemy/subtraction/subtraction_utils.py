# Imports
import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '/nfshome/store03/users/arthur.tolley/ArchEnemy/scattered-light-removal/scattered_light_removal/removal')
import removal_class
import frame_plotting_utils

from pycbc.filter import sigma, make_frequency_series
from pycbc.types.timeseries import TimeSeries

class load_artefacts:
    def __init__(self):
        
        self.artefacts = None
        
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
        logging.info('Number of Artefact Triggers: %s', len(all_artefacts))

        self.artefacts = all_artefacts
        
class search_objects:
    def __init__(self,
                 removal_instance=None):
        
        self.artefacts = None
        self.removal_instance = removal_instance
        
    def find_artefact_periods(self,
                              artefacts,
                              performed_chi_sq=True):
        
        self.artefacts = artefacts

        if performed_chi_sq is True:
            artefact_times = self.artefacts[self.artefacts[:, 6].argsort()][:,6]
        if performed_chi_sq is False:
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
                 frame_loc=None,
                 frame_type=None,
                 channel_name=None,
                 args=None):

        self.removal_instance.get_from_cli(args)

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
        
    def cut_artefacts_down(self,
                           artefacts,
                           period_start,
                           period_end):
        
        logging.info('Original number of artefacts: %s', len(artefacts))
        artefacts = artefacts[(artefacts[:,6] > period_start) & (artefacts[:,6] < period_end)]
        logging.info('Cut number of artefacts: %s', len(artefacts))
        self.artefacts = artefacts
               
    def cut_template_bank(self):

        logging.info('Length of initial bank: %s', len(self.template_bank))
        self.template_bank = self.template_bank[(self.template_bank[:,1] > (self.artefacts[:,1].min()-0.25)) & (self.template_bank[:,1] < (self.artefacts[:,1].max()+0.25))]

        self.template_bank = self.template_bank[(self.template_bank[:,0] > (self.artefacts[:,0].min()-1.0)) & (self.template_bank[:,0] < (self.artefacts[:,0].max()+1.0))]

        logging.info('Length of cut bank: %s', len(self.template_bank))

class glitch_subtraction:
    def __init__(self,
                 artefacts,
                 template_bank,
                 data,
                 psd,
                 removal_instance,
                 performed_chi_sq=True):
    
        self.artefacts = artefacts
        self.template_bank = template_bank
        self.data = data
        self.psd = psd
        self.performed_chi_sq = performed_chi_sq
        self.removal_instance = removal_instance
        
        self.subtracted_data = None
        self.subtracted_artefacts = []
                         
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
              
        gen_template = removal_class.ArtefactGeneration(start_time = self.data.start_time,
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

        #logging.info('Template Subtracted: %s', artefact)
        logging.info('SNR of Subtracted Artefact: %s', OG_SNR)
              
    def find_best_template(self,
                           data):
        
        if data is None:
            data = self.data

        # Data related processes
        fs_data = make_frequency_series(data)
        white_data = (fs_data / self.psd**0.5)
        overwhitened_data = fs_data / self.psd
        self.removal_instance.preallocate_memory()
        
        self.mf_instance = removal_class.MatchedFilter(data,
                                         overwhitened_data,
                                         self.psd,
                                         self.removal_instance.temp_mem,
                                         self.removal_instance.snr_mem,
                                         self.removal_instance.corr_mem)
        self.mf_instance.instantiate()
            
        self.chi_mf_instance = removal_class.MatchedFilter(data,
                                             white_data,
                                             self.psd,
                                             self.removal_instance.temp_mem,
                                             self.removal_instance.snr_mem,
                                             self.removal_instance.corr_mem)
        self.chi_mf_instance.instantiate()

        snrs = []
        
        i = 1
        for template in self.template_bank:
            if i % 50 == 0:
                logging.info('Template number: %s / %s', i, len(self.template_bank))
            i += 1
            # Import template parameters
            fringe_frequency = template[0]
            timeperiod = template[1]
            pregenerated_splits = [template[2], template[3], template[4], template[5]]

            # Generate all forms of the template
            gen_template = removal_class.ArtefactGeneration(start_time = data.start_time,
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
            chi_squared = removal_class.ChiSquared(ts_template,
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
            
            if max_rw_snr < 8.0:
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

              
    def subtract_period_hierarchically(self,
                                       initial_subtraction=False):
        
        # We are assuming the period has already been loaded into data
        
        # If initial_subtraction is True:
        #  Subtract the loudest signal first
        if initial_subtraction is True:
            highest_snr = self.artefacts[np.argmax(self.artefacts[:,2])]

            self.subtract_artefact(highest_snr, data=self.data)
        
        snr_val = np.inf
        counter = 0
        while snr_val > 8.0:
            best_template = self.find_best_template(self.subtracted_data)
            snr_val = best_template[2]

            if snr_val < 8.0:
                logging.info('SNR Value: %s', snr_val)
                logging.info('Breaking: Less than limit')
                break
                
            self.subtract_artefact(best_template, data=self.subtracted_data)
            counter += 1
            
        return counter
            
    def save_subtracted_artefacts(self,
                                  file_name):
        
        np.savetxt(f'{file_name}.txt',
                   self.subtracted_artefacts,
                   header='Ffringe | Timeperiod | OG SNR | RW SNR | Amplitude | Phase | Artefact Time | Chi Sq Value',
                   fmt='%1.3f %1.3f %1.3f %1.3f %1.3e %1.3f %1.3f %1.3f')
        frame_plotting_utils.plot_spectrogram(self.data,
                                              q_range=[40,40])
        plt.savefig(f'{file_name}_init.png')
        plt.close()
        frame_plotting_utils.plot_overlayed_spectrogram(self.subtracted_data,
                                                        np.asarray(self.subtracted_artefacts),
                                                        q_range=[40,40])
        plt.savefig(f'{file_name}_sub.png')
        plt.close()
