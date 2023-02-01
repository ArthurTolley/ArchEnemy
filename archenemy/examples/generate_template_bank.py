from bank.generation_class import GenerateTemplateBank
import time
import numpy as np

import pycbc.psd

# Time how long template bank generation takes
start_time = time.time()

# Define template bank parameters
glitch_frequency_range = [20.0, 80.0]
timeperiod_range = [1.8, 5.5]
match_limit = 0.97
max_temps = 2000000
sample_rate = 2048.0
f_low = 15.0
find_template_splits = True

# Generate a psd
tlen = int(timeperiod_range[1] * sample_rate)
delta_f = 1.0 / timeperiod_range[1]
flen = int(tlen//2 + 1)
psd = pycbc.psd.from_string('aLIGOZeroDetHighPower',
                            flen,
                            delta_f,
                            f_low)


print('Generating Bank: Started')
bank_gen = GenerateTemplateBank(glitch_frequency_range = glitch_frequency_range,
                                timeperiod_range = timeperiod_range,
                                match_limit = match_limit,
                                maximum_templates = max_temps,
                                psd = psd,
                                sample_rate = sample_rate,
                                f_low = f_low,
                                find_template_splits = find_template_splits)

template_bank, total_tried = bank_gen.generate_template_bank()
print('Generating Bank: Completed')
end_time = time.time()

# We only want the Timeperiods and Fringe Frequencies out, not the frequency series'
print('Saving bank to file: Started')
if find_template_splits is True:
    zipped_t_bank = list(zip(*template_bank))
    final_t_bank = list(zip(zipped_t_bank[0],
                            zipped_t_bank[1],
                            zipped_t_bank[3],
                            zipped_t_bank[4],
                            zipped_t_bank[5],
                            zipped_t_bank[6] ))
    np.savetxt('Template_Bank.txt',
            final_t_bank,
            header='Ffringe Timeperiod Frac1 Frac2 Frac3 Frac4| Time taken = {} | Total tried = {}'.format((end_time - start_time), total_tried))
if find_template_splits is False:
    zipped_t_bank = list(zip(*template_bank))
    final_t_bank = list(zip(zipped_t_bank[0],
                            zipped_t_bank[1]))
    np.savetxt('Template_Bank.txt',
               final_t_bank,
               header='Ffringe Timeperiod | Time taken = {} | Total tried = {}'.format((end_time - start_time), total_tried))

print('Saving bank to file: Completed')

