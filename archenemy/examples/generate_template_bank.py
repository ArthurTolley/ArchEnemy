from pathlib import Path
import sys
import os
p = str(Path(os.path.abspath('')).parents[0])
sys.path.insert(1, p)

from generation_class import GenerateTemplateBank
import time
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()

find_template_splits = True
ffr_low = 20.0
ffr_up = 80.0
T_low = 1.8
T_up = 5.5
match_lim = 0.97
max_temps = 2000000

print('Generating Bank: Started')
bank_gen = GenerateTemplateBank(fringe_frequency_lower = ffr_low,
                                fringe_frequency_upper = ffr_up,
                                timeperiod_lower = T_low,
                                timeperiod_upper = T_up,
                                match_limit = match_lim,
                                maximum_templates = max_temps,
                                psd = 'aLIGOZeroDetHighPower',
                                sample_rate = 2048.0,
                                f_low = 15.0,
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
    np.savetxt(f'{T_low}_{T_up}_{ffr_low}_{ffr_up}_{match_lim}.txt',
            final_t_bank,
            header='Ffringe Timeperiod Frac1 Frac2 Frac3 Frac4| Time taken = {} | Total tried = {}'.format((end_time - start_time), total_tried))
if find_template_splits is False:
    zipped_t_bank = list(zip(*template_bank))
    final_t_bank = list(zip(zipped_t_bank[0],
                            zipped_t_bank[1]))
    np.savetxt(f'{T_low}_{T_up}_{ffr_low}_{ffr_up}_{match_lim}.txt',
            final_t_bank,
            header='Ffringe Timeperiod | Time taken = {} | Total tried = {}'.format((end_time - start_time), total_tried))

print('Saving bank to file: Completed')

print('Plotting template bank: Starting')
t_bank = np.asarray(final_t_bank)
ffringe = t_bank[:,0]
timeperiod = t_bank[:,1]
plt.figure(figsize=(8,8))
plt.title(f'Timeperiod range: {T_low} - {T_up} \n Fringe Frequency range: {ffr_low} - {ffr_up} \n Match Limit: {match_lim} \n Number of Templates: {len(t_bank)}')
plt.xlabel('Timeperiod [s]')
plt.ylabel('Fringe Frequency [Hz]')
plt.scatter(timeperiod, ffringe, marker='.', s=2.0)
plt.savefig(f'{T_low}_{T_up}_{ffr_low}_{ffr_up}_{match_lim}.png')
plt.close()
print('Plotting Template Bank: Completed')
