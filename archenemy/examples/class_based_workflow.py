import removal_class
import argparse
import sys
import os
import logging
import time
import pycbc.strain
import numpy as np

from pycbc import init_logging


parser = argparse.ArgumentParser(description="__doc__")
parser.add_argument('--verbose', action='count',
                    help="Flag: Do we want to print logging messages?")

# Input data
parser.add_argument('--data-source', choices=['LIGO', 'local', 'from_cli'],
                    default='any', required=True,
                    help='Make a choice between retrieving data from LIGO '
                         'servers, using a local .gwf file or the pycbc'
                         'from_cli function.')

parser.add_argument('--LIGO-frame-type', type=str,
                    help='The LIGO frame type, used to fetch the data.')
parser.add_argument('--local-file-location', type=str,
                    help='The location of the file on your machine.')

parser.add_argument('--duration', type=float,
                    help="Start time and duration instead of start time and "
                         "end time.")

parser.add_argument('--optimize', type=bool,
                    help='If present, the code optimizes all artefacts found '
                         'post clustering.')

parser.add_argument('--snr-limit', type=float, default=8.0,
                    help='The snr limit to apply to triggers found by clustering.')

parser.add_argument('--template-bank-location', type=str,
                    required=True,
                    help='The location of the template bank within your file '
                         'system.')
parser.add_argument('--pregenerated-splits', action='store_true',
                    help='Add this argument if your template bank contains'
                    'pregenerated segments widths. Each row should look'
                    'like: [ffringe, timeperiod, frac1, frac2, frac3, frac4')
parser.add_argument('--lowest-timeperiod', type=float, default=0.5,
                    help='The lowest timeperiod is used for the clustering window.')

parser.add_argument('--output-location', type=str, required=True,
                    help='The location of the folder to save workflow outputs '
                         'to.')
parser.add_argument('--output-frame-name', type=str, required=True,
                    help='The name you wish the output subtracted frame file '
                         'to be called.')
parser.add_argument('--output-frame-channel', type=str, required=True,
                    help='The name of the frame channel to use when saving.')

parser.add_argument('--optional-plotting', action='store_true',
                    help='Add this argument to enable optional plotting.')
parser.add_argument('--disable-chi-sq', action='store_true',
                    help='Add this argument to disable the chi squared.')

pycbc.strain.insert_strain_option_group(parser)
args = parser.parse_args()

# Setup logging utilities
init_logging(args.verbose)
start_time = time.time()

if args.disable_chi_sq is True:
    perform_chi_sq = False
else:
    perform_chi_sq = True

if args.pregenerated_splits is True:
    pregenerated_splits = True
else:
    pregenerated_splits = False

# Set up output folder
print(args.output_location)
if not os.path.exists(f'{args.output_location}'):
    logging.info('Creating output folder at: %s', args.output_location)
    os.makedirs(f'{args.output_location}')

logging.info('Beginning Scattered Light workflow:')

logging.info('Initializing search class:')
workflow = removal_class.GlitchSearch()
logging.info('    Instance initialized.')

# Retrieve gravitational wave data frame
if args.data_source == 'LIGO':
    logging.info('Retrieving data from LIGO storage.')
    if (args.gps_end_time is None) & (args.duration is not None):
        workflow.get_LIGO_data(args.LIGO_frame_type,
                               args.channel_name,
                               args.gps_start_time,
                               args.gps_start_time + args.duration)
    elif args.gps_end_time is not None:
        workflow.get_LIGO_data(args.LIGO_frame_type,
                               args.channel_name,
                               args.gps_start_time,
                               args.gps_end_time)
    else:
        logging.info('Workflow requires: \n'
                      'start time & duration OR \n'
                      'start time & end time')
        logging.info('Ending workflow')
        logging.info(' Time taken: %s', time.time()-start_time)
        sys.exit(0)
if args.data_source == 'local':
    logging.info('Loading data from local frame file.')
    logging.info('Location: %s', args.local_file_location)
    workflow.get_local_data(args.local_file_location,
                            args.channel_name)
if args.data_source == 'from_cli':
    logging.info('Using pycbc.strain.from_cli to get data:')
    workflow.get_from_cli(args)
if workflow.data is None:
    logging.info("You've messed up somewhere mate")
    logging.info("Ending workflow")
logging.info('Data loaded.')

# Conditioned the data
logging.info('Conditioning data.')
workflow.condition_data(perform_highpass = True,
                        highpass_limit = 15.0,
                        perform_resample = True,
                        delta_t = 1/2048.0,
                        perform_crop = True,
                        crop_start = 2,
                        crop_end = 2)
logging.info('Data conditioned.')

if args.optional_plotting is True:
    import frame_plotting_utils
    import matplotlib.pyplot as plt

    logging.info('Plotting initial data.')
    logging.info(' Saving image as: %s_initial.png', args.output_frame_name)
    frame_plotting_utils.plot_spectrogram(workflow.data,
                                          q_range=[40.0, 40.0])
    plt.savefig(f'{args.output_location}/{args.output_frame_name}_initial.png')
    plt.close()

# Generate PSD
logging.info('Generating PSD from data.')
workflow.generate_psd(segment_duration = 4,
                      low_freq_cutoff = 15.0)
logging.info('PSD generated.')

# Import the template bank
logging.info('Loading template bank and splits from local file.')
workflow.import_template_bank(args.template_bank_location,
                              pregenerated_splits= pregenerated_splits)
logging.info('Template bank loaded.')

# Matched filter and cluster
logging.info('Begin matched filtering of template bank and data: ')
logging.info(' Length of template bank: %s', len(workflow.template_bank))
logging.info(' Duration of data: %s', workflow.data.duration)
workflow.matched_filter_bank_and_cluster(window = None,
                                         perform_chi_sq = perform_chi_sq,
                                         snr_limit = args.snr_limit)
logging.info('Matched filtering and first round of clustering complete: ')
logging.info(' Number of peaks found: %s', len(workflow.all_triggers))

if len(workflow.all_triggers) == 0:
    logging.info('No artefacts found.')
    logging.info('Ending workflow.')
    logging.info(' Time taken: %s', time.time()-start_time)
    sys.exit(0)

# Clustering all peaks in time
logging.info('Begin clustering all peaks in time:')
workflow.cluster_snr_peaks()
if workflow.peaks is False:
    logging.info('No artefacts found.')
    logging.info('Ending workflow.')
    logging.info(' Time taken: %s', time.time()-start_time)
    file = open(f"{args.output_location}/none.txt", "w")
    file.write("No artefacts found.")
    file.close()
    sys.exit(0)
logging.info(' Number of artefact periods found: %s ', len(workflow.peaks))

if args.optional_plotting is True:
    logging.info('Plotting all peaks and clusters found.')
    plt.figure(figsize=(4, 4))
    plt.scatter(workflow.all_triggers[:,6],
                abs(workflow.all_triggers[:,3]),
                s=2)
    plt.vlines([workflow.peaks[:,6]],
                ymin=8,
                ymax=max(abs(workflow.all_triggers[:,3])),
                color='r',
                alpha=0.6)
    plt.xlabel('Time [s]')
    plt.ylabel('Signal-to-Noise Ratio')
    plt.savefig(f'{args.output_location}/{args.output_frame_name}_peaks.png')
    plt.close()

# Begin looking for harmonics
logging.info('Begin harmonic search:')

# Generate artefact periods
logging.info(' Generate artefact periods:')
workflow.calculate_artefact_periods()
logging.info(' Artefact periods generated.')
logging.info(' Artefact periods: %s', workflow.artefact_periods)

# Cluster max snr values in frequency within artefact periods
logging.info(' Cluster triggers found in the time period band within the artefact periods: ')
workflow.cluster_in_frequency(snr_limit = args.snr_limit)
logging.info(' Clustering completed')
logging.info(' Number of harmonics found: %s', len(workflow.harmonics))
if len(workflow.harmonics) == 0:
    logging.info('No artefacts found.')
    logging.info('Ending workflow.')
    logging.info(' Time taken: %s', time.time()-start_time)
    sys.exit(0)
logging.info(' Saving harmonics to text file: harmonics.txt')
if perform_chi_sq is True:
    np.savetxt(f'{args.output_location}/harmonics.txt',
               workflow.harmonics,
               delimiter=',',
               header='Ffringe | Timeperiod | Original SNR | Reweighted SNR | Amplitude | Phase | Artefact Time | Chi Value',
               fmt='%1.3f %1.3f %1.3f %1.3f %1.3e %1.3f %1.3f %1.3f')
if perform_chi_sq is False:
    np.savetxt(f'{args.output_location}/harmonics.txt',
               workflow.harmonics,
               delimiter=',',
               header='Ffringe | Timeperiod | Original SNR | Amplitude | Phase | Artefact Time',
               fmt='%1.3f %1.3f %1.3f %1.3e %1.3f %1.3f' )

if args.optional_plotting is True:
    logging.info('Plotting frequency harmonics.')
    plt.figure(figsize=(8, 4))
    plt.scatter(workflow.harmonics[:,6], abs(workflow.harmonics[:,0]), s=2)
    plt.vlines([workflow.peaks[:,6]],
                ymin=8,
                ymax=max(abs(workflow.all_triggers[:,0])),
                color='r',
                alpha=0.6)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.savefig(f'{args.output_location}/{args.output_frame_name}_freq.png')
    plt.close()

# Subtraction of optimized templates from conditioned data
#  Subtraction class not implemented yet

if args.optional_plotting is True:
    logging.info('Plotting overlayed artefacts.')
    logging.info(' Saving image as: %s_overlay.png', args.output_frame_name)
    frame_plotting_utils.plot_overlayed_spectrogram(workflow.data,
                                                    workflow.harmonics,
                                                    chi_sq_performed = perform_chi_sq)
    plt.savefig(f'{args.output_location}/{args.output_frame_name}_overlay.png')
    plt.close()


logging.info('Ending workflow.')
logging.info(' Time taken: %s', time.time()-start_time)
