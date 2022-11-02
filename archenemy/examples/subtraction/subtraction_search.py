import os
import sys
import logging
from pycbc import init_logging

sys.path.insert(1, '/nfshome/store03/users/arthur.tolley/ArchEnemy/scattered-light-removal/scattered_light_removal/removal')
import removal_class

import subtraction_utils
import numpy
import argparse
import pycbc

parser = argparse.ArgumentParser(description="__doc__")
parser.add_argument('--verbose', action='count',
                    help="Flag: Do we want to print logging messages?")
parser.add_argument('--from-file', action='store_true',
                    help='Grabbing results from a single file.')
parser.add_argument('--artefacts-file', type=str,
                    help='The location of the artefacts file.')
parser.add_argument('--template-bank-file', type=str,
                    help='The location of the template bank file.')
parser.add_argument('--output-location', type=str,
                    help='The location of the output files.')

pycbc.strain.insert_strain_option_group(parser)
args = parser.parse_args()
init_logging(args.verbose)

artefact_loader = subtraction_utils.load_artefacts()
artefact_loader.load_triggers_from_file(args.artefacts_file)
artefacts = artefact_loader.artefacts
logging.info('Loaded Artefacts')

removal_instance = removal_class.GlitchSearch()

search_objects = subtraction_utils.search_objects(removal_instance)
search_objects.get_data(args = args)
logging.info('Loaded Data')
logging.info('Data Duration: %s', search_objects.data.duration)
search_objects.get_psd()
search_objects.get_template_bank(args.template_bank_file, 
                                 pregenerated_splits=True)
search_objects.cut_artefacts_down(artefacts,
                                  args.gps_start_time,
                                  args.gps_end_time)
search_objects.cut_template_bank()

subtraction_search = subtraction_utils.glitch_subtraction(search_objects.artefacts,
                                                          search_objects.template_bank,
                                                          search_objects.data,
                                                          search_objects.psd,
                                                          removal_instance,
                                                          performed_chi_sq = True)
logging.info('Subtraction Search:')
counter = subtraction_search.subtract_period_hierarchically()

if counter >= 1:
    subtraction_search.save_subtracted_artefacts(f'{args.output_location}/subtracted_artefacts')
logging.info('Subtraction Search: Finished')
logging.info(' Number of artefacts subtracted: %s', counter)