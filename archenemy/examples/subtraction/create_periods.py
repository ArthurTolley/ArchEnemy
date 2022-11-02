import subtraction_utils
import numpy
import argparse

parser = argparse.ArgumentParser(description="__doc__")
parser.add_argument('--verbose', action='count',
                    help="Flag: Do we want to print logging messages?")
parser.add_argument('--from-file', action='store_true',
                    help='Grabbing results from a single file.')
parser.add_argument('--from-folder', action='store_true',
                    help='Grabbing results from a results folder.')
parser.add_argument('--artefacts-file', type=str,
                    help='The location of the artefacts file.')
parser.add_argument('--results-folder', type=str,
                    help='The location of the results folder.')
parser.add_argument('--output-location', type=str,
                    help='The location of the output files.')
parser.add_argument('--output-name', type=str,
                    help='The name of the artefact period file.')
args = parser.parse_args()

artefact_loader = subtraction_utils.load_artefacts()

if args.from_file is True:
    if args.artefacts_file is False:
        print('You need to give the location of the artefacts file.')
    artefact_loader.load_triggers_from_file(args.artefacts_file)
    artefacts = artefact_loader.artefacts
    
if args.from_folder is True:
    if args.results_folder is False:
        print('You need to give the location of the results folder.')
    artefact_loader.load_triggers_from_results(args.results_folder)
    artefacts = artefact_loader.artefacts
    
search_objects = subtraction_utils.search_objects()
search_objects.find_artefact_periods(artefacts, performed_chi_sq=True)
artefact_periods = search_objects.artefact_periods

numpy.savetxt(f'{args.output_location}/{args.output_name}.txt', artefact_periods, fmt="%1.i")
