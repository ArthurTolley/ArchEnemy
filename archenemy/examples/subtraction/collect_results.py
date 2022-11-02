import numpy
from pathlib import Path

# Search for the right files

def load_triggers_from_results(folder_location, folder_names):

    all_artefacts = []

    for folder_name in folder_names:
        file_loc = (folder_location + str(folder_name))
        file_loc = file_loc + '/subtracted_artefacts.txt'
        my_file = Path(file_loc)

        if my_file.is_file():
            artefacts = numpy.loadtxt(file_loc, skiprows=1, delimiter=' ')

            if len(artefacts) == 0:
                continue

            if len(numpy.shape(artefacts)) == 1:
                all_artefacts.append(artefacts)

            else:
                for artefact in artefacts:
                    all_artefacts.append(artefact)

    all_artefacts = numpy.asarray(all_artefacts)
    print('Number of Artefact Triggers: ', len(all_artefacts))
    
    return all_artefacts

folder_location = '/home/arthur.tolley/ArchEnemy/scattered-light-removal/scattered_light_removal/subtraction/chunk23/BBH/'

folder_names = numpy.loadtxt('/home/arthur.tolley/ArchEnemy/scattered-light-removal/scattered_light_removal/subtraction/chunk23/artefact_periods.txt', dtype=int)[:,0]

all_subtracted_artefacts = load_triggers_from_results(folder_location, folder_names)

numpy.savetxt('all_subtracted_artefacts.txt', all_subtracted_artefacts)
