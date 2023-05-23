# ArchEnemy
Matched filter search for scattered light artefacts in Gravitational-Wave data.

The results for the ArchEnemy paper are contained with the directory "paper_results".  
These are grouped by the detector seen in (LIGO-Hanford - H1, or LIGO-Livingston, L1) alongside the injection set used.  
These are the injection sets used in the O3b GWTC-3: http://arxiv.org/abs/2111.03606 

To subtract these glitches from the PyCBC GW search you must modify your clone of the PyCBC github repository and include the new file `glitch_subtraction.py` and modify the file `strain.py` within the `pycbc/strain/` directory.

For any questions on how to use these results and how they are laid out, please don't hesistate to contact me at: arthur.tolley@port.ac.uk
