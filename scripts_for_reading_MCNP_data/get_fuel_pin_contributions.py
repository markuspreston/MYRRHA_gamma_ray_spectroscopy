from mctools.mcnp.mctal import MCTAL
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
from scipy.optimize import curve_fit
from scipy import integrate
import pandas

class tally_data:
    def __init__(self):
        self.tally_number = None
        self.n_bins = None


Energies = [0.5, 1.0, 1.5, 2.0, 2.5]
name = ['500keV', '1000keV', '1500keV', '2000keV', '2500keV']

N_seeds = 32

tally_IDs = [1, 3]           ##Tally 1: F18 (left detector), Tally 3: F38 (right detector)

photopeak_bin = np.zeros(len(name), dtype=int)


values = np.zeros(shape=(len(name), N_seeds, len(tally_IDs), 127, 504))                     ## 504 because I include the TOTAL bin as well as the energy bins (and epsilon bin)
d_values = np.zeros(shape=(len(name), N_seeds, len(tally_IDs), 127, 504))

for infile_no, infile in enumerate(name):
    for seed_no in range(N_seeds):
        seed = 1 + seed_no*100
        print('Seed ' + str(seed))
        filename = name[infile_no] + '/optimal_configuration_seed_' + str(seed) + 'm'
        m=MCTAL(filename)
        tallies = m.Read()      #get list of tallies from a mctal file

        for tally_no in range(2):
            tally = tallies[tally_IDs[tally_no]]      #F18
            n_bins_full = len(tally.erg)
            data_full = tally.valsErrors.reshape((128, n_bins_full + 1, 2))      ## N_bins + 1 because last bin is totals.
            
            bin_edges = tally.erg

            bin_lower_edges = tally.erg[:-1]       ## final entry is the upper edge of the last bin
            bin_upper_edges = tally.erg[1:]        ## ## First entry is negative entries (will not be incl)
            bin_width = bin_upper_edges - bin_lower_edges
            bin_centres = bin_lower_edges + 0.5*bin_width


            # For each energy, get the bin corresponding to the photopeak (e.g. 500 keV for the 500-keV monoenergetic source, ...)
            for i in range(len(bin_lower_edges)):
                if ((Energies[infile_no] >= bin_lower_edges[i]) and (Energies[infile_no] <= bin_lower_edges[i+1])):
                    photopeak_bin[infile_no] = i
                    break


            # Use the first 127 entries (the contributions from the pins, using SCX). The 128:th entry is the total over all pins
            entries = data_full[0:127,1:, 0]                ## do not include negative bin, DO INCLUDE total bin
            d_entries = data_full[0:127,1:, 1]                ## do not include negative bin, DO INCLUDE total bin
            
            values[infile_no, seed_no, tally_no] = entries
            d_values[infile_no, seed_no, tally_no] = d_entries*entries         ## it is the rel errors that are stored. conv to absolute



values_summed = np.sum(values, axis=1)/float(N_seeds)              ## average over all seeds
d_values_summed = np.sqrt(np.sum(np.power(d_values, 2), axis=1))/float(N_seeds)

# FOR WRITING THE PHOTOPEAK DATA:
column_names = ['Pin', '500keV', '1000keV', '1500keV', '2000keV', '2500keV']


## F18:
out_data = np.zeros((127, 1 + len(name)))
out_data[:,0] = np.arange(127)
out_data[:,1] = np.transpose(values_summed[0, 0, :, photopeak_bin[0]])
out_data[:,2] = np.transpose(values_summed[1, 0, :, photopeak_bin[1]])
out_data[:,3] = np.transpose(values_summed[2, 0, :, photopeak_bin[2]])
out_data[:,4] = np.transpose(values_summed[3, 0, :, photopeak_bin[3]])
out_data[:,5] = np.transpose(values_summed[4, 0, :, photopeak_bin[4]])

pd.DataFrame(out_data, columns=column_names).to_csv('../input_data/monoenergetic_F8_tally_values_leftdet.csv')

## d_F18:
out_data = np.zeros((127, 1 + len(name)))
out_data[:,0] = np.arange(127)
out_data[:,1] = np.transpose(d_values_summed[0, 0, :, photopeak_bin[0]])
out_data[:,2] = np.transpose(d_values_summed[1, 0, :, photopeak_bin[1]])
out_data[:,3] = np.transpose(d_values_summed[2, 0, :, photopeak_bin[2]])
out_data[:,4] = np.transpose(d_values_summed[3, 0, :, photopeak_bin[3]])
out_data[:,5] = np.transpose(d_values_summed[4, 0, :, photopeak_bin[4]])
pd.DataFrame(out_data, columns=column_names).to_csv('../input_data/monoenergetic_F8_tally_d_values_leftdet.csv')




## F38:
out_data = np.zeros((127, 1 + len(name)))
out_data[:,0] = np.arange(127)
out_data[:,1] = np.transpose(values_summed[0, 1, :, photopeak_bin[0]])
out_data[:,2] = np.transpose(values_summed[1, 1, :, photopeak_bin[1]])
out_data[:,3] = np.transpose(values_summed[2, 1, :, photopeak_bin[2]])
out_data[:,4] = np.transpose(values_summed[3, 1, :, photopeak_bin[3]])
out_data[:,5] = np.transpose(values_summed[4, 1, :, photopeak_bin[4]])
#out_data = np.concatenate([(np.arange(127))[:, None], np.transpose(values_summed[:, 1, :, int(photopeak_bin[0])])], axis=1)          ## [:,None] to cast to 2D array.

pd.DataFrame(out_data, columns=column_names).to_csv('../input_data/monoenergetic_F8_tally_values_rightdet.csv')


## d_F38:
out_data = np.zeros((127, 1 + len(name)))
out_data[:,0] = np.arange(127)
out_data[:,1] = np.transpose(d_values_summed[0, 1, :, photopeak_bin[0]])
out_data[:,2] = np.transpose(d_values_summed[1, 1, :, photopeak_bin[1]])
out_data[:,3] = np.transpose(d_values_summed[2, 1, :, photopeak_bin[2]])
out_data[:,4] = np.transpose(d_values_summed[3, 1, :, photopeak_bin[3]])
out_data[:,5] = np.transpose(d_values_summed[4, 1, :, photopeak_bin[4]])
pd.DataFrame(out_data, columns=column_names).to_csv('../input_data/monoenergetic_F8_tally_d_values_rightdet.csv')


