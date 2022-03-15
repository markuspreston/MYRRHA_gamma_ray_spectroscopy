import glob
from mctools.mcnp.mctal import MCTAL
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
from scipy.optimize import curve_fit
from scipy import integrate
import pandas
#import ROOT
#from ROOT import gROOT, TFile, TTree, TCanvas, TPad, TH1F, TH2F, TImage, TGraph
#from ROOT import TPaveLabel, TMultiGraph, TProfile, TSpectrum, TSpectrumFit


#seeds = np.arange(1, 6401, 100)
#N_seeds = len(seeds)

CT = [0.3, 1, 5]            ## NOTE: If CT < 1 (e.g. 0.3), that means 0.MONTHS. So 0.3 means 3 months, NOT 0.3 years!
r = [10, 12, 14]
d = [20, 40, 60]
h = [0.5, 1.0]

N_CT = len(CT)
N_r = len(r)
N_d = len(d)
N_h = len(h)


# The tallies F18 and F38 contain the unbroadened spectra of the two detectors. The data in these tallies shall be averaged (assume statistically independent, and done to improve statistics).
column_headers_F18 = ['CT', 'r', 'd', 'h', 'epsilon_bin']
column_headers_F18 = np.concatenate([column_headers_F18, np.arange(0, 500 + 2, 1), ['TOTAL']], axis=0)               ## 500 + 2 because I set up the bin upper edges as 1e-3 500I 3e0, which means 500 steps between 1e-3 and 3e0. Adding the bins ending with 1e-3 and 3e0 gives 502 bins. This does not include the epsilon bin (0 to 1e-5), which is added in a column called epsilon

final_data_values_F18 = np.zeros((N_CT, N_r, N_d, N_h, 508))                ## 508 to contain the CT, r, d, h values, the epsilon bin, the 502 "proper" energy bins, the total bin (=> 508 parameters in total)
final_data_d_values_F18 = np.zeros((N_CT, N_r, N_d, N_h, 508))

for CT_no, CT_val in enumerate(CT):
    print(CT_val)

    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + "months"
    else:
        CT_str = str(CT_val) + "yr"

    for r_no, r_val in enumerate(r):
        print(r_val)
        for d_no, d_val in enumerate(d):
            print(d_val)
            for h_no, h_val in enumerate(h):
                print(h_val)

                h_str = str(str(h_val).replace('.', '_'))                           ## so that decimal values of h will have the format 0_1 etc in the directory structure

                infile_base = 'CT_' + CT_str + '/r_' + str(r_val) + '/d_' + str(d_val) + '/h_' + h_str + '/final'

                all_m_files = glob.glob(infile_base + '*m*')        ## check how many files ending with m in directory. Each such file is a correct input. N_seeds is the number of such files:
                N_seeds = len(all_m_files)

                values_F18 = np.zeros(shape=(N_seeds, 504))
                d_values_F18 = np.zeros(shape=(N_seeds, 504))
                
                values_F38 = np.zeros(shape=(N_seeds, 504))
                d_values_F38 = np.zeros(shape=(N_seeds, 504))
                
                
                values_F18_and_F38 = np.zeros(shape=(N_seeds, 504))                     ## will contain the summed and averaged tally
                d_values_F18_and_F38 = np.zeros(shape=(N_seeds, 504))

       
                for seed_no, m_filename in enumerate(all_m_files):
                    if (seed_no % 30 == 0):
                        print(seed_no)

                    infile = m_filename
                    m=MCTAL(infile)
                    tallies = m.Read()      #get list of tallies from a mctal file.
        
                    for tally in tallies:
                        n_bins_full = len(tally.erg)            ## the number of energy bins. Includes the bins ending at 0, 1E-5, 1E-3, ...

                        tally_number = tally.tallyNumber
                        bin_lower_edges = tally.erg[:-1]       ## final entry is the upper edge of the last bin
                        bin_upper_edges = tally.erg[1:]         ## do not use the "negative" bin (upper edge at zero)
                        bin_width = bin_upper_edges - bin_lower_edges
                        bin_centres = bin_lower_edges + 0.5*bin_width

                        if (tally_number == 18):
                            bin_lower_edges_F18 = bin_lower_edges
                            bin_upper_edges_F18 = bin_upper_edges
                            bin_centres_F18 = bin_centres
                            data_full = tally.valsErrors.reshape((n_bins_full + 1, 2))      ## n_bins_full + 1 because last bin is totals.
                            values_F18[seed_no] = data_full[1:, 0]                ## do not include the "negative" bin (upper edge at zero)
                            d_values_F18[seed_no] = data_full[1:, 1]*values_F18[seed_no]          ## convert relative uncert to abs uncert. Note that data_full[1:, 1] are the relative std deviations (column 1 of the data_full array) of the tally data for all positive energy bins (that is, excluded bin zero).
                        elif (tally_number == 38):                  ## repeat the same for the other detector
                            bin_lower_edges_F38 = bin_lower_edges
                            bin_upper_edges_F38 = bin_upper_edges
                            bin_centres_F38 = bin_centres
                            data_full = tally.valsErrors.reshape((n_bins_full + 1, 2))      ## n_bins_full + 1 because last bin is totals.
                            values_F38[seed_no] = data_full[1:, 0]                ## do not include the "negative" bin (upper edge at zero)
                            d_values_F38[seed_no] = data_full[1:, 1]*values_F38[seed_no]          ## convert relative uncert to abs uncert
    
                # Sum over the two detectors and divide by two to get the average:
                values_F18_and_F38 = (values_F18 + values_F38)/2.
                d_values_F18_and_F38 = np.sqrt(np.power(d_values_F18, 2) + np.power(d_values_F38, 2))/2.                ## standard uncertainty propagation for sum, then divide by two because of averaging.
         
                ## Sum over all runs and divide by N_seeds (the number of runs summed over)
                run_summed_values_F18_and_F38 = np.sum(values_F18_and_F38, axis=0)/N_seeds
                run_summed_d_values_F18_and_F38 = np.sqrt(np.sum(np.power(d_values_F18_and_F38, 2), axis=0))/N_seeds

                # Multiply by 2 because the axial extent of the source in MCNP was [0, 32.5] cm (i.e. half the axial length of the "real" active FA length):
                run_summed_values_F18_and_F38 = run_summed_values_F18_and_F38*2.
                run_summed_d_values_F18_and_F38 = run_summed_d_values_F18_and_F38*2.


                ## Setup the data that is to be written to file. Note that final_data_values_F18 actually contains the average over the F18 and F38 tallies:
                final_data_values_F18[CT_no, r_no, d_no, h_no] = np.concatenate([np.array([CT_val, r_val, d_val, h_val]), run_summed_values_F18_and_F38])
                final_data_d_values_F18[CT_no, r_no, d_no, h_no] = np.concatenate([np.array([CT_val, r_val, d_val, h_val]), run_summed_d_values_F18_and_F38])



# Create Pandas dataframe:
final_data_values_F18_df = pd.DataFrame(data = np.reshape(final_data_values_F18, (N_CT*N_r*N_d*N_h, 508)), columns = column_headers_F18)
final_data_d_values_F18_df = pd.DataFrame(data = np.reshape(final_data_d_values_F18, (N_CT*N_r*N_d*N_h, 508)), columns = column_headers_F18)



bin_params_F18 = np.stack([bin_lower_edges_F18, bin_upper_edges_F18, bin_centres_F18], axis=1)
bin_params_F18_df = pd.DataFrame(data = bin_params_F18, columns=['Lower_edge', 'Upper_edge', 'Centre'])

# Note: the output files will be called something with F8 (to be clearer that this is F8 tally data)
final_data_values_F18_df.to_csv('../input_data/F8_tally_values.csv')
final_data_d_values_F18_df.to_csv('../input_data/F8_tally_d_values.csv')
bin_params_F18_df.to_csv('../input_data/F8_tally_bin_params.csv')
