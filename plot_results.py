import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.stats import multivariate_normal
from matplotlib.ticker import FormatStrFormatter
from scipy import special
import scipy.integrate as integrate
from matplotlib.lines import Line2D
import matplotlib.patches as patches


CT = [0.3, 1, 5]                ## NOTE: If CT < 1 (e.g. 0.3), that means 0.months. So 0.3 means 3 months, not 0.3 years!
r = [10, 12, 14]
d = [20, 40, 60]
h = [0.5, 1.0]

N_CT = len(CT)
N_r = len(r)
N_d = len(d)
N_h = len(h)


MAXIMUM_TOTAL_COUNT_RATE = 5e5              ## The maximum permissable count rate in the detector
MAXIMUM_NET_PEAK_UNCERTAINTY = 0.01         ## 1%

FIGURE_DIR = 'output_figures'      ## Directory for writing the output png figures to


Nuclide_colour = {'137Cs': 'navy', '154Eu': 'forestgreen', '144Ce': 'darkorange', '134Cs': 'magenta', '106Ru': 'red'}
Nuclide_half_lives = {'106Ru': 372./365., '134Cs': 2.064, '144Ce': 285./365., '137Cs': 30.1, '154Eu': 8.60}             ## half-lives (in years)


## The indices of the peaks of interest. Has to match the peaks_to_fit array in the analyse.py script. In this case, the 144Ce(2186 keV) peak is [3, 1] in the peaks_to_fit array, the 106Ru(1050 keV) peak is [1, 0], the 134Cs(1365 keV) peak is [2, 1], the 154Eu(1274 keV) peak is [2, 0] and the 137Cs(662 keV) peak is [0, 2]. Note that it should be possible to analyse different nuclides at different CTs, if desired.
peaks_of_interest_indices= [[[3, 1], [1, 0], [2, 1], [2, 0], [0, 2]], \
                            [[3, 1], [1, 0], [2, 1], [2, 0], [0, 2]], \
                            [[3, 1], [1, 0], [2, 1], [2, 0], [0, 2]]]

# The nuclide names, again matchin the order in the peaks_of_interest_indices array.
peaks_of_interest_nuclide = [['144Ce', '106Ru', '134Cs', '154Eu', '137Cs'], \
                             ['144Ce', '106Ru', '134Cs', '154Eu', '137Cs'], \
                             ['144Ce', '106Ru', '134Cs', '154Eu', '137Cs']]


# Because I want to plot the data in a certain order (by half life), set that here.
custom_lines = [Line2D([0], [0], color=Nuclide_colour['144Ce'], lw=4),
                Line2D([0], [0], color=Nuclide_colour['106Ru'], lw=4),
                Line2D([0], [0], color=Nuclide_colour['134Cs'], lw=4),
                Line2D([0], [0], color=Nuclide_colour['154Eu'], lw=4),
                Line2D([0], [0], color=Nuclide_colour['137Cs'], lw=4)]



# use different linestyles (solid/dashed) for the two h values in the plots:
h_linestyle = [':', '-']

data_last_meas = [[0] * len(r) for i in range(len(CT))]                     ## in most cases, I'm only interested in the data from the longest 'measurement' (in my analysis, I only used one measurement time, 600 s, so in that case this will be it). Here, create list to hold this data.


for CT_no, CT_val in enumerate(CT):
    for r_no, r_val in enumerate(r):
        data_in = pd.read_csv('output_data/analysed_peak_data_CT_' + str(float(CT_val)) + '_r_' + str(float(r_val)) + '.csv')
        #data_all_meas = data_in.copy()
        data_last_meas[CT_no][r_no] = (data_in.drop_duplicates(['CT', 'r', 'd', 'h', 'Multiplet_no', 'Peak_no'], keep='last')).copy()                ## When analysing as function of geometrical parameters, just need one row for each config (the data in the 10 "measurement rows" are the same for all data except the measurement data). Get the one corresponding to the longest measurement time (10 min). drop_duplicates ensures that all rows that have the same CT, r, d, h, Multiplet_no and Peak_no values (except the one with the longest measurement time) are dropped from the dataframe.






# Plot total count rate IN A SINGLE FIGURE
fig, ax = plt.subplots(3, N_r, sharey=True, sharex=True)
plt.subplots_adjust(hspace=0.25)

for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
        CT_years = CT_val*10*30./365.       ## convert 0.3 to 3 months (0.3*10**30 days), then to years
    elif (CT_val < 5):
        CT_str = str(CT_val) + ' year'
        CT_years = CT_val
    else:
        CT_str = str(CT_val) + ' years'
        CT_years = CT_val

    ax[CT_no, 1].set_title('Cooling time = ' + CT_str)
    for r_no, r_val in enumerate(r):
        ax[CT_no, r_no].plot([d[0]-5, d[-1]+5], [MAXIMUM_TOTAL_COUNT_RATE, MAXIMUM_TOTAL_COUNT_RATE], color='black', linestyle='--', label='500 kcps')
        ax[CT_no, r_no].set_xlim(d[0]-5, d[-1]+5)
        ax[CT_no, r_no].set_ylim(1e3, 1e8)
        ax[CT_no, r_no].text(0.36, 0.85, '$r=$'+str(r_val)+' cm', transform=ax[CT_no, r_no].transAxes)

        data = data_last_meas[CT_no][r_no]

        for h_no, h_val in enumerate(h):
            data_sel = data[(data['CT'] == CT_val) & (data['r'] == r_val) & (data['h'] == h_val) & (data['Multiplet_no'] == 0) & (data['Peak_no'] == 0)]            ## get the data for multiplet_no = 0, peak_no = 0 (total count rate is independent of peak #)
            ax[CT_no, r_no].errorbar(data_sel['d'], data_sel['TOTAL_count_rate_above_1keV'], data_sel['d_TOTAL_count_rate_above_1keV'], linestyle=h_linestyle[h_no], color='black', marker='.')

ax[0, 0].grid(which='both', alpha=0.2)
ax[0, 1].grid(which='both', alpha=0.2)
ax[0, 2].grid(which='both', alpha=0.2)
ax[1, 0].grid(which='both', alpha=0.2)
ax[1, 1].grid(which='both', alpha=0.2)
ax[1, 2].grid(which='both', alpha=0.2)
ax[2, 0].grid(which='both', alpha=0.2)
ax[2, 1].grid(which='both', alpha=0.2)
ax[2, 2].grid(which='both', alpha=0.2)



plt.yscale('log')
fig.text(0.03, 0.5, 'Total count rate > 1 keV [cps]', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.04, r'Collimator length $d$ [cm]', ha='center', va='center')
plt.xticks([20, 40, 60])
ax[0, 0].set_yticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])

plt.subplots_adjust(left=0.15, top=0.95, bottom=0.12)
fig.set_size_inches(5, 5)
plt.savefig(FIGURE_DIR + '/CT_total_count_rate_ALL_CT.png', dpi=400)





# Plot net peak uncert after 10 minutes IN A SINGLE FIGURE
fig, ax = plt.subplots(3, N_r, sharey=True, sharex=True)
plt.subplots_adjust(hspace=0.15)

for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
        CT_years = CT_val*10*30./365.       ## convert 0.3 to 3 months (0.3*10**30 days), then to years
    elif (CT_val < 5):
        CT_str = str(CT_val) + ' year'
        CT_years = CT_val
    else:
        CT_str = str(CT_val) + ' years'
        CT_years = CT_val

    ax[CT_no, 1].set_title('Cooling time = ' + CT_str)
    for r_no, r_val in enumerate(r):
        ax[CT_no, r_no].plot([d[0]-5, d[-1]+5], [100*MAXIMUM_NET_PEAK_UNCERTAINTY, 100*MAXIMUM_NET_PEAK_UNCERTAINTY], color='black', linestyle='--', label='1%')
        ax[CT_no, r_no].set_xlim(d[0]-5, d[-1]+5)
        ax[CT_no, r_no].set_ylim(0.01, 40)
        ax[CT_no, r_no].text(0.04, 0.92, '$r=$'+str(r_val)+' cm', transform=ax[CT_no, r_no].transAxes)
        for peaks_of_interest_no, peaks_of_interest_val in enumerate(peaks_of_interest_indices[CT_no]):
            multiplet_no = peaks_of_interest_val[0]
            peak_no = peaks_of_interest_val[1]
            nuclide_name = peaks_of_interest_nuclide[CT_no][peaks_of_interest_no]

            nuclide_half_life = Nuclide_half_lives[nuclide_name]
            if (20.*nuclide_half_life < CT_years):
                nuclide_half_life_OK = False
            else:
                nuclide_half_life_OK = True

            data = data_last_meas[CT_no][r_no]

            if (nuclide_half_life_OK == True):                      ## Only interested in peak if CT < 20*(nuclide half life). 
                for h_no, h_val in enumerate(h):
                    data_sel = data[(data['CT'] == CT_val) & (data['r'] == r_val) & (data['h'] == h_val) & (data['Multiplet_no'] == multiplet_no) & (data['Peak_no'] == peak_no)]
                    ax[CT_no, r_no].errorbar(data_sel['d'], 100*data_sel['d_A_rel_meas'], 100*data_sel['d_d_A_rel_meas'], linestyle=h_linestyle[h_no], color=Nuclide_colour[nuclide_name], marker='.')


ax[2, 1].legend(custom_lines, [r'$^{144}$Ce (2186 keV)', r'$^{106}$Ru (1050 keV)', r'$^{134}$Cs (1365 keV)', r'$^{154}$Eu (1274 keV)', r'$^{137}$Cs (662 keV)'], ncol=3, bbox_to_anchor=(2.6, -0.2), edgecolor='black', prop={'size': 9})

plt.yscale('log')
ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax[2,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.text(0.03, 0.5, 'Relative peak-area uncertainty after 10 minutes [%]', ha='center', va='center', rotation='vertical')
fig.text(0.5, 0.08, r'Collimator length $d$ [cm]', ha='center', va='center')
plt.xticks([20, 40, 60])
ax[0, 0].grid(which='both', alpha=0.2)
ax[0, 1].grid(which='both', alpha=0.2)
ax[0, 2].grid(which='both', alpha=0.2)
ax[1, 0].grid(which='both', alpha=0.2)
ax[1, 1].grid(which='both', alpha=0.2)
ax[1, 2].grid(which='both', alpha=0.2)
ax[2, 0].grid(which='both', alpha=0.2)
ax[2, 1].grid(which='both', alpha=0.2)
ax[2, 2].grid(which='both', alpha=0.2)

plt.subplots_adjust(left=0.15, top=0.95, bottom=0.12)
fig.set_size_inches(5, 9)
plt.savefig(FIGURE_DIR + '/CT_net_peak_uncertainty_ALL_CT.png', dpi=400)








# Plot the "OK values" map (UPDATED 2022-02-07)
x_coordinates = np.arange(2, 9, 3)
y_coordinates = np.arange(1, 4, 2)
print(x_coordinates)
#exit()

peaks_of_interest_x_offset = [-1.0, -0.5, 0, 0.5, 1.0]
CT_y_offset = [-0.5, 0, 0.5]

fig, ax = plt.subplots(3, 1, sharex=True)

for r_no, r_val in enumerate(r):

    ax[r_no].set_title('r = ' + str(r_val) + ' cm')



    for CT_no, CT_val in enumerate(CT):
        if (CT_val < 1):
            CT_str = str(int(CT_val*10)) + ' months'
            CT_years = CT_val*10*30./365.       ## convert 0.3 to 3 months (0.3*10**30 days), then to years
        elif (CT_val < 5):
            CT_str = str(CT_val) + ' year'
            CT_years = CT_val
        else:
            CT_str = str(CT_val) + ' years'
            CT_years = CT_val



        for peaks_of_interest_no, peaks_of_interest_val in enumerate(peaks_of_interest_indices[CT_no]):
            multiplet_no = peaks_of_interest_val[0]
            peak_no = peaks_of_interest_val[1]
            nuclide_name = peaks_of_interest_nuclide[CT_no][peaks_of_interest_no]
            nuclide_half_life = Nuclide_half_lives[nuclide_name]

            data = data_last_meas[CT_no][r_no]


            for d_no, d_val in enumerate(d):
                for h_no, h_val in enumerate(h):
                    data_sel = data[(data['CT'] == CT_val) & (data['r'] == r_val) & (data['h'] == h_val) & (data['d'] == d_val) & (data['Multiplet_no'] == multiplet_no) & (data['Peak_no'] == peak_no)]

                    if ((data_sel['TOTAL_count_rate_above_1keV'] + data_sel['d_TOTAL_count_rate_above_1keV']).to_numpy() <= MAXIMUM_TOTAL_COUNT_RATE):           ## if the total count rate at least one sigma below acceptable value
                        if ((data_sel['d_A_rel_meas'] + data_sel['d_d_A_rel_meas']).to_numpy() <= MAXIMUM_NET_PEAK_UNCERTAINTY):           ## if the net peak uncertainty at least one sigma below acceptable value
                            ax[r_no].scatter(x_coordinates[d_no] + peaks_of_interest_x_offset[peaks_of_interest_no], y_coordinates[h_no] + CT_y_offset[CT_no], marker='o', color=Nuclide_colour[nuclide_name])
                        else:                   ## the total count rate is OK, but peak has too high uncertainty
                            ax[r_no].scatter(x_coordinates[d_no] + peaks_of_interest_x_offset[peaks_of_interest_no], y_coordinates[h_no] + CT_y_offset[CT_no], marker='+', color=Nuclide_colour[nuclide_name])
                    else:                       ## too high count rate
                        if ((data_sel['d_A_rel_meas'] + data_sel['d_d_A_rel_meas']).to_numpy() <= MAXIMUM_NET_PEAK_UNCERTAINTY):           ## if the net peak uncertainty at least one sigma below acceptable value
                            ax[r_no].scatter(x_coordinates[d_no] + peaks_of_interest_x_offset[peaks_of_interest_no], y_coordinates[h_no] + CT_y_offset[CT_no], marker='x', color=Nuclide_colour[nuclide_name])
                        else:                   ## if total count rate is too high AND uncertainty too high:
                            ax[r_no].scatter(x_coordinates[d_no] + peaks_of_interest_x_offset[peaks_of_interest_no], y_coordinates[h_no] + CT_y_offset[CT_no], marker='s', color=Nuclide_colour[nuclide_name])


plt.legend(custom_lines, [r'$^{144}$Ce (2186 keV)', r'$^{106}$Ru (1050 keV)', r'$^{134}$Cs (1365 keV)', r'$^{154}$Eu (1274 keV)', r'$^{137}$Cs (662 keV)'], ncol=3, bbox_to_anchor=(1.28, 4.15), edgecolor='black', prop={'size': 9})


for ax1 in ax:
    # create a twin of the axis that shares the x-axis
    ax2 = ax1.twinx()

    ax1.set_xlim(0.5, 9.5)
    ax1.set_ylim(0, 4)
    ax1.set_yticks([1, 3])
    ax1.set_yticklabels(['0.5', '1.0'])

    ax1.set_xticks([2, 5, 8])
    ax1.set_xticklabels(['20', '40', '60'])

    ax2.set_ylim(0, 4)
    ax2.set_yticks([0.5, 1, 1.5, 2.5, 3, 3.5])
    ax2.set_yticklabels(['3 months' , '1 year', '5 years', '3 months', '1 year', '5 years'])

    ax1.plot([3.5, 3.5], [0, 4], color='black')
    ax1.plot([6.5, 6.5], [0, 4], color='black')
    ax1.plot([0.5, 9.5], [2, 2], color='black')
    ax1.plot([0.5, 9.5], [4, 4], color='black')

# Mark the best solution with dashed rectangle
p = patches.Rectangle((3.6, 0.1), 2.8, 1.8, fill=False, linestyle='dashed')
ax[1].add_artist(p)

fig.subplots_adjust(hspace=0.3, right=0.8, bottom=0.07)


fig.text(0.03, 0.5, r'Collimator slit height $h$ [cm]', ha='center', va='center', rotation='vertical')
fig.text(0.98, 0.5, r'Cooling time', ha='center', va='center', rotation='vertical')
ax[2].set_xlabel(r'Collimator length $d$ [cm]')

fig.set_size_inches(5, 7)
plt.savefig(FIGURE_DIR + '/all_r_acceptable_configurations_map_UPDATED.png', dpi=400)





plt.show()


