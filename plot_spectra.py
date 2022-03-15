import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import genfromtxt


CT = [0.3, 1.0, 5.0]                ## NOTE: If CT < 1 (e.g. 0.3), that means 0.months. So 0.3 means 3 months, NOT 0.3 years!
r = [10.0, 12.0, 14.0]
d = [20, 40, 60]
h = [0.5, 1.0]


peaks_of_interest_indices= [[[3, 1], [1, 0], [2, 1], [2, 0], [0, 2]], \
                            [[3, 1], [1, 0], [2, 1], [2, 0], [0, 2]], \
                            [[3, 1], [1, 0], [2, 1], [2, 0], [0, 2]]]

peaks_of_interest_nuclide = [['144Ce', '106Ru', '134Cs', '154Eu', '137Cs'], \
                             ['144Ce', '106Ru', '134Cs', '154Eu', '137Cs'], \
                             ['144Ce', '106Ru', '134Cs', '154Eu', '137Cs']]


# Plot multiplet fits and spectra for optimal geometry (r = 12, d = 40, h = 0.5) for the three CTs:
fig1, ax1 = plt.subplots(3, 1, sharex=True, gridspec_kw = {'hspace':0})

for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
    else:
        CT_str = str(CT_val) + ' yr'
    for r_no, r_val in enumerate([12.0]):
        for d_no, d_val in enumerate([40]):
            for h_no, h_val in enumerate([0.5]):
                # Read broadened spectrum:
                spectrum_filename = "output_data/broadened_spectrum_CT_" + str(CT_val) + "_r_" + str(r_val) + "_d_" + str(d_val) + "_h_" + str(h_val) + ".csv"
                broadened_spectrum_data = genfromtxt(spectrum_filename, delimiter=',')
                bin_centres_broadened = broadened_spectrum_data[:, 0]
                broadened_spectrum = broadened_spectrum_data[:, 1]
                d_broadened_spectrum = broadened_spectrum_data[:, 2]

                # Read fit to broadened spectrum:
                spectrum_filename = "output_data/fitted_broadened_spectrum_CT_" + str(CT_val) + "_r_" + str(r_val) + "_d_" + str(d_val) + "_h_" + str(h_val) + ".csv"
                broadened_spectrum_fit_data = genfromtxt(spectrum_filename, delimiter=',')
                bin_centres_broadened_fit = broadened_spectrum_fit_data[:, 0]
                broadened_spectrum_fit = broadened_spectrum_fit_data[:, 1]

                ax1[CT_no].errorbar(1000*bin_centres_broadened[1:], broadened_spectrum[1:], d_broadened_spectrum[1:], marker='o', linestyle='None', color='black', zorder=1, label='Broadened spectrum')         ## exclude the very first bin in plotting

                ax1[CT_no].plot(1000*bin_centres_broadened_fit[0:19], broadened_spectrum_fit[0:19], linewidth=3, color='red', label='Multiplet fit', zorder=100)              ## the multiplet containing 137Cs

                ax1[CT_no].plot([1000*5.786519999999999442e-01, 1000*5.786519999999999442e-01], [1e1, 1.5*np.max(broadened_spectrum[1:])], linewidth=2, color='black', linestyle=':')
                ax1[CT_no].plot([1000*6.864000000000000101e-01, 1000*6.864000000000000101e-01], [1e1, 1.5*np.max(broadened_spectrum[1:])], linewidth=2, color='black', linestyle=':')

                ## Plot the lines:
                ax1[CT_no].plot([1000*6.047210E-01, 1000*6.047210E-01], [1e1, 1.5*np.max(broadened_spectrum[1:])], linewidth=2, color='lightgray', linestyle=':')
                ax1[CT_no].plot([1000*6.219300E-01, 1000*6.219300E-01], [1e1, 1.5*np.max(broadened_spectrum[1:])], linewidth=2, color='lightgray', linestyle=':')
                ax1[CT_no].plot([1000*6.616570E-01, 1000*6.616570E-01], [1e1, 1.5*np.max(broadened_spectrum[1:])], linewidth=2, color='lightgray', linestyle=':')

                ax1[CT_no].set_ylim(0, 1.5*np.max(broadened_spectrum[1:]))
                ax1[CT_no].set_xlim(1000*0.550, 1000*0.850)

                if (CT_no == 0):
                    ax1[CT_no].text(1000*6.047210E-01-1000*8e-3, 1.3*np.max(broadened_spectrum[1:]), '(1)', ha='center')
                    ax1[CT_no].text(1000*6.219300E-01-1000*8e-3, 1.3*np.max(broadened_spectrum[1:]), '(2)', ha='center')
                    ax1[CT_no].text(1000*6.616570E-01-1000*8e-3, 1.3*np.max(broadened_spectrum[1:]), '(3)', ha='center', weight='bold')



ax1[0].legend(edgecolor='black', facecolor='white', ncol=2, bbox_to_anchor=(0.2, 1), prop={'size': 9}, framealpha=1)
plt.xlabel('Energy [keV]')
fig1.text(0.03, 0.5, 'Counts per second per 6 keV', ha='center', va='center', rotation='vertical')
ax1[0].text(0.95, 0.85, 'CT = 3 months', transform=ax1[0].transAxes, horizontalalignment='right')
ax1[1].text(0.95, 0.85, 'CT = 1 year', transform=ax1[1].transAxes, horizontalalignment='right')
ax1[2].text(0.95, 0.85, 'CT = 5 years', transform=ax1[2].transAxes, horizontalalignment='right')

plt.subplots_adjust(left=0.12, top=0.92, bottom=0.12)
fig1.set_size_inches(5, 5)
plt.savefig('output_figures/peak_fit_example.png', dpi=400)





# Plot broadened spectra for paper (CT = 3 months, r = [10, 12, 14] cm, d = 40, h = 0.5)
fig = plt.figure()
r_colour = ['silver', 'gray', 'black']
CT_linestyle = ['-', '--']
for CT_no, CT_val in enumerate([0.3, 5.0]):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
    else:
        CT_str = str(CT_val) + ' yr'
    for r_no, r_val in enumerate(r):
        for d_no, d_val in enumerate([40]):
            for h_no, h_val in enumerate([0.5]):
                # Read broadened spectrum:
                spectrum_filename = "output_data/broadened_spectrum_CT_" + str(CT_val) + "_r_" + str(r_val) + "_d_" + str(d_val) + "_h_" + str(h_val) + ".csv"
                broadened_spectrum_data = genfromtxt(spectrum_filename, delimiter=',')
                bin_centres_broadened = broadened_spectrum_data[:, 0]
                broadened_spectrum = broadened_spectrum_data[:, 1]
                d_broadened_spectrum = broadened_spectrum_data[:, 2]


                if (CT_no == 0):
                    plt.errorbar(1000*bin_centres_broadened[1:], broadened_spectrum[1:], d_broadened_spectrum[1:], linewidth=2, linestyle=CT_linestyle[CT_no], color=r_colour[r_no], label='r = ' + str(r_val) + ' cm')         ## exclude the very first bin in plotting
                else:
                    plt.errorbar(1000*bin_centres_broadened[1:], broadened_spectrum[1:], d_broadened_spectrum[1:], linewidth=2, linestyle=CT_linestyle[CT_no], color=r_colour[r_no])         ## exclude the very first bin in plotting. Do not add label


plt.ylim(0.1, 1e5)
plt.xlim(1000*0.5, 1000*2.25)
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts per second per 6 keV')

plt.plot([1000*6.616570E-01, 1000*6.616570E-01], [.1, 1e5], zorder=-100, linestyle=':', color='gray')
plt.text(1000*6.616570E-01-1000*5e-2, .15, '(a)', ha='center')

plt.plot([1000*1.050410E+00, 1000*1.050410E+00], [.1, 1e5], zorder=-100, linestyle=':', color='gray')
plt.text(1000*1.050410E+00-1000*5e-2, .15, '(b)', ha='center')

plt.plot([1000*1.274430E+00, 1000*1.274430E+00], [.1, 1e5], zorder=-100, linestyle=':', color='gray')
plt.text(1000*1.274430E+00-1000*5e-2, .15, '(c)', ha='center')

plt.plot([1000*1.365180E+00, 1000*1.365180E+00], [.1, 1e5], zorder=-100, linestyle=':', color='gray')
plt.text(1000*1.365180E+00+1000*5e-2, .15, '(d)', ha='center')

plt.plot([1000*2.185660E+00, 1000*2.185660E+00], [.1, 1e5], zorder=-100, linestyle=':', color='gray')
plt.text(1000*2.185660E+00-1000*5e-2, .15, '(e)', ha='center')

plt.grid(which='both', alpha=0.2)

plt.legend(edgecolor='black', facecolor='white', prop={'size': 9}, framealpha=1)

plt.subplots_adjust(left=0.12, top=0.92, bottom=0.12)
fig.set_size_inches(5, 5)
plt.savefig('output_figures/spectrum_vs_r.png', dpi=400)


plt.show()
