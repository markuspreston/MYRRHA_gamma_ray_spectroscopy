import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.stats import multivariate_normal
from matplotlib.ticker import FormatStrFormatter
from scipy import special
import scipy.integrate as integrate
import sys


## The multiplet fit function is a 2nd degree polynomial (to model the underlying bg due to scattering in LBE), a smoothed step function under each peak (to model scattering from photopeak gammas) and Gaussian (to model peak). The mu and sigma of the peak(s) is known a priori, and fixed.
def multiplet_fit_fcn(x, *pars):             ## the parameters are passed as an array to this function, to allow multiple peaks to be fitted at the same time.
    parameters = pars[0]
    N_peaks = int(parameters[0])

    mu = parameters[1:(1+N_peaks)]
    sigma = parameters[(1+N_peaks):(1+N_peaks+N_peaks)]
    S = parameters[(1+N_peaks+N_peaks):(1+N_peaks+N_peaks+N_peaks)]
    b0 = parameters[(1+N_peaks+N_peaks+N_peaks):(1+N_peaks+N_peaks+N_peaks+1)]
    b1 = parameters[(1+N_peaks+N_peaks+N_peaks+1):(1+N_peaks+N_peaks+N_peaks+2)]
    b2 = parameters[(1+N_peaks+N_peaks+N_peaks+2):(1+N_peaks+N_peaks+N_peaks+3)]
    a1 = parameters[(1+N_peaks+N_peaks+N_peaks+3):(1+N_peaks+N_peaks+N_peaks+3+N_peaks)]


    polynomial_background = b0 + b1*x + b2*x*x


    peak_background = 0.
    peak_signal = 0.

    for peak_no in range(N_peaks):
        peak_background += a1[peak_no]*(S[peak_no]*(1./(sigma[peak_no]*np.sqrt(2.*np.pi))))*special.erfc((x - mu[peak_no])/sigma[peak_no])            ## note: when S = 0, the peak-induced bg will also be zero. This ensures that the step-function amplitude is a fraction of the peak amplitude (a is limited to a range between 0 and 1 in the fitting below)
        peak_signal += S[peak_no]*(1./(sigma[peak_no]*np.sqrt(2.*np.pi)))*np.exp(-0.5*np.power((x - mu[peak_no])/sigma[peak_no], 2))            ## integral (peak area) S is the parameter

    return polynomial_background + peak_background + peak_signal


def find_bin(bin_lower_edges, bin_upper_edges, val):
    found_bin = -1
    for i in range(len(bin_upper_edges)):
        if (val >= bin_lower_edges[i] and val < bin_upper_edges[i]):
            found_bin = i
            break;

    return found_bin

def gaussian_convolution_kernel(x, mu, sigma, bin_width):
    bin_centre = x          ## this is a discrete convolution, so use the bin centres
    return bin_width*(1./(sigma*np.sqrt(2.*np.pi)))*np.exp(-0.5*np.power((bin_centre - mu)/sigma, 2))               ## bin_width in front needed to normalise, i.e. ensure that the sum of the kernel over all bins is one.


class Multiplet():                  ## used to store relevant data for each multiplet that can be fitted.
    def __init__(self, N_peaks, N_MCNP_realisations):

        self.peak_bin = np.zeros(N_peaks, dtype=int)           ## peak location(s), in units of bins (for unbroadened spec)
        self.peak_sigma = np.zeros(N_peaks)           ## peak width(s)
        self.peak_bin_mu = np.zeros(N_peaks)
        self.peak_bin_sigma = np.zeros(N_peaks)
        self.peak_analysis_limits_bin = np.zeros((N_peaks, 2), dtype=int)           ## peak fit range (bins)

    def add_peak(self, peak_no, peak_energy, peak_bin, peak_sigma, peak_bin_mu, peak_bin_sigma, peak_analysis_limits_bin):
        self.peak_bin[peak_no] = peak_bin
        self.peak_sigma[peak_no] = peak_sigma
        self.peak_bin_mu[peak_no] = peak_bin_mu
        self.peak_bin_sigma[peak_no] = peak_bin_sigma
        self.peak_analysis_limits_bin[peak_no] = peak_analysis_limits_bin

    def set_fit_x_range(self):
        self.fit_x_range_bin = np.array([np.min(self.peak_analysis_limits_bin), np.max(self.peak_analysis_limits_bin)], dtype=int)             ## fit range should cover the entire region (from lowest analysis limit to highest)

    def set_fit_popt_pcov(self, N_peaks, N_meas, N_MCNP_realisations):               ## one fit per multiplet
        N_free_params = N_peaks + 3 + N_peaks         ## N_peaks*S, b0, b1, b2, N_peaks*a1. Mu and sigma of each peak is fixed in the fit.
        self.popt = np.zeros(N_free_params)
        self.pcov = np.zeros((N_free_params, N_free_params))

        self.popt_meas_realisations = np.zeros((N_meas, N_MCNP_realisations, N_free_params))
        self.pcov_meas_realisations = np.zeros((N_meas, N_MCNP_realisations, N_free_params, N_free_params))

        self.A_fit_realisations = np.zeros((N_peaks, N_MCNP_realisations))
        self.A_fit_meas_realisations = np.zeros((N_peaks, N_meas, N_MCNP_realisations))
        self.d_A_fit_meas_realisations = np.zeros((N_peaks, N_meas, N_MCNP_realisations))
        self.d_A_rel_fit_meas_realisations = np.zeros((N_peaks, N_meas, N_MCNP_realisations))

        self.A_fit_meas = np.zeros((N_peaks, N_meas))
        self.d_A_fit_meas = np.zeros((N_peaks, N_meas))
        self.d_A_fit_meas_rel = np.zeros((N_peaks, N_meas))



DATA_DIR = './input_data/'

values = pd.read_csv(DATA_DIR + 'F8_tally_values.csv')
d_values = pd.read_csv(DATA_DIR + 'F8_tally_d_values.csv')
bins = pd.read_csv(DATA_DIR + 'F8_tally_bin_params.csv')

bin_lower_edges = bins['Lower_edge'].to_numpy()[1:]          ## [1:] to exclude epsilon bin
bin_upper_edges = bins['Upper_edge'].to_numpy()[1:]
bin_centres = bins['Centre'].to_numpy()[1:]


N_bins = len(bin_centres)

bin_width = bin_centres[100]-bin_centres[99]            ## each bin has the same width.


CT = [float(sys.argv[1])]                ## Get the CT from the command line argument
r = [float(sys.argv[2])]                ## Get the r from the command line argument

d = [20, 40, 60]                ## specify which d value(s) to analyse
h = [0.5, 1.0]               ## specify which h value(s) to analyse

N_CT = len(CT)
N_r = len(r)
N_d = len(d)
N_h = len(h)

MAXFEV = 50000000              ## maximum number of function evaluations for curve_fit

MEAS_TIME = [600]              ## I'm only interested in one measurement time (10 minutes <=> 600 seconds) in this work. But here one can specify other measurement times as well.
N_meas = len(MEAS_TIME)         ## the number of "measurements" to consider (note: not the same as the number of realisations to consider for each measurement)

FIT_RANGE_SIGMA = 3           ## how far outside the peak mean to include in the analysis. Is important for defining the outer limits of each multiplet.

N_MCNP_realisations = 1000      ## How many MCNP realisations used to propagate the MCNP statistical uncertainties to the end results.

ANALYSE_MEASURED_SPECTRA = True         ## if one wants, here the analysis of the "measured" spectra (i.e. with added Poisson stats) can be turned off to speed things up a bit.


# These are the multiplets that will be fitted. One row for each CT (allows removing some peaks/multiplets e.g. for long-CT fuel. However, for our analysis all peaks/multiplets were analysed for all CTs). The peaks to include in each multiplet were identified "by hand" by looking at the gamma-ray source term to identify which significant peaks sit close to each other in the spectrum. Because the fit range stretches 3*sigma outside the lowest/highest peaks, there should be no strong peaks within those limits that are not included in the multiplet.
peaks_to_fit = [[[6.047210E-01, 6.219300E-01, 6.616570E-01], [1.050410E+00, 1.12807E+00, 1.16797E+00, 1.19454E+00], [1.274430E+00, 1.365180E+00], [2.112540E+00, 2.185660E+00]], \
                [[6.047210E-01, 6.219300E-01, 6.616570E-01], [1.050410E+00, 1.12807E+00, 1.16797E+00, 1.19454E+00], [1.274430E+00, 1.365180E+00], [2.112540E+00, 2.185660E+00]], \
                [[6.047210E-01, 6.219300E-01, 6.616570E-01], [1.050410E+00, 1.12807E+00, 1.16797E+00, 1.19454E+00], [1.274430E+00, 1.365180E+00], [2.112540E+00, 2.185660E+00]]]

GEB_params = [1.1956E-4, 2.2320E-2, 1.2852E-1]                  ## These are the energy-broadening parameters. obtained by fitting to Saint Gobain FWHM data. From ~/Documents/MYRRHA_work/MYRRHA_assembly_NDA_May2021/LaBr3_137Cs_source/fit_GEB.py. ADD THIS CODE TO REPO?? XXX

multiplets = [ [ [ [ [] for i in range(N_h) ] for i in range(N_d) ] for i in range(N_r) ] for i in range(N_CT) ]                ## create a nested list that holds all multiplets for each geometry configuration (each CT, r, d, h combination).

peak_analysis_limits = np.zeros(2)
peak_analysis_limits_bin = np.zeros(2, dtype=int)



## The energy broadening is performed in this code. Therefore, need to find the sigma to use for each "input energy". That is, for each bin in the unbroadened spectrum, there is an associated Gaussian to broaden with, and each has a certain sigma. So, first get the broadening FWHM from the GEB_params relationship (assume the relationship FWHM = a + b*sqrt(E + c*E^2), as MCNP does). Then calculate the sigma (FWHM/2.355) for each bin.

broadening_sigma = np.zeros(N_bins)

for n in range(N_bins):             
    broadening_FWHM = GEB_params[0] + GEB_params[1]*np.sqrt(bin_centres[n] + GEB_params[2]*np.power(bin_centres[n], 2))
    broadening_sigma[n] = broadening_FWHM/2.355


# Now, we have the sigma to use for the Gaussian broadening of the data in each bin.


for CT_no, CT_val in enumerate(CT):
    for r_no, r_val in enumerate(r):
        for d_no, d_val in enumerate(d):
            for h_no, h_val in enumerate(h):
                N_multiplets = len(peaks_to_fit[CT_no])
                
                for multiplet_no in range(N_multiplets):
                    N_peaks = len(peaks_to_fit[CT_no][multiplet_no])
                    multiplets[CT_no][r_no][d_no][h_no].append(Multiplet(N_peaks, N_MCNP_realisations))     # Add a multiplet that can hold N_peaks and N_MCNP realisations

                    for peak_no in range(N_peaks):
                        peak_energy = peaks_to_fit[CT_no][multiplet_no][peak_no]
                        peak_bin = find_bin(bin_lower_edges, bin_upper_edges, peak_energy)
                
                        FWHM = GEB_params[0] + GEB_params[1]*np.sqrt(peak_energy + GEB_params[2]*np.power(peak_energy, 2))                  ## get the FWHM at the peak energy.
                        peak_sigma = FWHM/2.355                 ## convert FWHM -> sigma. Used to get the fit range.

                        peak_bin_mu = bin_centres[peak_bin]
                        peak_bin_sigma = broadening_sigma[peak_bin]


                        peak_analysis_limits[0] = peak_energy - FIT_RANGE_SIGMA*peak_sigma
                        peak_analysis_limits[1] = peak_energy + FIT_RANGE_SIGMA*peak_sigma
                
                        peak_analysis_limits_bin[0] = find_bin(bin_lower_edges, bin_upper_edges, peak_analysis_limits[0])
                        peak_analysis_limits_bin[1] = find_bin(bin_lower_edges, bin_upper_edges, peak_analysis_limits[1])
                
                        multiplets[CT_no][r_no][d_no][h_no][multiplet_no].add_peak(peak_no, peak_energy, peak_bin, peak_sigma, peak_bin_mu, peak_bin_sigma, peak_analysis_limits_bin)
                
                    multiplets[CT_no][r_no][d_no][h_no][multiplet_no].set_fit_x_range()
                    multiplets[CT_no][r_no][d_no][h_no][multiplet_no].set_fit_popt_pcov(N_peaks, N_meas, N_MCNP_realisations)


# Unbroadened spectrum container. Used for getting the total count rate
unbroadened_spectrum = np.zeros((len(CT), len(r), len(d), len(h), N_bins))
d_unbroadened_spectrum = np.zeros((len(CT), len(r), len(d), len(h), N_bins))

# Broadened spectrum container. Used to hold the broadened spectrum for each configuration
broadened_spectrum = np.zeros((len(CT), len(r), len(d), len(h), N_bins))
d_broadened_spectrum = np.zeros((len(CT), len(r), len(d), len(h), N_bins))

# Measured spectrum container. Used to hold the *scaled* broadened spectrum for each configuration (the multiplets are fitted to this initially, to provide better estimate of fit parameter values)
meas_spectrum = np.zeros((len(CT), len(r), len(d), len(h), N_meas, N_bins))
d_meas_spectrum = np.zeros((len(CT), len(r), len(d), len(h), N_meas, N_bins))


# Container for unbroadened spectrum realisations. Used to calculate the total count rate from sum of bin contents.
unbroadened_spectrum_realisations = np.zeros((len(CT), len(r), len(d), len(h), N_MCNP_realisations, N_bins))

# Container for broadened spectrum realisations. Contains the broadened spectrum (accounting for the MCNP statistical uncertainties via N_MCNP_realisations realisations)
broadened_spectrum_realisations = np.zeros((len(CT), len(r), len(d), len(h), N_MCNP_realisations, N_bins)) 

# Container for measured spectrum realisations. This is basically the broadened_spectrum_realisations scaled by the measurement time. These are the spectra that are fitted to get the peak-area uncertainties.
meas_spectrum_realisations = np.zeros((len(CT), len(r), len(d), len(h), N_meas, N_MCNP_realisations, N_bins))
# Container for the Poisson uncertainties in the "measured" spectrum
d_meas_spectrum_realisations = np.zeros((len(CT), len(r), len(d), len(h), N_meas, N_MCNP_realisations, N_bins))

# Total count rate container. Calculate for each MCNP realisation (from these, the mean and std dev can later be determined)
TOTAL_count_rate_above_1keV_realisations = np.zeros((len(CT), len(r), len(d), len(h), N_MCNP_realisations))


for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
    else:
        CT_str = str(CT_val) + ' yr'
    for r_no, r_val in enumerate(r):
        for d_no, d_val in enumerate(d):
            for h_no, h_val in enumerate(h):
                # Get the data (and the statistical uncertainties) for the correct CT, r, d, h
                data_sel = values[(values['CT'] == CT_val) & (values['r'] == r_val) & (values['d'] == d_val) & (values['h'] == h_val)]
                d_data_sel = d_values[(d_values['CT'] == CT_val) & (d_values['r'] == r_val) & (d_values['d'] == d_val) & (d_values['h'] == h_val)]


                spectrum_bins = np.arange(0, 502, 1)        ## the input data contains energy bins 0, 1, 2, 3, ..., 501.

                # Get the (raw, unbroadened) data in the columns called 0, 1, 2, ..., 501:
                unbroadened_spectrum[CT_no, r_no, d_no, h_no] = data_sel[spectrum_bins.astype(str)].to_numpy().flatten()
                d_unbroadened_spectrum[CT_no, r_no, d_no, h_no] = d_data_sel[spectrum_bins.astype(str)].to_numpy().flatten()


                ## I want to get N_MCNP_realisations realisations of the *unbroadened* spectra. Used to propagate the uncertainty in the *total* count rate in a way that is consistent with the other uncertainty propagation.
                unbroadened_spectrum_realisations_temp = np.random.normal(loc=unbroadened_spectrum[CT_no, r_no, d_no, h_no], scale=d_unbroadened_spectrum[CT_no, r_no, d_no, h_no], size=(N_MCNP_realisations, len(broadened_spectrum[CT_no, r_no, d_no, h_no])))               ## assume a normal distribution (with sigma from the MCNP data) for each bin. Sample N_MCNP_realisations realisations for each bin.

                ## Need to replace negative values with zero (because it is a spectrum with counts):
                unbroadened_spectrum_realisations[CT_no, r_no, d_no, h_no] = np.where(unbroadened_spectrum_realisations_temp < 0., 0., unbroadened_spectrum_realisations_temp)



# I want the broadened spectrum between 500 keV and 2800 keV. The reason is that the unbroadened spectrum goes from 0 to 3 MeV, but one cannot get the convolution with the Gaussian kernel correctly at the edges (this would include incorrect zeroes coming from outside the edges). To avoid this, only determine the convolution in this limited range. This does not affect the total-rate determination, and is OK for the spectrum analysis since I do not analyse anything below 500 keV or above 2800 keV.
bin_500keV = find_bin(bin_lower_edges, bin_upper_edges, 0.5)
bin_2800keV = find_bin(bin_lower_edges, bin_upper_edges, 2.8)



# Get the TOTAL count rate:
for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
    else:
        CT_str = str(CT_val) + ' yr'
    for r_no, r_val in enumerate(r):
        for d_no, d_val in enumerate(d):
            for h_no, h_val in enumerate(h):
                #TOTAL_count_rate_above_1keV[CT_no, r_no, d_no, h_no] = np.sum(unbroadened_spectrum[CT_no, r_no, d_no, h_no, 1:])            ## energy bin 1 starts at 1 keV. So sum above that to get total count rate above 1 keV
                TOTAL_count_rate_above_1keV_realisations[CT_no, r_no, d_no, h_no, :] = np.sum(unbroadened_spectrum_realisations[CT_no, r_no, d_no, h_no, :, 1:], axis=1)            ## energy bin 1 starts at 1 keV. So sum above that to get total count rate above 1 keV



### Do the broadening, i.e. convolve the unbroadened spectrum with a Gaussian kernel (having energy-dependent sigma)
for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
    else:
        CT_str = str(CT_val) + ' yr'
    for r_no, r_val in enumerate(r):
        for d_no, d_val in enumerate(d):
            for h_no, h_val in enumerate(h):
                ## loop over all energy bins. For each bin, determine the Gaussian kernel (energy-dependent sigma). Do the convolution.
                for n in range(bin_500keV, bin_2800keV+1):                 ## only do convolution above 500 keV and below 2800 keV (need padding around convolution to accomodate broadening from below and above)
                    for m in range(-20, 21):            ## Need to truncate the kernel at some point (cannot sum from -inf to inf). Number of bins to include is set here. I set this to 41 (so include 20 bins on each side of zero). The probability that far from the mean appears to be negligible (the bin width is ~6 keV, the broadening sigma at 2800 keV is 18.5 keV. So 20 bins from the mean is 60 keV from the mean (i.e. 3*sigma). Above that the likelihood given by the Gaussian is assumed to be negligible). For lower energies, e.g. 500 keV, the broadening sigma is even smaller, meaning that the convolution will take more than 3*sigma into account. So this should be a valid approximation for the considered energy range.
                        #So for each bin n in the new, broadened, spectrum, need to loop over bins m and add to the sum.
                        E_m = bin_width*m               ## the centre of the bin for which to get the kernel. Note that this is all centred at zero. So one bin is centred at 0, the first to the right at +bin_width, the first at the left at -bin_width, etc etc.
                        kernel = gaussian_convolution_kernel(E_m, 0., broadening_sigma[n], bin_width)           # get the value of the Gaussian convolution kernel (which has mu fixed at zero) at E = E_m, sigma = broadening_sigma[n]). bin_width is added to ensure normalisation of the convolution. That is, if we calculate the value of a standard normalised Gaussian at certain equidistant x values and sum that, the sum will not be 1 - we need to account for the bin width to get a discrete sum that is 1.

                        broadened_spectrum[CT_no, r_no, d_no, h_no, n] += unbroadened_spectrum[CT_no, r_no, d_no, h_no, n-m]*kernel             ## here we do the actual convolution. Because we now made sure that the kernel is normalised, the broadened spectrum will have the same integral as the unbroadened spectrum.
                        d_broadened_spectrum[CT_no, r_no, d_no, h_no, n] += np.power(d_unbroadened_spectrum[CT_no, r_no, d_no, h_no, n-m]*kernel, 2)            ## propagate statistical uncertainties through the convolution sum. For now, just sum the squares. When entire sum is complete, get sqrt below.

                        broadened_spectrum_realisations[CT_no, r_no, d_no, h_no, :, n] += unbroadened_spectrum_realisations[CT_no, r_no, d_no, h_no, :, n-m]*kernel


                    d_broadened_spectrum[CT_no, r_no, d_no, h_no, n] = np.sqrt(d_broadened_spectrum[CT_no, r_no, d_no, h_no, n])            ## sqrt of summed quadratures. These are then the statistical uncertainties in the broadened spectra.
                    #d_broadened_spectrum_realisations[CT_no, r_no, d_no, h_no, :, n] = d_broadened_spectrum[CT_no, r_no, d_no, h_no, n]         ## use the same uncertainty for each realisation

                
                ## For each "measurement", scale the broadened spectrum with the measurement time to get the "measured" spectrum. To this, Poisson noise will be applied and the uncertainty in each bin set to the sqrt of the (scaled) bin content. Each realisation will therefore be a separate "measurement".
                for meas_no in range(N_meas):
                    meas_spectrum[CT_no, r_no, d_no, h_no, meas_no] = MEAS_TIME[meas_no]*broadened_spectrum[CT_no, r_no, d_no, h_no]
                    d_meas_spectrum[CT_no, r_no, d_no, h_no, meas_no] = np.sqrt(meas_spectrum[CT_no, r_no, d_no, h_no, meas_no])

                    meas_spectrum_realisations[CT_no, r_no, d_no, h_no, meas_no] = MEAS_TIME[meas_no]*broadened_spectrum_realisations[CT_no, r_no, d_no, h_no,:]
                    d_meas_spectrum_realisations[CT_no, r_no, d_no, h_no, meas_no] = np.sqrt(meas_spectrum_realisations[CT_no, r_no, d_no, h_no, meas_no,:])



## Fit all peaks in the multiplets
# Step 1: directly fit the broadened spectrum from the MCNP simulation. This fit will then give good starting values for the parameters for later fits.
# Step 2: fit each "realisation" (with Poisson uncertainties as weights). Used to get the peak-area uncertainties.
for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
    else:
        CT_str = str(CT_val) + ' yr'
    for r_no, r_val in enumerate(r):
        for d_no, d_val in enumerate(d):
            for h_no, h_val in enumerate(h):
                print(CT_val)
                print(r_val)
                print(d_val)
                print(h_val)
                for multiplet_no in range(N_multiplets):
                    print('Fitting multiplet # ' + str(multiplet_no + 1) + ' of ' + str(N_multiplets))
                    N_peaks = len(peaks_to_fit[CT_no][multiplet_no])

                    ## Some parameters will be fixed in the fit:
                    fixed_params = np.hstack([N_peaks, multiplets[CT_no][r_no][d_no][h_no][multiplet_no].peak_bin_mu, multiplets[CT_no][r_no][d_no][h_no][multiplet_no].peak_bin_sigma])            # instead of using the "true" peak mu and sigma of the peak, get the mu from the (unbroadened spectrum) bin containing the peak and the corresponding broadened sigma. This is done because the convolution was based on the binned unbroadened spectrum.

                    multiplet_fit_fcn_fixed_mu_and_sigma = lambda x, *free_params: multiplet_fit_fcn(x, np.hstack([fixed_params, free_params]))         ## set up a function that can be fitted while keeping some parameters fixed (number of peaks in the multiplet, mu, sigma) and some free (S, b0, b1, b2, a1).


                    # Provide some initial guesses for the parameters:
                    S_init_guess = np.zeros(N_peaks)
                    b0_init_guess = 100000
                    b1_init_guess = -10000
                    b2_init_guess = 0
                    a1_init_guess = np.zeros(N_peaks)

                    for peak_no in range(N_peaks):
                        S_init_guess[peak_no] = 100.
                        a1_init_guess[peak_no] = 0.1

                    params_init_guess = np.hstack([S_init_guess, b0_init_guess, b1_init_guess, b2_init_guess, a1_init_guess])



                    # Some lower and upper bounds for the parameters:
                    S_lower_bound = np.zeros(N_peaks)
                    b0_lower_bound = 0.
                    b1_lower_bound = -np.inf
                    b2_lower_bound = -np.inf
                    a1_lower_bound = np.zeros(N_peaks)

                    S_upper_bound = np.zeros(N_peaks)
                    b0_upper_bound = np.inf
                    b1_upper_bound = np.inf
                    b2_upper_bound = np.inf
                    a1_upper_bound = np.ones(N_peaks)               ## a1 should be between 0 and 1 (defined as a fraction of the Gaussian-peak amplitude)

                    for peak_no in range(N_peaks):
                        S_lower_bound[peak_no] = 0.
                        S_upper_bound[peak_no] = np.inf


                    params_lower_bounds = np.hstack([S_lower_bound, b0_lower_bound, b1_lower_bound, b2_lower_bound, a1_lower_bound])
                    params_upper_bounds = np.hstack([S_upper_bound, b0_upper_bound, b1_upper_bound, b2_upper_bound, a1_upper_bound])

                    # the multiplet should be fitted between the leftmost (peak - 3*sigma) and the rightmost (peak + 3*sigma)
                    min_bin = multiplets[CT_no][r_no][d_no][h_no][multiplet_no].fit_x_range_bin[0]
                    max_bin = multiplets[CT_no][r_no][d_no][h_no][multiplet_no].fit_x_range_bin[1]

                    x_fit = bin_centres[min_bin:max_bin+1]          ## the fit range (in correct units, not just bin number)


                    
                    # Fit the multiplet to the broadened spectrum. The results of this are used as input to the fits to the "measured" spectra.
                    multiplets[CT_no][r_no][d_no][h_no][multiplet_no].popt, multiplets[CT_no][r_no][d_no][h_no][multiplet_no].pcov = curve_fit(multiplet_fit_fcn_fixed_mu_and_sigma, x_fit, broadened_spectrum[CT_no, r_no, d_no, h_no, min_bin:max_bin+1], sigma=d_broadened_spectrum[CT_no, r_no, d_no, h_no, min_bin:max_bin+1], absolute_sigma=True, p0=(params_init_guess), bounds=(params_lower_bounds, params_upper_bounds), maxfev=MAXFEV)

                    if (ANALYSE_MEASURED_SPECTRA == True):
                        print('Fit the "measured" spectra')

                        for meas_no in range(N_meas):           ## in the work so far, only use 10 minutes = 600 seconds, but in principle here one loops over several such measurement times.
                            print('Measurement ' + str(meas_no+1) + ' of ' + str(N_meas))


                            ## PARAMETER INITIAL GUESSES:
                            params_init_guess = MEAS_TIME[meas_no]*multiplets[CT_no][r_no][d_no][h_no][multiplet_no].popt                  ## use the results from the fit to the broadened MCNP spectrum (scaled by the measurement time) as starting values for the fit to the "meas" data

                            params_init_guess[-N_peaks:] = multiplets[CT_no][r_no][d_no][h_no][multiplet_no].popt[-N_peaks:]            ## For the a1 parameters (the last N_peaks parameters), I need a special approach. Cannot simply scale up by MEAS_TIME (as I do for the areas S and the polynomial parameters). a1 is supposed to be a fraction of the peak amplitude (i.e. a value between 0 and 1), and so I should use the same value as I got for the previous fit as a starting point (i.e. it should not depend on the measurement time)


                            # First, fit this to a scaled version of the broadened spectrum (with the correct Poisson uncertainties). This is however *not* the "realisations" of the measured spectrum - this is just to get possibly better initial guesses for the fit parameters before fitting to all realisations.

                            popt_meas, pcov_meas = curve_fit(multiplet_fit_fcn_fixed_mu_and_sigma, x_fit, meas_spectrum[CT_no, r_no, d_no, h_no, meas_no, min_bin:max_bin+1], sigma=d_meas_spectrum[CT_no, r_no, d_no, h_no, meas_no, min_bin:max_bin+1], absolute_sigma=True, p0=(params_init_guess), bounds=(params_lower_bounds, params_upper_bounds), maxfev=MAXFEV)          ## Fit of the multiplet to the "measured" spectrum, i.e. the MEAS_TIME-scaled version of the MCNP spectrum, with uncertainties from Poisson statistics. This fit is *only* used for getting initial guess for the fitting below.

                            ## Update the initial parameter guesses, to be the values from the fit to the scaled MCNP spectrum. These parameters will then be used as starting values when fitting to the individual realisations.
                            params_init_guess = popt_meas


                            for realisation_no in range(N_MCNP_realisations):
                                if ((realisation_no+1) % 50 == 0):
                                    print('Realisation ' + str(realisation_no+1) + ' of ' + str(N_MCNP_realisations))

                                # Do the fit to the spectrum. Note that d_meas_spectrum_realisations, which are the Poisson uncertainties that have been applied, are used as the sigma in the fit. This is to mimic a scenario where a real spectrum with these uncertainties has been collected.
                                try:
                                    multiplets[CT_no][r_no][d_no][h_no][multiplet_no].popt_meas_realisations[meas_no, realisation_no], multiplets[CT_no][r_no][d_no][h_no][multiplet_no].pcov_meas_realisations[meas_no, realisation_no] = curve_fit(multiplet_fit_fcn_fixed_mu_and_sigma, x_fit, meas_spectrum_realisations[CT_no, r_no, d_no, h_no, meas_no, realisation_no, min_bin:max_bin+1], sigma=d_meas_spectrum_realisations[CT_no, r_no, d_no, h_no, meas_no, realisation_no, min_bin:max_bin+1], absolute_sigma=True, p0=(params_init_guess), bounds=(params_lower_bounds, params_upper_bounds), maxfev=MAXFEV)              ## fit to the realisation of the measurement. These are stored in the "multiplets" structure, because we may want to access the fitted parameter values later.
                                except RuntimeError:
                                    print("Error - curve_fit failed")               ## to avoid halting the program if there is a problem.


                                # For each fitted peak, get the peak integral (S), which is the *peak_no* parameter in the multiplet. Note that I divide by bin_width to go from integral in [counts*MeV] to [counts]. (however, not an entirely necessary step because I do the same for the uncertainty in S and then calculate the ratio between the two).
                                for peak_no in range(N_peaks):
                                    multiplets[CT_no][r_no][d_no][h_no][multiplet_no].A_fit_meas_realisations[peak_no, meas_no, realisation_no] = multiplets[CT_no][r_no][d_no][h_no][multiplet_no].popt_meas_realisations[meas_no, realisation_no, peak_no]/bin_width                      ## popt_meas_realisations[meas_no, realisation_no, *peak_no*] gives me the parameter number peak_no (i.e. the peak area for the peak in question)
                                    multiplets[CT_no][r_no][d_no][h_no][multiplet_no].d_A_fit_meas_realisations[peak_no, meas_no, realisation_no] = np.sqrt(multiplets[CT_no][r_no][d_no][h_no][multiplet_no].pcov_meas_realisations[meas_no, realisation_no, peak_no, peak_no])/bin_width                      ## pcov_meas_realisations[meas_no, realisation_no, *peak_no*] gives me the parameter covariance matrix. Get the variance for the S parameter at element [peak_no, peak_no] and take square root to get the std deviation in S. Again, divide by bin_width to get in units of [counts]



# Write spectra to file (for possible plotting later on)
for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
    else:
        CT_str = str(CT_val) + ' yr'
    for r_no, r_val in enumerate(r):
        for d_no, d_val in enumerate(d):
            for h_no, h_val in enumerate(h):
                # Write unbroadened spectrum to file:
                # Column 1: bin centres, column 2: spectrum value, column 3: spectrum uncertainty
                spectrum_and_d_spectrum = np.stack([bin_centres, unbroadened_spectrum[CT_no, r_no, d_no, h_no], d_unbroadened_spectrum[CT_no, r_no, d_no, h_no]], axis=1)
                spectrum_filename = "unbroadened_spectrum_CT_" + str(CT_val) + "_r_" + str(r_val) + "_d_" + str(d_val) + "_h_" + str(h_val) + ".csv"
                np.savetxt(spectrum_filename, spectrum_and_d_spectrum, delimiter=",")

                # Write broadened spectrum to file:
                spectrum_and_d_spectrum = np.stack([bin_centres, broadened_spectrum[CT_no, r_no, d_no, h_no], d_broadened_spectrum[CT_no, r_no, d_no, h_no]], axis=1)
                spectrum_filename = "broadened_spectrum_CT_" + str(CT_val) + "_r_" + str(r_val) + "_d_" + str(d_val) + "_h_" + str(h_val) + ".csv"
                np.savetxt(spectrum_filename, spectrum_and_d_spectrum, delimiter=",")


                # Write the fit(s) to file. Done by obtaining the x and y values for all fits and putting together into a long array:
                x_fit_TOTAL = np.empty(0)
                y_fit_TOTAL = np.empty(0)

                for multiplet_no in range(N_multiplets):
                    N_peaks = len(peaks_to_fit[CT_no][multiplet_no])
                    fixed_params = np.hstack([N_peaks, multiplets[CT_no][r_no][d_no][h_no][multiplet_no].peak_bin_mu, multiplets[CT_no][r_no][d_no][h_no][multiplet_no].peak_bin_sigma])

                    min_bin = multiplets[CT_no][r_no][d_no][h_no][multiplet_no].fit_x_range_bin[0]
                    max_bin = multiplets[CT_no][r_no][d_no][h_no][multiplet_no].fit_x_range_bin[1]
                    x_fit = bin_centres[min_bin:max_bin+1]

                    x_fit_TOTAL = np.append(x_fit_TOTAL, x_fit)
                    y_fit_TOTAL = np.append(y_fit_TOTAL, multiplet_fit_fcn_fixed_mu_and_sigma(x_fit, *multiplets[CT_no][r_no][d_no][h_no][multiplet_no].popt))

                # Write fit to broadened spectrum to file:
                spectrum_fit = np.stack([x_fit_TOTAL, y_fit_TOTAL], axis=1)
                spectrum_filename = "fitted_broadened_spectrum_CT_" + str(CT_val) + "_r_" + str(r_val) + "_d_" + str(d_val) + "_h_" + str(h_val) + ".csv"
                np.savetxt(spectrum_filename, spectrum_fit, delimiter=",")



output_data = np.ones((N_CT, N_r, N_d, N_h, N_multiplets, 10, N_meas, 7+2+2+2+2))           ## "10" is a placeholder for the maximum number of peaks per multiplet
output_data = -1*output_data
#output_data_columns =  ['CT', 'r', 'd', 'h', 'Multiplet_no', 'Peak_no', 'Measurement_time', 'A_raw', 'd_A_raw', 'A_fit', 'd_A_fit', 'diff_A_raw_A_fit', 'd_diff_A_raw_A_fit', 'mean_A_meas', 'sigma_A_meas', 'mean_d_A_meas', 'sigma_d_A_meas', 'd_A_rel_meas', 'd_d_A_rel_meas', 'peak_to_bg', 'd_peak_to_bg', 'TOTAL_count_rate_above_1keV', 'd_TOTAL_count_rate_above_1keV']
output_data_columns =  ['CT', 'r', 'd', 'h', 'Multiplet_no', 'Peak_no', 'Measurement_time', 'mean_A_meas', 'sigma_A_meas', 'mean_d_A_meas', 'sigma_d_A_meas', 'd_A_rel_meas', 'd_d_A_rel_meas', 'TOTAL_count_rate_above_1keV', 'd_TOTAL_count_rate_above_1keV']


for CT_no, CT_val in enumerate(CT):
    if (CT_val < 1):
        CT_str = str(int(CT_val*10)) + ' months'
    else:
        CT_str = str(CT_val) + ' yr'
    for r_no, r_val in enumerate(r):
        for d_no, d_val in enumerate(d):
            for h_no, h_val in enumerate(h):
                for multiplet_no in range(N_multiplets):
                    N_peaks = len(peaks_to_fit[CT_no][multiplet_no])
                    for peak_no in range(N_peaks):
                        print('Peak ' + str(peak_no))

                        mean_A_meas = np.mean(multiplets[CT_no][r_no][d_no][h_no][multiplet_no].A_fit_meas_realisations[peak_no, :, :], axis=1)
                        sigma_A_meas = np.std(multiplets[CT_no][r_no][d_no][h_no][multiplet_no].A_fit_meas_realisations[peak_no, :, :], axis=1)

                        mean_d_A_meas = np.mean(multiplets[CT_no][r_no][d_no][h_no][multiplet_no].d_A_fit_meas_realisations[peak_no, :, :], axis=1)
                        sigma_d_A_meas = np.std(multiplets[CT_no][r_no][d_no][h_no][multiplet_no].d_A_fit_meas_realisations[peak_no, :, :], axis=1)

                        d_A_rel_meas = np.mean(np.divide(multiplets[CT_no][r_no][d_no][h_no][multiplet_no].d_A_fit_meas_realisations[peak_no, :, :], multiplets[CT_no][r_no][d_no][h_no][multiplet_no].A_fit_meas_realisations[peak_no, :, :]), axis=1)                     ## for each realisation, calculate the relative uncertainty in the fitted area. The mean of all realisations is the average rel. uncert.
                        d_d_A_rel_meas = np.std(np.divide(multiplets[CT_no][r_no][d_no][h_no][multiplet_no].d_A_fit_meas_realisations[peak_no, :, :], multiplets[CT_no][r_no][d_no][h_no][multiplet_no].A_fit_meas_realisations[peak_no, :, :]), axis=1)                   ## the std dev of all relative uncertainties describes the variation between different realisations (i.e. due to MCNP).


                        TOTAL_count_rate_above_1keV_out = np.mean(TOTAL_count_rate_above_1keV_realisations[CT_no, r_no, d_no, h_no, :])
                        d_TOTAL_count_rate_above_1keV_out = np.std(TOTAL_count_rate_above_1keV_realisations[CT_no, r_no, d_no, h_no, :])


                        for meas_no in range(N_meas):
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 0] = CT_val
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 1] = r_val
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 2] = d_val
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 3] = h_val
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 4] = multiplet_no
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 5] = peak_no
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 6] = MEAS_TIME[meas_no]
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 7] = mean_A_meas[meas_no]
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 8] = sigma_A_meas[meas_no]
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 9] = mean_d_A_meas[meas_no]
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 10] = sigma_d_A_meas[meas_no]
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 11] = d_A_rel_meas[meas_no]
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 12] = d_d_A_rel_meas[meas_no]
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 13] = TOTAL_count_rate_above_1keV_out
                            output_data[CT_no, r_no, d_no, h_no, multiplet_no, peak_no, meas_no, 14] = d_TOTAL_count_rate_above_1keV_out

output_data_df = pd.DataFrame(data = np.reshape(output_data, (N_CT*N_r*N_d*N_h*N_multiplets*10*N_meas, 7+2+2+2+2)), columns = output_data_columns)
#output_data_df = output_data_df.replace({'Nuclide': })

## I have defined 10 peaks per multiplet in the df to be able to hold all. I set all to -1 by default, so now I know that the elements which are still -1 are not "real". So, remove them to produce the final df. Also, replace float with int in the Multiplet_no and Peak_no columns
output_data_df = output_data_df[output_data_df['CT'] > 0.].copy()
output_data_df[['Multiplet_no', 'Peak_no']] = output_data_df[['Multiplet_no', 'Peak_no']].astype(int)


print(output_data_df)

output_data_df.to_csv('output_data/analysed_peak_data_CT_' + str(CT[0]) + '_r_' + str(r[0]) + '.csv')          ## if only processing one CT and r (from sys.argv)




