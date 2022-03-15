import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm
from matplotlib.patches import Polygon, Circle

class fuel_pin:
    def __init__(self, orientation, pin_pitch, pin_radius):
        if (orientation == 'flat_topped' or orientation == 'pointy_topped'):
            self.orientation = orientation
        else:
            print('Strange orientation! Exiting.')
            exit()

        r = pin_pitch/2.                ## the hexagon radius is half the pin pitch

        self.hex_r = r
        self.hex_t = r*(2./np.sqrt(3.))         ## hexagon side
        self.hex_R = r*(2./np.sqrt(3.))         ## hexagon larger radius

        self.pin_r = pin_radius

        self.cmap = cm.get_cmap('Reds')

    def set_color(self, zval, zmin, zmax):
        self.zval_norm = (zval - zmin)/(zmax - zmin)

    def define_geom(self, q_coord, r_coord):
        self.centre = np.array([q_coord*2.*self.hex_r + r_coord*self.hex_r, -np.sqrt(3.)*r_coord*self.hex_r])           ## since angle between sides is 60 degrees, to get step in y direction I take tan(60) = Delta_y/r => Delta_y = r*tan(60) = r*sqrt(3). Minus sign because going to negative r moves me up (in y direction), and reverse.

        self.centre[1] = -1*self.centre[1]          ### reflect the y value, because the r coordinate on https://www.redblobgames.com/grids/hexagons/ does not match the "y" coord in MCNP, see Fig. 4-37 of manual

        if (self.orientation == 'pointy_topped'):
            hexagon_corners = np.array([[self.hex_r, 0.5*self.hex_t], [0., self.hex_R], [-self.hex_r, 0.5*self.hex_t], [-self.hex_r, -0.5*self.hex_t], [0., -self.hex_R], [self.hex_r, -0.5*self.hex_t]])

        hexagon_corners = hexagon_corners + self.centre             ## offset by the position of the centre of this hexagon.
        self.polygon = Polygon(hexagon_corners, closed=True, edgecolor='black', facecolor='None')

        self.marker = Circle(self.centre, radius = self.pin_r, facecolor=self.cmap(self.zval_norm))




fuel_pins = []

pin_pitch = 8.4             ## mm
pin_radius = 5.42/2.             ## mm
pin_counter = 0

pin_centres = np.zeros((127, 2))

for i in range(0, 127):
    fuel_pins.append(fuel_pin('pointy_topped', pin_pitch, pin_radius))
    fuel_pins[i].set_color(5., 10., 100.)

for q in range(0, 7):
    fuel_pins[pin_counter].define_geom(q, -6)
    pin_counter += 1

for q in range(-1, 7):
    fuel_pins[pin_counter].define_geom(q, -5)
    pin_counter += 1

for q in range(-2, 7):
    fuel_pins[pin_counter].define_geom(q, -4)
    pin_counter += 1

for q in range(-3, 7):
    fuel_pins[pin_counter].define_geom(q, -3)
    pin_counter += 1

for q in range(-4, 7):
    fuel_pins[pin_counter].define_geom(q, -2)
    pin_counter += 1

for q in range(-5, 7):
    fuel_pins[pin_counter].define_geom(q, -1)
    pin_counter += 1

for q in range(-6, 7):
    fuel_pins[pin_counter].define_geom(q, 0)
    pin_counter += 1

for q in range(-6, 6):
    fuel_pins[pin_counter].define_geom(q, 1)
    pin_counter += 1

for q in range(-6, 5):
    fuel_pins[pin_counter].define_geom(q, 2)
    pin_counter += 1

for q in range(-6, 4):
    fuel_pins[pin_counter].define_geom(q, 3)
    pin_counter += 1

for q in range(-6, 3):
    fuel_pins[pin_counter].define_geom(q, 4)
    pin_counter += 1

for q in range(-6, 2):
    fuel_pins[pin_counter].define_geom(q, 5)
    pin_counter += 1

for q in range(-6, 1):
    fuel_pins[pin_counter].define_geom(q, 6)
    pin_counter += 1


for i in range(0, 127):
    pin_centres[i, 0] = fuel_pins[i].centre[0]
    pin_centres[i, 1] = fuel_pins[i].centre[1]


energy = ['500keV', '1000keV', '1500keV', '2000keV', '2500keV']
N_pins_per_row = [7, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 7]


# These input files contain the *photopeak* intensities for each of the energies of interest from all 127 source pins in the FA.
infile = ['values_leftdet.csv', 'values_rightdet.csv']


P = np.zeros(shape=(len(infile), len(energy), 127))
d_P = np.zeros(shape=(len(infile), len(energy), 127))

P_tot = np.zeros(shape=(len(infile), len(energy)))
d_P_tot = np.zeros(shape=(len(infile), len(energy)))

P_rel_tally_summed = np.zeros(shape=(len(energy), 127))
d_P_rel_tally_summed = np.zeros(shape=(len(energy), 127))

P_rel_row_sum_tally_summed = np.zeros(shape=(len(energy), len(N_pins_per_row)))
d_P_rel_row_sum_tally_summed = np.zeros(shape=(len(energy), len(N_pins_per_row)))


for infile_no in range(len(infile)):                ## loop over the two detectors.
    infile_name = 'input_data/monoenergetic_F8_tally_' + infile[infile_no]
    df = pd.read_csv(infile_name)

    d_infile_name = 'input_data/monoenergetic_F8_tally_d_' + infile[infile_no]
    d_df = pd.read_csv(d_infile_name)

    for i in range(len(energy)):
        # Get the intensity data from the input file.
        P_df = df[energy[i]]
        d_P_df = d_df[energy[i]]

        P[infile_no, i] = P_df.to_numpy()
        d_P[infile_no, i] = d_P_df.to_numpy()

        # The *total* intensity summed over all pins
        P_tot[infile_no, i] = P_df.sum()                ## the total count rate in the peak (summed over all pins)
        d_P_tot[infile_no, i] = np.sqrt(np.power(d_P_df, 2).sum())



## Because the two detectors were on the opposite y side, need to flip all the pin entries for P[1]. Axis 0 in P[1] is the energy, axis 1 is the pin number.
P[1] = np.flip(P[1], axis=1)
d_P[1] = np.flip(d_P[1], axis=1)

## Sum the *pin-wise* rate from the two tallies, then average
P_tally_summed = np.add(P[0], P[1])/2.              ## divide by two to get average
d_P_tally_summed = np.sqrt(np.power(d_P[0], 2) + np.power(d_P[1], 2))/2.              ## propagate uncertainty, divide by two to get average

# Sum the *total* rate from all pins over the two tallies, then average
P_tot_tally_summed = np.add(P_tot[0], P_tot[1])/2.              ## divide by two to get average
d_P_tot_tally_summed = np.sqrt(np.power(d_P_tot[0], 2) + np.power(d_P_tot[1], 2))/2.              ## propagate uncertainty, divide by two to get average



for i in range(len(energy)):
    P_rel_tally_summed[i] = P_tally_summed[i]/P_tot_tally_summed[i]         ## the relative contribution from each pin (including data from both tallies)

    d_P_rel_tally_summed[i] = P_rel_tally_summed[i]*np.sqrt(np.power(d_P_tally_summed[i]/P_tally_summed[i], 2) + np.power(d_P_tot_tally_summed[i]/P_tot_tally_summed[i], 2))


# In cases where P = 0 (due to low stats), the d_P_rel will become nan (because of division by zero in d_P_tally_summed/P_tally_summed). So, replace those nan:s with zeroes:
d_P_rel_tally_summed = np.nan_to_num(d_P_rel_tally_summed, nan=0.)


# Plot the fuel assembly (FA) with the relative intensities of all pins.
for i in range(len(energy)):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    
    sc = ax.scatter(pin_centres[:,0], pin_centres[:, 1], s=80, c=100.*P_rel_tally_summed[i], cmap='Reds', norm=mpl.colors.LogNorm())
    plt.colorbar(sc, label='Fractional contribution to X-keV peak area [%]')
    plt.xlabel('x position [mm]')
    plt.ylabel('y position [mm]')

plt.plot(pin_centres[0, 0], pin_centres[0, 1], marker='o')
plt.plot(pin_centres[3, 0], pin_centres[3, 1], marker='o')

#NOTE: Row 12 (and pin 127) is the closest to the detector!

for i in range(len(energy)):
    counter = 0
    for row_no in range(len(N_pins_per_row)):
        index = np.arange(0, N_pins_per_row[row_no])

        P_rel_sel = P_rel_tally_summed[i, counter:(counter+N_pins_per_row[row_no])]     ## the relative P values (the relative contribution from each pin compared with the total P from the entire FA) for the current energy and current row
        P_rel_row_sum_tally_summed[i, row_no] = np.sum(P_rel_sel)                       ## how much each row contributes

        ## should also plot the relative contribution of each row (not divided by number of pins per row). This is np.sum(P_rel_sel) ((contribution per pin/total count rate) summed per row)

        d_P_rel_sel = d_P_rel_tally_summed[i, counter:(counter+N_pins_per_row[row_no])]         # get the standard deviation of the relative P value for each pin in this row.
        d_P_rel_row_sum_tally_summed[i, row_no] = np.sqrt(np.sum(np.power(d_P_rel_sel, 2)))     # get the uncertainty in the relative P for the *entire row* using quadrature summation.
    
        counter += N_pins_per_row[row_no]


# The relative contribution from each row:
P_rel_row_sum_tally_summed_flipped = np.flip(P_rel_row_sum_tally_summed, axis=1)        ## to get row 0 to be closest to detector
d_P_rel_row_sum_tally_summed_flipped = np.flip(d_P_rel_row_sum_tally_summed, axis=1)        ## to get row 0 to be closest to detector



## Get the relative contribution of different rows can answer the question "if I detect X counts at energy E, what fraction comes from row Y?
for i in range(len(energy)):
    xdata = np.arange(1, len(N_pins_per_row)+1)                 ## start counting at row 1
    ydata = (100*P_rel_row_sum_tally_summed_flipped[i])
    d_ydata = (100*d_P_rel_row_sum_tally_summed_flipped[i])

    print('Relative contribution by rows at ' + energy[i] + ' [%]')
    print(ydata)
    print('Std dev of Relative contribution by rows at ' + energy[i] + ' [%]')
    print(d_ydata)




plt.show()
