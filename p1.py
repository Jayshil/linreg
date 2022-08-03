import numpy as np
import matplotlib.pyplot as plt
import utils as utl
import os

roll, fl, fle = np.loadtxt(os.getcwd() + '/Data/roll-angle-trend-kelt11.dat', usecols=(0,1,2), unpack=True)

design = utl.des_mat(50, roll)

bets, covs = utl.gls(design, fl, fle)

model_y = design @ bets

plt.errorbar(roll, fl, yerr=fle, fmt='.', zorder=5)
plt.plot(roll, model_y, 'k', zorder=10)
plt.show()