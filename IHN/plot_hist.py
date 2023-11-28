import matplotlib.pyplot as plt
import numpy as np

# An "interface" to matplotlib.axes.Axes.hist() method
data = np.load('IHN_results/satellite_thermal_ext_dense/resnpy.npy', allow_pickle=True)
n, bins, patches = plt.hist(x=data, bins=20)
plt.title("Test MACE")
plt.ylim(0, 20000)
plt.xlabel("MACE")
plt.ylabel("Freqency")
plt.savefig("hist.png")