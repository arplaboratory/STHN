import matplotlib.pyplot as plt
import numpy as np

path = "satellite_thermal_ext_sparse_128"
# An "interface" to matplotlib.axes.Axes.hist() method
plt.figure()
data = np.load(f'IHN_results/{path}/resnpy.npy', allow_pickle=True)
n, bins, patches = plt.hist(x=data, bins=np.linspace(0, 100, 20))
plt.title("Test MACE")
plt.ylim(0, 20000)
plt.xlabel("MACE")
plt.ylabel("Frequency")
plt.savefig("hist.png")
plt.close()

plt.figure()
flow_data = np.load(f'IHN_results/{path}/flownpy.npy', allow_pickle=True)
n, bins, patches = plt.hist(x=flow_data, bins=np.linspace(0, 100, 20))
plt.title("Test Flow")
plt.ylim(0, 20000)
plt.xlabel("Flow")
plt.ylabel("Frequency")
plt.savefig("flowhist.png")
plt.close()