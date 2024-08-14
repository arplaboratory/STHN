import matplotlib.pyplot as plt
import numpy as np

def plot_hist_helper(path):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    data = np.load(f'{path}/resnpy.npy', allow_pickle=True)
    n, bins, patches = plt.hist(x=data, bins=np.linspace(0, 100, 20))
    plt.title("Test MACE")
    plt.ylim(0, 20000)
    plt.xlabel("MACE")
    plt.ylabel("Frequency")
    plt.savefig(f"{path}/hist.png")
    plt.close()

    # plt.figure()
    # flow_data = np.load(f'{path}/flownpy.npy', allow_pickle=True)
    # n, bins, patches = plt.hist(x=flow_data, bins=np.linspace(0, 100, 20))
    # plt.title("Test Flow")
    # plt.ylim(0, 20000)
    # plt.xlabel("Flow")
    # plt.ylabel("Frequency")
    # plt.savefig(f"{path}/flowhist.png")
    # plt.close()

if __name__ == '__main__':
    path = "IHN_results/satellite_thermal_dense"
    plot_hist_helper(path)