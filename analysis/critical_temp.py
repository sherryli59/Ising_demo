
import numpy as np
from ising.ising_model import IsingModel2D
import matplotlib.pyplot as plt
import os

def susceptibility(magnetization, temperature):
    return np.var(magnetization) / temperature


def find_critical_point(nrows, ncols, temperature_range, num_steps):
    avg_m = []
    for temperature in temperature_range:
        ising_model = IsingModel2D(nrows, ncols, temperature, random_init=False)
        trajectory,_ = ising_model.simulate(num_steps)
        magnetizations = np.abs(np.mean(trajectory,axis=(1,2)))
        #suscep = susceptibility(magnetizations, temperature)
        avg_magnetization = np.mean(magnetizations)
        avg_m.append(avg_magnetization)
    return np.array(avg_m)

# Example usage:
nrows, ncols = 50, 50
temperature_range = np.linspace(0.1, 4, 50)
num_steps = 500

avg_m = find_critical_point(nrows, ncols, temperature_range, num_steps)


plt.plot(temperature_range,avg_m , marker='o')
plt.xlabel('Temperature')
plt.ylabel('Average Magnetization')
plt.title('Average Magnetization vs Temperature')
#make a directory called results in the current directory if it does not exist
os.makedirs('../results', exist_ok=True)
plt.savefig('../results/average_magnetization_vs_temperature.png')