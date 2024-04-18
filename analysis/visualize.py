from ising.ising_model import IsingModel2D
import celluloid
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    nrows = 50
    ncols = 50
    temperature = 2.2
    model = IsingModel2D(nrows, ncols, temperature)
    steps = 500
    traj, energies = model.simulate(steps, burn_in=0, save_every=1)
    plt.plot(energies)
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy vs Step")
    plt.savefig("../results/ising_energy_%dx%d_t_%.3f.png"%(nrows, ncols, temperature))
    plt.close()
    np.save("../results/ising_traj_%dx%d_t_%.3f.npy"%(nrows, ncols, temperature), traj)
    # animate the simulation
    fig, ax = plt.subplots()
    camera = celluloid.Camera(fig)
    for i in range(steps):
        plt.imshow(traj[i], animated=True, cmap='binary')
        camera.snap()
    animation = camera.animate(interval=5, blit=True)
    animation.save("../results/ising_traj_%dx%d_t_%.3f.gif"%(nrows, ncols, temperature))
    
