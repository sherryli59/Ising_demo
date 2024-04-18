import numpy as np

class IsingModel2D:
    def __init__(self, nrows, ncols, temperature, J=1, H=0, random_init=True):
        self.nrows = nrows
        self.ncols = ncols
        self.temperature = temperature
        if random_init:
            self.spins = np.random.choice([-1, 1], size=(nrows, ncols))
        else:
            self.spins = np.ones((nrows, ncols), dtype=int)
        self.H = H
        self.J = J
    
    def energy(self):
        return -self.J* np.sum(self.spins * (np.roll(self.spins, 1, axis=0) + np.roll(self.spins, 1, axis=1))) - self.H * np.sum(self.spins)
    
    def update(self):
        for _ in range(self.nrows * self.ncols):
            i = np.random.randint(self.nrows)
            j = np.random.randint(self.ncols)
            dE = 2 * self.J* self.spins[i, j] * (
                self.spins[(i - 1) % self.nrows, j] +
                self.spins[(i + 1) % self.nrows, j] +
                self.spins[i, (j - 1) % self.ncols] +
                self.spins[i, (j + 1) % self.ncols]
            ) + 2 * self.H * self.spins[i, j]
            if dE <= 0 or np.random.rand() < np.exp(-dE / self.temperature):
                self.spins[i, j] *= -1
    
    def simulate(self, steps, burn_in=100, save_every=1):
        trajectories = []
        energies = []
        for i in range(steps+burn_in):
            self.update()
            if i >= burn_in and i % save_every == 0:
                trajectories.append(np.copy(self.spins))
                energies.append(self.energy())
        return trajectories, energies

