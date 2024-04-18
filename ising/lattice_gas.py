import numpy as np

class LatticeGas2D:
    def __init__(self, nrows, ncols, temperature, A=1, B=1):
        self.nrows = nrows
        self.ncols = ncols
        self.temperature = temperature
        self.particles = np.ones((nrows, ncols), dtype=int) # 1 for particle, 0 for empty
        self.A = A
        self.B = B
    
    def energy(self):
        return self.B* np.sum(self.particles * (np.roll(self.particles, 1, axis=0) + np.roll(self.particles, 1, axis=1))) + self.A * np.sum(self.particles)
    
    def update(self):
        for _ in range(self.nrows * self.ncols):
            i = np.random.randint(self.nrows)
            j = np.random.randint(self.ncols)
            e_before = self.energy()
            neighbors_sum = (
                self.particles[(i - 1) % self.nrows, j]
                + self.particles[(i + 1) % self.nrows, j]
                + self.particles[i, (j - 1) % self.ncols]
                + self.particles[i, (j + 1) % self.ncols]
            )
            diff = 1 if self.particles[i, j] == 0 else -1
            dE = diff*(self.A + self.B * neighbors_sum)
            if dE <= 0 or np.random.rand() < np.exp(-dE / self.temperature):
                self.particles[i, j] += diff
                e_after = self.energy()
                assert e_after == e_before + dE
    
    def simulate(self, steps, burn_in=1000, save_every=1):
        trajectories = []
        energies = []
        for i in range(steps+burn_in):
            self.update()
            if i >= burn_in and i % save_every == 0:
                trajectories.append(np.copy(self.particles))
                energies.append(self.energy())
        return trajectories, energies



