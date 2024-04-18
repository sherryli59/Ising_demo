import pytest
import numpy as np

from ising.lattice_gas import LatticeGas2D

class TestLatticegAS2D:
    @pytest.fixture
    def model(self):
        return LatticeGas2D(nrows=3, ncols=2,A=1, B=2, temperature=1.0)

    def test_initialization(self, model):
        assert model.particles.shape == (3, 2)
        assert np.all(np.abs(model.particles) == 1)

    def test_energy(self, model):
        assert model.energy() == pytest.approx(30)

        # For a specific configuration
        model.particles = np.array([[1, 1], [0, 0], [0, 1]])
        assert model.energy() == pytest.approx(9)

    def test_dE(self, model):
        for _ in range(10):
            i = np.random.randint(model.nrows)
            j = np.random.randint(model.ncols)
            e_before = model.energy()
            neighbors_sum = (
                model.particles[(i - 1) % model.nrows, j]
                + model.particles[(i + 1) % model.nrows, j]
                + model.particles[i, (j - 1) % model.ncols]
                + model.particles[i, (j + 1) % model.ncols]
            )
            diff = 1 if model.particles[i, j] == 0 else -1
            dE = diff*(model.A +  model.B * neighbors_sum)
            if dE <= 0 or np.random.rand() < np.exp(-dE / model.temperature):
                model.particles[i, j] += diff
                e_after = model.energy()
                assert e_after == pytest.approx(e_before + dE)
    
    def test_simulate(self, model):
        trajectories, energies = model.simulate(steps=200, burn_in=100, save_every=1)
        assert len(trajectories) == 200
        assert len(energies) == 200
        #check the trajectories only contain 0 and 1
        assert np.all(np.isin(trajectories, [0, 1]))


