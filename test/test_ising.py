import pytest
import numpy as np

from ising.ising_model import IsingModel2D

class TestIsingModel2D:
    @pytest.fixture
    def model(self):
        return IsingModel2D(nrows=3, ncols=2, temperature=1.0, random_init=False)

    def test_initialization(self, model):
        assert model.spins.shape == (3, 2)
        assert np.all(np.abs(model.spins) == 1)

    def test_energy(self, model):
        assert model.energy() == pytest.approx(-12)

        # For a specific configuration
        model.spins = np.array([[1, 1], [0, 0], [0, 1]])
        assert model.energy() == pytest.approx(-3)

    def test_dE(self, model):
        for _ in range(10):
            i = np.random.randint(model.nrows)
            j = np.random.randint(model.ncols)
            e_before = model.energy()
            dE = 2 * model.J* model.spins[i, j] * (
                model.spins[(i - 1) % model.nrows, j] +
                model.spins[(i + 1) % model.nrows, j] +
                model.spins[i, (j - 1) % model.ncols] +
                model.spins[i, (j + 1) % model.ncols]
            ) + 2 * model.H * model.spins[i, j]
            if dE <= 0 or np.random.rand() < np.exp(-dE / model.temperature):
                model.spins[i, j] *= -1
                e_after = model.energy()
                assert e_after == pytest.approx(e_before + dE)
    
    def test_simulate(self, model):
        trajectories, energies = model.simulate(steps=200, burn_in=100, save_every=1)
        assert len(trajectories) == 200
        assert len(energies) == 200
        #check the trajectories only contain -1 and 1
        assert np.all(np.isin(trajectories, [-1, 1]))