import numpy as np
import scipy.stats
import pytest
import os
import sys
import functools

"""
Run by py.test [dir] > [output_dir] on Windows, Python 3.6.
The output to command prompt does not work on Python 3.6:
See https://github.com/pytest-dev/pytest/pull/2462.
"""

# Appending directory above the current one to the sys.path
cur_dir = os.path.dirname(os.path.realpath(__file__))
split_dir = cur_dir.split('\\')
above_dir = '\\'.join(split_dir[:-1])
sys.path.append(above_dir)
import pyhawkes


dim = 3
seed = 123
T = 100000
limit = 200000
plot_range = (0, 100)
grid = 0.05


@pytest.fixture
def parameters2():
    mu = np.array([0.15, 0.15], dtype=float)
    rho = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=float)
    m = np.array([[5, 5], [5, 5]], dtype=float)
    M = np.array([[5, 5], [5, 5]], dtype=int)
    epsilon = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=float)
    n = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=float)
    return mu, rho, m, M, epsilon, n


@pytest.fixture
def parameters3():
    mu = np.array([0.15, 0.15, 0.15], dtype=float)
    rho = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                   dtype=float)
    m = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]],
                 dtype=float)
    M = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]],
                 dtype=int)
    epsilon = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                       dtype=float)
    n = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                 dtype=float)
    return mu, rho, m, M, epsilon, n


@pytest.fixture
def hawkes_list(parameters2):
    mu, rho, m, M, epsilon, n = parameters2
    hawkes = pyhawkes.sim_power_hawkes(mu, rho, m, M, epsilon, n, length=T,
                                       max=limit, rseed=seed)
    return hawkes


@pytest.fixture
def hawkes_list_stat(parameters3):
    mu, rho, m, M, epsilon, n = parameters3
    hawkes = pyhawkes.sim_power_hawkes(mu, rho, m, M, epsilon, n, T, max=limit,
                                       rseed=seed)
    return hawkes


@pytest.fixture
def comp_list_stat(parameters3):
    def comp_list_stat_int(pos):
        mu, rho, m, M, epsilon, n = parameters3
        hawkes = pyhawkes.sim_power_hawkes(mu, rho, m, M, epsilon, n, T, max=limit,
                                           rseed=seed)
        comp = pyhawkes.comp_power_hawkes(mu, rho, m, M, epsilon, n, hawkes, T)
        return comp[pos]
    return comp_list_stat_int

sim_partial = functools.partial(pyhawkes.sim_power_hawkes, max=limit, rseed=seed)
comp_partial = functools.partial(pyhawkes.comp_power_hawkes, events=hawkes_list)
lik_partial = functools.partial(pyhawkes.lik_power_hawkes, events=hawkes_list)
plot_partial = functools.partial(pyhawkes.plot_power_hawkes, events=hawkes_list,
                                 begin=plot_range[0], end=plot_range[1], grid=grid)
functions = [sim_partial, comp_partial, lik_partial, plot_partial]


@pytest.mark.usefixtures('parameters2')
@pytest.mark.parametrize('function', functions)
class TestExponential:
    def test_type_mu(self, function, parameters2):
        _, rho, m, M, epsilon, n = parameters2
        mu = np.array([1, 1], dtype=int)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_type_rho(self, function, parameters2):
        mu, _, m, M, epsilon, n = parameters2
        rho = np.array([[1, 1], [1, 1]], dtype=int)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_type_m(self, function, parameters2):
        mu, rho, _, M, epsilon, n = parameters2
        m = np.array([[1, 1], [1, 1]], dtype=int)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_type_M(self, function, parameters2):
        mu, rho, m, _, epsilon, n = parameters2
        M = np.array([[1, 1], [1, 1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_type_epsilon(self, function, parameters2):
        mu, rho, m, M, _, n = parameters2
        epsilon = np.array([[1, 1], [1, 1]], dtype=int)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_type_n(self, function, parameters2):
        mu, rho, m, M, epsilon, _ = parameters2
        n = np.array([[1, 1], [1, 1]], dtype=int)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_pos_mu(self, function, parameters2):
        _, rho, m, M, epsilon, n = parameters2
        mu = np.array([-0.15, 0.15], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_pos_rho(self, function, parameters2):
        mu, _, m, M, epsilon, n = parameters2
        rho = np.array([[-0.1, 0.1], [0.1, 0.1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_pos_m(self, function, parameters2):
        mu, rho, _, M, epsilon, n = parameters2
        m = np.array([[-1, 1], [1, 1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_pos_M(self, function, parameters2):
        mu, rho, m, _, epsilon, n = parameters2
        M = np.array([[-1, 1], [1, 1]], dtype=int)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_pos_epsilon(self, function, parameters2):
        mu, rho, m, M, _, n = parameters2
        epsilon = np.array([[-1, 1], [1, 1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_pos_n(self, function, parameters2):
        mu, rho, m, M, epsilon, _ = parameters2
        n = np.array([[-1, 1], [1, 1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_shape_mu(self, function, parameters2):
        _, rho, m, M, epsilon, n = parameters2
        mu = np.array([0.15, 0.15, 0.15], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False

    def test_shape_rho(self, function, parameters2):
        mu, _, m, M, epsilon, n = parameters2
        rho = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False

    def test_shape_m(self, function, parameters2):
        mu, rho, _, M, epsilon, n = parameters2
        m = np.array([[1, 1, 1], [1, 1, 1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False

    def test_shape_M(self, function, parameters2):
        mu, rho, m, _, epsilon, n = parameters2
        M = np.array([[1, 1, 1], [1, 1, 1]], dtype=int)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False

    def test_shape_epsilon(self, function, parameters2):
        mu, rho, m, M, _, n = parameters2
        epsilon = np.array([[1, 1, 1], [1, 1, 1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False

    def test_shape_n(self, function, parameters2):
        mu, rho, m, M, epsilon, _ = parameters2
        n = np.array([[1, 1, 1], [1, 1, 1]], dtype=float)
        try:
            function(mu, rho, m, M, epsilon, n, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False


def test_stationarity(parameters2):
    mu, rho, m, M, epsilon, _ = parameters2
    n = np.array([[1, 1], [1, 1]], dtype=float)
    try:
        pyhawkes.sim_power_hawkes(mu, rho, m, M, epsilon, n, length=T, max=limit,
                                  rseed=seed)
    except RuntimeError:
        assert True
    else:
        assert False


def test_limit(parameters2):
    mu, rho, m, M, epsilon, n = parameters2
    loc_limit = 10
    try:
        pyhawkes.sim_power_hawkes(mu, rho, m, M, epsilon, n, length=T, max=loc_limit,
                                  rseed=seed)
    except RuntimeError:
        assert True
    else:
        assert False


def test_ks(comp_list_stat):
    p_value = [[], [], []]
    for i in range(3):
        _, p_value[i] = scipy.stats.kstest(comp_list_stat(i), 'expon', args=(0, 1))
    assert all([i > 0.05 for i in p_value])


def test_ll(parameters3, hawkes_list_stat):
    mu, rho, m, M, epsilon, n = parameters3
    log_likelihood = pyhawkes.lik_power_hawkes(mu, rho, m, M, epsilon, n,
                                               events=hawkes_list_stat, length=T)
    assert np.round(log_likelihood) == 202261
