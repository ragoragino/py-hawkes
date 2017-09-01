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

T = 100000
limit = 200000
seed = 123
dim = 3
plot_range = (0, 100)
grid = 0.05

@pytest.fixture
def parameters2():
    mu = np.array([0.15, 0.15], dtype=float)
    alpha = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=float)
    beta = np.array([[1, 1], [1, 1]], dtype=float)
    return mu, alpha, beta


@pytest.fixture
def parameters3():
    mu = np.array([0.05, 0.3, 0.25])
    alpha = np.array([[0.4, 0.3, 0.1], [0.3, 0.4, 0.23], [0.18, 0.43, 0.31]])
    beta = np.array([[1, 2, 9], [1.2, 1.8, 1.4], [4.9, 1.15, 8.5]])
    return mu, alpha, beta


@pytest.fixture
def hawkes_list_stat(parameters3):
    mu, alpha, beta = parameters3
    hawkes = pyhawkes.sim_exp_hawkes(mu, alpha, beta, T, max=limit, rseed=seed)
    return hawkes


@pytest.fixture
def comp_list_stat(parameters3):
    def comp_list_stat_int(pos):
        mu, alpha, beta = parameters3
        hawkes = pyhawkes.sim_exp_hawkes(mu, alpha, beta, T, max=limit, rseed=seed)
        comp = pyhawkes.comp_exp_hawkes(mu, alpha, beta, hawkes, T)
        return comp[pos]
    return comp_list_stat_int


@pytest.fixture
def hawkes_list(parameters2):
    mu, alpha, beta = parameters2
    hawkes = pyhawkes.sim_exp_hawkes(mu, alpha, beta, length=T, max=limit, rseed=seed)
    return hawkes

sim_partial = functools.partial(pyhawkes.sim_exp_hawkes, max=limit, rseed=seed)
comp_partial = functools.partial(pyhawkes.comp_exp_hawkes, events=hawkes_list)
lik_partial = functools.partial(pyhawkes.lik_exp_hawkes, events=hawkes_list)
plot_partial = functools.partial(pyhawkes.plot_exp_hawkes, events=hawkes_list,
                                 begin=plot_range[0], end=plot_range[1], grid=grid)
functions = [sim_partial, comp_partial, lik_partial, plot_partial]


@pytest.mark.parametrize('function', functions)
class TestExponential:
    def test_type_mu(self, function):
        mu = np.array([1, 1], dtype=int)
        alpha = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=float)
        beta = np.array([[1, 1], [1, 1]], dtype=float)
        try:
            function(mu, alpha, beta, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_type_alpha(self, function):
        mu = np.array([0.15, 0.15], dtype=float)
        alpha = np.array([[1, 1], [1, 1]], dtype=int)
        beta = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=float)
        try:
            function(mu, alpha, beta, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_type_beta(self, function):
        mu = np.array([0.15, 0.15], dtype=float)
        alpha = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=float)
        beta = np.array([[1, 1], [1, 1]], dtype=int)
        try:
            function(mu, alpha, beta, length=T)
        except ValueError:
            assert True
        else:
            assert False

    def test_pos_mu(self, function):
        mu = np.array([-0.15, 0.15], dtype=float)
        alpha = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=float)
        beta = np.array([[1, 1], [1, 1]], dtype=float)
        try:
            function(mu, alpha, beta, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_pos_alpha(self, function):
        mu = np.array([0.15, 0.15], dtype=float)
        alpha = np.array([[0.3, 0.3], [0.3, -0.3]], dtype=float)
        beta = np.array([[1, 1], [1, 1]], dtype=float)
        try:
            function(mu, alpha, beta, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_pos_beta(self, function):
        mu = np.array([0.15, 0.15], dtype=float)
        alpha = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=float)
        beta = np.array([[1, 1], [1, -1]], dtype=float)
        try:
            function(mu, alpha, beta, length=T)
        except pyhawkes.ParametersConstraintError:
            assert True
        else:
            assert False

    def test_shape_mu(self, function):
        mu = np.array([0.15, 0.15, 0.15], dtype=float)
        alpha = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=float)
        beta = np.array([[1, 1], [1, 1]], dtype=float)
        try:
            function(mu, alpha, beta, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False

    def test_shape_alpha(self, function):
        mu = np.array([0.15, 0.15], dtype=float)
        alpha = np.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]], dtype=float)
        beta = np.array([[1, 1], [1, 1]], dtype=float)
        try:
            function(mu, alpha, beta, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False

    def test_shape_beta(self, function):
        mu = np.array([0.15, 0.15], dtype=float)
        alpha = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=float)
        beta = np.array([[1, 1, 1], [1, 1, 1]], dtype=float)
        try:
            function(mu, alpha, beta, length=T)
        except pyhawkes.IncompatibleShapeError:
            assert True
        else:
            assert False


def test_stationarity():
    mu = np.array([0.15, 0.15], dtype=float)
    alpha = np.array([[100, 100], [100, 100]], dtype=float)
    beta = np.array([[1, 1], [1, 1]], dtype=float)
    try:
        pyhawkes.sim_exp_hawkes(mu, alpha, beta, length=T, max=limit, rseed=seed)
    except RuntimeError:
        assert True
    else:
        assert False


def test_limit(parameters2):
    mu, alpha, beta = parameters2
    loc_limit = 10
    try:
        pyhawkes.sim_exp_hawkes(mu, alpha, beta, length=T, max=loc_limit, rseed=seed)
    except RuntimeError:
        assert True
    else:
        assert False


def test_ks(comp_list_stat):
    p_value = [[], [], []]
    for i in range(3):
        _, p_value[i] = scipy.stats.kstest(comp_list_stat(i), 'expon', args=(0, 1))
    assert all([i > 0.05 for i in p_value])


def test_ll(hawkes_list_stat, parameters3):
    mu, alpha, beta = parameters3
    mu = np.array([0.05, 0.3, 0.25])
    alpha = np.array([[0.4, 0.3, 0.1], [0.3, 0.4, 0.23], [0.18, 0.43, 0.31]])
    beta = np.array([[1, 2, 9], [1.2, 1.8, 1.4], [4.9, 1.15, 8.5]])
    hawkes = pyhawkes.sim_exp_hawkes(mu, alpha, beta, length=100000, max=200000, rseed=123)
    log_likelihood = pyhawkes.lik_exp_hawkes(mu, alpha, beta, hawkes, T)
    assert np.round(log_likelihood) == 204777