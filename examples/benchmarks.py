import sys
import os
import numpy as np
import logging
import time

# Appending directory above the current one to the sys.path
cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)
split_dir = cur_dir.split('\\')
above_dir = '\\'.join(split_dir[:-1])
sys.path.append(above_dir)
import pyhawkes

if __name__ == '__main__':
    logger = logging.getLogger('hawkes')
    handler = logging.FileHandler(r'benchmarks.txt', mode='w')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    T = 5000000
    seed = 123
    max_jumps = 10000000

    mu1d = np.array([0.45, 0.4], dtype=float)

    def base1d(container, loc, val):
        return container[2 * loc] + container[2 * loc + 1] * \
                                    ((val / T - 0.5) ** 2)

    mu2d = np.array([0.25, 0.2, 0.25, 0.2], dtype=float)

    def base2d(container, loc, val):
        return container[2 * loc] + container[2 * loc + 1] * \
                                    ((val / T - 0.5) ** 2)

    mu3d = np.array([0.13, 0.17, 0.13, 0.17, 0.13, 0.17], dtype=float)

    def base3d(container, loc, val):
        return container[2 * loc] + container[2 * loc + 1] * \
                                    ((val / T - 0.5) ** 2)

    """
    1D
    """

    # EXPONENTIAL WITH CONSTANT BASE INTENSITY
    mu = np.array([0.5])
    alpha = np.array([[0.4]])
    beta = np.array([[0.8]])
    begin = time.time()
    hawkes = pyhawkes.sim_exp_hawkes(mu, alpha, beta, length=T, max=max_jumps, rseed=seed)
    end = time.time()
    logger.info("1D Hawkes simulation with an exponential kernel with constant base intensity of length "
                "{} lasted for {} s".format(len(hawkes[0]), end - begin))

    # EXPONENTIAL WITH QUADRATIC BASE INTENSITY
    rho = 1 / beta
    m = np.array([[1]], dtype=float)
    M = np.array([[1]], dtype=int)
    epsilon = np.array([[0.2]], dtype=float)
    n = alpha / beta
    begin = time.time()
    hawkes = pyhawkes.sim_gen_hawkes(mu1d, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                     rseed=seed, pfunc=base1d)
    end = time.time()
    logger.info("1D Hawkes simulation with an exponential kernel with quadratic base intensity of length "
                "{} lasted for {} s".format(len(hawkes[0]), end - begin))

    # POWER-LAW WITH CONSTANT BASE INTENSITY
    mu = np.array([0.5], dtype=float)
    rho = np.array([[0.1]], dtype=float)
    m = np.array([[5]], dtype=float)
    M = np.array([[5]], dtype=int)
    epsilon = np.array([[0.2]], dtype=float)
    n = np.array([[0.5]], dtype=float)
    begin = time.time()
    hawkes = pyhawkes.sim_power_hawkes(mu, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                       rseed=seed)
    end = time.time()
    logger.info("1D Hawkes simulation with a power-law kernel with constant base intensity of length "
                "{} lasted for {} s".format(len(hawkes[0]), end - begin))

    # POWER-LAW WITH QUADRATIC BASE INTENSITY
    begin = time.time()
    hawkes = pyhawkes.sim_gen_hawkes(mu1d, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                        rseed=seed, pfunc=base1d)
    end = time.time()
    logger.info("1D Hawkes simulation with a power-law kernel with quadratic base intensity of length "
                "{} lasted for {} s".format(len(hawkes[0]), end - begin))

    """
    2D
    """

    # EXPONENTIAL WITH CONSTANT BASE INTENSITY
    mu = np.array([0.25, 0.25])
    alpha = np.array([[0.3, 0.4], [0.4, 0.5]])
    beta = np.array([[1, 2.4], [1.5, 2]])
    begin = time.time()
    hawkes = pyhawkes.sim_exp_hawkes(mu, alpha, beta, length=T, max=max_jumps, rseed=seed)
    end = time.time()
    logger.info("2D Hawkes simulation with an exponential kernel with constant base intensity of length {} "
                "lasted for {} s".format(len(hawkes[0]) + len(hawkes[1]), end - begin))

    # EXPONENTIAL WITH QUADRATIC BASE INTENSITY
    rho = 1 / beta
    m = np.array([[1, 1], [1, 1]], dtype=float)
    M = np.array([[1, 1], [1, 1]], dtype=int)
    epsilon = np.array([[1, 1], [1, 1]], dtype=float)
    n = alpha / beta
    begin = time.time()
    hawkes = pyhawkes.sim_gen_hawkes(mu2d, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                     rseed=seed, pfunc=base2d)
    end = time.time()
    logger.info("2D Hawkes simulation with an exponential kernel with quadratic base intensity of length "
                "{} lasted for {} s".format(len(hawkes[0]) + len(hawkes[1]), end - begin))

    # POWER-LAW WITH CONSTANT BASE INTENSITY
    mu = np.array([0.25, 0.25], dtype=float)
    rho = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=float)
    m = np.array([[5, 5], [5, 5]], dtype=float)
    M = np.array([[5, 5], [5, 5]], dtype=int)
    epsilon = np.array([[0.2, 0.2], [0.2, 0.2]], dtype=float)
    n = np.array([[0.25, 0.25], [0.25, 0.25]], dtype=float)
    begin = time.time()
    hawkes = pyhawkes.sim_power_hawkes(mu, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                       rseed=seed)
    end = time.time()
    logger.info("2D Hawkes simulation with a power-law kernel with constant base intensity of length {} "
                "lasted for {} s".format(len(hawkes[0]) + len(hawkes[1]), end - begin))

    # POWER-LAW WITH QUADRATIC BASE INTENSITY
    begin = time.time()
    hawkes = pyhawkes.sim_gen_hawkes(mu2d, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                         rseed=seed, pfunc=base2d)
    end = time.time()
    logger.info("2D Hawkes simulation with a power-law kernel with quadratic base intensity of length "
                "{} lasted for {} s".format(len(hawkes[0]) + len(hawkes[1]), end - begin))

    """
    3D
    """

    # EXPONENTIAL WITH CONSTANT BASE INTENSITY
    mu = np.array([0.15, 0.15, 0.15])
    alpha = np.array([[0.4, 0.3, 0.1], [0.3, 0.4, 0.23], [0.18, 0.43, 0.31]])
    beta = np.array([[1, 2, 9], [1.2, 1.8, 1.4], [4.9, 1.15, 8.5]])
    begin = time.time()
    hawkes = pyhawkes.sim_exp_hawkes(mu, alpha, beta, length=T, max=max_jumps, rseed=seed)
    end = time.time()
    logger.info("3D Hawkes simulation with an exponential kernel with constant base intensity of length {} "
                "lasted for {} s".format(len(hawkes[0]) + len(hawkes[1]) + len(hawkes[2]),
                                         end - begin))

    # EXPONENTIAL WITH QUADRATIC BASE INTENSITY
    rho = 1 / beta
    m = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)
    M = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)
    epsilon = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)
    n = alpha / beta
    begin = time.time()
    hawkes = pyhawkes.sim_gen_hawkes(mu3d, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                     rseed=seed, pfunc=base3d)
    end = time.time()
    logger.info("3D Hawkes simulation with an exponential kernel with quadratic base intensity of length "
                "{} lasted for {} s".format(len(hawkes[0]) + len(hawkes[1]) + len(hawkes[2]),
                                                    end - begin))

    # POWER-LAW WITH CONSTANT BASE INTENSITY
    mu = np.array([0.15, 0.15, 0.15], dtype=float)
    rho = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=float)
    m = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=float)
    M = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=int)
    epsilon = np.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], dtype=float)
    n = np.array([[0.18, 0.18, 0.18], [0.18, 0.18, 0.18], [0.18, 0.18, 0.18]], dtype=float)
    begin = time.time()
    hawkes = pyhawkes.sim_power_hawkes(mu, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                       rseed=seed)
    end = time.time()
    logger.info("3D Hawkes simulation with a power-law kernel with constant base intensity of length {} "
                "lasted for {} s".format(len(hawkes[0]) + len(hawkes[1]) + len(hawkes[2]),
                                         end - begin))

    # POWER-LAW WITH QUADRATIC BASE INTENSITY
    begin = time.time()
    hawkes = pyhawkes.sim_gen_hawkes(mu3d, rho, m, M, epsilon, n, length=T, max=max_jumps,
                                         rseed=seed, pfunc=base3d)
    end = time.time()
    logger.info("3D Hawkes simulation with a power-law kernel with quadratic base intensity of length "
                "{} lasted for {} s".format(len(hawkes[0]) + len(hawkes[1]) + len(hawkes[2]),
                                                end - begin))