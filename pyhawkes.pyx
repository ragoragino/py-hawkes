import numpy as np
cimport numpy as np
cimport pyhawkes
import cython
import ctypes
from libc.time cimport time, time_t



"""
DEFINITIONS
"""


DTYPE = np.double
ITYPE = np.int
ctypedef np.double_t DTYPE_t
ctypedef np.int_t ITYPE_t
ctypedef double (*cfunction) (int loc, double t)
ctypedef double (*cfunction_int) (int loc, double t1, double t2)

cdef object f
cdef object g
cdef object h
cdef int climit = 1000000
cdef unsigned int cseed = time(NULL)

cdef double pfunc_call(int loc, double t):
    global f
    return (<object>f)(h, loc, t)

cdef double pfunc_call_int(int loc, double t1, double t2):
    global g
    return (<object>g)(h, loc, t1, t2)


cdef control_exp(mu, alpha, beta):
    if not (mu.shape[0], mu.shape[0]) == alpha.shape == beta.shape:
        raise IncompatibleShapeError()

    if np.any(mu <= 0) or np.any(alpha <= 0) or np.any(beta <= 0):
        raise ParametersConstraintError()


cdef control_pl(mu, rho, m, M, epsilon, n):
    if not (mu.shape[0], mu.shape[0]) == rho.shape == m.shape \
    == M.shape == epsilon.shape == n.shape:
        raise IncompatibleShapeError()

    if np.any(mu <= 0) or np.any(rho <= 0) or np.any(m <= 0) or \
    np.any(M <= 0) or np.any(epsilon <= 0) or np.any(n <= 0):
        raise ParametersConstraintError()


cdef control_gen(mu, rho, m, M, epsilon, n):
    if not rho.shape == m.shape == M.shape == epsilon.shape == n.shape:
        raise IncompatibleShapeError()

    if np.any(rho <= 0) or np.any(m <= 0) or np.any(M <= 0) \
    or np.any(epsilon <= 0) or np.any(n <= 0):
        raise ParametersConstraintError()


cdef class IncompatibleShapeError(Exception):
    def __init__(self):
        super().__init__("Input parameters have incompatible shapes!")


cdef class ParametersConstraintError(Exception):
    def __init__(self):
        super().__init__("Parameters are not positive!")


"""
EXPONENTIAL HAWKES
"""


@cython.boundscheck(False)
@cython.wraparound(False)
def sim_exp_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                   np.ndarray[dtype=double, ndim=2, mode="c"] alpha not None,
                   np.ndarray[dtype=double, ndim=2, mode="c"] beta not None,
                   length, max = climit, rseed = cseed):
    control_exp(mu, alpha, beta)

    cdef:
        int limit = max
        int seed = rseed
        int dim = mu.shape[0]
        double clength = length
        np.ndarray[double, ndim = 2, mode = 'c'] events = \
            np.ascontiguousarray(np.zeros((dim, limit), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    pyhawkes.simulate_exponential_hawkes(&mu[0], &alpha[0,0], &beta[0,0],
        clength, dim, limit, &events[0,0], &tracker[0], seed)
    hawkes = [events[i, :tracker[i]] for i in range(dim)]

    return hawkes


@cython.boundscheck(False)
@cython.wraparound(False)
def comp_exp_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                    np.ndarray[dtype=double, ndim=2, mode="c"] alpha not None,
                    np.ndarray[dtype=double, ndim=2, mode="c"] beta not None,
                    events, length):
    control_exp(mu, alpha, beta)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length

    for i in range(dim):
        max_len = max(max_len, len(events[i]))

    cdef:
        np.ndarray[double, ndim = 2, mode = 'c'] compensator_series = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    for i in range(dim):
        pyhawkes.compensator_exponential_hawkes(&mu[0], &alpha[0,0],
            &beta[0,0], clength, i, dim, max_len, &events_new[0,0],
            &tracker[0], &compensator_series[0,0] + i * max_len)
    compensator = [compensator_series[i, :tracker[i]] for i in range(dim)]

    return compensator


@cython.boundscheck(False)
@cython.wraparound(False)
def lik_exp_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                   np.ndarray[dtype=double, ndim=2, mode="c"] alpha not None,
                   np.ndarray[dtype=double, ndim=2, mode="c"] beta not None,
                   events, length):
    control_exp(mu, alpha, beta)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        double likelihood = 0

    for i in range(dim):
        max_len = max(max_len, len(events[i]) + 1)
        # +1 because of need for one more space for compensator routine

    cdef:
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    for i in range(dim):
        likelihood += pyhawkes.loglikelihood_exponential_hawkes(&mu[0],
            &alpha[0,0], &beta[0,0], clength, i, dim, max_len,
            &events_new[0,0], &tracker[0])

    return likelihood


@cython.boundscheck(False)
@cython.wraparound(False)
def lik_exp_hawkes_optim(np.ndarray[dtype=double, ndim=1, mode="c"] parameters
                         not None, events, length):
    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        double likelihood = 0
        int step = dim + ipow(dim, 2)

    for i in range(dim):
        max_len = max(max_len, len(events[i]) + 1)
        # +1 because of need for one more space for compensator routine

    cdef:
        np.ndarray[double, ndim = 2, mode = 'c'] events_new \
            = np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker \
            = np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    for i in range(dim):
        likelihood += pyhawkes.loglikelihood_exponential_hawkes(&parameters[0],
            &parameters[0] + dim, &parameters[0] + step, clength, i, dim,
            max_len, &events_new[0,0], &tracker[0])

    return likelihood


@cython.boundscheck(False)
@cython.wraparound(False)
def plot_exp_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] alpha not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] beta not None,
                     events, length, begin, end, grid):
    control_exp(mu, alpha, beta)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        int cend = end

    for i in range(dim):
        max_len = max(max_len, len(events[i]))

    cdef:
        int points = (int)(end - begin) / grid
        int plt_limit = max_len + points + 1
        np.ndarray[double, ndim = 2, mode = 'c'] plt_events = \
            np.ascontiguousarray(np.zeros((dim, plt_limit), dtype = DTYPE))
        np.ndarray[double, ndim = 2, mode = 'c'] plt_x = \
            np.ascontiguousarray(np.zeros((dim, plt_limit), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] plt_end_tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    pyhawkes.plt_exponential_hawkes(&mu[0], &alpha[0,0], &beta[0,0], begin,
        end, grid, dim, max_len, &events_new[0,0], &tracker[0], &plt_events[0,0],
        &plt_x[0,0], &plt_end_tracker[0])
    plot_hawkes = [plt_events[i, :plt_end_tracker[i]] for i in range(dim)], \
        [plt_x[i, :plt_end_tracker[i]] for i in range(dim)]

    return plot_hawkes


"""
POWER-LAW HAWKES
"""


@cython.boundscheck(False)
@cython.wraparound(False)
def sim_power_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] rho not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                     np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] epsilon not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] n not None,
                     length, max = climit, rseed = cseed):
    control_pl(mu, rho, m, M, epsilon, n)

    cdef:
        int limit = max
        dim = mu.shape[0]
        int seed = rseed
        double clength = length
        np.ndarray[double, ndim = 2, mode = 'c'] events = \
            np.ascontiguousarray(np.zeros((dim, limit), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    pyhawkes.simulate_power_hawkes(&mu[0], &rho[0,0], &m[0,0], &M[0,0],
        &epsilon[0,0], &n[0,0], clength, dim, limit, &events[0,0], &tracker[0],
        seed)
    hawkes = [events[i, :tracker[i]] for i in range(dim)]

    return hawkes


@cython.boundscheck(False)
@cython.wraparound(False)
def comp_power_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                      np.ndarray[dtype=double, ndim=2, mode="c"] rho not None,
                      np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                      np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                      np.ndarray[dtype=double, ndim=2, mode="c"] epsilon not None,
                      np.ndarray[dtype=double, ndim=2, mode="c"] n not None,
                      events, length):
    control_pl(mu, rho, m, M, epsilon, n)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length

    for i in range(dim):
        max_len = max(max_len, len(events[i]))

    cdef:
        np.ndarray[double, ndim = 2, mode = 'c'] compensator_series = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))
        double * Z = NULL
        double * alpha = NULL

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    for i in range(dim):
        pyhawkes.compensator_power_hawkes(&mu[0], &rho[0,0], &m[0,0],
            &M[0,0], &epsilon[0,0], &n[0,0], clength, i, dim, max_len,
            &events_new[0,0], &tracker[0], &compensator_series[0,0] + i * max_len,
            Z, alpha)
    compensator = [compensator_series[i, :tracker[i]] for i in range(dim)]

    return compensator


@cython.boundscheck(False)
@cython.wraparound(False)
def lik_power_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] rho not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                     np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] epsilon not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] n not None,
                     events, length):
    control_pl(mu, rho, m, M, epsilon, n)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        double likelihood = 0

    for i in range(dim):
        max_len = max(max_len, len(events[i]) + 1)
        # +1 because of need for one more space for compensator routine

    cdef:
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    for i in range(dim):
        likelihood += pyhawkes.loglikelihood_power_hawkes(&mu[0], &rho[0,0],
            &m[0,0], &M[0,0], &epsilon[0,0], &n[0,0], clength, i, dim,
            max_len, &events_new[0,0], &tracker[0])

    return likelihood


@cython.boundscheck(False)
@cython.wraparound(False)
def lik_power_hawkes_optim(np.ndarray[dtype=double, ndim=1, mode="c"] parameters
                           not None, np.ndarray[dtype=double, ndim=2, mode="c"] m
                           not None, np.ndarray[dtype=int, ndim=2, mode="c"] M
                           not None, events, length):

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        double likelihood = 0

    for i in range(dim):
        max_len = max(max_len, len(events[i]) + 1)
        # +1 because of need for one more space for compensator routine

    cdef:
        int step = ipow(dim, 2)
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    for i in range(dim):
        likelihood += pyhawkes.loglikelihood_power_hawkes(&parameters[0],
            &parameters[0] + dim, &m[0,0], &M[0,0], &parameters[0] + dim + step,
            &parameters[0] + dim + 2 * step, clength, i, dim, max_len,
            &events_new[0,0], &tracker[0])

    return likelihood


@cython.boundscheck(False)
@cython.wraparound(False)
def plot_power_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                      np.ndarray[dtype=double, ndim=2, mode="c"] rho not None,
                      np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                      np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                      np.ndarray[dtype=double, ndim=2, mode="c"] epsilon not None,
                      np.ndarray[dtype=double, ndim=2, mode="c"] n not None,
                      events, length, begin, end, grid):
    control_pl(mu, rho, m, M, epsilon, n)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        int cend = end

    for i in range(dim):
        max_len = max(max_len, len(events[i]))

    cdef:
        int points = (int)(end - begin) / grid
        int plt_limit = max_len + points + 1
        np.ndarray[double, ndim = 2, mode = 'c'] plt_events = \
            np.ascontiguousarray(np.zeros((dim, plt_limit), dtype = DTYPE))
        np.ndarray[double, ndim = 2, mode = 'c'] plt_x = \
            np.ascontiguousarray(np.zeros((dim, plt_limit), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] plt_end_tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = \
            np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = \
            np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    pyhawkes.plt_power_hawkes(&mu[0], &rho[0,0], &m[0,0], &M[0,0],
        &epsilon[0,0], &n[0,0], begin, end, grid, dim, max_len,
        &events_new[0,0], &tracker[0], &plt_events[0,0], &plt_x[0,0],
        &plt_end_tracker[0])
    plot_hawkes = [plt_events[i, :plt_end_tracker[i]] for i in range(dim)], \
        [plt_x[i, :plt_end_tracker[i]] for i in range(dim)]

    return plot_hawkes


"""
GENERAL HAWKES
"""


@cython.boundscheck(False)
@cython.wraparound(False)
def sim_gen_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] rho not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                     np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] epsilon not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] n not None,
                     length, pfunc, max = climit, rseed = cseed):
    control_gen(mu, rho, m, M, epsilon, n)

    cdef:
        int limit = max
        int seed = rseed
        dim = rho.shape[0]
        double clength = length
        np.ndarray[double, ndim = 2, mode = 'c'] events = np.ascontiguousarray(np.zeros((dim, limit), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    global f
    f = pfunc
    global h
    h = mu

    pyhawkes.simulate_general_hawkes(&mu[0], &rho[0,0], &m[0,0], &M[0,0], &epsilon[0,0], &n[0,0], clength, dim, limit, &events[0,0],
        &tracker[0], seed, <cfunction> pfunc_call)
    hawkes = [events[i, :tracker[i]] for i in range(dim)]

    return hawkes


@cython.boundscheck(False)
@cython.wraparound(False)
def comp_gen_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] rho not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                     np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] epsilon not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] n not None,
                     events, length, pfunc_int):
    control_gen(mu, rho, m, M, epsilon, n)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length

    for i in range(dim):
        max_len = max(max_len, len(events[i]))

    cdef:
        np.ndarray[double, ndim = 2, mode = 'c'] compensator_series = np.ascontiguousarray(np.zeros((dim, max_len), dtype = DTYPE))
        double * Z = NULL
        double * alpha = NULL
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = np.ascontiguousarray(np.zeros((dim, max_len),
        dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    global g
    g = pfunc_int
    global h
    h = mu

    for i in range(dim):
        pyhawkes.compensator_general_hawkes(&mu[0], &rho[0,0], &m[0,0], &M[0,0], &epsilon[0,0], &n[0,0], clength, i, dim, max_len,
        &events_new[0,0], &tracker[0], &compensator_series[0,0] + i * max_len, Z, alpha, <cfunction_int> pfunc_call_int)
    compensator = [compensator_series[i, :tracker[i]] for i in range(dim)]

    return compensator


@cython.boundscheck(False)
@cython.wraparound(False)
def lik_gen_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] rho not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                     np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] epsilon not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] n not None,
                     events, length, pfunc, pfunc_int):
    control_gen(mu, rho, m, M, epsilon, n)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        double likelihood = 0

    for i in range(dim):
        max_len = max(max_len, len(events[i]) + 1) # +1 because of need for one more space for compensator routine

    cdef:
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = np.ascontiguousarray(np.zeros((dim, max_len),
        dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    global f
    f = pfunc
    global g
    g = pfunc_int
    global h
    h = mu

    for i in range(dim):
        likelihood += pyhawkes.loglikelihood_general_hawkes(&mu[0], &rho[0,0], &m[0,0], &M[0,0], &epsilon[0,0], &n[0,0], clength,
                                                i, dim, max_len, &events_new[0,0], &tracker[0], <cfunction> pfunc_call,
                                                <cfunction_int> pfunc_call_int)

    return likelihood


@cython.boundscheck(False)
@cython.wraparound(False)
def lik_gen_hawkes_optim(np.ndarray[dtype=double, ndim=1, mode="c"] parameters not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                     np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                     events, length, base_length, pfunc, pfunc_int):
    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        int c_base_l = base_length
        double likelihood = 0
        int step = ipow(dim, 2)

    for i in range(dim):
        max_len = max(max_len, len(events[i]) + 1) # +1 because of need for one more space for compensator routine

    cdef:
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = np.ascontiguousarray(np.zeros((dim, max_len),
        dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    global f
    f = pfunc
    global g
    g = pfunc_int
    global h
    h = parameters[:c_base_l]

    for i in range(dim):
        likelihood += pyhawkes.loglikelihood_general_hawkes(&parameters[0], &parameters[0] + c_base_l, &m[0,0], &M[0,0], &parameters[0] + c_base_l + step,
                                                &parameters[0] + c_base_l + 2 * step, clength, i, dim, max_len, &events_new[0,0], &tracker[0],
                                                <cfunction> pfunc_call, <cfunction_int> pfunc_call_int)

    return likelihood


@cython.boundscheck(False)
@cython.wraparound(False)
def plot_gen_hawkes(np.ndarray[dtype=double, ndim=1, mode="c"] mu not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] rho not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] m not None,
                     np.ndarray[dtype=int, ndim=2, mode="c"] M not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] epsilon not None,
                     np.ndarray[dtype=double, ndim=2, mode="c"] n not None,
                     events, length, begin, end, grid, pfunc):
    control_gen(mu, rho, m, M, epsilon, n)

    cdef:
        int i
        int dim = len(events)
        int max_len = 0
        double clength = length
        int cbegin = begin
        int cend = end

    for i in range(dim):
        max_len = max(max_len, len(events[i]))

    cdef:
        double cgrid = grid
        int points = (int)(cend - cbegin) / grid
        int plt_limit = max_len + points + 1
        np.ndarray[double, ndim = 2, mode = 'c'] plt_events = np.ascontiguousarray(np.zeros((dim, plt_limit), dtype = DTYPE))
        np.ndarray[double, ndim = 2, mode = 'c'] plt_x = np.ascontiguousarray(np.zeros((dim, plt_limit), dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] plt_end_tracker = np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))
        np.ndarray[double, ndim = 2, mode = 'c'] events_new = np.ascontiguousarray(np.zeros((dim, max_len),
        dtype = DTYPE))
        np.ndarray[int, ndim = 1, mode = 'c'] tracker = np.ascontiguousarray(np.zeros(dim, dtype = ITYPE))

    for i in range(dim):
        tracker[i] = len(events[i])
        events_new[i, :tracker[i]] = events[i]

    global f
    f = pfunc
    global h
    h = mu

    pyhawkes.plt_general_hawkes(&mu[0], &rho[0,0], &m[0,0], &M[0,0], &epsilon[0,0], &n[0,0], cbegin, cend, cgrid, dim, max_len, &events_new[0,0], &tracker[0],
                            &plt_events[0,0], &plt_x[0,0], &plt_end_tracker[0], <cfunction> pfunc_call)
    plot_hawkes = [plt_events[i, :plt_end_tracker[i]] for i in range(dim)], [plt_x[i, :plt_end_tracker[i]] for i in range(dim)]

    return plot_hawkes

