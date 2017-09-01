"""
DECLARATIONS
"""


cdef extern from "exp_hawkes.h":
    void simulate_exponential_hawkes(const double mu[], const double alpha[],
        const double beta[], double T, int dim, int max,
	    double * process_list, int * process_track, unsigned int seed) except +

    void compensator_exponential_hawkes(const double mu[], const double alpha[],
        const double beta[], double T, int pos, int dim, int max, double *
        process_list, int * process_track, double * compensator_series)

    double loglikelihood_exponential_hawkes(double mu[], double alpha[],
        double beta[], double T, int pos, int dim, int max, double *
        process_list, int * process_track)

    void plt_exponential_hawkes(const double mu[], const double alpha[],
        const double beta[], int begin, int end, double grid, int dim, int max,
        double * process_list, int * process_track, double * plt_intensity,
        double * plt_list, int * plt_end)


cdef extern from "power_hawkes.h":
    void simulate_power_hawkes(const double mu[], const double rho[],
        const double m[], const int M[], const double epsilon[],
        const double n[], double T, int dim, int max, double *
        process_list, int * process_track, unsigned int seed) except +

    void compensator_power_hawkes(const double mu[], const double rho[],
        const double m[], const int M[], const double epsilon[], const double n[],
	    double T, int pos, int dim, int max, double * process_list, int *
	    process_track, double * compensator_series, double * Z,
	    double * alpha)

    double loglikelihood_power_hawkes(double mu[], double rho[], double m[],
        int M[], double epsilon[], double n[], double T, int pos,
	    int dim, int max, double * process_list, int * process_track)

    void plt_power_hawkes(const double mu[], const double rho[], const double m[],
        const int M[], const double epsilon[], const double n[],
	    int begin, int end, double grid, int dim, int max, double * process_list,
	    int * process_track, double * lt_intensity, double * plt_list,
	    int * plt_end)


cdef extern from "general_hawkes.h":
    void simulate_general_hawkes(const double mu[], const double rho[], const double m[],
        const int M[], const double epsilon[], const double n[], double T, int dim,
        int max, double * process_list, int * process_track, unsigned int seed,
        double (*func)(int loc, double t)) except +

    void compensator_general_hawkes(const double mu[], const double rho[], const double m[],
        const int M[], const double epsilon[], const double n[], double T, int pos, int dim,
        int max, double * process_list, int * process_track, double * compensator_series,
        double * Z, double * alpha, double (*func)(int loc, double arg1, double arg2))

    double loglikelihood_general_hawkes(double mu[], double rho[], double m[], int M[],
        double epsilon[], double n[], double T, int pos, int dim, int max, double *
        process_list, int * process_track, double (*func)(int loc, double t),
        double (*func_int)(int loc, double arg1, double arg2))

    void plt_general_hawkes(const double mu[], const double rho[], const double m[],
        const int M[], const double epsilon[], const double n[], int begin, int end,
        double grid, int dim, int max, double * process_list, int * process_track,
        double * plt_intensity, double * plt_list, int * plt_end,
        double (*func)(int loc, double t))


cdef extern from "additional_functions.h":
    int ipow(int x, int y)

