/*
Function that simulates jumps for a d-dimensional Hawkes process
with an approximated power-law kernel and arbitrary base intensity
by Ogata's thinning method

@param
mu: an array of length d containing the base intensity values
rho: an array of length d x d containing parameters of scale of approximation. D rows of the
	rho matrix are concatenated to a single d x d array
m: an array of length d x d containing parameters of precision of approximation. D rows of the
	m matrix are concatenated to a single d x d array
M: an array of length d x d containing parameters of range of approximation. D rows of the
	M matrix are concatenated to a single d x d array
epsilon: an array of length d x d containing power-law parameters. D rows of the
	epsilon matrix are concatenated to a single d x d array
n: an array of length d x d containing values for branching ratios. D rows of the
	n matrix are concatenated to a single d x d array
T: time horizon of the simulation
dim: d
max: maximum allowed length of jumps in a single dimension
process_list: an empty array of length dim x max
process_track: an empty array of length dim
seed: seed for pseudo-random number generator
func: function pointer to a base intensity function. The function takes location
	for the mu array and current time position in the simulation as parameters.

@return
process_list: updated with simulated values
process_track: updated with indices of last simulated values in each dimension
*/

void simulate_general_hawkes(const double mu[], const double rho[], const double m[], 
	const int M[], const double epsilon[], const double n[], double T, int dim, int max, 
	double * process_list, int * process_track, unsigned int seed, 
	double(*func)(int loc, double t));


/*
Function that computes the compensator of a single dimension of a
d-dimensional Hawkes process with an approximated power-law kernel
and arbitrary base intensity

@param
mu: an array of length d containing the base intensity values
rho: an array of length d x d containing parameters of scale of approximation. D rows of the
	rho matrix are concatenated to a single d x d array
m: an array of length d x d containing parameters of precision of approximation. D rows of the
	m matrix are concatenated to a single d x d array
M: an array of length d x d containing parameters of range of approximation. D rows of the
	M matrix are concatenated to a single d x d array
epsilon: an array of length d x d containing power-law parameters. D rows of the
	epsilon matrix are concatenated to a single d x d array
n: an array of length d x d containing values for branching ratios. D rows of the
	n matrix are concatenated to a single d x d array
T: time horizon of the simulation
pos: current index of the process
dim: d
max: maximum length of jumps in a single dimension
process_list: an array containing simulated values of the process
process_track: an array containing last indices of the simulated values in
each dimension
compensator_series: an empty array of dimension max x dim
Z: an array of length max(M) x dim. Z is chosen such that the
	absolute value of the spectral radius of the integral of the kernel is lower
	than 1. Row Z for a given dimension pos should be passed in. The array should be valid
	only when passed from a log-likelihood function, otherwise should be set to NULL.
alpha: an array containing rho * pow(m, i) for i = 0^{M_{pos}-1}values for a given dimension pos.
	Should be valid only when passed from a log-likelihood function, otherwise
	should be set to NULL.
func_int: function pointer to an integral of a base intensity function. The function takes location
	for the mu array and two closest time positions in the computation as parameters.

@return
compensator_series: updated with values of the compensator series for dimension pos.
*/

void compensator_general_hawkes(const double mu[], const double rho[], const double m[], 
	const int M[], const double epsilon[], const double n[], double T, int pos, int dim,
	int max, double * process_list, int * process_track, double * compensator_series, 
	double * Z, double * alpha,	double(*func)(int loc, double arg1, double arg2));


/*
Function that computes negative log-likehlood of a single dimension of a
d-dimensional Hawkes process with an approximated power-law kernel and
arbitrary base intensity

@param
mu: an array of length d containing the base intensity values
rho: an array of length d x d containing parameters of scale of approximation. D rows of the
	rho matrix are concatenated to a single d x d array
m: an array of length d x d containing parameters of precision of approximation. D rows of the
	m matrix are concatenated to a single d x d array
M: an array of length d x d containing parameters of range of approximation. D rows of the
	M matrix are concatenated to a single d x d array
epsilon: an array of length d x d containing power-law parameters. D rows of the
	epsilon matrix are concatenated to a single d x d array
n: an array of length d x d containing values for branching ratios. D rows of the
	n matrix are concatenated to a single d x d arrayT: time horizon of the simulation
pos: current index of the process
dim: d
max: maximum length of jumps in a single dimension +1
process_list: an array containing simulated values of the process
process_track: an array containing last indices of the simulated values in
	each dimension
func: function pointer to a base intensity function. The function takes location
	for the mu array and current time position in the computation as parameters.
func_int: function pointer to an integral of a base intensity function. The function takes location
	for the mu array and two closest time positions in the computation as parameters. 


@return
double: negative log-likelihood
*/

double loglikelihood_general_hawkes(double mu[], double rho[], double m[], int M[], 
	double epsilon[], double n[], double T, int pos, int dim, int max,
	double * process_list, int * process_track, double(*func)(int loc, double t), 
	double(*func_int)(int loc, double arg1, double arg2));


/*
Function that computes values of the conditional intensity function \lambda
for a d-dimensional Hawkes process with an approximated power-law kernel 
and arbitrary base intensity in intervals of a pre-specified gap
together with its values at jump times.

@param
mu: an array of length d containing the base intensity values
rho: an array of length d x d containing parameters of scale of approximation. D rows of the
	rho matrix are concatenated to a single d x d array
m: an array of length d x d containing parameters of precision of approximation. D rows of the
	m matrix are concatenated to a single d x d array
M: an array of length d x d containing parameters of range of approximation. D rows of the
	M matrix are concatenated to a single d x d array
epsilon: an array of length d x d containing power-law parameters. D rows of the
	epsilon matrix are concatenated to a single d x d array
n: an array of length d x d containing values for branching ratios. D rows of the
	n matrix are concatenated to a single d x d array
begin: the beginning time for required values of \lambda, can be in [0, T]
end: the end time for required values of \lambda, can be in [0, T]
grid: specified grid at which the \lambda values will be computed, e.g. each 0.1
dim: d
max: maximum length of jumps in a single dimension
process_list: an array containing simulated values of the process
process_track: an array containing last indices of the simulated values in
	each dimension
plt_intensity: empty array of length dim x (max + (int)(end - begin) / grid + 1)
plt_list: empty array of length dim x (max + (int)(end - begin) / grid + 1)
plt_end: empty array of length dim
func: function pointer to a base intensity function. The function takes location
	for the mu array and current time position in the computation as parameters.

@return
plt_intensity: updated by values of \lambda
plt_list: updated by values of timestamps corresponding to \lambda in plt_intensity
plt_end: updated by indices of last values in plt_intensity and plt_list
	for each dimension
*/

void plt_general_hawkes(const double mu[], const double rho[], const double m[], 
	const int M[], const double epsilon[], const double n[], int begin, int end, 
	double grid, int dim, int max, double * process_list, int * process_track, 
	double * plt_intensity, double * plt_list, int * plt_end, 
	double(*func)(int loc, double t));