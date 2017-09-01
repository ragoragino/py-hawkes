/*
Function that simulates jumps for a d-dimensional Hawkes process
with an exponential kernel by Ogata's thinning method

@param
mu: an array of length d containing the base intensity values
alpha: an array of length d x d containing instantaneous intensity values. D rows of the
	alpha matrix are concatenated to a single d x d array
beta: an array of length d x d containing decay parameters. D rows of the
	beta matrix are concatenated to a single d x d array
T: time horizon of the simulation
dim: d
max: maximum allowed length of jumps in a single dimension
process_list: an empty array of length dim x max 
process_track: an empty array of length dim
seed: seed for pseudo-random number generator

@return
process_list: updated with simulated values
process_track: updated with indices of last simulated values in each dimension 
*/

void simulate_exponential_hawkes(const double mu[], const double alpha[], const double beta[], double T, 
	int dim, int max, double * process_list, int * process_track, unsigned int seed);


/*
Function that computes the compensator of a single dimension of a 
d-dimensional Hawkes process with an exponential kernel

@param
mu: an array of length d containing the base intensity values
alpha: an array of length d x d containing instantaneous intensity values. D rows of the
	alpha matrix are concatenated to a single d x d array
beta: an array of length d x d containing decay parameters. D rows of the
	beta matrix are concatenated to a single d x d array
T: time horizon of the simulation
pos: current index of the process
dim: d
max: maximum length of jumps in a single dimension 
process_list: an array containing simulated values of the process
process_track: an array containing last indices of the simulated values in 
	each dimension
compensator_series: an empty array of dimension max x dim

@return
compensator_series: updated with values of the compensator series for dimension pos
*/

void compensator_exponential_hawkes(const double mu[], const double alpha[], const double beta[], double T, 
	int pos, int dim, int max, double * process_list, int * process_track, double * compensator_series);


/*
Function that computes negative log-likehlood of a single dimension of a
d-dimensional Hawkes process with an exponential kernel

@param
mu: an array of length d containing the base intensity values
alpha: an array of length d x d containing instantaneous intensity values. D rows of the
	alpha matrix are concatenated to a single d x d array
beta: an array of length d x d containing decay parameters. D rows of the
	beta matrix are concatenated to a single d x d array
T: time horizon of the simulation
pos: current index of the process
dim: d
max: maximum length of jumps in a single dimension +1
process_list: an array containing simulated values of the process
process_track: an array containing last indices of the simulated values in
	each dimension

@return
double: negative log-likelihood
*/

double loglikelihood_exponential_hawkes(double mu[], double alpha[], double beta[], double T, int pos,
	int dim, int max, double * process_list, int * process_track);


/*
Function that computes values of the conditional intensity function \lambda
of a d-dimensional Hawkes process in intervals of a pre-specified gap 
together with its values at jump times.

@param
mu: an array of length d containing the base intensity values
alpha: an array of length d x d containing instantaneous intensity values. D rows of the
	alpha matrix are concatenated to a single d x d array
beta: an array of length d x d containing decay parameters. D rows of the
	beta matrix are concatenated to a single d x d array
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

@return
plt_intensity: updated by values of \lambda
plt_list: updated by values of timestamps corresponding to \lambda in plt_intensity
plt_end: updated by indices of last values in plt_intensity and plt_list
	for each dimension
*/

void plt_exponential_hawkes(const double mu[], const double alpha[], const double beta[], int begin, int end, 
	double grid, int dim, int max, double * process_list, int * process_track, double * plt_intensity, 
	double * plt_list, int * plt_end);
