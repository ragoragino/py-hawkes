#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stdexcept>

/*
Function that checks for stationarity of a d-dimensional Hawkes process 
with an exponential kernel.

@param
alpha: an array of length d x d containing instantaneous intensity values. D rows of the
	alpha matrix are concatenated to a single d x d array
beta: an array of length d x d containing decay parameters. D rows of the
	beta matrix are concatenated to a single d x d array
dim: d

@return
bool: indicator whether the process is stationary or not
*/

static bool stationaritycheck(const double alpha[], const double beta[], int dim); 


/*
Function that checks for stationarity of a d-dimensional Hawkes process
with an approximated power-law kernel.

@param
n: an array of length d x d containing values for branching ratios. D rows of the
	n matrix are concatenated to a single d x d array
dim: d

@return
bool: indicator whether the process is stationary or not
*/

static bool stationaritycheck(const double n[], int dim); 


/*
Function that excludes numbers 0 or 1 from the pseudo-random
number generator picks.

@param

@return
double: random number from (0, 1)
*/

static double random_check(); 


/*
Integer equivalent of pow function

@param
a: base
b: exponent

@return
int: integer equivalent of pow(a, b)
*/

static int ipow(int a, int b);


struct holder
{
	int index;
	double stamp;
};


class StationarityError : public std::runtime_error {
public:
	StationarityError(std::string message)
		: std::runtime_error(message) { }
};


class LimitError : public std::runtime_error {
public:
	LimitError(std::string message)
		: std::runtime_error(message) { }
};


static int ipow(int base, int exp)
{
    int result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}


static bool stationaritycheck(const double n[], int d)
{
	bool indicator = false;
	Eigen::MatrixXd stat_mat(d, d);
	for (int i = 0; i != d; ++i)
	{
		for (int j = 0; j != d; ++j)
			stat_mat(i, j) = n[i * d + j];
	}
	Eigen::EigenSolver<Eigen::MatrixXd> eigsol;
	eigsol.compute(stat_mat, false);
	Eigen::VectorXd eigsol_real = eigsol.eigenvalues().real();
	for (int i = 0; i != d; ++i)
	{
		if (fabs(eigsol_real(i)) >= 1)
			indicator = true;
	}

	return indicator;
}


static bool stationaritycheck(const double x[], const double y[], int d)
{
	bool indicator = false;
	Eigen::MatrixXd stat_mat(d, d);
	for (int i = 0; i != d; ++i)
	{
		for (int j = 0; j != d; ++j)
		{
			stat_mat(i, j) = x[i * d + j] / y[i * d + j];
		}
	}
	Eigen::EigenSolver<Eigen::MatrixXd> eigsol;
	eigsol.compute(stat_mat, false);
	Eigen::VectorXd eigsol_real = eigsol.eigenvalues().real();
	for (int i = 0; i != d; ++i)
	{
		if (fabs(eigsol_real(i)) >= 1)
			indicator = true;
	}

	return indicator;
}


static double random_check() {
	double random = rand();
	if (random == 0 || random == RAND_MAX)
		return random_check();
	else
		return random;
}
