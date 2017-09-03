#include "additional_functions.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


int ipow(int base, int exp)
{
	int result = 1;
	while (exp)
	{
		if (exp & 1)
		{
			result *= base;
		}
		exp >>= 1;
		base *= base;
	}

	return result;
}


bool stationaritycheck(const double n[], int d)
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
		{
			indicator = true;
		}
	}

	return indicator;
}


bool stationaritycheck(const double x[], const double y[], int d)
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
		{
			indicator = true;
		}
	}

	return indicator;
}


double random_check() {
	double random = rand();
	if (random == 0 || random == RAND_MAX)
	{
		return random_check();
	}
	else
	{
		return random;
	}
}
