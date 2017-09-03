#include "power_hawkes.h"
#include "additional_functions.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <vector> // Enclose struct holder in the vector in plot_exponential_hawkes
#include <algorithm> // use std::sort for ordering elements in std::vector<holder> in plot_power_hawkes 
#include <new> // std::bad_alloc

void simulate_power_hawkes(const double mu[], const double rho[], const double m[], const int M[], const double epsilon[], 
	const double n[], double T, int dim, int max, double * process_list, int * process_track, unsigned int seed)
{
	bool indicator = stationaritycheck(n, dim);
	if (indicator)
	{
		throw StationarityError("Non-stationary power-law kernel!");
	}

	srand(seed);
	double s{ 0.0 };
	int l{ 0 };
	double upper_lambda{ 0.0 };
	double candidate_unif{ 0.0 };
	double candidate{ 0.0 };
	double acceptance{ 0.0 };
	double random{ 0.0 };
	double random_acc{ 0.0 };
	double other_process_rec{ 0.0 };
	double * individual_lambda = (double*)calloc(dim, sizeof(double));
	double * cumulative_lambda = (double*)calloc(dim, sizeof(double));
	int * other_process_track = (int*)calloc(dim * dim, sizeof(int));
	double * Z = (double*)calloc(dim * dim, sizeof(double));

	// Creating a matrix of alpha_i = rho * pow(m, i) and calculating Z values
	int max_M{ 0 };
	for (int i = 0; i != dim * dim; ++i)
		max_M = max_M >= M[i] ? max_M : M[i];
	double * alpha = (double*)calloc(dim * max_M * dim, sizeof(double));
	double * recursive_array = (double*)calloc(dim * max_M * dim, sizeof(double));
	double * other_process = (double*)calloc(dim * max_M * dim, sizeof(double));
	for (int i = 0; i != dim; ++i)
	{
		for (int j = 0; j != dim; ++j)
		{
			for (int k = 0; k != M[i * dim + j]; ++k)
			{
				alpha[max_M * (i * dim + j) + k] = rho[i * dim + j] * pow(m[i * dim + j], k);
				Z[i * dim + j] += pow(alpha[max_M * (i * dim + j) + k], -epsilon[i * dim + j]);
			}
		}
	}

	while (s < T)
	{
		upper_lambda = 0.0;
		for (int i = 0; i != dim; ++i)
		{
			if (process_track[i] == max)
			{
				free(other_process);
				free(recursive_array);
				free(individual_lambda);
				free(other_process_track);
				free(cumulative_lambda);
				free(Z);
				free(alpha);
				throw LimitError("Given limit of jumps exceeded!");
			}

			// Calculating the sum of lambda(s)
			upper_lambda += mu[i];
			for (int j = 0; j != dim; ++j)
			{
				if (process_track[j] < 1)
				{
					continue;
				}
				if (process_track[i] >= 1)
				{
					for (int k = 0; k != M[i * dim + j]; ++k)
					{
						other_process[max_M * (i * dim + j) + k] = 0.0;
						for (int l = max * j + other_process_track[i * dim + j]; l != max * j + process_track[j]; ++l)
							other_process[max_M * (i * dim + j) + k] += exp(-(s - process_list[l]) / alpha[max_M * (i * dim + j) + k]);
						upper_lambda += (n[i * dim + j] / Z[i * dim + j]) * pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j]))
							* (exp(-(s - process_list[max * i + process_track[i] - 1]) / alpha[max_M * (i * dim + j) + k]) *
								recursive_array[max_M * (i * dim + j) + k] + other_process[max_M * (i * dim + j) + k]);
					}
				}
				else
				{
					for (int k = 0; k != M[i * dim + j]; ++k)
					{
						for (int l = max * j; l != max * j + process_track[j]; ++l)
						{
							upper_lambda += (n[i * dim + j] / Z[i * dim + j]) * pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j])) *
								exp(-(s - process_list[l]) / alpha[max_M * (i * dim + j) + k]);
						}
					}
				}
			}
		}

		random = random_check();
		candidate_unif = random / RAND_MAX;
		candidate = -log(candidate_unif) / upper_lambda;
		s += candidate;
		random_acc = random_check();
		acceptance = random_acc / RAND_MAX;

		// Calculating the sum of updated lambda(s_new)
		for (int i = 0; i != dim; ++i)
		{
			individual_lambda[i] = mu[i];
			for (int j = 0; j != dim; ++j)
			{
				if (process_track[j] < 1)
				{
					continue;
				}
				if (process_track[i] >= 1)
				{
					for (int k = 0; k != M[i * dim + j]; ++k)
					{
						individual_lambda[i] += (n[i * dim + j] / Z[i * dim + j]) * pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j])) *
							(exp(-(s - process_list[max * i + process_track[i] - 1]) / alpha[max_M * (i * dim + j) + k]) * 
								recursive_array[max_M * (i * dim + j) + k] + exp(-candidate / alpha[max_M * (i * dim + j) + k]) * 
								other_process[max_M * (i * dim + j) + k]);
					}
				}
				else
				{
					for (int k = 0; k != M[i * dim + j]; ++k)
					{
						for (int l = max * j; l != max * j + process_track[j]; ++l)
						{
							individual_lambda[i] += (n[i * dim + j] / Z[i * dim + j]) * pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j])) *
								exp(-(s - process_list[l]) / alpha[max_M * (i * dim + j) + k]);
						}
					}
				}
			}
		}

		cumulative_lambda[0] = individual_lambda[0];
		for (int i = 1; i != dim; ++i)
			cumulative_lambda[i] = cumulative_lambda[i - 1] + individual_lambda[i];

		// Checking the acceptance condition and finding the dimension of acceptance
		if (acceptance * upper_lambda <= cumulative_lambda[dim - 1])
		{
			l = 0;
			while (acceptance * upper_lambda > cumulative_lambda[l])
				l += 1;
			process_list[max * l + process_track[l]] = s;
			process_track[l] += 1;

			// Adjusting reursive arrays in the accepted dimension
			for (int i = 0; i != dim; ++i)
			{
				if (i != l)
				{
					for (int j = 0; j != M[l * dim + i]; ++j)
					{
						if (process_track[l] >= 2)
						{
							recursive_array[max_M * (l * dim + i) + j] = exp(-(s - process_list[max * l + process_track[l] - 2]) / alpha[max_M * (l * dim + i) + j]) *
								recursive_array[max_M * (l * dim + i) + j] + exp(-candidate / alpha[max_M * (l * dim + i) + j]) * other_process[max_M * (l * dim + i) + j];
						}
						else if (process_track[l] == 1)
						{
							other_process_rec = 0.0;
							for (int k = max * i; k != max * i + process_track[i]; ++k)
								other_process_rec += exp(-(s - process_list[k]) / alpha[max_M * (l * dim + i) + j]);
							recursive_array[max_M * (l * dim + i) + j] = other_process_rec;
						}
					}
				}
				else
				{
					if (process_track[l] >= 2)
					{
						for (int k = 0; k != M[l * dim + i]; ++k)
						{
							recursive_array[max_M * (l * dim + i) + k] = exp(-(s - process_list[max * l + process_track[l] - 2]) / alpha[max_M * (l * dim + i) + k]) *
								(1.0 + recursive_array[max_M * (l * dim + i) + k]);
						}
					}
				}

				if (i != l)
				{
					other_process_track[l * dim + i] = process_track[i];
				}
				else
				{
					other_process_track[l * dim + i] = process_track[i] - 1;
				}
			}
		}
	}

	if (process_list[max * l + process_track[l] - 1] > T)
	{
		process_list[max * l + process_track[l] - 1] = 0;
		process_track[l] -= 1;
	}

	free(other_process);
	free(recursive_array);
	free(individual_lambda);
	free(cumulative_lambda);
	free(other_process_track);
	free(Z);
	free(alpha);
}


void compensator_power_hawkes(const double mu[], const double rho[], const double m[], const int M[], const double epsilon[], 
	const double n[], double T, int pos, int dim, int max, double * process_list, int * process_track, double * compensator_series, 
	double * Z, double * alpha)
{
	double compensator{ 0.0 };
	double other_processes{ 0.0 };
	double counter{ 0.0 };
	int index{ 0 };
	bool indic{ false };
	int * other_process_track = (int*)calloc(dim, sizeof(int));

	int max_M{ 0 };
	for (int i = 0; i != dim * dim; ++i)
		max_M = max_M >= M[i] ? max_M : M[i];
	double * recursive_array = (double*)calloc(dim * max_M, sizeof(double));

	// Checking if the parameters Z and alpha passed are valid.
	// Shall be valid if passed by the loglikelihood function, 
	// otherwise should be pointers set to NULL.
	if (Z == 0 || alpha == 0)
	{
		// Creating a matrix of alpha = rho * pow(m, i) and calculating Z values
		Z = (double*)realloc(Z, dim * sizeof(double));
		alpha = (double*)realloc(alpha, dim * max_M * sizeof(double));
		if (Z && alpha)
		{
			indic = true;
			for (int i = 0; i != dim; ++i)
			{
				Z[i] = 0;
				for (int j = 0; j != M[pos * dim + i]; ++j)
				{
					alpha[max_M * i + j] = rho[pos * dim + i] * pow(m[pos * dim + i], j);
					Z[i] += pow(alpha[max_M * i + j], -epsilon[pos * dim + i]);
				}
			}
		}
		else
		{
			free(Z);
			free(alpha);
			free(other_process_track);
			free(recursive_array);
			throw std::bad_alloc();
		}
	}

	// Main routine for the compensator calculation
	for (int i = 0; i != process_track[pos]; ++i)
	{
		if (i != 0)
		{
			compensator = mu[pos] * (process_list[max * pos + i] - process_list[max * pos + i - 1]);
		}
		else
		{
			compensator = mu[pos] * process_list[max * pos + i];
		}

		for (int j = 0; j != dim; ++j)
		{
			if (j == pos)
			{
				if (i != 0)
				{
					for (int k = 0; k != M[pos * dim + j]; ++k)
					{
						compensator += (n[pos * dim + j] / Z[j]) * pow(alpha[max_M * j + k], -epsilon[pos * dim + j]) *
							(1 - exp(-(process_list[pos * max + i] - process_list[max * pos + i - 1]) / alpha[max_M * j + k])) *
							(recursive_array[max_M * j + k] + 1);
						if (i >= 1)
						{
							recursive_array[max_M * j + k] = (1 + recursive_array[max_M * j + k]) * exp(-(process_list[pos * max + i] -
								process_list[max * pos + i - 1]) / alpha[max_M * j + k]);
						}
					}
				}
			}
			else
			{
				if (i == 0)
				{
					for (int k = 0; k != M[pos * dim + j]; ++k)
					{
						index = other_process_track[j];
						counter = 0.0;
						other_processes = 0.0;
						while (process_list[j * max + index] < process_list[pos * max + i])
						{
							other_processes += exp(-(process_list[pos * max + i] - process_list[j * max + index]) / alpha[max_M * j + k]);
							counter += 1.0;
							index += 1;
							if (index == process_track[j])
							{
								index -= 1;
								break;
							}
						}
						compensator += (n[pos * dim + j] / Z[j]) * pow(alpha[max_M * j + k], -epsilon[pos * dim + j]) * 
							(counter - other_processes);
						recursive_array[max_M * j + k] = other_processes;
					}
					other_process_track[j] = index;
				}
				else
				{
					for (int k = 0; k != M[pos * dim + j]; ++k)
					{
						counter = 0.0;
						other_processes = 0.0;
						index = other_process_track[j];
						while (process_list[j * max + index] < process_list[pos * max + i])
						{
							other_processes += exp(-(process_list[pos * max + i] - process_list[j * max + index]) / alpha[max_M * j + k]);
							counter += 1.0;
							index += 1;
							if (index == process_track[j])
							{
								index -= 1;
								break;
							}
						}
						compensator += (n[pos * dim + j] / Z[j]) * pow(alpha[max_M * j + k], -epsilon[pos * dim + j]) * 
							(((1 - exp(-(process_list[pos * max + i] - process_list[pos * max + i - 1]) / alpha[max_M * j + k])) * 
								recursive_array[max_M * j + k]) + counter - other_processes);
						recursive_array[max_M * j + k] = exp(-(process_list[pos * max + i] - process_list[pos * max + i - 1]) / alpha[max_M * j + k]) * 
							recursive_array[max_M * j + k] + other_processes;
					}
					other_process_track[j] = index;
				}
			}
		}
		compensator_series[i] = compensator;
	}

	free(recursive_array);
	free(other_process_track);
	if (indic)
	{
		free(Z);
		free(alpha);
	}
}


double loglikelihood_power_hawkes(double mu[], double rho[], double m[], int M[], double epsilon[], double n[], double T, int pos, 
	int dim, int max, double * process_list, int * process_track)
{
	double recursive_sum{ 0.0 };
	double other_process{ 0.0 };
	int index{ 0 };
	double intensity{ 0 };
	double loglikelihood{ 0.0 };
	int * other_process_track = (int*)calloc(dim, sizeof(int));
	double * compensator_series = (double*)calloc(process_track[pos] + 1, sizeof(double)); // add correct diomensionality
	double * Z = (double*)calloc(dim, sizeof(double));

	// Creating a matrix of alpha = rho * pow(m, i) and calculating Z values
	int max_M{ 0 };
	for (int i = 0; i != dim * dim; ++i)
		max_M = max_M >= M[i] ? max_M : M[i];
	double * alpha = (double*)calloc(dim * max_M, sizeof(double));
	double * recursive_array = (double*)calloc(dim * max_M, sizeof(double));
	for (int i = 0; i != dim; ++i)
	{
		for (int j = 0; j != M[pos * dim + i]; ++j)
		{
			alpha[max_M * i + j] = rho[pos * dim + i] * pow(m[pos * dim + i], j);
			Z[i] += pow(alpha[max_M * i + j], -epsilon[pos * dim + i]);
		}
	}

	for (int i = 0; i != dim; ++i)
	{
		process_list[max * i + process_track[i]] = T;
		process_track[i] += 1;
	}

	// Adding the sum of residual series to the likelihood
	compensator_power_hawkes(mu, rho, m, M, epsilon, n, T, pos, dim, max, process_list, process_track, compensator_series, Z, alpha);
	for (int i = 0; i != process_track[pos] + 1; ++i)
		loglikelihood -= compensator_series[i];
	for (int i = 0; i != dim; ++i)
	{
		process_track[i] -= 1;
		process_list[max * i + process_track[i]] = 0;
	}

	// Adding the log(lambda(t_i)) for all t_i for given dimension to the likelihood
	for (int i = 0; i != process_track[pos]; ++i)
	{
		recursive_sum = 0.0;
		for (int j = 0; j != dim; ++j)
		{
			if (j == pos && i >= 1)
			{
				for (int k = 0; k != M[pos * dim + j]; ++k)
				{
					recursive_array[max_M * j + k] = (1 + recursive_array[max_M * j + k]) * exp(-(process_list[pos * max + i] - 
						process_list[max * pos + i - 1]) / alpha[max_M * j + k]);
					recursive_sum += recursive_array[max_M * j + k] * (n[pos * dim + j] / Z[j]) * 
						pow(alpha[max_M * j + k], -(1 + epsilon[pos * dim + j]));
				}
			}
			else
			{
				if (i != 0)
				{
					for (int k = 0; k != M[pos * dim + j]; ++k)
					{
						other_process = 0.0;
						index = other_process_track[j];
						while (process_list[max * j + index] < process_list[max * pos + i])
						{
							other_process += exp(-(process_list[max * pos + i] - process_list[max * j + index]) 
								/ alpha[max_M * j + k]);
							index += 1;
							if (index == process_track[j])
							{
								index -= 1;
								break;
							}
						}
						recursive_array[max_M * j + k] = exp(-(process_list[max * pos + i] - process_list[max * pos + i - 1]) / 
							alpha[max_M * j + k]) * recursive_array[max_M * j + k] + other_process;
						recursive_sum += recursive_array[max_M * j + k] * (n[pos * dim + j] / Z[j]) * 
							pow(alpha[max_M * j + k], -(1 + epsilon[pos * dim + j]));
					}
					other_process_track[j] = index;
				}
				else
				{
					for (int k = 0; k != M[pos * dim + j]; ++k)
					{
						index = other_process_track[j];
						while (process_list[max * j + index] < process_list[max * pos + i])
						{
							other_process += exp(-(process_list[max * pos + i] - process_list[max * j + index]) / 
								alpha[max_M * j + k]);
							index += 1;
							if (index == process_track[j])
							{
								index -= 1;
								break;
							}
						}
						recursive_array[max_M * j + k] = other_process;
						recursive_sum += recursive_array[max_M * j + k] * (n[pos * dim + j] / Z[j]) * 
							pow(alpha[max_M * j + k], -(1 + epsilon[pos * dim + j]));
					}
					other_process_track[j] = index;
				}
			}
		}
		loglikelihood += log(mu[pos] + recursive_sum);
	}

	free(other_process_track);
	free(recursive_array);
	free(compensator_series);
	free(alpha);
	free(Z);

	return -loglikelihood;
}


void plt_power_hawkes(const double mu[], const double rho[], const double m[], const int M[], const double epsilon[], const double n[], 
	int begin, int end, double grid, int dim, int max, double * process_list, int * process_track, double * plt_intensity, double * plt_list,
	int * plt_end)
{
	double s{ 0.0 };
	int points = (int)(end - begin) / grid;
	int plt_length = max + points + 1;
	int * int_track = (int*)calloc(dim, sizeof(int));
	int * other_process_track = (int*)calloc(dim * dim, sizeof(int));
	bool * indic = (bool*)calloc(dim * dim, sizeof(bool));
	double * Z = (double*)calloc(dim * dim, sizeof(double));

	// Creating a matrix of alpha = rho * pow(m, i) and calculating Z values
	int max_M{ 0 };
	for (int i = 0; i != dim * dim; ++i)
		max_M = max_M >= M[i] ? max_M : M[i];
	double * alpha = (double*)calloc(dim * max_M * dim, sizeof(double));
	double * recursive_array = (double*)calloc(dim * max_M * dim, sizeof(double));
	double * other_process = (double*)calloc(dim * max_M * dim, sizeof(double));
	for (int i = 0; i != dim; ++i)
	{
		for (int j = 0; j != dim; ++j)
		{
			for (int k = 0; k != M[i * dim + j]; ++k)
			{
				alpha[max_M * (i * dim + j) + k] = rho[i * dim + j] * pow(m[i * dim + j], k);
				Z[i * dim + j] += pow(alpha[max_M * (i * dim + j) + k], -epsilon[i * dim + j]);
			}
		}
	}

	std::vector<holder> ordering_vec;
	int order_i;
	holder holder_obj;

	if (s >= begin)
	{
		for (int i = 0; i != dim; ++i)
		{
			plt_intensity[i * plt_length + plt_end[i]] = mu[i];
			plt_list[i * plt_length + plt_end[i]] = s;
			plt_end[i] += 1;
		}
	}

	while (s - 0.00001 <= end - grid)
	{
		s += grid;
		// Next for loop and std::sort is used for finding the order of events in the process in the given time frame
		ordering_vec.clear();
		for (int i = 0; i != dim; ++i)
		{
			if (int_track[i] == process_track[i])
			{
				continue;
			}
			order_i = int_track[i];
			while (process_list[i * max + order_i] < s)
			{
				holder_obj = { i, process_list[i * max + order_i] };
				ordering_vec.push_back(holder_obj);
				++order_i;
				if (order_i == process_track[i])
				{
					break;
				}
			}
		}
		std::sort(ordering_vec.begin(), ordering_vec.end(), [](holder &a, holder &b) {return a.stamp <= b.stamp; });

		// Adding sorted lambda values at jumps in the current timeframe (of length grid) 
		for (holder h : ordering_vec)
		{
			int i = h.index;
			if (s >= begin)
			{
				plt_intensity[i * plt_length + plt_end[i]] = mu[i];
			}
			int_track[i] += 1;

			for (int j = 0; j != dim; ++j)
			{
				if (i != j)
				{
					for (int k = 0; k != M[i * dim + j]; ++k)
					{
						other_process[max_M * (i * dim + j) + k] = 0.0;
						for (int l = max * j + other_process_track[i * dim + j]; l != max * j + int_track[j]; ++l)
						{
							other_process[max_M * (i * dim + j) + k] += exp(-(process_list[max * i + int_track[i] - 1] - process_list[l]) /
								alpha[max_M * (i * dim + j) + k]);
						}
						if (int_track[i] >= 2)
						{
							recursive_array[max_M * (i * dim + j) + k] = exp(-(process_list[i * max + int_track[i] - 1] - 
								process_list[i * max + int_track[i] - 2]) / alpha[max_M * (i * dim + j) + k]) * 
								recursive_array[max_M * (i * dim + j) + k] + other_process[max_M * (i * dim + j) + k];
						}
						else
						{
							recursive_array[max_M * (i * dim + j) + k] = other_process[max_M * (i * dim + j) + k];
						}
					}
				}
				else if (int_track[i] >= 2)
				{
					for (int k = 0; k != M[i * dim + j]; ++k)
					{
						recursive_array[max_M * (i * dim + j) + k] = exp(-(process_list[i * max + int_track[i] - 1] - 
							process_list[i * max + int_track[i] - 2]) / alpha[max_M * (i * dim + j) + k]) * 
							(recursive_array[max_M * (i * dim + j) + k] + 1.0);
					}
				}

				if (i != j)
				{
					other_process_track[i * dim + j] = int_track[j];
				}
				else
				{
					other_process_track[i * dim + j] = int_track[j];
				}

				if (s < begin || int_track[j] < 1)
				{
					continue;
				}

				if (i != j)
				{
					for (int k = 0; k != M[i * dim + j]; ++k)
					{
						plt_intensity[i * plt_length + plt_end[i]] += (n[i * dim + j] / Z[i * dim + j]) * 
							pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j])) * recursive_array[max_M * (i * dim + j) + k];
					}
				}
				else if (i == j)
				{
					for (int k = 0; k != M[i * dim + j]; ++k)
					{
						plt_intensity[i * plt_length + plt_end[i]] += (n[i * dim + j] / Z[i * dim + j]) * 
							pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j])) *
							(recursive_array[max_M * (i * dim + j) + k] + 1.0);
					}
				}
			}
			if (s >= begin)
			{
				plt_list[i * plt_length + plt_end[i]] = process_list[max * i + int_track[i] - 1];
				plt_end[i] += 1;
			}
		}

		// Adding lambda values at specified grid points
		if (s + 0.00001 >= begin)   // Floating point imprecision
		{
			for (int i = 0; i != dim; ++i)
			{
				plt_intensity[i * plt_length + plt_end[i]] = mu[i];

				for (int j = 0; j != dim; ++j)
				{
					if (int_track[j] < 1)
					{
						continue;
					}
					if (int_track[i] >= 1)
					{
						if (i != j)
						{
							for (int k = 0; k != M[i * dim + j]; ++k)
							{
								other_process[max_M * (i * dim + j) + k] = 0.0;
								for (int l = max * j + other_process_track[i * dim + j]; l != max * j + int_track[j]; ++l)
									other_process[max_M * (i * dim + j) + k] += exp(-(s - process_list[l]) / alpha[max_M * (i * dim + j) + k]);
								plt_intensity[i * plt_length + plt_end[i]] += (n[i * dim + j] / Z[i * dim + j]) * 
									pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j])) * 
									(exp(-(s - process_list[max * i + int_track[i] - 1]) / alpha[max_M * (i * dim + j) + k]) * 
										recursive_array[max_M * (i * dim + j) + k] + other_process[max_M * (i * dim + j) + k]);
							}
						}
						else
						{
							for (int k = 0; k != M[i * dim + j]; ++k)
							{
								plt_intensity[i * plt_length + plt_end[i]] += (n[i * dim + j] / Z[i * dim + j]) * 
									pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j])) * exp(-(s - process_list[max * i + int_track[i] - 1]) / 
										alpha[max_M * (i * dim + j) + k]) * (1.0 + recursive_array[max_M * (i * dim + j) + k]);
							}
						}
					}
					else
					{
						for (int k = 0; k != M[i * dim + j]; ++k)
						{
							for (int l = max * j; l != max * j + int_track[j]; ++l)
							{
								plt_intensity[i * plt_length + plt_end[i]] += (n[i * dim + j] / Z[i * dim + j]) * 
									pow(alpha[max_M * (i * dim + j) + k], -(1 + epsilon[i * dim + j])) * 
									exp(-(s - process_list[l]) / alpha[max_M * (i * dim + j) + k]);
							}
						}
					}
				}
				plt_list[i * plt_length + plt_end[i]] = s;
				plt_end[i] += 1;
			}
		}
	}

	free(other_process);
	free(recursive_array);
	free(int_track);
	free(other_process_track);
	free(indic);
	free(alpha);
	free(Z);
}