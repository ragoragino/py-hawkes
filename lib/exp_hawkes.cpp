#include "exp_hawkes.h"
#include "additional_functions.h"
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <vector> // Enclose struct holder in the vector in plot_exponential_hawkes
#include <algorithm> // use std::sort for ordering elements in std::vector<holder> in plot_exponential_hawkes 

void simulate_exponential_hawkes(const double mu[], const double alpha[], const double beta[], double T, int dim, int max,
    double * process_list, int * process_track, unsigned int seed)
{
	bool indicator = stationaritycheck(alpha, beta, dim);
	if (indicator)
	{
		throw StationarityError("Non-stationary exponential kernel!");
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
    double * recursive_array = (double*)calloc(dim * dim, sizeof(double));
    double * other_process = (double*)calloc(dim * dim, sizeof(double));
    double * individual_lambda = (double*)calloc(dim, sizeof(double));
    double * cumulative_lambda = (double*)calloc(dim, sizeof(double));
    int * other_process_track = (int*)calloc(dim * dim, sizeof(int));

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
                throw LimitError("Given limit of jumps exceeded!");
            }

			// Calculating the sum of lambda(s)
            upper_lambda += mu[i];
            for (int j = 0; j != dim; ++j)
            {
                other_process[i * dim + j] = 0.0;
				if (process_track[j] < 1)
				{
					continue;
				}

                if (process_track[i] >= 1)
                {
                    for (int k = max * j + other_process_track[i * dim + j]; k != max * j + process_track[j]; ++k)
                        other_process[i * dim + j] += exp(-(s - process_list[k]) * beta[i * dim + j]);
                    upper_lambda += alpha[i * dim + j] * (exp(-(s - process_list[max * i + process_track[i] - 1]) * 
                        beta[i * dim + j]) * recursive_array[i * dim + j] + other_process[i * dim + j]);

                }
                else
                {
                    for (int k = max * j; k != max * j + process_track[j]; ++k)
                        upper_lambda += alpha[i * dim + j] * exp(-beta[i * dim + j] * (s - process_list[k]));
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
                    individual_lambda[i] += alpha[i * dim + j] * (exp(-(s - process_list[max * i + process_track[i] - 1]) * 
                        beta[i * dim + j]) * recursive_array[i * dim + j] + exp(-candidate * beta[i * dim + j]) * other_process[i * dim + j]);
                }
                else
                {
                    for (int k = max * j; k != max * j + process_track[j]; ++k)
                        individual_lambda[i] += alpha[i * dim + j] * exp(-beta[i * dim + j] * (s - process_list[k]));
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
                    if (process_track[l] >= 2)
                    {
                        recursive_array[l * dim + i] = exp(-beta[l * dim + i] * (s - process_list[max * l + process_track[l] - 2])) *
                            recursive_array[l * dim + i] + exp(-candidate * beta[l * dim + i]) * other_process[l * dim + i];
                    }
                    else if (process_track[l] == 1)
                    {
                        other_process_rec = 0.0;
                        for (int j = max * i; j != max * i + process_track[i]; ++j)
                            other_process_rec += exp(-(s - process_list[j]) * beta[l * dim + i]);
                        recursive_array[l * dim + i] = other_process_rec;
                    }
                    other_process_track[l * dim + i] = process_track[i];
                }
                else
                {
                    if (process_track[l] >= 2)
                    {
                        recursive_array[l * dim + i] = exp(-beta[l * dim + i] * (s - process_list[max * l + process_track[l] - 2])) 
                            * (1.0 + recursive_array[l * dim + i]);
                    }
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
}


void compensator_exponential_hawkes(const double mu[], const double alpha[], const double beta[], double T, int pos, int dim, int max, 
    double * process_list, int * process_track, double * compensator_series)
{
    double compensator{ 0.0 };
    double other_processes{ 0.0 };
    double counter{ 0.0 };
    int k{ 0 };
    double * recursive_array = (double*)calloc(dim, sizeof(double));
    int * other_process_track = (int*)calloc(dim, sizeof(int));

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
                    compensator += (alpha[pos * dim + j] / beta[pos * dim + j]) * (1 - exp(-beta[pos * dim + j] * 
                        (process_list[pos * max + i] - process_list[max * pos + i - 1]))) * (recursive_array[j] + 1);
                    if (i >= 1)
                    {
                        recursive_array[j] = (1 + recursive_array[j]) * (exp(-beta[pos * dim + j] * 
                            (process_list[pos * max + i] - process_list[max * pos + i - 1])));
                    }
                }
            }
            else
            {
                counter = 0.0;
                other_processes = 0.0;
                if (i == 0)
                {                
                    k = other_process_track[j];
                    while (process_list[j * max + k] < process_list[pos * max + i])
                    {
                        other_processes += exp(-beta[pos * dim + j] * (process_list[pos * max + i] - process_list[j * max + k]));
                        counter += 1.0;
                        k += 1;
                        if (k == process_track[j])
                        {
                            k -= 1;
                            break;
                        }
                    }
                    other_process_track[j] = k;
                    compensator += (alpha[pos * dim + j] / beta[pos * dim + j]) * (counter - other_processes);
                    recursive_array[j] = other_processes;
                }
                else
                {
                    k = other_process_track[j];
                    while (process_list[j * max + k] < process_list[pos * max + i])
                    {
                        other_processes += exp(-beta[pos * dim + j] * (process_list[pos * max + i] - process_list[j * max + k]));
                        counter += 1.0;
                        k += 1;
                        if (k == process_track[j])
                        {
                            k -= 1;
                            break;
                        }
                    }
                    other_process_track[j] = k;
                    compensator += (alpha[pos * dim + j] / beta[pos * dim + j]) * (((1 - exp(-beta[pos * dim + j] * 
                        (process_list[pos * max + i] - process_list[pos * max + i - 1]))) * recursive_array[j]) + 
                        counter - other_processes);
                    recursive_array[j] = exp(-beta[pos * dim + j] * (process_list[pos * max + i] - process_list[pos * max + i - 1])) * 
                        recursive_array[j] + other_processes;
                }
            }
        }

        compensator_series[i] = compensator;
    }

    free(other_process_track);
    free(recursive_array);
}


double loglikelihood_exponential_hawkes(double mu[], double alpha[], double beta[], double T, int pos, int dim, int max,
    double * process_list, int * process_track) 
{
    double recursive_sum{ 0.0 };
    double other_process{ 0.0 };
    double loglikelihood{ 0.0 };
    int k{ 0 };
    int * other_process_track = (int*)calloc(dim, sizeof(int));
    double * compensator_series = (double*)calloc(process_track[pos] + 1, sizeof(double));
    double * recursive_array = (double*)calloc(dim, sizeof(double));
    
    for (int i = 0; i != dim; ++i) 
    {
        process_list[max * i + process_track[i]] = T; // Add T for correct MLE computation. It has to be disregarded in all other procedures!   
        process_track[i] += 1;
    }

	// Adding the sum of residual series to the likelihood
    compensator_exponential_hawkes(mu, alpha, beta, T, pos, dim, max, process_list, process_track, compensator_series);
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
                recursive_array[j] = exp(-beta[pos * dim + j] * (process_list[max * pos + i] - process_list[max * pos  + i - 1])) *
                    (recursive_array[j] + 1);
                recursive_sum += recursive_array[j] * alpha[pos * dim + j];
            }
            else
            {
                other_process = 0.0;
                if (i != 0) 
                {
                    k = other_process_track[j];
                    while (process_list[max * j + k] < process_list[max * pos + i]) 
                    {
                        other_process += exp(-beta[pos * dim + j] * (process_list[max * pos + i] - process_list[max * j + k]));
                        k += 1;
                        if (k == process_track[j])
                        {
                            k -= 1;
                            break;
                        }
                    }
                    other_process_track[j] = k;
                    recursive_array[j] = exp(-beta[pos * dim + j] * (process_list[max * pos + i] - process_list[max * pos + i - 1])) * 
                        recursive_array[j] + other_process;
                    recursive_sum += recursive_array[j] * alpha[pos * dim + j];
                }
                else
                {
                    k = other_process_track[j];
                    while (process_list[max * j + k] < process_list[max * pos + i])
                    {
                        other_process += exp(-beta[pos * dim + j] * (process_list[max * pos + i] - process_list[max * j + k]));
                        k += 1;
                        if (k == process_track[j])
                        {
                            k -= 1;
                            break;
                        }
                    }
                    other_process_track[j] = k;
                    recursive_array[j] = other_process;
                    recursive_sum += recursive_array[j] * alpha[pos * dim + j];
                }
            }
        }
        loglikelihood += log(mu[pos] + recursive_sum);
    }

    free(other_process_track);
    free(recursive_array);
    free(compensator_series);

    return -loglikelihood;
}

void plt_exponential_hawkes(const double mu[], const double alpha[], const double beta[], int begin, int end, double grid, int dim, int max,
    double * process_list, int * process_track, double * plt_intensity, double * plt_list, int * plt_end)
{
    double s{ 0.0 };
    int points = (int)(end - begin) / grid;
    int plt_length = max + points + 1;
    int * int_track = (int*)calloc(dim, sizeof(int));
    double * recursive_array = (double*)calloc(dim * dim, sizeof(double));
    double * other_process = (double*)calloc(dim * dim, sizeof(double));
    int * other_process_track = (int*)calloc(dim * dim, sizeof(int));
    bool * indic = (bool*)calloc(dim * dim, sizeof(bool));

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

    while (s - 0.00001 <= end - grid)   // Floating point imprecision
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
        std::sort(ordering_vec.begin(), ordering_vec.end(), [](holder &a, holder &b) {return a.stamp <= b.stamp;});
        
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
					other_process[i * dim + j] = 0.0;
					for (int k = max * j + other_process_track[i * dim + j]; k != max * j + int_track[j]; ++k)
						other_process[i * dim + j] += exp(-(process_list[max * i + int_track[i] - 1] - process_list[k]) * beta[i * dim + j]);

					if (int_track[i] >= 2)
					{
						recursive_array[i * dim + j] = exp(-beta[i * dim + j] * (process_list[i * max + int_track[i] - 1] -
							process_list[i * max + int_track[i] - 2])) * recursive_array[i * dim + j] + other_process[i * dim + j];
					}
					else
					{
						recursive_array[i * dim + j] = other_process[i * dim + j];
					}
				}
                else if (int_track[i] >= 2)
                {
                    recursive_array[i * dim + j] = exp(-beta[i * dim + j] * (process_list[i * max + int_track[i] - 1] - 
                        process_list[i * max + int_track[i] - 2])) * (recursive_array[i * dim + j] + 1.0);
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
					plt_intensity[i * plt_length + plt_end[i]] += alpha[i * dim + j] * recursive_array[i * dim + j];
				}
				else if (i == j)
				{
					plt_intensity[i * plt_length + plt_end[i]] += alpha[i * dim + j] * (recursive_array[i * dim + j] + 1.0);
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
                    other_process[i * dim + j] = 0.0;
					if (int_track[j] < 1)
					{
						continue;
					}
                    if (int_track[i] >= 1)
                    {
                        if (i != j)
                        {
                            for (int k = max * j + other_process_track[i * dim + j]; k != max * j + int_track[j]; ++k)
                                other_process[i * dim + j] += exp(-(s - process_list[k]) * beta[i * dim + j]);
                            plt_intensity[i * plt_length + plt_end[i]] += alpha[i * dim + j] * (exp(-(s - process_list[max * i + int_track[i] - 1]) *
                                beta[i * dim + j]) * recursive_array[i * dim + j] + other_process[i * dim + j]);
                        }
                        else 
                        {
                            plt_intensity[i * plt_length + plt_end[i]] += alpha[i * dim + j] * exp(-(s - process_list[max * i + int_track[i] - 1]) *
                                beta[i * dim + j]) * (1.0 + recursive_array[i * dim + j]);
                        }
                    }
                    else
                    {
                        for (int k = max * j; k != max * j + int_track[j]; ++k)
                            plt_intensity[i * plt_length + plt_end[i]] += alpha[i * dim + j] * exp(-beta[i * dim + j] * (s - process_list[k]));
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
}
