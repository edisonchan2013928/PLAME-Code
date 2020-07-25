#include "vec_operation.h"

double ip(double*q, double*p, int dim)
{
	double ip_value = 0;
	for (int d = 0; d < dim; d++)
		ip_value += q[d] * p[d];

	return ip_value;
}

void update_weight(double*w, double alpha, double bar_alpha, double y_i, double*x_i, int dim)
{
	for (int d = 0; d < dim; d++)
		w[d] += (alpha - bar_alpha)*y_i*x_i[d];
}