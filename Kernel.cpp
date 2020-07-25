#include "Kernel.h"

struct entry
{
	int index;
	double value;
};

bool sort_entryfunction(entry& l_entry, entry& r_entry)
{
	return (l_entry.value < r_entry.value);
}

double pow_term(double value, int index)
{
	double return_value = 1;
	for (int i = 0; i < index; i++)
		return_value = return_value * value;
	return return_value;
}

double K_function(double*q, double*p, int dim, int method)
{
	double value = 0;
	if (method == 1) //Chi2 Kernel
	{
		for (int d = 0; d < dim; d++)
		{
			if (fabs(q[d]) < eps || fabs(p[d]) < eps)
				continue;
			value += (2.0*q[d] * p[d]) / (q[d] + p[d]);
		}
	}

	return value;
}

double K_aggregation(double*q, model& our_model)
{
	double KA_Value = 0;

	if (our_model.method == 1) //Chi2 Kernel
	{
		for (int i = 0; i < our_model.n; i++)
			KA_Value += our_model.labelVector[i] * our_model.alphaVector[i] * K_function(q, our_model.dataMatrix[i], our_model.dim, our_model.method);
	}
	if (our_model.method == 2) //Piecewise-linear Chi2 Kernel
	{
		int beta;
		for (int d = 0; d < our_model.dim; d++)
		{
			beta = binary_search_beta(q[d], our_model.S, 0, our_model.S.size() - 1);
			KA_Value += our_model.AP[d][beta] * q[d] + our_model.A[d][beta];
		}
	}

	return KA_Value;
}

void find_sequence(double q_fixed, model& our_model)
{
	double ell;
	double u;
	double q = q_fixed;
	double H;
	double alpha, alpha_square;
	double Delta;
	double q_square, q_cubic, q_quartic;
	double middle_term;
	interval inter;

	ell = 0;
	while (ell < 1)
	{
		inter.ell = ell;
		if (our_model.method == 2) //Piecewise-linear Chi2 kernel
		{
			alpha = sqrt(q + ell);
			H = (2 * q*ell) / (q + ell);
			alpha_square = alpha * alpha;
			q_square = q * q;
			q_cubic = q_square * q;
			q_quartic = q_cubic * q;
			middle_term = 2 * q - H - our_model.delta;
			Delta = 16 * q_quartic*alpha_square - 8 * middle_term * alpha_square * (q_cubic + q_square * ell);
			u = pow_term((4 * q_square*alpha + sqrt(Delta)) / (2 * middle_term*alpha_square), 2) - q;

			inter.u = u;
			our_model.S.push_back(inter);
		}
		ell = u;
	}
}

void find_sequence(int d, model& our_model)
{
	double ell;
	double u, u_best;
	double q;
	double H;
	double alpha, alpha_square;
	double Delta;
	double q_square, q_cubic, q_quartic;
	double middle_term;
	interval inter;

	ell = 0;
	while (ell < 1)
	{
		inter.ell = ell;
		if (our_model.method == 2) //Piecewise-linear Chi2 kernel
		{
			u_best = 1;
			for (int i = 0; i < our_model.n; i++)
			{
				q = our_model.dataMatrix[i][d];
				alpha = sqrt(q + ell);
				H = (2 * q * ell) / (q + ell);
				alpha_square = alpha * alpha;
				q_square = q * q;
				q_cubic = q_square * q;
				q_quartic = q_cubic * q;
				middle_term = 2 * q - H - our_model.delta;
				Delta = 16 * q_quartic*alpha_square - 8 * middle_term * alpha_square * (q_cubic + q_square * ell);
				u = pow_term((4 * q_square*alpha + sqrt(Delta)) / (2 * middle_term*alpha_square), 2) - q;

				if (u < u_best)
					u_best = u;
			}

			inter.u = u_best;
			our_model.S_array[d].push_back(inter);
		}
		ell = u;
	}
}

int binary_search_beta(double x, vector<interval>& S, int low, int high)
{
	int middle = (int)floor((double)(low + high) / 2.0);
	if (S[middle].ell<=x && S[middle].u>=x)
		return middle;
	if (middle + 1 == high)
	{
		if (S[high].ell<x && S[high].u + eps>x)
			return high;
	}
	if (middle - 1 == low)
	{
		if (S[low].ell - eps<x && S[low].u>x)
			return low;
	}

	if (S[middle].u < x)
		return binary_search_beta(x, S, middle, high);
	if (S[middle].ell > x)
		return binary_search_beta(x, S, low, middle);

	cout << "ERROR!" << endl;
	exit(0);
}

/*void preprocess_partition(model& our_model)
{
	our_model.point_partition = new int*[our_model.n];
	for (int i = 0; i < our_model.n; i++)
		our_model.point_partition[i] = new int[our_model.dim];

	for (int i = 0; i < our_model.n; i++)
		for (int d = 0; d < our_model.dim; d++)
			our_model.point_partition[i][d] = binary_search_beta(our_model.dataMatrix[i][d], our_model.S, 0, our_model.S.size() - 1);
}*/

void preprocess_PL_kernel(double q_fixed, model& our_model)
{
	int beta;
	double m, c, q;
	vector<interval> S;
	//double debugValue;
	//find_sequence(q_fixed, our_model);
	for (int d = 0; d < our_model.dim; d++)
	{
		our_model.S_array.push_back(S);
		find_sequence(d, our_model);
	}

	our_model.AP = new double*[our_model.dim];
	our_model.A = new double*[our_model.dim];
	for (int d = 0; d < our_model.dim; d++)
	{
		our_model.AP[d] = new double[our_model.S.size()];
		our_model.A[d] = new double[our_model.S.size()];
	}

	for (int d = 0; d < our_model.dim; d++)
	{
		for (int beta = 0; beta < our_model.S.size(); beta++)
		{
			our_model.AP[d][beta] = 0;
			our_model.A[d][beta] = 0;
		}
	}

	//preprocess_partition(our_model);
	//obtain Q_ii
	our_model.Q_ii = new double[our_model.n];
	for (int i = 0; i < our_model.n; i++)
	{
		//debugValue = K_function(our_model.dataMatrix[i], our_model.dataMatrix[i], our_model.dim, 2);
		our_model.Q_ii[i] = 0;
		for (int d = 0; d < our_model.dim; d++)
		{
			//beta = our_model.point_partition[i][d];
			beta = binary_search_beta(our_model.dataMatrix[i][d], our_model.S, 0, our_model.S.size() - 1);
			if (our_model.method == 2)
			{
				q = our_model.dataMatrix[i][d];
				if (q < eps)
					continue;
				m = (2 * q * q) / ((q + our_model.S[beta].ell)*(q + our_model.S[beta].u));
				c = (2 * q * our_model.S[beta].ell*our_model.S[beta].u) / ((q + our_model.S[beta].ell)*(q + our_model.S[beta].u));
				our_model.Q_ii[i] += m * q + c;
			}
		}
	}
}