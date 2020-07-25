#include "svm_train.h"

//We focus on hinge-loss (L1-SVM, the most traditional one)
void dual_coord_descent(model& our_model)
{
	double U = our_model.C; 
	int num_iter;
	double G;
	double PG;
	double temp_alpha;
	double*Q = new double[our_model.n];
	double q, m, c, denominator; //Used in piecewise linear kernel
	int beta; //Used in piecewise linear kernel

	//initialization
	num_iter = 1000;
	for (int i = 0; i < our_model.n; i++)
	{
		our_model.alphaVector[i] = 0;
		our_model.dataMatrix[i][our_model.dim] = 1;
		if (our_model.method == 0) //Linear kernel
			Q[i] = our_model.labelVector[i] * our_model.labelVector[i] * ip(our_model.dataMatrix[i], our_model.dataMatrix[i], our_model.dim + 1);
		if (our_model.method == 1) //Chi2 kernel
			Q[i] = our_model.labelVector[i] * our_model.labelVector[i] * K_function(our_model.dataMatrix[i], our_model.dataMatrix[i], our_model.dim, our_model.method);
	}

	if (our_model.method == 2)
	{
		preprocess_PL_kernel(1, our_model); //Set q_fixed to be 1 (we need to think about it later.)
		for (int i = 0; i < our_model.n; i++)
			Q[i] = our_model.Q_ii[i];
	}

	for (int d = 0; d < our_model.dim + 1; d++)
		our_model.weight[d] = 0;

	for (int it = 0; it < num_iter; it++)
	{
		for (int i = 0; i < our_model.n; i++)
		{
			if (our_model.method == 0) //Linear kernel
				G = our_model.labelVector[i] * ip(our_model.weight, our_model.dataMatrix[i], our_model.dim + 1) - 1;
			if (our_model.method == 1) //Chi2 kernel
				G = our_model.labelVector[i] * K_aggregation(our_model.dataMatrix[i], our_model) - 1;
			if (our_model.method == 2)
			{
				//G = our_model.labelVector[i] * K_aggregation(our_model.dataMatrix[i], our_model) - 1;
				G = 0;
				for (int d = 0; d < our_model.dim; d++)
				{
					q = our_model.dataMatrix[i][d];
					if (q < eps)
						continue;

					beta = binary_search_beta(q, our_model.S, 0, our_model.S.size() - 1);
					G = G + our_model.AP[d][beta] * q + our_model.A[d][beta];

					/*for (int beta = 0; beta < (int)our_model.S.size(); beta++)
					{
						m = (2 * q * q) / ((q + our_model.S[beta].ell)*(q + our_model.S[beta].u));
						c = (2* q * our_model.S[beta].ell*our_model.S[beta].u) / ((q + our_model.S[beta].ell)*(q + our_model.S[beta].u));
						G = G + m * our_model.AP[d][beta] + c * our_model.A[d][beta];
					}*/
				}
				G = our_model.labelVector[i] * G - 1;
			}

			if (our_model.alphaVector[i] < eps) //alpha_i=0
				PG = min(G, 0.0);
			if (our_model.alphaVector[i] > U - eps) //alpha_i=U
				PG = max(G, 0.0);
			if (our_model.alphaVector[i] > 0 && our_model.alphaVector[i] < U)
				PG = G;

			if (fabs(PG) > eps) //|PG|!=0
			{
				temp_alpha = our_model.alphaVector[i];
				our_model.alphaVector[i] = min(max(our_model.alphaVector[i] - G / Q[i], 0.0), U);

				if (our_model.method == 0) //Linear kernel (No need to update for non-linear kernel functions)
					update_weight(our_model.weight, our_model.alphaVector[i], temp_alpha, our_model.labelVector[i], our_model.dataMatrix[i], our_model.dim + 1);
				if (our_model.method == 2)//piecewise-linear additive kernel
				{
					for (int d = 0; d < our_model.dim; d++)
					{
						for (int beta = 0; beta < our_model.S.size(); beta++)
						{
							denominator = (our_model.dataMatrix[i][d] + our_model.S[beta].ell) * (our_model.dataMatrix[i][d] + our_model.S[beta].u);
							if (denominator == 0)
								continue;

							m = (2 * our_model.dataMatrix[i][d] * our_model.dataMatrix[i][d]) / denominator;
							c = (2 * our_model.dataMatrix[i][d] * our_model.S[beta].ell*our_model.S[beta].u) / denominator;

							our_model.AP[d][beta] += (our_model.alphaVector[i] - temp_alpha) * our_model.labelVector[i] * m;
							our_model.A[d][beta] += (our_model.alphaVector[i] - temp_alpha) * our_model.labelVector[i] * c;
						}
					}

				}
			}

		}
	}
}

void train_SVM(model& our_model)
{
	//clock_t start_t;
	//clock_t end_t;
	double totalTime;

	auto start_t = chrono::high_resolution_clock::now();
	dual_coord_descent(our_model);
	auto end_t = chrono::high_resolution_clock::now();

	totalTime = (chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count()) / 1000000000.0;
	//totalTime = (end_t - start_t) / CLOCKS_PER_SEC;
	cout << "Time(sec): " << totalTime << endl;
}

void out_model(model& our_model)
{
	fstream out_file;
	out_file.open(our_model.model_fileName, ios::in | ios::out | ios::trunc);

	if (out_file.is_open() == false)
	{
		cout << "Cannot open out_file!" << endl;
		exit(0);
	}

	out_file << our_model.method << endl;
	out_file << our_model.dim << endl;

	if (our_model.method == 0)
	{
		//out_file << "LINEAR" << endl;
		for (int d = 0; d < our_model.dim + 1; d++)
			out_file << our_model.weight[d] << " ";
	}

	if (our_model.method == 1 || our_model.method == 2)
	{
		double w_i;
		int sv_on_line_index;
		int num_sv = 0;
		for (int i = 0; i < our_model.n; i++)
		{
			if (our_model.alphaVector[i] < our_model.C - eps && our_model.alphaVector[i] > eps)
				sv_on_line_index = i;

			if (our_model.alphaVector[i] > eps)
				num_sv++;
		}

		out_file << num_sv << endl;
		out_file << K_aggregation(our_model.dataMatrix[sv_on_line_index], our_model) << endl;

		for (int i = 0; i < our_model.n; i++)
		{
			w_i = our_model.labelVector[i] * our_model.alphaVector[i];
			if (fabs(w_i) > eps)
			{
				out_file << w_i << " ";
				for (int d = 0; d < our_model.dim; d++)
					out_file << our_model.dataMatrix[i][d] << " ";
				out_file << endl;
			}
		}
	}

	if (our_model.method == 2)
	{
		out_file << "delta: " << our_model.delta << endl;
		out_file << "number_of_intervals: " << our_model.S.size() << endl;
		out_file << "Intervals:" << endl;
		for (int i = 0; i < our_model.S.size(); i++)
			out_file << our_model.S[i].ell << " " << our_model.S[i].u << " ";
		out_file << endl;
		
		//output AP[j,beta] and A[j,beta]
		out_file << "AP[j,beta]:" << endl;
		for (int j = 0; j < our_model.dim; j++)
		{
			for (int beta = 0; beta < our_model.S.size(); beta++)
				out_file << our_model.AP[j][beta] << " ";
			out_file << endl;
		}
		out_file << "A[j,beta]" << endl;
		for (int j = 0; j < our_model.dim; j++)
		{
			for (int beta = 0; beta < our_model.S.size(); beta++)
				out_file << our_model.A[j][beta] << " ";
			out_file << endl;
		}
	}
}