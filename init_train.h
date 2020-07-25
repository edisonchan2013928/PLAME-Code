#pragma once
#ifndef INIT_TRAIN_H
#define INIT_TRAIN_H

#include "Library.h"

const double inf = 99999999999;
const double eps = 0.000000001;

struct interval
{
	double ell;
	double u;
};

struct model
{
	int n;
	int dim;
	int method;
	double**dataMatrix;
	double*labelVector;
	double train_epsilon; //not useful
	char*model_fileName;

	//two-class SVM implementation
	//***************************************************//
	double*alphaVector;
	double C;

	//linear SVM
	double*PG_grad;
	double*weight;

	//piecewise_linear additive kernel
	//double**dataMatrix_dim;
	vector<interval> S;
	vector< vector<interval> > S_array;
	double*Q_ii;
	double**AP;
	double**A;
	//int**point_partition;
	double delta; //tolerance
	//vector< vector<double> > sort_dataMatrix_dim;
	//double*sort_dataMatrix_dim;
	//***************************************************//

	//multi-class SVM implementation
	//***************************************************//
	int num_class;
	double alphaVector_class; //num_class x n
	double**weight_class; //num_class x d matrix
	//***************************************************//
};

void loadData(char*dataFileName, model& our_model);
void init_model(int argc, char**argv, model& our_model);

#endif