#pragma once
#ifndef KERNEL_H
#define KERNEL_H

#include "init_train.h"

double K_function(double*q, double*p, int dim, int method);
double K_aggregation(double*q, model& our_model);
void find_sequence(double q_fixed, model& our_model);
void find_sequence(int d, model& our_model);
int binary_search_beta(double x, vector<interval>& S, int low, int high);
void preprocess_PL_kernel(double q_fixed, model& our_model);

#endif