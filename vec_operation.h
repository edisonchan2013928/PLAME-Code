#pragma once
#ifndef VEC_OPERATION_H
#define VEC_OPERATION_H

#include "Library.h"

double ip(double*q, double*p, int dim);
void update_weight(double*w, double alpha, double bar_alpha, double y_i, double*x_i, int dim);

#endif