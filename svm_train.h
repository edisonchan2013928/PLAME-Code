#pragma once
#ifndef SVM_TRAIN_H
#define SVM_TRAIN_H

#include "init_train.h"
#include "vec_operation.h"
#include "Kernel.h"

void dual_coord_descent(model& our_model);
void train_SVM(model& our_model);
void out_model(model& our_model);

#endif