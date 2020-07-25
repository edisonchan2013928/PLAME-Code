#include "svm_train.h"

int main(int argc, char**argv)
{
	model our_model;
	init_model(argc, argv, our_model);
	train_SVM(our_model);
	out_model(our_model);
}