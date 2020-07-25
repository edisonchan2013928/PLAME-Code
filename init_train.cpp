#include "init_train.h"

void loadData(char*dataFileName, model& our_model)
{
	fstream dataFile;

	dataFile.open(dataFileName);
	if (dataFile.is_open() == false)
	{
		cout << "Cannot open dataFile!" << endl;
		exit(0);
	}

	dataFile >> our_model.n;
	dataFile >> our_model.dim;

	our_model.dataMatrix = new double*[our_model.n];
	our_model.labelVector = new double[our_model.n];

	for (int i = 0; i < our_model.n; i++)
		our_model.dataMatrix[i] = new double[our_model.dim + 1];

	for (int i = 0; i < our_model.n; i++)
	{
		dataFile >> our_model.labelVector[i];
		for (int d = 0; d < our_model.dim; d++)
			dataFile >> our_model.dataMatrix[i][d];
	}

	//two-class SVM
	if (our_model.num_class == 2)
	{
		our_model.alphaVector = new double[our_model.n];
		our_model.weight = new double[our_model.dim + 1];
	}
	//multi-class SVM
	if (our_model.num_class > 2)
	{
		//code here
	}

	dataFile.close();
}

void init_model(int argc, char**argv, model& our_model)
{
	/*char*dataFileName = (char*)"../../../Datasets/a6a/a6a_tr_scale_ours_v2";
	our_model.method = 0;
	our_model.C = 1;
	our_model.train_epsilon = 0.001;
	our_model.model_fileName = (char*)"../../../Datasets/a6a/a6a_train_linear";
	our_model.delta = 0.01;*/
	char*dataFileName = argv[1];
	our_model.method = atoi(argv[2]);
	our_model.C = atof(argv[3]);
	//our_model.train_epsilon = atof(argv[4]); //default should be 0.001
	our_model.model_fileName = argv[4];
	our_model.delta = atof(argv[5]);
	our_model.num_class = atoi(argv[6]);

	loadData(dataFileName, our_model);
}