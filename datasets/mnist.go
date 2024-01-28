package datasets

import "github.com/Hukyl/mlgo/matrix"

// MnistDataset reads the CSV files for MNIST dataset, in the form:
// [2]{X_train, Y_train}, [2]{X_test, Y_test}, err
func MnistDataset(trainPath, testPath string, batchSize int) ([2][]matrix.Matrix[float64], [2][]matrix.Matrix[float64], error) {
	train := [2][]matrix.Matrix[float64]{}
	test := [2][]matrix.Matrix[float64]{}

	labels, data, err := readAndBatchCSV(trainPath, batchSize)
	if err != nil {
		return train, test, err
	}
	X_train, Y_train := translateToMatrix(labels, data)

	labels, data, err = readAndBatchCSV(testPath, batchSize)
	if err != nil {
		return train, test, err
	}
	X_test, Y_test := translateToMatrix(labels, data)

	train[0], train[1] = X_train, Y_train
	test[0], test[1] = X_test, Y_test

	return train, test, nil
}
