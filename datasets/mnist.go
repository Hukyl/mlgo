package datasets

import "github.com/Hukyl/mlgo/matrix"

func translateToMatrix(labels []float64, data [][]float64) (matrix.Matrix[float64], matrix.Matrix[float64]) {
	matrixLabels := matrix.NewZeroMatrix[float64](len(labels), 10)
	for i, label := range labels {
		matrixLabels.Set(i, int(label), 1)
	}
	matrixData, _ := matrix.NewMatrix(data)
	return matrixData, matrixLabels
}

// MnistDataset reads the CSV files for MNIST dataset, in the form:
// [2]{X_train, Y_train}, [2]{X_test, Y_test}, err
func MnistDataset(trainPath, testPath string) ([2]matrix.Matrix[float64], [2]matrix.Matrix[float64], error) {
	train := [2]matrix.Matrix[float64]{}
	test := [2]matrix.Matrix[float64]{}

	labels, data, err := ReadMnistCSV(trainPath)
	if err != nil {
		return train, test, err
	}
	X_train, Y_train := translateToMatrix(labels, data)

	labels, data, err = ReadMnistCSV(testPath)
	if err != nil {
		return train, test, err
	}
	X_test, Y_test := translateToMatrix(labels, data)

	train[0], train[1] = X_train, Y_train
	test[0], test[1] = X_test, Y_test

	return train, test, nil
}
