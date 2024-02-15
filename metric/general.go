// Package metric provides a set of metrics to calculate the accuracy of ANN predictions.
package metric

import "github.com/Hukyl/mlgo/matrix"

const DefaultEpsilon = 1e-5

// Metric is an interface for calculating accuracies for the neural network output.
//
// Calculate accepts two matrices, of size {outputSize, sampleCount} to produce a single accuracy
// result. If sampleCount > 1, the accuracy between different samples is averaged.
//
// Example:
//
//	yTrue, _ := matrix.NewMatrix([][]float64{{2.3, 2.4}, {0, 1}})
//	yTrue = yTrue.T()
//	yHat, _ := matrix.NewMatrix([][]float64{{1.8, 2.5}, {0, 1}})
//	yHat = yHat.T()
//	var metric Metric
//	fmt.Println(metric.Calculate(yTrue, yHat)) // 0.5
type Metric interface {
	Calculate(yTrue, yHat matrix.Matrix[float64]) float64
}
