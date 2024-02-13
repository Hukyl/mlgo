// Package datasets comprises of several dataset file processing functions
// to present them in useful manner.
package datasets

import (
	"github.com/Hukyl/mlgo/matrix"
	. "golang.org/x/exp/constraints"
)

// BatchMatrix accepts a slice of inputs to the neural network, and produces
// a slice of matrices with row count equal to batchSize.
func BatchMatrix[T Signed | Float](input [][]T, batchSize int) []matrix.Matrix[T] {
	var result []matrix.Matrix[T]

	for i := 0; i < len(input); i += batchSize {
		end := i + batchSize
		if end > len(input) {
			end = len(input)
		}
		m, _ := matrix.NewMatrix(input[i:end])
		result = append(result, m)
	}

	return result
}

// OneHotEncode accepts a slice of labels, and produces slices of
// one-hot encoded values (i.e. slices with 0s, and 1 in ith position)
// of length classCount.
//
//	labels := []float64{0, 2, 1}
//	encoded := OneHotEncode(labels, 3) // [ [1, 0, 0], [0, 0, 1], [0, 1, 0] ]
func OneHotEncode(labels []float64, classCount int) [][]float64 {
	output := make([][]float64, len(labels))
	for i, v := range labels {
		vector := make([]float64, classCount)
		vector[int(v)] = 1
		output[i] = vector
	}
	return output
}
