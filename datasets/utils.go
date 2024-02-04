package datasets

import (
	"github.com/Hukyl/mlgo/matrix"
	. "golang.org/x/exp/constraints"
)

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

func OneHotEncode(labels []float64) [][]float64 {
	output := make([][]float64, len(labels))
	for i, v := range labels {
		vector := make([]float64, 10)
		vector[int(v)] = 1
		output[i] = vector
	}
	return output
}
