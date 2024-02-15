package metric

import (
	"sync"

	"github.com/Hukyl/mlgo/matrix"
)

func oneHotEncodingToValues(m matrix.Matrix[float64]) []int {
	predictions := make([]int, m.ColumnCount())

	wg := sync.WaitGroup{}
	wg.Add(m.ColumnCount())
	for j := 0; j < m.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			largestIndex := 0
			largestValue, _ := m.At(0, j)

			for i := 0; i < m.RowCount(); i++ {
				value, _ := m.At(i, j)
				if value > largestValue {
					largestValue = value
					largestIndex = i
				}
			}
			predictions[j] = largestIndex
		}(j)
	}
	wg.Wait()

	return predictions
}

// CategoriacalAccuracy is a probabilistic accuracy metric, used to compare most probable
// guess of a neural network to the neural network output. True labels must be presented as
// one-hot encoded values.
//
// Example:
//
//	ca := metrics.CategoricalAccuracy{}
//	yTrue, _ := matrix.NewMatrix([][]float64{{1, 0, 0}, {0, 1, 0}})
//	yTrue = yTrue.T()
//	yHat, _ := matrix.NewMatrix([][]float64{{0.78, 0.20, 0.02}, {0.10, 0.11, 0.79}})
//	yHat = yHat.T()
//	fmt.Println(ca.Calculate(yTrue, yHat)) // 0.5
type CategoricalAccuracy struct{}

func (c CategoricalAccuracy) Calculate(yTrue, yHat matrix.Matrix[float64]) float64 {
	correct := 0

	predictions := oneHotEncodingToValues(yHat)
	trueValues := oneHotEncodingToValues(yTrue)

	for i := 0; i < len(predictions); i++ {
		if predictions[i] == trueValues[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(predictions))
}

// SparseCategoriacalAccuracy is a probabilistic accuracy metric, used to compare most probable
// guess of a neural network to the neural network output. The only difference from
// CategoriacalAccuracy is that true labels are presented as actual labels.
//
// Example:
//
//	ca := metrics.SparseCategoricalAccuracy{}
//	yTrue, _ := matrix.NewMatrix([][]float64{{0, 1}})
//	yHat, _ := matrix.NewMatrix([][]float64{{0.78, 0.20, 0.02}, {0.10, 0.11, 0.79}})
//	yHat = yHat.T()
//	fmt.Println(ca.Calculate(yTrue, yHat)) // 0.5
type SparseCategoricalAccuracy struct{}

func (s SparseCategoricalAccuracy) Calculate(yTrue, yHat matrix.Matrix[float64]) float64 {
	correct := 0

	predictions := oneHotEncodingToValues(yHat)

	for i := 0; i < len(predictions); i++ {
		yTrueI, _ := yTrue.At(0, i)
		if predictions[i] == int(yTrueI) {
			correct++
		}
	}

	return float64(correct) / float64(len(predictions))
}
