package nn_test

import (
	"math"
	"testing"

	"github.com/Hukyl/mlgo/activation"
	"github.com/Hukyl/mlgo/loss"
	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/nn"
)

func relativelyEqual(m1, m2 matrix.Matrix[float64]) bool {
	if m1.Size() != m2.Size() {
		return false
	}
	for i := 0; i < m1.RowCount(); i++ {
		for j := 0; j < m1.ColumnCount(); j++ {
			m1Item, _ := m1.At(i, j)
			m2Item, _ := m2.At(i, j)
			if math.Abs(m1Item-m2Item) > 1e-4 {
				return false
			}
		}
	}
	return true
}

func TestTrain(t *testing.T) {
	TrainX, _ := matrix.NewMatrix([][]float64{
		{1, 5},
		{2, 3},
		{3, -1},
		{6, 4},
		{10, 10},
		{11, 5},
	})
	TrainX = TrainX.T()
	TrainY, _ := matrix.NewMatrix([][]float64{{2, 2, 4.5, 4, 8, 20}})

	model := nn.NewRandomNeuralNetwork(
		[]int{2, 1},
		[]activation.ActivationFunction{activation.Linear{}},
		loss.SquareLoss[float64]{},
	)
	parameters := nn.NeuralNetworkParameters{
		LearningRate:   0.0005,
		WeightDecay:    0,
		IterationCount: 100_000,
	}
	model.Train(TrainX, TrainY, parameters)

	X, _ := matrix.NewMatrix([][]float64{{6, 1}})
	X = X.T()
	got := model.Predict(X)

	want, _ := matrix.NewMatrix([][]float64{{9.77306421}})
	if !relativelyEqual(got, want) {
		t.Fatalf("%s != %s", got, want)
	}
}
