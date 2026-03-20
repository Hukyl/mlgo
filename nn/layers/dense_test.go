package layers_test

import (
	"math"
	"testing"

	"github.com/Hukyl/mlgo/activation"
	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/nn/layers"
	"github.com/Hukyl/mlgo/utils"
)

func TestDenseBackPropagate_UsesOldWeights(t *testing.T) {
	// Arrange
	W, _ := matrix.NewMatrix([][]float64{
		{0.5, 0.3},
		{0.2, 0.8},
	})
	b := matrix.NewZeroMatrix[float64](2, 1)
	layer, _ := layers.NewDense(W, b, activation.Linear{})

	input, _ := matrix.NewMatrix([][]float64{{1.0}, {1.0}})
	output := [2]matrix.Matrix[float64]{input, input}
	dLdA, _ := matrix.NewMatrix([][]float64{{1.0}, {1.0}})

	params := utils.NeuralNetworkParameters{
		EpochCount:          1,
		InitialLearningRate: 0.5,
	}
	params.Validate()

	// Compute expected propagation using OLD weights: W.T @ dLdZ
	// With Linear activation, dAdZ = ones, so dLdZ = dLdA * ones = dLdA
	// Expected = W_old.T @ dLdA = [[0.5, 0.2], [0.3, 0.8]].T @ [[1], [1]]
	//          = [[0.5, 0.2], [0.3, 0.8]].T @ [[1], [1]]
	//          = [[0.7], [1.1]]
	wantPropRow0 := 0.5 + 0.2
	wantPropRow1 := 0.3 + 0.8

	// Act
	result := layer.BackPropagate(dLdA, input, output, params)

	// Assert
	gotRow0, _ := result.At(0, 0)
	gotRow1, _ := result.At(1, 0)

	if math.Abs(gotRow0-wantPropRow0) > 1e-10 {
		t.Errorf("propagation[0] = %v, want %v (old weights used)", gotRow0, wantPropRow0)
	}
	if math.Abs(gotRow1-wantPropRow1) > 1e-10 {
		t.Errorf("propagation[1] = %v, want %v (old weights used)", gotRow1, wantPropRow1)
	}
}
