package nn_test

import (
	"math"
	"testing"

	"github.com/Hukyl/mlgo/activation"
	"github.com/Hukyl/mlgo/loss"
	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/nn"
	"github.com/Hukyl/mlgo/nn/layers"
)

func TestComputeCost_SquareLoss(t *testing.T) {
	testCases := []struct {
		desc string
		yHat [][]float64
		y    [][]float64
		want float64
	}{
		{
			desc: "single-sample",
			yHat: [][]float64{{0.5}},
			y:    [][]float64{{1.0}},
			want: 0.125, // 0.5 * (1.0 - 0.5)^2 = 0.125
		},
		{
			desc: "two-samples",
			yHat: [][]float64{{0.5, 0.8}},
			y:    [][]float64{{1.0, 0.0}},
			// sample 1: 0.5*(1.0-0.5)^2 = 0.125
			// sample 2: 0.5*(0.0-0.8)^2 = 0.32
			// avg = (0.125 + 0.32) / 2 = 0.2225
			want: 0.2225,
		},
		{
			desc: "multi-row-two-samples",
			yHat: [][]float64{{0.5, 0.2}, {0.3, 0.7}},
			y:    [][]float64{{1.0, 0.0}, {0.0, 1.0}},
			// sample 1: 0.5*(1.0-0.5)^2 + 0.5*(0.0-0.3)^2 = 0.125 + 0.045 = 0.17
			// sample 2: 0.5*(0.0-0.2)^2 + 0.5*(1.0-0.7)^2 = 0.02 + 0.045 = 0.065
			// avg = (0.17 + 0.065) / 2 = 0.1175
			want: 0.1175,
		},
	}

	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			// Arrange
			W, _ := matrix.NewMatrix([][]float64{{1.0}})
			b := matrix.NewZeroMatrix[float64](1, 1)
			layer, _ := layers.NewDense(W, b, activation.Linear{})

			model := nn.NewNeuralNetwork(
				[]layers.Layer{layer},
				loss.SquareLoss[float64]{},
			)

			yHatM, _ := matrix.NewMatrix(tC.yHat)
			yM, _ := matrix.NewMatrix(tC.y)

			// Act
			got := model.ComputeCost(yHatM, yM)

			// Assert
			if math.Abs(got-tC.want) > 1e-10 {
				t.Errorf("ComputeCost = %v, want %v", got, tC.want)
			}
		})
	}
}

func TestComputeCost_CCELoss(t *testing.T) {
	// Arrange
	W, _ := matrix.NewMatrix([][]float64{{1.0}})
	b := matrix.NewZeroMatrix[float64](1, 1)
	layer, _ := layers.NewDense(W, b, activation.Linear{})

	model := nn.NewNeuralNetwork(
		[]layers.Layer{layer},
		loss.CategoricalCrossEntropyLoss[float64]{Epsilon: 1e-7},
	)

	// Two samples with 3 classes
	yHat, _ := matrix.NewMatrix([][]float64{
		{0.7, 0.1},
		{0.2, 0.8},
		{0.1, 0.1},
	})
	y, _ := matrix.NewMatrix([][]float64{
		{1.0, 0.0},
		{0.0, 1.0},
		{0.0, 0.0},
	})

	// Act
	got := model.ComputeCost(yHat, y)

	// Assert
	// sample 1: -1.0*log(0.7) = 0.35667...
	// sample 2: -1.0*log(0.8) = 0.22314...
	// avg = (0.35667 + 0.22314) / 2 = 0.28991...
	want := (-math.Log(0.7) + -math.Log(0.8)) / 2
	if math.Abs(got-want) > 1e-5 {
		t.Errorf("ComputeCost = %v, want %v", got, want)
	}
}
