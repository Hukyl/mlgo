package metric_test

import (
	"testing"

	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/metric"
)

func TestCategoricalAccuracy(t *testing.T) {
	testCases := []struct {
		yTrue [][]float64
		yHat  [][]float64
		want  float64
		desc  string
	}{
		{
			yTrue: [][]float64{{0, 1}, {1, 0}},
			yHat:  [][]float64{{0.04, 0.96}, {0.6, 0.4}},
			want:  1.0,
			desc:  "binary",
		},
		{
			yTrue: [][]float64{{0, 1, 0, 0}, {0, 0, 0, 1}},
			yHat:  [][]float64{{0.01, 0.8, 0.18, 0.01}, {0.9, 0.05, 0.025, 0.025}},
			want:  0.5,
			desc:  "quaternary",
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			yTrueM, _ := matrix.NewMatrix(tC.yTrue)
			yTrueM = yTrueM.T()
			yHatM, _ := matrix.NewMatrix(tC.yHat)
			yHatM = yHatM.T()
			got := metric.CategoricalAccuracy{}.Calculate(yTrueM, yHatM)
			if tC.want != got {
				t.Fail()
			}
		})
	}
}

func TestSparseCategoricalAccuracy(t *testing.T) {
	testCases := []struct {
		yTrue [][]float64
		yHat  [][]float64
		want  float64
		desc  string
	}{
		{
			yTrue: [][]float64{{1, 0}},
			yHat:  [][]float64{{0.04, 0.96}, {0.6, 0.4}},
			want:  1.0,
			desc:  "binary",
		},
		{
			yTrue: [][]float64{{1, 3}},
			yHat:  [][]float64{{0.01, 0.8, 0.18, 0.01}, {0.9, 0.05, 0.025, 0.025}},
			want:  0.5,
			desc:  "quaternary",
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			yTrueM, _ := matrix.NewMatrix(tC.yTrue)
			yHatM, _ := matrix.NewMatrix(tC.yHat)
			yHatM = yHatM.T()
			got := metric.SparseCategoricalAccuracy{}.Calculate(yTrueM, yHatM)
			if tC.want != got {
				t.Fail()
			}
		})
	}
}
