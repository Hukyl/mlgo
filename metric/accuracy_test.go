package metric_test

import (
	"testing"

	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/metric"
)

func TestAccuracy(t *testing.T) {
	testCases := []struct {
		yTrue   [][]float64
		yHat    [][]float64
		want    float64
		epsilon float64
		desc    string
	}{
		{
			yTrue:   [][]float64{{0.04, 0.1}},
			yHat:    [][]float64{{0.04, 0.1}},
			want:    1.0,
			epsilon: 1e-20,
			desc:    "exact",
		},
		{
			yTrue:   [][]float64{{0.04, 0.1, 0.9, 2.4}},
			yHat:    [][]float64{{0.04000001, 0.099999999, 0.9, 2.0}},
			want:    0.75,
			epsilon: 1e-5,
			desc:    "small-margin",
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			yTrueM, _ := matrix.NewMatrix(tC.yTrue)
			yHatM, _ := matrix.NewMatrix(tC.yHat)
			got := metric.Accuracy{Epsilon: tC.epsilon}.Calculate(yTrueM, yHatM)
			if tC.want != got {
				t.Fail()
			}
		})
	}
}
