package activation

import (
	"math"

	. "github.com/Hukyl/mlgo/matrix"
)

type ReLU struct{}

func (r ReLU) Apply(x float64) float64 {
	return math.Max(x, 0)
}

func (r ReLU) ApplyMatrix(M Matrix[float64]) {
	ApplyByElement(M, r.Apply)
}

func (r ReLU) Derivative(x float64) float64 {
	if x >= 0 {
		return 1.0
	}
	return 0.0
}

func (r ReLU) DerivativeMatrix(M Matrix[float64]) Matrix[float64] {
	result := M.DeepCopy()
	ApplyByElement(result, r.Derivative)
	return result
}
