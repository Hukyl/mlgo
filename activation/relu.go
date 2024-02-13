package activation

import (
	"math"

	. "github.com/Hukyl/mlgo/matrix"
)

// ReLU, or Rectified Linear Unit, is an activation function which works similarly to
// the actual neuron, either suppresses the value completely, or propagates it further.
//
//	ReLU(x) = max(x, 0)
//	dReLU/dx = 1 if x >= 0 else 0
//
// Computation of derivative in such manner is mathematically incorrect, as ReLU is
// not differentiable at x = 0, but in practice either 1 or 0 (1 in case of this implementation)
// is used.
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
