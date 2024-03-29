package activation

import (
	"math"

	"github.com/Hukyl/mlgo/matrix"
)

// Sigmoid is a continuous non-linear activation function which maps
// the rational numbers to [0;1] range.
//
//	Sigmoid(x) = 1 / (1 + exp(-x))
//	dSigmoid/dx = Sigmoid(x) * (1 - Sigmoid(x))
type Sigmoid struct{}

func (s Sigmoid) Apply(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func (s Sigmoid) ApplyMatrix(M matrix.Matrix[float64]) {
	matrix.ApplyByElement(M, s.Apply)
}

func (s Sigmoid) Derivative(x float64) float64 {
	sigm := s.Apply(x)
	return sigm * (1 - sigm)
}

func (s Sigmoid) DerivativeMatrix(M matrix.Matrix[float64]) matrix.Matrix[float64] {
	result := M.DeepCopy()
	matrix.ApplyByElement(result, s.Derivative)
	return result
}
