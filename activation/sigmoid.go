package activation

import (
	"math"

	"github.com/Hukyl/mlgo/matrix"
)

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

func (s Sigmoid) BackPropagate(z float64) float64 {
	return z * (1 - z) // d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z))
}

func (s Sigmoid) BackPropagateMatrix(M matrix.Matrix[float64]) {
	matrix.ApplyByElement(M, s.BackPropagate)
}
