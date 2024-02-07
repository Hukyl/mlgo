package activation

import (
	"math"

	. "github.com/Hukyl/mlgo/matrix"
)

const lambda = 1.0507
const alpha = 1.6733

// Scaled Exponential Linear Units, or SELUs, are activation functions
// that induce self-normalizing properties.
type SELU struct{}

func (s SELU) Apply(x float64) float64 {
	if x >= 0 {
		return lambda * x
	} else {
		return lambda * alpha * (math.Exp(x) - 1)
	}
}

func (s SELU) ApplyMatrix(m Matrix[float64]) {
	ApplyByElement(m, s.Apply)
}

func (s SELU) Derivative(x float64) float64 {
	if x >= 0 {
		return lambda
	}
	return lambda * alpha * math.Exp(x)
}

func (s SELU) DerivativeMatrix(m Matrix[float64]) Matrix[float64] {
	result := m.DeepCopy()
	ApplyByElement(result, s.Derivative)
	return result
}

func (s SELU) BackPropagate(z float64) float64 {
	if z >= 0 {
		return lambda * z
	}
	return (z + lambda*alpha) * z
}

func (s SELU) BackPropagateMatrix(m Matrix[float64]) {
	ApplyByElement(m, s.BackPropagate)
}
