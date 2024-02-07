package activation

import (
	"math"

	"github.com/Hukyl/mlgo/matrix"
)

// Softmax is a probability-generative vector function.
//
// As it is a *vector* function, Apply() and BackPropagate() methods return NaN
type Softmax struct{}

func (s Softmax) Apply(z float64) float64 {
	return math.NaN()
}

func (s Softmax) ApplyMatrix(M matrix.Matrix[float64]) {
	for j := 0; j < M.ColumnCount(); j++ {
		exponents := make([]float64, M.RowCount())
		sumExponents := float64(0.0)
		for i := 0; i < M.RowCount(); i++ {
			z, _ := M.At(i, j)
			exponents[i] = math.Exp(z)
			sumExponents += exponents[i]
		}
		for i := 0; i < M.RowCount(); i++ {
			M.Set(i, j, exponents[i]/sumExponents)
		}
	}
}

// FIXME
func (s Softmax) Derivative(x float64) float64 {
	return math.NaN()
}

func (s Softmax) DerivativeMatrix(M matrix.Matrix[float64]) matrix.Matrix[float64] {
	return M
}

//

func (s Softmax) BackPropagate(z float64) float64 {
	return math.NaN()
}

func (s Softmax) BackPropagateMatrix(M matrix.Matrix[float64]) {
	M.Broadcast(M.RowCount(), M.RowCount())
	for j := 0; j < M.ColumnCount(); j++ {
		s_j, _ := M.At(j, j)
		for i := 0; i < M.RowCount(); i++ {
			value, _ := M.At(i, j)
			if i == j {
				value *= 1 - s_j
			} else {
				value *= -s_j
			}
			M.Set(i, j, value)
		}
	}
}

// SoftmaxWithCCE is an activation function, which is used only together with categorical
// cross-entropy loss. The main difference between this function and Softmax is that
// it does not produce a proper back propagation derivative, but instead relies fully on
// categorical cross-entropy loss function.
//
// IMPORTANT: should be only used with CategoricalCrossEntropyLossWithSoftmax loss function!
//
// As it is a *vector* function, Apply() and BackPropagate() methods return NaN
type SoftmaxWithCCE struct {
	Softmax
}

func (s SoftmaxWithCCE) BackPropagateMatrix(M matrix.Matrix[float64]) {}
