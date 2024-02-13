package activation

import (
	"math"

	. "github.com/Hukyl/mlgo/matrix"
)

// Softmax is a probability-generative vector activation function, i.e.
// generates probabilities from the input matrix. The number of classes
// is the row count of the input matrix, and columns are treated as separate
// outputs of the batch.
//
//	Softmax(x)_j = exp(x_j) / sum(x_i, i ∈ [1, classCount])
//
// As Softmax is a *vector* function, Apply() and Derivative() methods return NaN.
//
// Due to implementation of the matrix class, and the fact that dSoftmax/dx returns
// a Jacobian matrix (i.e. a 3D-tensor for a list of vecotrs),
// the DerivativeMatrix() is not implemented and has to be overriden.
type Softmax struct{}

func (s Softmax) Apply(z float64) float64 {
	return math.NaN()
}

func (s Softmax) ApplyMatrix(M Matrix[float64]) {
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

func (s Softmax) Derivative(x float64) float64 {
	return math.NaN()
}

func (s Softmax) DerivativeMatrix(M Matrix[float64]) Matrix[float64] {
	panic("not implemented")
}

// SoftmaxWithCCE is an activation function, which is used only together with categorical
// cross-entropy loss. The main difference between this function and Softmax is that
// it does not produce a proper derivative, but instead relies fully on
// the derivative of categorical cross-entropy loss function.
//
// IMPORTANT: should be only used with CategoricalCrossEntropyLossWithSoftmax loss function!
//
// As SoftmaxWithCCE is a *vector* function, Apply() and Derivative() methods return NaN.
type SoftmaxWithCCE struct {
	Softmax
}

func (s SoftmaxWithCCE) DerivativeMatrix(M Matrix[float64]) Matrix[float64] {
	return NewOnesMatrix(M.RowCount(), M.ColumnCount())
}
