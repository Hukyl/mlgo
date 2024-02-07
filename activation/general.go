package activation

import (
	"fmt"

	. "github.com/Hukyl/mlgo/matrix"
)

// ActivationFunction is the interface for the output for the entire layer in
// a neural network.
//
// Apply applies the function to a scalar. As some activation functions are vector
// functions, this method may return NaN as a placeholder.
//
// ApplyMatrix applies the function to a nx1 matrix. Depending on the function,
// the function can be applied individually to each element, which would be
// equivalent to calling Apply() to each element, or apply to the whole matrix,
// using multiple elements to compute the value.
// This method modifies the given matrix in-place, which may result in change
// of dimensions of the matrix.
//
// Derivative produces a derivative with respect to the input of the activation function.
//
// DerivativeMatrix produces a derivative matrix. As some
// activation functions are vector functions, this function may use the whole matrix.
//
// BackPropagate uses the computed function output (e.g. result of Apply()) to
// produce the first derivative with respect to the input. As some
// activation functions are vector functions, this method may return NaN as a placeholder.
//
// BackPropagateMatrix calculates derivatives with respect to the activation function
// for nx1 matrix using computed function output (e.g. result of ApplyMatrix()).
// Depending on the function, the derivative may be applied individually
// to each element, which would be equivalent to calling BackPropagate() to each element,
// or apply to the whole matrix, using multiple elements to compute the value.
// This method modifies the given matrix in-place, which may result in change
// of dimensions of the matrix.
//
// IMPORTANT: BackPropagate and BackPropagateMatrix actually modify the data given,
// and the output does not get multiplied by any matrix.
// Therefore, Linear activation function BackPropagate(z) would return z instead of 1,
// as the linear function does not modify it in any way.
type ActivationFunction interface {
	Apply(float64) float64
	ApplyMatrix(Matrix[float64])

	Derivative(float64) float64
	DerivativeMatrix(Matrix[float64]) Matrix[float64]

	BackPropagate(float64) float64
	BackPropagateMatrix(Matrix[float64])
}

var activationMap = map[string]func() ActivationFunction{
	"Linear":         func() ActivationFunction { return Linear{} },
	"Sigmoid":        func() ActivationFunction { return Sigmoid{} },
	"ReLU":           func() ActivationFunction { return ReLU{} },
	"Softmax":        func() ActivationFunction { return Softmax{} },
	"SoftmaxWithCCE": func() ActivationFunction { return SoftmaxWithCCE{} },
}

func DynamicActivation(activationName string) (ActivationFunction, error) {
	f, ok := activationMap[activationName]
	if !ok {
		return nil, fmt.Errorf("unknown activation function: %s", activationName)
	}
	return f(), nil
}
