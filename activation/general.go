// Package activation containts the list of all
// activation functions for layers of the neural network.
//
// Activation fucnton are applied to the output of the layer (i.e. W*X + b)
// to enhance its prediction capability.
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
type ActivationFunction interface {
	Apply(float64) float64
	ApplyMatrix(Matrix[float64])

	Derivative(float64) float64
	DerivativeMatrix(Matrix[float64]) Matrix[float64]
}

var activationMap = map[string]func() ActivationFunction{
	"Linear":         func() ActivationFunction { return Linear{} },
	"Sigmoid":        func() ActivationFunction { return Sigmoid{} },
	"ReLU":           func() ActivationFunction { return ReLU{} },
	"SELU":           func() ActivationFunction { return SELU{} },
	"Softmax":        func() ActivationFunction { return Softmax{} },
	"SoftmaxWithCCE": func() ActivationFunction { return SoftmaxWithCCE{} },
}

// DynamicActivation returns the activation function based on the name.
// Identical to importing and initializing the activation function directly.
func DynamicActivation(activationName string) (ActivationFunction, error) {
	f, ok := activationMap[activationName]
	if !ok {
		return nil, fmt.Errorf("unknown activation function: %s", activationName)
	}
	return f(), nil
}
