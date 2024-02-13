// Package loss contains loss functions for different ways of computing the degree of error
// between the ANN prediction and the actual labels to the data.
package loss

import (
	"fmt"

	. "github.com/Hukyl/mlgo/matrix"
	. "golang.org/x/exp/constraints"
)

// LossFunction is a general interface for all loss functions for the final output of ANN.
//
// Apply applies the loss function on scalars, which is in fact a respective cost function.
//
// ApplyMatrix applies the loss function to the prediction and label matrices.
// Separate columns are treated as separate outputs in the batch, and therefore for
// a batch a 1xN matrix is produced.
//
// ApplyDerivative applies derivative function with respect to the ANN prediction.
//
// ApplyDerivativeMatrix applies the derivative w.r.t. ANN prediction. Separate columns
// are treated as separate outputs of the ANN for the batch.
type LossFunction[T Float] interface {
	Apply(y T, yHat T) T
	ApplyMatrix(y Matrix[T], yHat Matrix[T]) Matrix[T]

	ApplyDerivative(y T, yHat T) T
	ApplyDerivativeMatrix(y Matrix[T], yHat Matrix[T]) Matrix[T]
}

// DynamicLoss returns a loss function by fully corresponding name.
// Identical to importing and initializing the function directly.
func DynamicLoss[T Float](lossName string) (LossFunction[T], error) {
	var f LossFunction[T]
	switch lossName {
	case "SquareLoss":
		f = SquareLoss[T]{}
	case "LogLoss":
		f = LogLoss[T]{}
	case "CategoricalCrossEntropyLoss":
		f = CategoricalCrossEntropyLoss[T]{}
	case "CCELossWithSoftmax":
		f = CCELossWithSoftmax[T]{}
	default:
		return nil, fmt.Errorf("unknown activation function: %s", lossName)
	}
	return f, nil
}
