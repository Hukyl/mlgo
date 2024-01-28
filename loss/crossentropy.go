package loss

import (
	"math"

	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/utils"
)

type CategoricalCrossEntropyLoss[T utils.Float] struct{}

func (l CategoricalCrossEntropyLoss[T]) Apply(y, yHat T) T {
	return -y * T(math.Log(float64(yHat)))
}

func (l CategoricalCrossEntropyLoss[T]) ApplyDerivative(y, yHat T) T {
	return yHat - y
}

func (l CategoricalCrossEntropyLoss[T]) ApplyDerivativeMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	denominator := yHat.MultiplyByScalar(-1)
	matrix.ApplyByElement(denominator, func(x T) T { return 1 / x })
	result, _ := y.MultiplyElementwise(denominator)
	return result
}

/************************************************************************/

// CategoricalCrossEntropyLossWithSoftmax is a loss function which is used to determine
// the amount of error the weights should be corrected by, i.e. the cost.
//
// As CCE is often combined with Softmax, used in the last layer, this loss function
// combines the effect of the two to efficiently compute the derivative.
//
// IMPORTANT: should be only used with SoftmaxWithCCE activation function!
type CCELossWithSoftmax[T utils.Float] struct {
	CategoricalCrossEntropyLoss[T]
}

func (l CCELossWithSoftmax[T]) ApplyDerivativeMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	result, _ := yHat.Add(y.MultiplyByScalar(-1))
	return result
}

/************************************************************************/
