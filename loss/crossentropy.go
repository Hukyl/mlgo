package loss

import (
	"math"

	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/utils"
)

const clipValue = 1e-10

type CategoricalCrossEntropyLoss[T utils.Float] struct{}

func (l CategoricalCrossEntropyLoss[T]) Apply(y, yHat T) T {
	return -y * T(math.Log(float64(yHat)))
}

func (l CategoricalCrossEntropyLoss[T]) ApplyMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	result := matrix.NewZeroMatrix[T](1, y.ColumnCount())
	for j := 0; j < y.ColumnCount(); j++ {
		sum := T(0)

		for i := 0; i < y.RowCount(); i++ {
			yValue, _ := y.At(i, j)
			yHatValue, _ := yHat.At(i, j)
			sum += -yValue * T(math.Log(float64(yHatValue)))
		}

		result.Set(0, j, sum)
	}
	return result
}

func (l CategoricalCrossEntropyLoss[T]) ApplyDerivative(y, yHat T) T {
	return yHat - y
}

func (l CategoricalCrossEntropyLoss[T]) ApplyDerivativeMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	yClipped := matrix.Clip(y, clipValue, 1-clipValue)
	denominator := yHat.MultiplyByScalar(-1)
	matrix.ApplyByElement(denominator, func(x T) T { return 1 / x })
	result, _ := yClipped.MultiplyElementwise(denominator)
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
	result, _ := yHat.Add(matrix.Clip(y, clipValue, 1-clipValue).MultiplyByScalar(-1))
	return result
}

/************************************************************************/
