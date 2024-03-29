package loss

import (
	"math"

	. "github.com/Hukyl/mlgo/matrix"
	. "golang.org/x/exp/constraints"
)

// CategoricalCrossEntropyLoss is a loss for comparing two probability distributions.
//
//	CCE(pred, label) = -label*Log(pred)
type CategoricalCrossEntropyLoss[T Float] struct {
	Epsilon float64
}

func (l CategoricalCrossEntropyLoss[T]) Apply(y, yHat T) T {
	return -y * T(math.Log(float64(yHat)))
}

func (l CategoricalCrossEntropyLoss[T]) ApplyMatrix(y Matrix[T], yHat Matrix[T]) Matrix[T] {
	result := NewZeroMatrix[T](1, y.ColumnCount())
	for j := 0; j < y.ColumnCount(); j++ {
		sum := T(0)

		for i := 0; i < y.RowCount(); i++ {
			yValue, _ := y.At(i, j)
			yHatValue, _ := yHat.At(i, j)
			// In case yHatValue is close to 0
			sum += -yValue * T(math.Log(math.Max(
				float64(yHatValue),
				l.Epsilon/float64(y.ColumnCount()*10),
			)))
		}

		result.Set(0, j, sum)
	}
	return result
}

func (l CategoricalCrossEntropyLoss[T]) ApplyDerivative(y, yHat T) T {
	return -y * T(math.Log(math.Max(float64(yHat), l.Epsilon)))
}

func (l CategoricalCrossEntropyLoss[T]) ApplyDerivativeMatrix(y Matrix[T], yHat Matrix[T]) Matrix[T] {
	smoothedLabel := Clip(
		y,
		T(l.Epsilon)/T(y.ColumnCount()),
		1-T(l.Epsilon),
	)
	denominator := yHat.MultiplyByScalar(-1)
	ApplyByElement(denominator, func(x T) T { return 1 / x })
	result, _ := smoothedLabel.MultiplyElementwise(denominator)
	return result
}

// CCELossWithSoftmax is a loss function which is used to determine
// the amount of error the weights should be corrected by, i.e. the cost.
//
// As CCE is often combined with Softmax, used in the last layer, this loss function
// combines the effect of the two to efficiently compute the derivative.
//
// IMPORTANT: should be only used with SoftmaxWithCCE activation function!
type CCELossWithSoftmax[T Float] struct {
	CategoricalCrossEntropyLoss[T]
}

func (l CCELossWithSoftmax[T]) ApplyDerivativeMatrix(y Matrix[T], yHat Matrix[T]) Matrix[T] {
	smoothedLabel := Clip(
		y,
		T(l.Epsilon)/T(y.ColumnCount()),
		1-T(l.Epsilon),
	)
	result, _ := yHat.Add(smoothedLabel.MultiplyByScalar(-1))
	return result
}
