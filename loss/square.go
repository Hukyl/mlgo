package loss

import (
	. "github.com/Hukyl/mlgo/matrix"
	. "golang.org/x/exp/constraints"
)

// SquareLoss is a MSE loss function.
//
//	SquareLoss(pred, label) = 0.5 * (pred-label)**2
//	dSquareLoss/dpred = pred - label
//
// For convenience of producing the derivative, SquareLoss is multiplied by 1/2.
type SquareLoss[T Float] struct{}

func (s SquareLoss[T]) Apply(y, yHat T) T {
	return (y - yHat) * (y - yHat) / 2
}

func (s SquareLoss[T]) ApplyMatrix(y Matrix[T], yHat Matrix[T]) Matrix[T] {
	diff, _ := y.Add(yHat.MultiplyByScalar(-1))
	ApplyByElement(diff, func(t T) T { return t * t / 2 })
	return diff
}

func (s SquareLoss[T]) ApplyDerivative(y, yHat T) T {
	return yHat - y
}

func (s SquareLoss[T]) ApplyDerivativeMatrix(y Matrix[T], yHat Matrix[T]) Matrix[T] {
	result, _ := yHat.Add(y.MultiplyByScalar(-1))
	return result
}
