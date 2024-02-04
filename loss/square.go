package loss

import (
	"github.com/Hukyl/mlgo/matrix"
	. "golang.org/x/exp/constraints"
)

type SquareLoss[T Float] struct{}

func (s SquareLoss[T]) Apply(y, yHat T) T {
	return (y - yHat) * (y - yHat) / 2
}

func (s SquareLoss[T]) ApplyMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	diff, _ := y.Add(yHat.MultiplyByScalar(-1))
	matrix.ApplyByElement(diff, func(t T) T { return t * t / 2 })
	return diff
}

func (s SquareLoss[T]) ApplyDerivative(y, yHat T) T {
	return yHat - y
}

func (s SquareLoss[T]) ApplyDerivativeMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	result, _ := yHat.Add(y.MultiplyByScalar(-1))
	return result
}
