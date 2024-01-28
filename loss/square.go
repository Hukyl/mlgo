package loss

import (
	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/utils"
)

type SquareLoss[T utils.Float] struct{}

func (s SquareLoss[T]) Apply(y, yHat T) T {
	return (y - yHat) * (y - yHat) / 2
}

func (s SquareLoss[T]) ApplyDerivative(y, yHat T) T {
	return yHat - y
}

func (s SquareLoss[T]) ApplyDerivativeMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	result, _ := yHat.Add(y.MultiplyByScalar(-1))
	return result
}
