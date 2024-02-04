package loss

import (
	"fmt"

	matrix "github.com/Hukyl/mlgo/matrix"
	. "golang.org/x/exp/constraints"
)

type LossFunction[T Float] interface {
	Apply(y T, yHat T) T
	ApplyMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T]

	ApplyDerivative(y T, yHat T) T
	ApplyDerivativeMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T]
}

func DynamicLoss[T Float](lossName string) (LossFunction[T], error) {
	var lossMap = map[string]func() LossFunction[T]{
		"SquareLoss":                  func() LossFunction[T] { return SquareLoss[T]{} },
		"LogLoss":                     func() LossFunction[T] { return LogLoss[T]{} },
		"CategoricalCrossEntropyLoss": func() LossFunction[T] { return CategoricalCrossEntropyLoss[T]{} },
		"CCELossWithSoftmax":          func() LossFunction[T] { return CCELossWithSoftmax[T]{} },
	}
	f, ok := lossMap[lossName]
	if !ok {
		return nil, fmt.Errorf("unknown activation function: %s", lossName)
	}
	return f(), nil
}
