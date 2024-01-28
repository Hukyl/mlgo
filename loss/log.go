package loss

import (
	"math"

	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/utils"
)

type LogLoss[T utils.Float] struct{}

func (l LogLoss[T]) Apply(y, yHat T) T {
	return -y*T(math.Log(float64(yHat))) - (1-y)*T(math.Log(1-float64(yHat)))
}

func (l LogLoss[T]) ApplyDerivative(y, yHat T) T {
	return (yHat - y) / (yHat - yHat*yHat)
}

func (l LogLoss[T]) ApplyDerivativeMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	yHatSquared, _ := yHat.MultiplyElementwise(yHat)
	denominator, _ := yHat.Add(yHatSquared.MultiplyByScalar(-1))
	matrix.ApplyByElement(denominator, func(x T) T { return 1 / x })
	numerator, _ := yHat.Add(y.MultiplyByScalar(-1))
	result, _ := numerator.MultiplyElementwise(denominator)
	return result // (yHat - y) / (yHat - yHat*yHat)
}
