package loss

import (
	"math"

	"github.com/Hukyl/mlgo/matrix"
	. "golang.org/x/exp/constraints"
)

type LogLoss[T Float] struct{}

func (l LogLoss[T]) Apply(y, yHat T) T {
	return -y*T(math.Log(float64(yHat))) - (1-y)*T(math.Log(1-float64(yHat)))
}

func (l LogLoss[T]) ApplyMatrix(y matrix.Matrix[T], yHat matrix.Matrix[T]) matrix.Matrix[T] {
	ln_yHat := yHat.DeepCopy() // Log(yHat)
	matrix.ApplyByElement(ln_yHat, func(t T) T { return T(math.Log(float64(t))) })

	ln_1_yHat := yHat.DeepCopy() // Log(1 - yHat)
	matrix.ApplyByElement(ln_1_yHat, func(t T) T { return T(math.Log(1 - float64(t))) })

	p1, _ := y.MultiplyElementwise(ln_yHat) // y * Log(yHat)
	p2 := y.DeepCopy()
	matrix.ApplyByElement(p2, func(t T) T { return 1 - t })
	p2, _ = p2.MultiplyElementwise(ln_1_yHat) // (1-y) * Log(1 - yHat)

	result, _ := p1.Add(p2)
	return result.MultiplyByScalar(-1) // -y*Log(yHat) - (1-y)*Log(1-yHat)

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
