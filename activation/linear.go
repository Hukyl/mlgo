package activation

import (
	"github.com/Hukyl/mlgo/matrix"
)

type Linear struct{}

func (l Linear) Apply(z float64) float64 {
	return z
}

func (l Linear) ApplyMatrix(M matrix.Matrix[float64]) {}

func (l Linear) BackPropagate(z float64) float64 {
	return z // d/dz = 1
}

func (l Linear) BackPropagateMatrix(M matrix.Matrix[float64]) {}
