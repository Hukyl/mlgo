package activation

import (
	"math"

	"github.com/Hukyl/mlgo/matrix"
)

type ReLU struct{}

func (r ReLU) Apply(z float64) float64 {
	return math.Max(z, 0)
}

func (r ReLU) ApplyMatrix(M matrix.Matrix[float64]) {
	matrix.ApplyByElement(M, r.Apply)
}

func (r ReLU) BackPropagate(z float64) float64 {
	return math.Max(z, 0) // This is a bit of simplification, as d(ReLU)/dx at x=0 is undefined.
}

func (r ReLU) BackPropagateMatrix(M matrix.Matrix[float64]) {
	matrix.ApplyByElement(M, r.BackPropagate)
}
