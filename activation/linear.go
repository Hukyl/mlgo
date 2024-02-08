package activation

import (
	. "github.com/Hukyl/mlgo/matrix"
)

type Linear struct{}

func (l Linear) Apply(x float64) float64 {
	return x
}

func (l Linear) ApplyMatrix(M Matrix[float64]) {}

func (l Linear) Derivative(x float64) float64 {
	return 1.0
}

func (l Linear) DerivativeMatrix(m Matrix[float64]) Matrix[float64] {
	return NewOnesMatrix(m.RowCount(), m.ColumnCount())
}
