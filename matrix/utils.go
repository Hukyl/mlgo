package matrix

import (
	"errors"
	"sync"

	. "golang.org/x/exp/constraints"
)

// NewMatrix returns a matrix implementation using the slice data.
//
// if the data is empty, or column count is incosistnent, returns an error,
func NewMatrix[T Signed | Float](data [][]T) (Matrix[T], error) {
	if len(data) == 0 {
		return nil, errors.New("at least one row")
	}
	for i := 1; i < len(data); i++ {
		if len(data[i]) != len(data[0]) {
			return nil, errors.New("incosistent column count")
		}
	}
	m := new(matrix[T])
	m.data = data
	return m, nil
}

// NewZeroMatrix returns a matrix implementation filled with zeros with required size.
func NewZeroMatrix[T Signed | Float](rowCount int, columnCount int) Matrix[T] {
	m := new(matrix[T])
	m.data = make([][]T, rowCount)
	for i := 0; i < rowCount; i++ {
		m.data[i] = make([]T, columnCount)
	}
	return m
}

// NewOnesMatrix returns a matrix implementation filled with ones with required size.
func NewOnesMatrix(rowCount int, columnCount int) Matrix[float64] {
	m := NewZeroMatrix[float64](rowCount, columnCount)
	for i := 0; i < rowCount; i++ {
		for j := 0; j < columnCount; j++ {
			m.Set(i, j, 1)
		}
	}
	return m
}

// IdentityMatrix returns a sqaure identity matrix implementation.
func IdentityMatrix(rowCount int) Matrix[float64] {
	m := NewZeroMatrix[float64](rowCount, rowCount)
	for i := 0; i < rowCount; i++ {
		m.Set(i, i, 1)
	}
	return m
}

/****************************************************************************/

// ApplyByElement applies some function elementwise to the matrix.
// This opeartion changes the matrix, so consider using Matrix.Deepcopy()
// depending on your needs.
func ApplyByElement[T Signed | Float](M Matrix[T], f func(T) T) {
	var value T
	for i := 0; i < M.RowCount(); i++ {
		for j := 0; j < M.ColumnCount(); j++ {
			value, _ = M.At(i, j)
			M.Set(i, j, f(value))
		}
	}
}

// Clip clips all the values in the matrix using lower and upper bound.
//
//	 if value > upper {
//		value = upper
//	 } else if value < lower {
//		value = lower
//	 }
func Clip[T Signed | Float](M Matrix[T], lower, upper T) Matrix[T] {
	result := M.DeepCopy()

	wg := sync.WaitGroup{}
	wg.Add(result.ColumnCount())
	for j := 0; j < M.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < M.RowCount(); i++ {
				v, _ := result.At(i, j)
				if v < lower {
					v = lower
				} else if v > upper {
					v = upper
				}
				result.Set(i, j, v)
			}
		}(j)
	}
	wg.Wait()

	return result
}
