package matrix

import (
	"errors"
	"sync"

	"github.com/Hukyl/mlgo/utils"
)

func NewMatrix[T utils.Number](data [][]T) (Matrix[T], error) {
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

func NewZeroMatrix[T utils.Number](rowCount int, columnCount int) Matrix[T] {
	m := new(matrix[T])
	m.data = make([][]T, rowCount)
	for i := 0; i < rowCount; i++ {
		m.data[i] = make([]T, columnCount)
	}
	return m
}

func NewOnesMatrix(rowCount int, columnCount int) Matrix[float64] {
	m := NewZeroMatrix[float64](rowCount, columnCount)
	for i := 0; i < rowCount; i++ {
		for j := 0; j < columnCount; j++ {
			m.Set(i, j, 1)
		}
	}
	return m
}

func IdentityMatrix(rowCount int) Matrix[float64] {
	m := NewZeroMatrix[float64](rowCount, rowCount)
	for i := 0; i < rowCount; i++ {
		m.Set(i, i, 1)
	}
	return m
}

/****************************************************************************/

func ApplyByElement[T utils.Number](M Matrix[T], f func(T) T) {
	var value T
	for i := 0; i < M.RowCount(); i++ {
		for j := 0; j < M.ColumnCount(); j++ {
			value, _ = M.At(i, j)
			M.Set(i, j, f(value))
		}
	}
}

func Clip[T utils.Number](M Matrix[T], lower, upper T) Matrix[T] {
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
