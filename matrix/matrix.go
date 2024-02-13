// Package matrix provides Matrix interface and matrix implementation
// along with a few util functions.
package matrix

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"

	. "golang.org/x/exp/constraints"
)

// Matrix is an interface for matrix implementations.
//
// Implementations have to implement Stringer and Marshaler/Unmarshaler.
//
// Size returns the dimensions of the matrix in the format [2]{rows, columns}
//
// RowCount returns the row number (i.e. Size()[0])
//
// ColumnCount returns the column number (i.e. Size()[1])
//
// AreSameSize compares Size() of two matrices by dimension.
//
// Broadcast broadcasts the matrix to the desired size. This is implemented
// by copying the values in existing rows and columns to fit the new.
//
//	var M Matrix[int]  // let it hold 1x1 matrix with value 3.
//	fmt.Println(M.Broadcast(2, 3))  // [ [3, 3, 3], [3, 3, 3] ]
//
// Equals compares to matrices by exact value. Returns false on incompatible
// matrices sizes. Depending on the implementation, comparisons can be exact or
// using some epsilon error.
//
// At gets the value at given indices. On the counterpart to math notation,
// indexing starts at 0.
//
// Set sets the value at given indices. Indexing starts at 0.
//
// Add adds two matrices in math manner, elementwise. Returns an error
// if the matrices are non-conformable.
//
// AddScalar add a scalar to each element of the matrix. Equivalent to
// Add(M.Broadcast(rows, columns)), where M is a 1x1 matrix with given scalar.
//
// Multiply multiplies two matrices in the math manner. If the matrices are
// non-conformable (columns_1 != rows_2), returns an error.
//
// MultiplyByScalar multiples the matrix by a scalar, i.e. multiply each
// element by a given scalar.
//
// MultiplyElementwise multiplies two matrices elementwise. If matrices
// are not of the same size, returns an error.
//
// T transposes the matrix by the main diagonal.
//
// Minor returns the minor from a certain element, returning
// (rows-1)x(columns-1) matrix. Returns error if invalid indices.
//
// Determinant returns the determinant of the matrix.
//
// Inverse returns the inverse matrix for a given matrix.
// Only possible for square matrices. If the determinant is equal
// to zero, returns an error.
//
// Copy and Deepcopy copy the matrix to a new location.
type Matrix[T Signed | Float] interface {
	json.Marshaler
	json.Unmarshaler
	fmt.Stringer

	Size() [2]int
	RowCount() int
	ColumnCount() int
	AreSameSize(Matrix[T]) bool
	Broadcast(int, int) error

	Equals(Matrix[T]) bool

	At(int, int) (T, error)
	Set(int, int, T) error

	Add(Matrix[T]) (Matrix[T], error)
	AddScalar(T) Matrix[T]
	Multiply(Matrix[T]) (Matrix[T], error)
	MultiplyByScalar(T) Matrix[T]
	MultiplyElementwise(Matrix[T]) (Matrix[T], error)

	T() Matrix[T]

	Minor(int, int) (Matrix[T], error)
	Determinant() (T, error)
	Inverse() (Matrix[T], error)

	Copy() Matrix[T]
	DeepCopy() Matrix[T]
}

type matrix[T Signed | Float] struct {
	data [][]T
}

/************************************************************************/

func (m *matrix[T]) RowCount() int { return len(m.data) }

func (m *matrix[T]) ColumnCount() int {
	if len(m.data) > 0 {
		return len(m.data[0])
	}
	return 0
}

func (m1 *matrix[T]) Size() [2]int {
	return [2]int{m1.RowCount(), m1.ColumnCount()}
}

func (m1 *matrix[T]) AreSameSize(m2 Matrix[T]) bool {
	otherSize := m2.Size()
	return m1.RowCount() == otherSize[0] && m1.ColumnCount() == otherSize[1]
}

func (m1 *matrix[T]) inRange(i, j int) bool {
	return 0 <= i && i < m1.RowCount() && 0 <= j && j < m1.ColumnCount()
}

func (m *matrix[T]) Broadcast(newRows, newColumns int) error {
	rows := m.RowCount()
	columns := m.ColumnCount()
	if newRows < rows || newColumns < columns || newRows%rows != 0 || newColumns%columns != 0 {
		return errors.New("invalid broadcast size (must be scalable by a positive factor)")
	}
	for i := 0; i < rows; i++ {
		m.data[i] = slices.Grow(m.data[i], newColumns-columns)
		for j := 0; j < newColumns/columns-1; j++ {
			m.data[i] = append(m.data[i], m.data[i][:columns]...)
		}
	}
	m.data = slices.Grow(m.data, newRows-rows)
	for i := 0; i < newRows-rows; i++ {
		m.data = append(m.data, make([]T, newColumns))
	}
	for i := rows; i < newRows; i += rows {
		for j := 0; j < rows; j++ {
			copy(m.data[i+j], m.data[j])
		}
	}
	return nil
}

/************************************************************************/

func (m1 *matrix[T]) Equals(m2 Matrix[T]) bool {
	if !m1.AreSameSize(m2) {
		return false
	}
	for i := 0; i < m1.RowCount(); i++ {
		for j := 0; j < m1.ColumnCount(); j++ {
			m1Item, _ := m1.At(i, j)
			m2Item, _ := m2.At(i, j)
			if m1Item != m2Item {
				return false
			}
		}
	}
	return true
}

/************************************************************************/

func (m1 *matrix[T]) At(i, j int) (T, error) {
	if !m1.inRange(i, j) {
		return 0, errors.New("indices are not in range")
	}
	return m1.data[i][j], nil
}

func (m1 *matrix[T]) Set(i, j int, value T) error {
	if !m1.inRange(i, j) {
		return errors.New("indices are not in range")
	}
	m1.data[i][j] = value
	return nil
}

/************************************************************************/

func (m1 *matrix[T]) Add(m2 Matrix[T]) (Matrix[T], error) {
	if !m1.AreSameSize(m2) {
		return nil, errors.New("matrices are not the same size")
	}
	m3 := NewZeroMatrix[T](m1.RowCount(), m1.ColumnCount())

	wg := sync.WaitGroup{}
	wg.Add(m3.ColumnCount())
	for j := 0; j < m3.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < m3.RowCount(); i++ {
				m1ItemValue, _ := m1.At(i, j)
				m2ItemValue, _ := m2.At(i, j)
				m3.Set(i, j, m1ItemValue+m2ItemValue)
			}
		}(j)
	}
	wg.Wait()

	return m3, nil
}

func (m1 *matrix[T]) AddScalar(k T) Matrix[T] {
	m3 := NewZeroMatrix[T](m1.RowCount(), m1.ColumnCount())

	wg := sync.WaitGroup{}
	wg.Add(m3.ColumnCount())
	for j := 0; j < m3.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < m3.RowCount(); i++ {
				item, _ := m1.At(i, j)
				m3.Set(i, j, item+k)
			}
		}(j)
	}
	wg.Wait()

	return m3
}

func (m1 *matrix[T]) Multiply(m2 Matrix[T]) (Matrix[T], error) {
	otherSize := m2.Size()
	if m1.ColumnCount() != otherSize[0] {
		return nil, errors.New("matrices are not conformable under multiplication")
	}
	m3 := NewZeroMatrix[T](m1.RowCount(), otherSize[1])

	wg := sync.WaitGroup{}
	wg.Add(m3.ColumnCount())
	for j := 0; j < m3.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < m3.RowCount(); i++ {
				value := T(0)
				for k := 0; k < m1.ColumnCount(); k++ {
					m1Value, _ := m1.At(i, k)
					m2Value, _ := m2.At(k, j)
					value += m1Value * m2Value
				}
				m3.Set(i, j, value)
			}
		}(j)
	}
	wg.Wait()

	return m3, nil
}

func (m1 *matrix[T]) MultiplyByScalar(k T) Matrix[T] {
	m3 := NewZeroMatrix[T](m1.RowCount(), m1.ColumnCount())

	wg := sync.WaitGroup{}
	wg.Add(m3.ColumnCount())
	for j := 0; j < m3.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < m3.RowCount(); i++ {
				item, _ := m1.At(i, j)
				m3.Set(i, j, item*k)
			}
		}(j)
	}
	wg.Wait()

	return m3
}

func (m1 *matrix[T]) MultiplyElementwise(m2 Matrix[T]) (Matrix[T], error) {
	if !m1.AreSameSize(m2) {
		return nil, errors.New("matrices are not the same size")
	}
	m3 := NewZeroMatrix[T](m1.RowCount(), m1.ColumnCount())

	wg := sync.WaitGroup{}
	wg.Add(m3.ColumnCount())
	for j := 0; j < m3.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < m3.RowCount(); i++ {
				m1Value, _ := m1.At(i, j)
				m2Value, _ := m2.At(i, j)
				m3.Set(i, j, m1Value*m2Value)
			}
		}(j)
	}
	wg.Wait()

	return m3, nil
}

/************************************************************************/

func (m *matrix[T]) T() Matrix[T] {
	result := NewZeroMatrix[T](m.ColumnCount(), m.RowCount())

	wg := sync.WaitGroup{}
	wg.Add(m.ColumnCount())
	for j := 0; j < m.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < m.RowCount(); i++ {
				item, _ := m.At(i, j)
				result.Set(j, i, item)
			}
		}(j)
	}
	wg.Wait()

	return result
}

/************************************************************************/

func (m *matrix[T]) Minor(i, j int) (Matrix[T], error) {
	if !m.inRange(i, j) {
		return nil, errors.New("indices are not in range")
	}
	result := NewZeroMatrix[T](m.RowCount()-1, m.ColumnCount()-1)
	var resultRow, resultColumn int
	for row := 0; row < m.RowCount(); row++ {
		resultRow = row
		if row == i {
			continue
		} else if row > i {
			resultRow--
		}
		for column := 0; column < m.ColumnCount(); column++ {
			resultColumn = column
			if column == j {
				continue
			} else if column > j {
				resultColumn--
			}
			value, _ := m.At(row, column)
			result.Set(resultRow, resultColumn, value)
		}
	}
	return result, nil
}

func (m *matrix[T]) Determinant() (T, error) {
	if m.RowCount() != m.ColumnCount() {
		return 0, errors.New("matrix is not square (n x n)")
	}
	if m.RowCount() == 1 {
		return m.At(0, 0)
	} else if m.RowCount() == 2 {
		a, _ := m.At(0, 0)
		b, _ := m.At(0, 1)
		c, _ := m.At(1, 0)
		d, _ := m.At(1, 1)
		return a*d - b*c, nil
	} else {
		det := T(0)
		for column := 0; column < m.ColumnCount(); column++ {
			minor, _ := m.Minor(0, column)
			minorDeterminant, _ := minor.Determinant()
			value, _ := m.At(0, column)
			det += T(math.Pow(-1, float64(column))) * value * minorDeterminant
		}
		return det, nil
	}
}

func (m *matrix[T]) Inverse() (Matrix[T], error) {
	if m.RowCount() != m.ColumnCount() {
		return nil, errors.New("matrix is not square (n x n)")
	}
	result := NewZeroMatrix[T](m.RowCount(), m.ColumnCount())
	var determinant T // calculate it here to reduce computations
	for i := 0; i < m.RowCount(); i++ {
		determinant = 0
		for j := 0; j < m.ColumnCount(); j++ {
			minor, _ := m.Minor(i, j)
			minorDeterminant, _ := minor.Determinant()
			result.Set(i, j, T(math.Pow(-1, float64(i+j)))*minorDeterminant)

			value, _ := m.At(i, j)
			determinant += T(math.Pow(-1, float64(i+j))) * value * minorDeterminant
		}
	}
	if determinant == 0 {
		return nil, errors.New("matrix is singular")
	}
	return result.T().MultiplyByScalar(1 / determinant), nil
}

/************************************************************************/

func (m matrix[T]) String() string {
	return fmt.Sprint(m.data)
}

func (m matrix[T]) Copy() Matrix[T] {
	result, _ := NewMatrix[T](m.data)
	return result
}

func (m matrix[T]) DeepCopy() Matrix[T] {
	result := NewZeroMatrix[T](m.RowCount(), m.ColumnCount())

	wg := sync.WaitGroup{}
	wg.Add(result.ColumnCount())
	for j := 0; j < m.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < m.RowCount(); i++ {
				item, _ := m.At(i, j)
				result.Set(i, j, item)
			}
		}(j)
	}
	wg.Wait()

	return result
}

func (m *matrix[T]) MarshalJSON() ([]byte, error) {
	return json.Marshal(m.data)
}

func (m *matrix[T]) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &m.data); err != nil {
		return err
	}
	return nil
}
