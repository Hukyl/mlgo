package activation_test

import (
	"math"
	"testing"

	"github.com/Hukyl/mlgo/activation"
	"github.com/Hukyl/mlgo/matrix"
)

func TestSoftmaxApplyMatrix(t *testing.T) {
	testCases := []struct {
		desc  string
		input [][]float64
		want  [][]float64
	}{
		{
			desc:  "simple-single-column",
			input: [][]float64{{1.0}, {2.0}, {3.0}},
			want: [][]float64{
				{math.Exp(1) / (math.Exp(1) + math.Exp(2) + math.Exp(3))},
				{math.Exp(2) / (math.Exp(1) + math.Exp(2) + math.Exp(3))},
				{math.Exp(3) / (math.Exp(1) + math.Exp(2) + math.Exp(3))},
			},
		},
		{
			desc:  "two-columns",
			input: [][]float64{{1.0, 0.0}, {0.0, 1.0}},
			want: [][]float64{
				{math.Exp(1) / (math.Exp(1) + 1), 1 / (1 + math.Exp(1))},
				{1 / (math.Exp(1) + 1), math.Exp(1) / (1 + math.Exp(1))},
			},
		},
	}

	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			// Arrange
			m, _ := matrix.NewMatrix(tC.input)
			wantM, _ := matrix.NewMatrix(tC.want)

			// Act
			activation.Softmax{}.ApplyMatrix(m)

			// Assert
			for i := 0; i < m.RowCount(); i++ {
				for j := 0; j < m.ColumnCount(); j++ {
					got, _ := m.At(i, j)
					want, _ := wantM.At(i, j)
					if math.Abs(got-want) > 1e-10 {
						t.Errorf("At(%d,%d): got %v, want %v", i, j, got, want)
					}
				}
			}
		})
	}
}

func TestSoftmaxApplyMatrix_SumsToOne(t *testing.T) {
	// Arrange
	input := [][]float64{{1.0, -2.0}, {2.0, 0.5}, {3.0, 1.0}}
	m, _ := matrix.NewMatrix(input)

	// Act
	activation.Softmax{}.ApplyMatrix(m)

	// Assert
	for j := 0; j < m.ColumnCount(); j++ {
		sum := 0.0
		for i := 0; i < m.RowCount(); i++ {
			v, _ := m.At(i, j)
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("column %d sum = %v, want 1.0", j, sum)
		}
	}
}

func TestSoftmaxApplyMatrix_LargeValues(t *testing.T) {
	// Arrange
	input := [][]float64{{1000.0}, {1001.0}, {1002.0}}
	m, _ := matrix.NewMatrix(input)

	// Act
	activation.Softmax{}.ApplyMatrix(m)

	// Assert
	sum := 0.0
	for i := 0; i < m.RowCount(); i++ {
		v, _ := m.At(i, 0)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("At(%d,0) = %v, expected finite value", i, v)
		}
		if v < 0 || v > 1 {
			t.Errorf("At(%d,0) = %v, expected value in [0,1]", i, v)
		}
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("sum = %v, want 1.0", sum)
	}
}

func TestSoftmaxApplyMatrix_NegativeLargeValues(t *testing.T) {
	// Arrange
	input := [][]float64{{-1000.0}, {-999.0}, {-998.0}}
	m, _ := matrix.NewMatrix(input)

	// Act
	activation.Softmax{}.ApplyMatrix(m)

	// Assert
	sum := 0.0
	for i := 0; i < m.RowCount(); i++ {
		v, _ := m.At(i, 0)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("At(%d,0) = %v, expected finite value", i, v)
		}
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("sum = %v, want 1.0", sum)
	}
}

func TestSoftmaxWithCCEApplyMatrix_LargeValues(t *testing.T) {
	// Arrange
	input := [][]float64{{500.0}, {501.0}, {502.0}}
	m, _ := matrix.NewMatrix(input)

	// Act
	activation.SoftmaxWithCCE{}.ApplyMatrix(m)

	// Assert
	for i := 0; i < m.RowCount(); i++ {
		v, _ := m.At(i, 0)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("At(%d,0) = %v, expected finite value", i, v)
		}
	}
}
