package layers_test

import (
	"math"
	"testing"

	"github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/nn/layers"
)

func TestDropout_MasksElements(t *testing.T) {
	// Arrange
	rate := 0.5
	d := layers.NewDropout(4, rate)
	input, _ := matrix.NewMatrix([][]float64{
		{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
		{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
	})

	// Act
	output, err := d.ForwardPropagate(input)
	if err != nil {
		t.Fatalf("ForwardPropagate error: %v", err)
	}

	// Assert
	zeroCount := 0
	totalCount := 0
	for i := 0; i < output[0].RowCount(); i++ {
		for j := 0; j < output[0].ColumnCount(); j++ {
			v, _ := output[0].At(i, j)
			if v == 0 {
				zeroCount++
			}
			totalCount++
		}
	}
	maskedRatio := float64(zeroCount) / float64(totalCount)
	if maskedRatio < 0.1 || maskedRatio > 0.9 {
		t.Errorf("masked ratio = %v, expected roughly %v", maskedRatio, rate)
	}
}

func TestDropout_ScalesSurvivingValues(t *testing.T) {
	// Arrange
	rate := 0.5
	keepProb := 1.0 - rate
	d := layers.NewDropout(4, rate)
	input, _ := matrix.NewMatrix([][]float64{
		{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
		{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
		{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
		{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
	})

	// Act
	output, err := d.ForwardPropagate(input)
	if err != nil {
		t.Fatalf("ForwardPropagate error: %v", err)
	}

	// Assert - surviving values should be scaled by 1/keepProb
	expectedScaled := 2.0 / keepProb
	for i := 0; i < output[0].RowCount(); i++ {
		for j := 0; j < output[0].ColumnCount(); j++ {
			v, _ := output[0].At(i, j)
			if v != 0.0 && math.Abs(v-expectedScaled) > 1e-10 {
				t.Errorf("At(%d,%d) = %v, want 0.0 or %v", i, j, v, expectedScaled)
			}
		}
	}
}

func TestDropout_PreservesExpectedValue(t *testing.T) {
	// Arrange
	rate := 0.3
	d := layers.NewDropout(100, rate)

	rows := make([][]float64, 100)
	for i := range rows {
		row := make([]float64, 1000)
		for j := range row {
			row[j] = 5.0
		}
		rows[i] = row
	}
	input, _ := matrix.NewMatrix(rows)

	// Act
	output, err := d.ForwardPropagate(input)
	if err != nil {
		t.Fatalf("ForwardPropagate error: %v", err)
	}

	// Assert - mean of output should be close to mean of input (5.0)
	sum := 0.0
	count := 0
	for i := 0; i < output[0].RowCount(); i++ {
		for j := 0; j < output[0].ColumnCount(); j++ {
			v, _ := output[0].At(i, j)
			sum += v
			count++
		}
	}
	mean := sum / float64(count)
	if math.Abs(mean-5.0) > 0.5 {
		t.Errorf("mean output = %v, expected close to 5.0 (inverted dropout scaling)", mean)
	}
}

func TestDropout_ZeroRate(t *testing.T) {
	// Arrange
	d := layers.NewDropout(3, 0.0)
	input, _ := matrix.NewMatrix([][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	})

	// Act
	output, err := d.ForwardPropagate(input)
	if err != nil {
		t.Fatalf("ForwardPropagate error: %v", err)
	}

	// Assert - with rate=0, all values should pass through unchanged
	for i := 0; i < output[0].RowCount(); i++ {
		for j := 0; j < output[0].ColumnCount(); j++ {
			got, _ := output[0].At(i, j)
			want, _ := input.At(i, j)
			if math.Abs(got-want) > 1e-10 {
				t.Errorf("At(%d,%d) = %v, want %v", i, j, got, want)
			}
		}
	}
}
