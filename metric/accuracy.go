package metric

import (
	"math"
	"sync"

	"github.com/Hukyl/mlgo/matrix"
)

// Accuracy compares the prediction to the actual values.
//
// If Epsilon is provided, the check is performed to some level of accuracy.
//
//	diff := math.Abs(pred - label)
//	if diff < Epsilon {
//		correct++
//	}
//
// Different columns are treated as different samples. The correct counter goes up
// only if all the values per one prediction are equal to the label.
type Accuracy struct {
	Epsilon float64
}

func (a Accuracy) Calculate(yTrue, yHat matrix.Matrix[float64]) float64 {
	correct := 0
	precision := a.Epsilon
	if precision == 0.0 {
		precision = DefaultEpsilon
	}

	m := sync.Mutex{}
	wg := sync.WaitGroup{}
	wg.Add(yTrue.ColumnCount())
	for j := 0; j < yTrue.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < yTrue.RowCount(); i++ {
				yTrueV, _ := yTrue.At(i, j)
				yHatV, _ := yHat.At(i, j)
				if math.Abs(yHatV-yTrueV) > precision {
					return
				}
			}
			m.Lock()
			correct++
			m.Unlock()
		}(j)
	}
	wg.Wait()

	return float64(correct) / float64(yTrue.ColumnCount())
}
