package layers

import (
	"fmt"
	"sync"

	"github.com/Hukyl/mlgo/activation"
	. "github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/utils"
)

type dropout struct {
	inputSize int

	rate float64
}

func (d *dropout) String() string {
	return fmt.Sprintf("Dropout{rate: %v, input: %d}", d.rate, d.inputSize)
}

func (d *dropout) MarshalJSON() ([]byte, error) {
	return []byte{}, nil
}

func (d *dropout) UnmarshalJSON(_ []byte) error {
	return nil
}

func (d *dropout) InputSize() [2]int {
	return [2]int{d.inputSize, 1}
}

func (d *dropout) OutputSize() [2]int {
	return [2]int{d.inputSize, 1}
}

func (d *dropout) IsTraining() bool {
	return true
}

func (d *dropout) Weights() Matrix[float64] {
	return nil
}

func (d *dropout) Bias() Matrix[float64] {
	return nil
}

func (d *dropout) Activation() activation.ActivationFunction {
	return nil
}

func (d *dropout) ForwardPropagate(X Matrix[float64]) (Y Matrix[float64], err error) {
	keepProb := 1.0 - d.rate
	Y = uniformMatrix(X.Size(), 0.0, 1.0)

	wg := sync.WaitGroup{}
	wg.Add(Y.ColumnCount())
	for j := 0; j < Y.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < Y.RowCount(); i++ {
				prob, _ := Y.At(i, j)
				if prob < keepProb {
					v, _ := X.At(i, j)
					Y.Set(i, j, v)
				} else {
					Y.Set(i, j, 0.0)
				}
			}
		}(j)
	}
	wg.Wait()

	return Y, nil
}

func (d *dropout) BackPropagate(nextLayerPropagation Matrix[float64], output Matrix[float64]) [2]Matrix[float64] {
	return [2]Matrix[float64]{nextLayerPropagation, nextLayerPropagation}
}

func (d *dropout) UpdateWeights(_, _ Matrix[float64], _ utils.NeuralNetworkParameters) {}
