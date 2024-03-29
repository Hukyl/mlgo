package layers

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/Hukyl/mlgo/activation"
	. "github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/utils"
)

type dropout struct {
	inputSize int
	rate      float64
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

/***************************************************************************/

func (d *dropout) ForwardPropagate(X Matrix[float64]) (Y [2]Matrix[float64], err error) {
	keepProb := 1.0 - d.rate
	output := uniformMatrix(X.Size(), 0.0, 1.0)

	wg := sync.WaitGroup{}
	wg.Add(output.ColumnCount())
	for j := 0; j < output.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < d.inputSize; i++ {
				prob, _ := output.At(i, j)
				if prob < keepProb {
					v, _ := X.At(i, j)
					output.Set(i, j, v)
				} else {
					output.Set(i, j, 0.0)
				}
			}
		}(j)
	}
	wg.Wait()

	Y = [2]Matrix[float64]{output, output}

	return Y, nil
}

func (d *dropout) BackPropagate(nextLayerPropagation, X Matrix[float64], A [2]Matrix[float64], parameters utils.NeuralNetworkParameters) Matrix[float64] {
	return nextLayerPropagation
}

func (d *dropout) updateWeights(_, _ Matrix[float64], _ utils.NeuralNetworkParameters) {}

/***************************************************************************/

func (d *dropout) String() string {
	return fmt.Sprintf("Dropout{%[1]d -> %[1]d, rate: %v}", d.inputSize, d.rate)
}

func (d *dropout) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		InputSize int
		Rate      float64
		Type      string
	}{
		InputSize: d.inputSize,
		Rate:      d.rate,
		Type:      "Dropout",
	})
}

func (d *dropout) UnmarshalJSON(data []byte) error {
	var v struct {
		InputSize int
		Rate      float64
	}
	err := json.Unmarshal(data, &v)
	if err != nil {
		return errors.Join(
			fmt.Errorf("invalid dropout layer"),
			err,
		)
	}
	d.inputSize = v.InputSize
	d.rate = v.Rate
	return nil
}
