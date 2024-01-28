package nn

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"reflect"
	"regexp"
	"strings"

	"github.com/Hukyl/mlgo/loss"
	"github.com/Hukyl/mlgo/matrix"
)

type NeuralNetwork interface {
	json.Marshaler
	json.Unmarshaler

	InputSize() [2]int
	OutputSize() [2]int

	Predict(X matrix.Matrix[float64]) (Y matrix.Matrix[float64])
	Train(X matrix.Matrix[float64], Y matrix.Matrix[float64], numIterations int, learningRate float64) error
}

/************************************************************************/

type nn struct {
	layers       []Layer
	LossFunction loss.LossFunction[float64]
}

func (n *nn) InputSize() [2]int {
	return n.layers[0].InputSize()
}

func (n *nn) OutputSize() [2]int {
	return n.layers[len(n.layers)-1].OutputSize()
}

func (n *nn) Predict(X matrix.Matrix[float64]) matrix.Matrix[float64] {
	Y := X
	for _, l := range n.layers {
		Y, _ = l.ForwardPropagate(Y)
	}
	return Y
}

func (n *nn) validateTrainSamples(X, Y matrix.Matrix[float64]) error {
	var errorText string
	switch {
	case X.RowCount() != n.InputSize()[0]:
		errorText = "invalid input size"
	case Y.RowCount() != n.OutputSize()[0]:
		errorText = "invalid output size"
	case X.ColumnCount() != Y.ColumnCount():
		errorText = "incosistent sample count"
	}
	if len(errorText) > 0 {
		return errors.New(errorText)
	}
	return nil
}

// forwardPropagate propagates the input through the layers of the network.
//
// Returns slice, with size of (N layers)+1, where [0] is the input to the network,
// and each element is the result of propagation through the next layer.
func (n *nn) forwardPropagate(X matrix.Matrix[float64]) []matrix.Matrix[float64] {
	inputCache := make([]matrix.Matrix[float64], len(n.layers)+1)
	inputCache[0] = X

	for j, layer := range n.layers {
		soFar, _ := layer.ForwardPropagate(inputCache[j])
		inputCache[j+1] = soFar
	}
	return inputCache
}

// backPropagate propagates the derivatives from the end of the neural network.
// Each (i-1)-th element of returned slices corresponds to the back propagation derivative
// for i-th layer.
//
// For example:
//
//	L1 -> dL/dZ1 = dL/dA3 * dA3/dZ3 * dZ3/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1
//	L2 -> dL/dZ2 = dL/dA3 * dA3/dZ3 * dZ3/dA2 * dA2/dZ2
//	L3 -> dL/dZ3 = dL/dA3 * dA3/dZ3
func (n *nn) backPropagate(Y matrix.Matrix[float64], inputCache []matrix.Matrix[float64]) [][2]matrix.Matrix[float64] {
	layerCount := len(n.layers)
	backPropagationCache := make([][2]matrix.Matrix[float64], layerCount+1)

	yHat := inputCache[layerCount]
	dL := n.LossFunction.ApplyDerivativeMatrix(Y, yHat)
	backPropagationCache[layerCount] = [2]matrix.Matrix[float64]{dL, dL}

	for j := layerCount - 1; j >= 0; j-- {
		layer := n.layers[j]
		backPropagationCache[j] = layer.BackPropagate(backPropagationCache[j+1][1], inputCache[j+1])
	}

	return backPropagationCache
}

func (n *nn) computeCost(yHat, Y matrix.Matrix[float64]) float64 {
	cost := float64(0)

	losses := n.LossFunction.ApplyMatrix(Y, yHat)
	for column := 0; column < losses.ColumnCount(); column++ {
		for row := 0; row < losses.RowCount(); row++ {
			v, _ := losses.At(column, row)
			cost += v
		}
	}
	return cost / float64(losses.ColumnCount())
}

func (n *nn) updateWeights(inputCache []matrix.Matrix[float64], backPropagationCache [][2]matrix.Matrix[float64], learningRate float64) {
	for j, layer := range n.layers {
		layer.UpdateWeights(backPropagationCache[j][0], inputCache[j], learningRate)
	}
}

func (n *nn) Train(X matrix.Matrix[float64], Y matrix.Matrix[float64], numIterations int, learningRate float64) error {
	err := n.validateTrainSamples(X, Y)
	if err != nil {
		return err
	}

	tenthIteration := float64(numIterations) / 10.0

	for i := 1; i <= numIterations; i++ {
		// Forward propagate and store inputs
		inputCache := n.forwardPropagate(X)

		// Print cost
		if i == 1 || math.Mod(float64(i), tenthIteration) == 0.0 {
			cost := n.computeCost(inputCache[len(inputCache)-1], Y)
			log.Printf("Cost after %d iter: %v\n", i, cost)
		}

		// Backpropagation to fill the derivatives
		backPropagationCache := n.backPropagate(Y, inputCache)

		// Update parameters
		n.updateWeights(inputCache, backPropagationCache, learningRate)
	}
	return nil
}

/************************************************************************/

func (n nn) String() string {
	b := strings.Builder{}
	for i, layer := range n.layers {
		b.WriteString(fmt.Sprintf("Layer %d: %s\n", i, layer))
	}
	return b.String()
}

func (n *nn) MarshalJSON() ([]byte, error) {
	lossFullName := reflect.TypeOf(n.LossFunction).Name()
	r := regexp.MustCompile(`^(.+)\[.+\]$`)

	return json.Marshal(&struct {
		Layers       []Layer
		LossFunction string
	}{
		Layers: n.layers, LossFunction: r.FindStringSubmatch(lossFullName)[1],
	})
}

func (n *nn) UnmarshalJSON(data []byte) error {
	var v struct {
		Layers       []*layer
		LossFunction string
	}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	n.layers = make([]Layer, len(v.Layers))
	for i, l := range v.Layers {
		n.layers[i] = l
	}
	n.LossFunction, _ = loss.DynamicLoss[float64](v.LossFunction)
	return nil
}
