package nn

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"path/filepath"
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

	// Training functions
	ComputeCost(yHat, Y matrix.Matrix[float64]) float64
	ForwardPropagate(X matrix.Matrix[float64]) []matrix.Matrix[float64]
	BackPropagate(Y matrix.Matrix[float64], inputCache []matrix.Matrix[float64]) [][2]matrix.Matrix[float64]
	UpdateWeights(
		inputCache []matrix.Matrix[float64],
		backPropagationCache [][2]matrix.Matrix[float64],
		parameters NeuralNetworkParameters,
	)

	// Prediction functions
	Predict(X matrix.Matrix[float64]) (Y matrix.Matrix[float64])
	Train(X, Y []matrix.Matrix[float64], parameters NeuralNetworkParameters) error
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

func (n *nn) validateTrainSamples(X, Y []matrix.Matrix[float64]) error {
	var errorText string

	for i := 0; i < len(X); i++ {
		X_batch, Y_batch := X[i], Y[i]
		switch {
		case X_batch.RowCount() != n.InputSize()[0]:
			errorText = "invalid input size"
		case Y_batch.RowCount() != n.OutputSize()[0]:
			errorText = "invalid output size"
		case X_batch.ColumnCount() != Y_batch.ColumnCount():
			errorText = "incosistent sample count"
		}
		if len(errorText) > 0 {
			return errors.New(errorText)
		}
	}
	return nil
}

// forwardPropagate propagates the input through the layers of the network.
//
// Returns slice, with size of (N layers)+1, where [0] is the input to the network,
// and each element is the result of propagation through the next layer.
func (n *nn) ForwardPropagate(X matrix.Matrix[float64]) []matrix.Matrix[float64] {
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
func (n *nn) BackPropagate(Y matrix.Matrix[float64], inputCache []matrix.Matrix[float64]) [][2]matrix.Matrix[float64] {
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

func (n *nn) ComputeCost(yHat, Y matrix.Matrix[float64]) float64 {
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

func (n *nn) UpdateWeights(inputCache []matrix.Matrix[float64], backPropagationCache [][2]matrix.Matrix[float64], parameters NeuralNetworkParameters) {
	for j, layer := range n.layers {
		layer.UpdateWeights(backPropagationCache[j][0], inputCache[j], parameters)
	}
}

func (n *nn) Train(X, Y []matrix.Matrix[float64], parameters NeuralNetworkParameters) error {
	err := n.validateTrainSamples(X, Y)
	if err != nil {
		return err
	}
	validateParameters(&parameters)

	parameters.currentEpoch = 0

	DumpNeuralNetwork(n, filepath.Join(parameters.DumpPath, "start.json"))
	for e := 0; e < int(parameters.EpochCount); e++ {
		log.Printf("Epoch %d out of %d\n", e+1, parameters.EpochCount)

		for i := 0; i < len(X); i++ {
			X_batch, Y_batch := X[i], Y[i]

			// Forward propagate and store inputs
			inputCache := n.ForwardPropagate(X_batch)

			cost := n.ComputeCost(inputCache[len(inputCache)-1], Y_batch)
			log.Printf("Cost after %d iter: %v\n", i+1, cost)

			// Backpropagation to fill the derivatives
			backPropagationCache := n.BackPropagate(Y_batch, inputCache)

			// Update parameters
			n.UpdateWeights(inputCache, backPropagationCache, parameters)
		}

		parameters.currentEpoch++

		log.Printf(
			"Accuracy: %v\n",
			parameters.AccuracyMetric.Calculate(Y[0], n.Predict(X[0])),
		)
		DumpNeuralNetwork(n, filepath.Join(parameters.DumpPath, fmt.Sprintf("epoch_%d.json", e+1)))
	}
	DumpNeuralNetwork(n, filepath.Join(parameters.DumpPath, "finish.json"))

	parameters.currentEpoch = 0

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
