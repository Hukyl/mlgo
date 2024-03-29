// Package nn provides neural network building blocks, as layers
// and the general struct for ANN.
package nn

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"

	"github.com/Hukyl/mlgo/activation"
	. "github.com/Hukyl/mlgo/loss"
	. "github.com/Hukyl/mlgo/matrix"
	. "github.com/Hukyl/mlgo/nn/layers"
	"github.com/Hukyl/mlgo/utils"
)

// NeuralNetwork is an interface for a general ANN.
//
// InputSize returns the dimensions of the input to the first layer.
// Returned as [2]{rowCount, 1}, but in practice columnCount can also differ.
//
// OutputSize returns the dimensions of the output of the last layer.
// Returned as [2]{rowCount, 1}, but in practice columnCount can also differ.
//
// ComputeCost uses the LossFunction of the last layer to compute cost, i.e. the error
// of the prediction. Separate columns of y and yHat represent different samples,
// and the cost is averaged between them.
//
// ForwardPropagate produces a slices of outputs for each layer. First element in array
// for each slice item is the linear combination using weights and biases, the second -
// linear combination passed through the activation function. The length of the returned
// slice is numberOfLayers+1, as the 0th element is the initial input to the ANN.
//
//	item[0] = W*X + b
//	item[1] = activation(item[0])
//
// BackPropagate uses the inputCache, produced by ForwardPropagate, to update the layers.
// Utilizes parameters for the neural network.
//
// Predict produces ANN prediction for the given data. Equivalent to ForwardPropagate()[-1][1].
//
// Train uses training functions (ComputeCost, ForwardPropagate, BackPropagate) to update
// the weights of the layers to decrease the cost and loss.
type NeuralNetwork interface {
	json.Marshaler
	json.Unmarshaler

	InputSize() [2]int
	OutputSize() [2]int

	// Training functions
	ComputeCost(yHat, Y Matrix[float64]) float64
	ForwardPropagate(X Matrix[float64]) (inputCache [][2]Matrix[float64])
	BackPropagate(Y Matrix[float64], inputCache [][2]Matrix[float64], parameters utils.NeuralNetworkParameters)

	// Prediction functions
	Predict(X Matrix[float64]) (Y Matrix[float64])
	Train(X, Y []Matrix[float64], parameters utils.NeuralNetworkParameters) error
}

/************************************************************************/

type nn struct {
	layers       []Layer
	LossFunction LossFunction[float64]
}

func (n *nn) InputSize() [2]int {
	return n.layers[0].InputSize()
}

func (n *nn) OutputSize() [2]int {
	return n.layers[len(n.layers)-1].OutputSize()
}

func (n *nn) Predict(X Matrix[float64]) Matrix[float64] {
	Y := X
	for _, l := range n.layers {
		if !l.IsTraining() {
			output, _ := l.ForwardPropagate(Y)
			Y = output[1]
		}
	}
	return Y
}

func (n *nn) validateTrainSamples(X, Y []Matrix[float64]) error {
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
func (n *nn) ForwardPropagate(X Matrix[float64]) [][2]Matrix[float64] {
	inputCache := make([][2]Matrix[float64], len(n.layers)+1)
	inputCache[0] = [2]Matrix[float64]{X, X}

	for j, layer := range n.layers {
		soFar, _ := layer.ForwardPropagate(inputCache[j][1])
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
func (n *nn) BackPropagate(Y Matrix[float64], inputCache [][2]Matrix[float64], parameters utils.NeuralNetworkParameters) {
	layerCount := len(n.layers)

	var backPropagation Matrix[float64]

	yHat := inputCache[layerCount][1]
	dL := n.LossFunction.ApplyDerivativeMatrix(Y, yHat)

	backPropagation = dL

	for j := layerCount - 1; j >= 0; j-- {
		layer := n.layers[j]
		backPropagation = layer.BackPropagate(
			backPropagation,
			inputCache[j][1],
			inputCache[j+1],
			parameters,
		)
		backPropagation = Clip(
			backPropagation,
			-parameters.ClipValue,
			parameters.ClipValue,
		)
	}
}

func (n *nn) ComputeCost(yHat, Y Matrix[float64]) float64 {
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

func (n *nn) Train(X, Y []Matrix[float64], parameters utils.NeuralNetworkParameters) error {
	err := n.validateTrainSamples(X, Y)
	if err != nil {
		return err
	}
	parameters.Validate()
	parameters.ResetEpoch()

	for e := 0; e < int(parameters.EpochCount); e++ {
		cost := float64(0.0)
		accuracy := float64(0.0)

		for i := 0; i < len(X); i++ {
			X_batch, Y_batch := X[i], Y[i]

			// Forward propagate and store inputs
			inputCache := n.ForwardPropagate(X_batch)

			// Calculate cost and accuracy
			prediction := inputCache[len(inputCache)-1][1]
			cost += n.ComputeCost(prediction, Y_batch) / float64(len(X))
			if math.IsNaN(cost) || math.IsInf(cost, 0) || cost == 0.0 {
				return errors.New("cost is an invalid number")
			}
			accuracy += parameters.AccuracyMetric.Calculate(Y_batch, prediction) / float64(len(X))

			// Updating the weights
			n.BackPropagate(Y_batch, inputCache, parameters)
		}
		log.Printf("Epoch %d/%d, avg_cost: %-10.5g avg_accuracy: %-10.5g\n", e+1, parameters.EpochCount, cost, accuracy)

		parameters.IncrementEpoch()
		if parameters.Backups.ToCreate {
			err = DumpNeuralNetwork(
				n,
				filepath.Join(
					parameters.Backups.Path,
					fmt.Sprintf("epoch_%d.json", e+1),
				),
			)
			if err != nil {
				log.Printf("dump saving error: %s", err)
			}
		}

	}

	parameters.ResetEpoch()
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
	var err error
	var v struct {
		Layers       []json.RawMessage
		LossFunction string
	}
	if err := json.Unmarshal(data, &v); err != nil {
		return errors.New("invalid general NN unmarshalling")
	}
	n.layers = make([]Layer, len(v.Layers))
	for i, lData := range v.Layers {
		var layer Layer
		var layerType struct {
			Type string
		}
		json.Unmarshal(lData, &layerType)
		switch layerType.Type {
		case "Dense":
			W, _ := NewMatrix([][]float64{{}})
			b, _ := NewMatrix([][]float64{{}})
			layer, _ = NewDense(W, b, activation.Linear{})
			err = layer.UnmarshalJSON(lData)
		case "Dropout":
			layer = NewDropout(0, 0)
			err = layer.UnmarshalJSON(lData)
		}
		if err != nil {
			return errors.Join(
				fmt.Errorf("error on parsing layer #%d", i+1),
				err,
			)
		}
		n.layers[i] = layer
	}
	n.LossFunction, err = DynamicLoss[float64](v.LossFunction)
	if err != nil {
		return fmt.Errorf("invalid loss name: %s", v.LossFunction)
	}
	return nil
}
