package layers

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"reflect"
	"strconv"

	"github.com/Hukyl/mlgo/activation"
	. "github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/utils"
)

type dense struct {
	weights    Matrix[float64]
	bias       Matrix[float64]
	activation activation.ActivationFunction
}

func (d *dense) InputSize() [2]int {
	return [2]int{d.Weights().ColumnCount(), 1}
}

func (d *dense) OutputSize() [2]int {
	return [2]int{d.Weights().RowCount(), 1}
}

func (d *dense) Weights() Matrix[float64] {
	return d.weights
}

func (d *dense) Bias() Matrix[float64] {
	return d.bias
}

func (d *dense) Activation() activation.ActivationFunction {
	return d.activation
}

func (d *dense) IsTraining() bool {
	return false
}

// ForwardPropagate applies produces two matrices as a result of forward propagation on X:
//   - A linear combination of the weights, inputs and biases (i.e. W*X + b)
//   - The actual output of the layer, which is activated linear combination (i.e. activation(W*X + b))
//
// Input has to be of d.InputSize() size, otherwise error is returned.
func (d *dense) ForwardPropagate(X Matrix[float64]) (output [2]Matrix[float64], err error) {
	linearCombination, err := d.Weights().Multiply(X)
	if err != nil {
		return output, err
	}
	broadcastedBias, _ := d.Bias().Multiply(NewOnesMatrix(1, X.ColumnCount()))
	linearCombination, err = linearCombination.Add(broadcastedBias)
	if err != nil {
		return output, err
	}
	output[0] = linearCombination

	activatedLinearCombination := linearCombination.DeepCopy()
	d.Activation().ApplyMatrix(activatedLinearCombination)
	output[1] = activatedLinearCombination

	return output, nil
}

// BackPropagate applies derivatives of the input and activation function to
// propagate the result further to previous layers.
//
//   - nextLayerPropagation is the result for `nextLayer.BackPropagate()`, where
//     `nextLayer` is the next layer in the neural network.
//   - X is the cached input for current layer, which was used to obtain `nextLayerPropagation“
//   - A is the cached output of current layer, produced via forward propagation on `X`.
//   - output is the cached output that this dense produced via forward propagation.
//
// Method is based on the fact, that the previous dense derivative is based on the next
// layer's partial derivative.
// For example (using log-loss and sigmoid activation function):
//
//	dL/dW3 = dL/dA3 * dA3/dZ3 * dZ3/dW3
//	= (((A3 - Y) / (A3 - A3^2)) * (A3 - A3^2)) @ A2.T()
//	= (A3 - Y) @ A2.T()
//
//	dL/dW2 = dL/dA3 * dA3/dZ3 * dZ3/dA2 * dA2/dZ2 * dZ2/dW2
//	= (W3.T() @ [A3 - Y] . A2(1-A2)) @ A1.T()
//	[FOUND IN dL/dW3]
//
//	dL/dW1 = dL/dA3 * dA3/dZ3 * dZ3/dA2 * dA2/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1
//	= (W2.T() @ [(W3.T() @ (A3 - Y) . A2(1-A2))] . A1(1-A1)) @ X.T()
//	[FOUND in dL/dW2]
//
// Then, for L3 (with weights W3) BackPropagate does the following:
//
//   - produce the partial derivative for this layer.
//
//     dLdZ = dLdA * dAdZ = nextLayerPropagation * layer.Activation.BackPropagateMatrix()
//     dLdb = dLdZ * dZdb = dLdZ * 1
//     dLdW = dLdZ * dZdW = dLdZ * X.T()
//
//   - update the weights and bias as the main goal of backpropagation.
//
//     W = W - learningRate * dLdW
//     b = b - learningRate * dLdb
//
//   - produce the propagation base for the next layer.
//
//     thisLayerPropagation = W.T() @ dLdZ
func (d *dense) BackPropagate(nextLayerPropagation, X Matrix[float64], A [2]Matrix[float64], parameters utils.NeuralNetworkParameters) Matrix[float64] {
	dAdZ := d.Activation().DerivativeMatrix(A[0]) // derivative for activation function

	dLdZ, _ := nextLayerPropagation.MultiplyElementwise(dAdZ)
	d.updateWeights(dLdZ, X, parameters)

	result, _ := d.Weights().T().Multiply(dLdZ)
	return result
}

// UpdateWeights updates the weights based on the BackPropagate(), where
// `nextLayer` is the next dense in the neural network, and the input to the current dense.
//
// Method is based on that for each next dense, derivative is going to be based on the next dense's
// backpropagation derivative.
func (d *dense) updateWeights(dLdZ, input Matrix[float64], parameters utils.NeuralNetworkParameters) {
	db, _ := dLdZ.Multiply(NewOnesMatrix(input.ColumnCount(), 1))
	dW, _ := dLdZ.Multiply(input.T())

	columns := float64(input.ColumnCount())

	if w, _ := dW.At(0, 0); math.IsNaN(w) {
		log.Printf("NaN dW")
	}
	decayed_dW, _ := dW.MultiplyByScalar(1 / columns).Add(
		d.weights.MultiplyByScalar(parameters.WeightDecay),
	)
	if w, _ := decayed_dW.At(0, 0); math.IsNaN(w) {
		log.Printf("NaN decayed_dW")
	}
	d.weights, _ = d.Weights().Add(decayed_dW.MultiplyByScalar(-parameters.LearningRate()))
	if w, _ := d.weights.At(0, 0); math.IsNaN(w) {
		log.Printf("NaN weight")
	}

	decayed_db, _ := db.MultiplyByScalar(1 / columns).Add(
		d.bias.MultiplyByScalar(parameters.WeightDecay),
	)
	d.bias, _ = d.Bias().Add(decayed_db.MultiplyByScalar(-parameters.LearningRate()))
}

/************************************************************************/

func (d dense) String() string {
	return fmt.Sprintf(
		"Dense{%d -> %d, activation: %s}",
		d.InputSize()[0],
		d.OutputSize()[0],
		reflect.TypeOf(d.activation).Name(),
	)
}

func (d *dense) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		Weights    Matrix[float64]
		Bias       Matrix[float64]
		Activation string
		Type       string
	}{
		Weights:    d.weights,
		Bias:       d.bias,
		Activation: reflect.TypeOf(d.activation).Name(),
		Type:       "Dense",
	})
}

func (d *dense) UnmarshalJSON(data []byte) error {
	var err error
	var v map[string]json.RawMessage

	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}

	w, _ := NewMatrix([][]float64{{}})
	if err := w.UnmarshalJSON(v["Weights"]); err != nil {
		return errors.Join(
			errors.New("invalid weight initializing"),
			err,
		)
	}
	d.weights = w

	b, _ := NewMatrix([][]float64{{}})
	if err := b.UnmarshalJSON(v["Bias"]); err != nil {
		return errors.Join(
			errors.New("invalid bias initializing"),
			err,
		)
	}
	d.bias = b

	activationLiteral, _ := strconv.Unquote(string(v["Activation"]))
	d.activation, err = activation.DynamicActivation(activationLiteral)
	return err // can return either actual error or nil
}
