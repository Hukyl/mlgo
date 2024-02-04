package layers

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"

	"github.com/Hukyl/mlgo/activation"
	. "github.com/Hukyl/mlgo/matrix"
	"github.com/Hukyl/mlgo/utils"
)

type Layer interface {
	fmt.Stringer
	json.Marshaler
	json.Unmarshaler

	InputSize() [2]int
	OutputSize() [2]int

	IsTraining() bool

	Weights() Matrix[float64]
	Bias() Matrix[float64]
	Activation() activation.ActivationFunction

	ForwardPropagate(X Matrix[float64]) (Y Matrix[float64], err error)
	BackPropagate(nextLayerPropagation, output Matrix[float64]) [2]Matrix[float64]
	UpdateWeights(nextLayerPropagation, input Matrix[float64], parameters utils.NeuralNetworkParameters)
}

/************************************************************************/

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

// ForwardPropagate applies the weights and the bias of given dense to produce
// an output.
// Input has to be of d.InputSize() size, otherwise error is returned.
func (d *dense) ForwardPropagate(X Matrix[float64]) (Matrix[float64], error) {
	linearCombination, err := d.Weights().Multiply(X)
	if err != nil {
		return nil, err
	}
	broadcastedBias, _ := d.Bias().Multiply(NewOnesMatrix(1, X.ColumnCount()))
	result, err := linearCombination.Add(broadcastedBias)
	if err != nil {
		return nil, err
	}
	d.Activation().ApplyMatrix(result)
	return result, nil
}

// BackPropagate applies derivatives of the input and activation function to
// propagate the result further to previous layers.
//
//   - nextLayerPropagation is the result for `nextLayer.BackPropagate()[1]`, where
//     `nextLayer` is the next dense in the neural network.
//   - output is the cached output that this dense produced via forward propagation.
//
// Method is based on the fact, that the previous dense derivative is based on the next
// dense's partial derivative.
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
// Then, for L3 (with weights W3) this method returns:
//
//	result[0] = ([(A3 - Y) / (A3 - A3^2)] . (A3-A3^2))
//	result[0] = (nextLayerPropagation . dA/dZ)
//	result[1] = W3.T() @ ([(A3 - Y) / (A3 - A3^2)] . (A3-A3^2))
//	result[1] = weights.T() @ result[0]
//
// Where result[0] can be used (with addition of one multiplication) to update this
// dense weights (in this case, W3), and result[1] can be used for previous dense
// (in our case, L2) to update its (in our case, W2) weights, using ```.UpdateWeights()``` method.
func (d *dense) BackPropagate(nextLayerPropagation, A Matrix[float64]) [2]Matrix[float64] {
	dAdZ := A.DeepCopy() // derivative for activation function
	d.Activation().BackPropagateMatrix(dAdZ)

	result := [2]Matrix[float64]{}
	result[0], _ = nextLayerPropagation.MultiplyElementwise(dAdZ)
	result[1], _ = d.Weights().T().Multiply(result[0])
	return result
}

// UpdateWeights updates the weights based on the `nextLayer.BackPropagate()[0]`, where
// `nextLayer` is the next dense in the neural network, and the input to the current dense.
//
// Method is based on that for each next dense, derivative is going to be based on the next dense's
// backpropagation derivative.
func (d *dense) UpdateWeights(dAdZ, input Matrix[float64], parameters utils.NeuralNetworkParameters) {
	db, _ := dAdZ.Multiply(NewOnesMatrix(input.ColumnCount(), 1))
	dW, _ := dAdZ.Multiply(input.T())

	columns := float64(input.ColumnCount())

	decayed_dW, _ := dW.MultiplyByScalar(1 / columns).Add(
		d.weights.MultiplyByScalar(parameters.WeightDecay),
	)
	d.weights, _ = d.Weights().Add(decayed_dW.MultiplyByScalar(-parameters.LearningRate()))

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
	var v map[string]json.RawMessage

	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}

	w, _ := NewMatrix([][]float64{{}})
	if err := w.UnmarshalJSON(v["Weights"]); err != nil {
		return err
	}
	d.weights = w

	b, _ := NewMatrix([][]float64{{}})
	if err := b.UnmarshalJSON(v["Bias"]); err != nil {
		return err
	}
	d.bias = b

	activationLiteral, _ := strconv.Unquote(string(v["Activation"]))
	d.activation, _ = activation.DynamicActivation(activationLiteral)
	return nil
}
