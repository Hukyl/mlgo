package nn

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/Hukyl/mlgo/activation"
	"github.com/Hukyl/mlgo/matrix"
)

type Layer interface {
	fmt.Stringer
	json.Marshaler
	json.Unmarshaler

	InputSize() [2]int
	OutputSize() [2]int

	Weights() matrix.Matrix[float64]
	Bias() matrix.Matrix[float64]
	Activation() activation.ActivationFunction

	ForwardPropagate(X matrix.Matrix[float64]) (Y matrix.Matrix[float64], err error)
	BackPropagate(nextLayerPropagation, output matrix.Matrix[float64]) [2]matrix.Matrix[float64]
	UpdateWeights(nextLayerPropagation, input matrix.Matrix[float64], learningRate float64)
}

/************************************************************************/

type layer struct {
	weights    matrix.Matrix[float64]
	bias       matrix.Matrix[float64]
	activation activation.ActivationFunction
}

func (l *layer) InputSize() [2]int {
	return [2]int{l.Weights().ColumnCount(), 1}
}

func (l *layer) OutputSize() [2]int {
	return [2]int{l.Weights().RowCount(), 1}
}

func (l *layer) Weights() matrix.Matrix[float64] {
	return l.weights
}

func (l *layer) Bias() matrix.Matrix[float64] {
	return l.bias
}

func (l *layer) Activation() activation.ActivationFunction {
	return l.activation
}

// ForwardPropagate applies the weights and the bias of given layer to produce
// an output.
// Input has to be of l.InputSize() size, otherwise error is returned.
func (l *layer) ForwardPropagate(X matrix.Matrix[float64]) (matrix.Matrix[float64], error) {
	linearCombination, err := l.Weights().Multiply(X)
	if err != nil {
		return nil, err
	}
	broadcastedBias, _ := l.Bias().Multiply(matrix.NewOnesMatrix(1, X.ColumnCount()))
	result, err := linearCombination.Add(broadcastedBias)
	if err != nil {
		return nil, err
	}
	l.Activation().ApplyMatrix(result)
	return result, nil
}

// BackPropagate applies derivatives of the input and activation function to
// propagate the result further to previous layers.
//
//   - nextLayerPropagation is the result for `nextLayer.BackPropagate()[1]`, where
//     `nextLayer` is the next layer in the neural network.
//   - output is the cached output that this layer produced via forward propagation.
//
// Method is based on the fact, that the previous layer derivative is based on the next
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
// Then, for L3 (with weights W3) this method returns:
//
//	result[0] = ([(A3 - Y) / (A3 - A3^2)] . (A3-A3^2))
//	result[0] = (nextLayerPropagation . dA/dZ)
//	result[1] = W3.T() @ ([(A3 - Y) / (A3 - A3^2)] . (A3-A3^2))
//	result[1] = weights.T() @ result[0]
//
// Where result[0] can be used (with addition of one multiplication) to update this
// layer weights (in this case, W3), and result[1] can be used for previous layer
// (in our case, L2) to update its (in our case, W2) weights, using ```.UpdateWeights()``` method.
func (l *layer) BackPropagate(nextLayerPropagation, A matrix.Matrix[float64]) [2]matrix.Matrix[float64] {
	dAdZ := A.DeepCopy() // derivative for activation function
	l.Activation().BackPropagateMatrix(dAdZ)

	result := [2]matrix.Matrix[float64]{}
	result[0], _ = nextLayerPropagation.MultiplyElementwise(dAdZ)
	result[1], _ = l.Weights().T().Multiply(result[0])
	return result
}

// UpdateWeights updates the weights based on the `nextLayer.BackPropagate()[0]`, where
// `nextLayer` is the next layer in the neural network, and the input to the current layer.
//
// Method is based on that for each next layer, derivative is going to be based on the next layer's
// backpropagation derivative.
func (l *layer) UpdateWeights(dAdZ, input matrix.Matrix[float64], learningRate float64) {
	db, _ := dAdZ.Multiply(matrix.NewOnesMatrix(input.ColumnCount(), 1))
	dW, _ := dAdZ.Multiply(input.T())

	columns := float64(input.ColumnCount())
	l.weights, _ = l.Weights().Add(dW.MultiplyByScalar(-learningRate / columns))
	l.bias, _ = l.Bias().Add(db.MultiplyByScalar(-learningRate / columns))
}

/************************************************************************/

func (l layer) String() string {
	return fmt.Sprintf("W = %s, b = %s", l.Weights(), l.Bias())
}

func (l *layer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		Weights    matrix.Matrix[float64]
		Bias       matrix.Matrix[float64]
		Activation string
	}{
		Weights: l.weights, Bias: l.bias, Activation: reflect.TypeOf(l.activation).Name(),
	})
}

func (l *layer) UnmarshalJSON(data []byte) error {
	// var v struct {
	// 	Weights    *matrix[float64]
	// 	Bias       *matrix[float64]
	// 	Activation string
	// }
	var v map[string]interface{}

	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}

	w, _ := matrix.NewMatrix([][]float64{{}})
	wData, err := json.Marshal(v["Weights"])
	if err != nil {
		return err
	}
	if err := w.UnmarshalJSON(wData); err != nil {
		return err
	}
	l.weights = w

	b, _ := matrix.NewMatrix([][]float64{{}})
	bData, err := json.Marshal(v["Bias"])
	if err != nil {
		return err
	}
	if err := b.UnmarshalJSON(bData); err != nil {
		return err
	}
	l.bias = b

	l.activation, _ = activation.DynamicActivation(v["Activation"].(string))
	return nil
}
