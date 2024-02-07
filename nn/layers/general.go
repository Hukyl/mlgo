package layers

import (
	"encoding/json"
	"fmt"

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
	BackPropagate(nextLayerPropagation, input, output Matrix[float64], parameters utils.NeuralNetworkParameters) Matrix[float64]
	updateWeights(nextLayerPropagation, input Matrix[float64], parameters utils.NeuralNetworkParameters)
}
