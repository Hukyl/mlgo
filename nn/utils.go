package nn

import (
	"encoding/json"
	"errors"
	"log"
	"math/rand"
	"os"

	"github.com/Hukyl/mlgo/activation"
	"github.com/Hukyl/mlgo/loss"
	"github.com/Hukyl/mlgo/matrix"
)

func jsonifyObject(obj interface{}, path string) error {
	f1, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f1.Close()
	enc := json.NewEncoder(f1)
	enc.SetIndent("", "    ")
	err = enc.Encode(obj)
	if err != nil {
		return err
	}
	return nil
}

func DumpNeuralNetwork(nn NeuralNetwork, path string) {
	jsonifyObject(nn, path)
	log.Printf("Dumped to %q\n", path)
}

func LoadNeuralNetwork(path string) (NeuralNetwork, error) {
	fileContent, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	nn := new(nn)

	// Unmarshal JSON data into the struct
	err = json.Unmarshal(fileContent, nn)
	if err != nil {
		return nil, err
	}

	return nn, nil
}

/************************************************************************/

func NewNeuralNetwork(layers []Layer, lossFunction loss.LossFunction[float64]) NeuralNetwork {
	return &nn{layers: layers, LossFunction: lossFunction}
}

func NewRandomNeuralNetwork(inputSize []int, activationFunctions []activation.ActivationFunction, lossFunction loss.LossFunction[float64]) NeuralNetwork {
	layers := make([]Layer, 0, len(inputSize)-1)
	for j := 1; j < len(inputSize); j++ {
		layer := NewRandomizedLayer([2]int{inputSize[j], inputSize[j-1]}, activationFunctions[j-1])
		layers = append(layers, layer)
	}
	return &nn{layers: layers, LossFunction: lossFunction}
}

/************************************************************************/

func NewLayer(W, b matrix.Matrix[float64], a activation.ActivationFunction) (Layer, error) {
	l := &layer{weights: W, bias: b, activation: a}
	if b.Size() != l.OutputSize() {
		return nil, errors.New("invalid bias size")
	}
	return l, nil
}

func NewRandomizedLayer(weightSize [2]int, a activation.ActivationFunction) Layer {
	W := matrix.NewZeroMatrix[float64](weightSize[0], weightSize[1])
	for i := 0; i < weightSize[0]; i++ {
		for j := 0; j < weightSize[1]; j++ {
			W.Set(i, j, rand.Float64()-0.5)
		}
	}
	b := matrix.NewZeroMatrix[float64](weightSize[0], 1)
	return &layer{weights: W, bias: b, activation: a}
}
