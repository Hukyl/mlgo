package nn

import (
	"encoding/json"
	"os"

	"github.com/Hukyl/mlgo/loss"
	"github.com/Hukyl/mlgo/nn/layers"
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

func NewNeuralNetwork(layers []layers.Layer, lossFunction loss.LossFunction[float64]) NeuralNetwork {
	return &nn{layers: layers, LossFunction: lossFunction}
}
