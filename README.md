# mlgo - A Golang Machine Learning Library

Welcome to mlgo, a Golang machine learning library crafted from derived mathematical proofs obtained through DeepLearning.AI courses. This library serves as a proof-of-concept as a learning project for deeper understanding of the underlying concepts. Was greatly inspired by Keras, machine learning library for Python, while adapting to the idioms and conventions of Golang.

## Key Features

1. **Mathematical Foundation:** Handwritten from derived mathematical proofs obtained from DeepLearning.AI courses, ensuring a solid theoretical basis.

2. **Open-Source Origins:** Information was gathered from free resources on the internet, articles and posts.

3. **Proof-of-Concept:** Not designed for extensive optimization, the library serves as a proof-of-concept to demonstrate the feasibility of implementing machine learning concepts in Golang.

## Example Code

```go
package main

import (
	"fmt"
	"log"

	"github.com/Hukyl/mlgo/activation"
	"github.com/Hukyl/mlgo/datasets"
	"github.com/Hukyl/mlgo/loss"
	"github.com/Hukyl/mlgo/metric"
	"github.com/Hukyl/mlgo/nn"
	"github.com/Hukyl/mlgo/nn/layers"
)


func main() {
	parameters := utils.NeuralNetworkParameters{
		EpochCount:          30,
		InitialLearningRate: 0.01,
        AccuracyMetric:      metric.CategoricalAccuracy{},
	}

    // Preprocess data
    X_train, Y_train := prepareData(datasets.MnistDataset("your/path"))

    // Define the structure of the layers
    l := []layers.Layer{
        layers.NewRandomDense(
            [2]int{784, 20},
            activation.Sigmoid{},
            layers.XavierUniformInitialization{},
        ),
        layers.NewRandomDense(
            [2]int{20, 10},
            activation.Sigmoid{},
            layers.XavierUniformInitialization{},
        ),
    }
    model := nn.NewNeuralNetwork(
        l,
        loss.LogLoss[float64]{},
    )

    // Train your network
	err = model.Train(X_train, Y_train, parameters)
	if err != nil {
		log.Fatal(err)
	}

	Y_pred := model.Predict(X_train[0])
	fmt.Println(parameters.AccuracyMetric.Calculate(Y_test[0], Y_pred))
}
```

## Author and Copyright

Andrii Shalaiev  
Copyright © 2024 Andrii Shalaiev. All rights reserved.

## References

- [DeepLearning.AI course](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science)

- [Debugging a Neural Network](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607)

## Disclaimer

This library is a proof-of-concept and may not be suitable for production use. Use at your own risk.
