package nn

import (
	"math"
	"math/rand"
)

type WeightInitialization interface {
	Generate(layerSize [2]int) float64
}

/*****************************************************/

// RandomInitialization randomly generates a number in range [-0.5, 0.5)
type RandomInitialization struct{}

func (r RandomInitialization) Generate(layerSize [2]int) float64 {
	return rand.Float64() - 0.5
}

/******************************************************/

// XavierInitialization is a weight initialization technique, also known as Glorot initialization.
//
// Mainly used for tanh activation function.
type XavierInitialization struct{}

func (x XavierInitialization) Generate(layerSize [2]int) float64 {
	return rand.NormFloat64() * math.Sqrt(float64(2)/float64(layerSize[0]+layerSize[1]))
}

/******************************************************/

// HeInitialization is a weight initialization technique, which uses the normal distribution
// with a standard deviation using the previous layer size.
//
// Mainly used with ReLU activation function to account for the zeros in (-inf;0] range.
type HeInitialization struct{}

func (h HeInitialization) Generate(layerSize [2]int) float64 {
	return rand.NormFloat64() * math.Sqrt(float64(2)/float64(layerSize[1]))
}

/******************************************************/
