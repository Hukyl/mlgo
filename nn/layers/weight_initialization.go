package layers

import (
	"math"
	"math/rand"
)

type WeightInitialization interface {
	Generate(layerSize [2]int) float64
}

/*****************************************************/

// RandomInitialization randomly generates a number in range [-0.5, 0.5)
type RandomInitialization struct {
	Min float64
	Max float64
}

func (r RandomInitialization) Generate(layerSize [2]int) float64 {
	return r.Min + rand.Float64()*(r.Max-r.Min)
}

/******************************************************/

// XavierNormalInitialization is a weight initialization technique, also known as Glorot initialization.
//
// Mainly used for tanh activation function.
type XavierNormalInitialization struct{}

func (x XavierNormalInitialization) Generate(layerSize [2]int) float64 {
	return rand.NormFloat64() * math.Sqrt(float64(2)/float64(layerSize[0]+layerSize[1]))
}

/******************************************************/

type XavierUniformInitialization struct{}

func (x XavierUniformInitialization) Generate(layerSize [2]int) float64 {
	limit := math.Sqrt(float64(6) / float64(layerSize[0]+layerSize[1]))

	// Generate a random number in the range [-x, x]
	return (rand.Float64()*2 - 1.0) * limit
}

/******************************************************/

// HeInitialization is a weight initialization technique, which uses the normal distribution
// with a standard deviation using the previous layer size.
//
// Mainly used with ReLU activation function to account for the zeros in (-inf;0] range.
type HeInitialization struct{}

func (h HeInitialization) Generate(layerSize [2]int) float64 {
	return rand.NormFloat64() * math.Sqrt(float64(2)/float64(layerSize[0]))
}

/******************************************************/
