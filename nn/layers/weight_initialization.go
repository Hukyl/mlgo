package layers

import (
	"math"
	"math/rand"
)

// WeightInitialization, even though contains only one method,
// is used to produce values for weights in the layer based on its size.
//
// Generate is used to generate a float value as a weight for the layer.
// Note, that some weight initialization techniques may utilize only one
// layer dimension, like He initialization.
type WeightInitialization interface {
	Generate(layerSize [2]int) float64
}

// RandomInitialization randomly generates a number in the range given.
//
// If the limits are not set, Generate produces zeros.
//
//	weight := r.Min + rand.Float64()*(r.Max-r.Min)
type RandomInitialization struct {
	Min float64
	Max float64
}

func (r RandomInitialization) Generate(layerSize [2]int) float64 {
	return r.Min + rand.Float64()*(r.Max-r.Min)
}

// XavierNormalInitialization (also known as Glorot) is a weight initialization technique
// based on the normal distribution.
//
// Mainly used for tanh activation function.
type XavierNormalInitialization struct{}

func (x XavierNormalInitialization) Generate(layerSize [2]int) float64 {
	return rand.NormFloat64() * math.Sqrt(float64(2)/float64(layerSize[0]+layerSize[1]))
}

// XavierNormalInitialization (also known as Glorot) is a weight initialization technique
// based on the uniform distribution.
//
// Mainly used for sigmoid activation function.
type XavierUniformInitialization struct{}

func (x XavierUniformInitialization) Generate(layerSize [2]int) float64 {
	limit := math.Sqrt(float64(6) / float64(layerSize[0]+layerSize[1]))

	// Generate a random number in the range [-x, x]
	return (rand.Float64()*2 - 1.0) * limit
}

// HeInitialization is a weight initialization technique, which uses the normal distribution
// with a standard deviation using the previous layer size.
//
// Mainly used with ReLU activation function to account for the zeros in (-inf;0] range.
type HeInitialization struct{}

func (h HeInitialization) Generate(layerSize [2]int) float64 {
	return rand.NormFloat64() * math.Sqrt(float64(2)/float64(layerSize[0]))
}

/******************************************************/
