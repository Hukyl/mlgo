package layers

import (
	"errors"
	"math"
	"math/rand"
	"sync"

	"github.com/Hukyl/mlgo/activation"
	. "github.com/Hukyl/mlgo/matrix"
)

// NewDense produces a new fully-connected layer of neurons using given weights and biases.
//
// Returns error if weights and biases sizes are non-conformable.
func NewDense(W, b Matrix[float64], a activation.ActivationFunction) (Layer, error) {
	l := &dense{weights: W, bias: b, activation: a}
	if b.Size()[0] != l.OutputSize()[0] {
		return nil, errors.New("invalid bias size")
	}
	return l, nil
}

// NewRandomDense produces a dense layer using a given weight initialization method.
//
// Beware that different weight initialization techniques are better suited for different
// activation functions, for example:
//   - ReLU - HeInitialization
//   - Sigmoid - XavierUniformInitializations
//
// and so on.
func NewRandomDense(weightSize [2]int, a activation.ActivationFunction, wi WeightInitialization) Layer {
	W := NewZeroMatrix[float64](weightSize[1], weightSize[0])
	for i := 0; i < weightSize[1]; i++ {
		for j := 0; j < weightSize[0]; j++ {
			W.Set(i, j, wi.Generate(weightSize))
		}
	}
	b := NewZeroMatrix[float64](weightSize[1], 1)
	return &dense{weights: W, bias: b, activation: a}
}

// NewDropout produces a dropout layer, which nullifies random neurons to reduce
// overfitting of the model.
//
// Neurons are nullified at random with different neurons being deactivated in
// different samples with some `rate`.
func NewDropout(inputSize int, rate float64) Layer {
	return &dropout{inputSize: inputSize, rate: rate}
}

/**********************************************************************/

func uniformMatrix(size [2]int, min, max float64) Matrix[float64] {
	m := NewZeroMatrix[float64](size[0], size[1])

	wg := sync.WaitGroup{}
	wg.Add(m.ColumnCount())
	for j := 0; j < m.ColumnCount(); j++ {
		go func(j int) {
			defer wg.Done()
			for i := 0; i < m.RowCount(); i++ {
				f := rand.NormFloat64()
				normalized := 0.5 * (1 + math.Erf(f/math.Sqrt2))
				projected := min + normalized*(max-min)
				m.Set(i, j, projected)
			}
		}(j)
	}
	wg.Wait()

	return m
}
