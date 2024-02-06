package utils

import (
	"math"

	"github.com/Hukyl/mlgo/metric"
)

const defaultEpochCount = 5
const defaultLearningRate = 0.01

type NeuralNetworkParameters struct {
	currentEpoch uint64
	EpochCount   uint64

	LearningRateDecay   float64
	InitialLearningRate float64
	WeightDecay         float64
	ClipValue           float64

	AccuracyMetric metric.Metric

	DumpPath string
}

func (nnp NeuralNetworkParameters) LearningRate() float64 {
	return nnp.InitialLearningRate / (1 + nnp.LearningRateDecay*float64(nnp.currentEpoch))
}

func (nnp *NeuralNetworkParameters) Validate() {
	if nnp.InitialLearningRate == 0 {
		nnp.InitialLearningRate = defaultLearningRate
	}
	if nnp.EpochCount == 0 {
		nnp.EpochCount = defaultEpochCount
	}
	if nnp.ClipValue == 0 {
		nnp.ClipValue = math.Inf(1)
	}
}

func (nnp *NeuralNetworkParameters) ResetEpoch() {
	nnp.currentEpoch = 0
}

func (nnp *NeuralNetworkParameters) IncrementEpoch() {
	nnp.currentEpoch++
}
