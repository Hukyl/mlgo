package nn

import (
	"github.com/Hukyl/mlgo/metric"
)

const defaultEpochCount = 5
const defaultLearningRate = 0.01
const defaultWeightDecay = 0.1

type NeuralNetworkParameters struct {
	currentEpoch uint64
	EpochCount   uint64

	LearningRateDecay   float64
	InitialLearningRate float64
	WeightDecay         float64

	AccuracyMetric metric.Metric

	DumpPath string
}

func (nnp NeuralNetworkParameters) LearningRate() float64 {
	return nnp.InitialLearningRate / (1 + nnp.LearningRateDecay*float64(nnp.currentEpoch))
}

func validateParameters(nnp *NeuralNetworkParameters) {
	if nnp.InitialLearningRate == 0 {
		nnp.InitialLearningRate = defaultLearningRate
	}
	if nnp.WeightDecay == 0 {
		nnp.WeightDecay = defaultWeightDecay
	}
	if nnp.EpochCount == 0 {
		nnp.EpochCount = defaultEpochCount
	}
}
