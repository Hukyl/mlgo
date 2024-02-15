package utils

import (
	"math"

	"github.com/Hukyl/mlgo/metric"
)

const defaultEpochCount = 5
const defaultLearningRate = 0.01

// BackupParameters manages data associated to creating the dumps for an ANN.
//
// ToCreate determines whether to create the dumps at all.
//
// Path specifies the folder, where the dumps should be stored.
type BackupParameters struct {
	ToCreate bool
	Path     string
}

// NeuralNetworkParameters containes some parameters to be applied to an ANN
// during its training.
//
// EpochCount specifies the total epochs for the ANN to train. If not set, initialized
// to 5.
//
// LearningRateDecay is used to degrade the learning rate after some epochs. Algorithm
// is based on the inverse of epochs passed during training.
//
// InitialLearningRate is the starting value for the learning rate. If LearningRateDecay
// is 0, this learning rate is kept during the whole training.
//
// WeightDecay is a L2 reguralization technique to enforce the model to improve weights
// using smaller absolute values. This helps in preventing gradient exploding and
// overfitting of the model, though if the value is too large can cause to underfit.
//
// ClipValue is the absolute value by which the gradient must be clipped to reduce
// the sudden changes in the weights.
//
// AccuracyMetric is a metric of calculating how many correct outputs were guessed during
// training. Output for this function is usually used in the logs for the epoch summary.
//
// Backups is a struct containing backup variables to manages ANN dumps.
type NeuralNetworkParameters struct {
	currentEpoch uint64
	EpochCount   uint64

	LearningRateDecay   float64
	InitialLearningRate float64
	WeightDecay         float64
	ClipValue           float64

	AccuracyMetric metric.Metric

	Backups BackupParameters
}

// LearningRate returns the current learning rate of the ANN. The return value
// of this funciton may depend on the epoch passed and learning rate decay value.
func (nnp NeuralNetworkParameters) LearningRate() float64 {
	return nnp.InitialLearningRate / (1 + nnp.LearningRateDecay*float64(nnp.currentEpoch))
}

// Validate updates the values of the hyperparameters to be valid.
// Currently updated parameters are:
//   - InitialLearningRate: set to 0.01
//   - EpochCount: set to 5
//   - ClipValue: if not provided, set to +inf
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

// ResetEpoch resets current epoch count to 0. Epoch count may influence
// the learning rate, based on learning rate decay.
func (nnp *NeuralNetworkParameters) ResetEpoch() {
	nnp.currentEpoch = 0
}

// IncrementEpoch increments current epoch count by 1. Epoch count may influence
// the learning rate, based on learning rate decay.
func (nnp *NeuralNetworkParameters) IncrementEpoch() {
	nnp.currentEpoch++
}
