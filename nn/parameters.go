package nn

const defaultLearningRate = 0.01
const defaultWeightDecay = 0.1
const defaultIterationCount = 100

type NeuralNetworkParameters struct {
	LearningRate   float64
	WeightDecay    float64
	IterationCount uint64
}

func validateParameters(nnp *NeuralNetworkParameters) {
	if nnp.LearningRate == 0 {
		nnp.LearningRate = defaultLearningRate
	}
	if nnp.WeightDecay == 0 {
		nnp.WeightDecay = defaultWeightDecay
	}
	if nnp.IterationCount == 0 {
		nnp.IterationCount = defaultIterationCount
	}
}
