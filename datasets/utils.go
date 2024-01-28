package datasets

import (
	"encoding/csv"
	"os"
	"strconv"

	"github.com/Hukyl/mlgo/matrix"
)

func readAndBatchCSV(filename string, batchSize int) ([][]float64, [][][]float64, error) {
	// Open the CSV file
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	// Create a CSV reader
	reader := csv.NewReader(file)

	// Initialize variables
	var data [][][]float64
	var labels [][]float64
	var currentDataBatch [][]float64
	var currnetLabelBatch []float64
	var currentLine []float64

	reader.Read() // skip header
	// Read records from the CSV file
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		label, _ := strconv.ParseFloat(record[0], 64)
		// Convert string values to float64 and append to the current batch
		for _, col := range record[1:] {
			value, err := strconv.ParseFloat(col, 64)
			if err != nil {
				return nil, nil, err
			}
			currentLine = append(currentLine, value)
		}

		currentDataBatch = append(currentDataBatch, currentLine)
		currentLine = make([]float64, 0)
		currnetLabelBatch = append(currnetLabelBatch, label)

		// If the current batch reaches the desired size, append to data and reset the batch
		if len(currentDataBatch) >= batchSize {
			data = append(data, currentDataBatch)
			labels = append(labels, currnetLabelBatch)
			currentDataBatch = make([][]float64, 0)
			currnetLabelBatch = make([]float64, 0)
		}

	}

	// Append the remaining elements of the last batch
	if len(currentDataBatch) > 0 {
		data = append(data, currentDataBatch)
		labels = append(labels, currnetLabelBatch)
	}

	return labels, data, nil
}

func translateToMatrix(labels [][]float64, data [][][]float64) ([]matrix.Matrix[float64], []matrix.Matrix[float64]) {
	matrixLabels := make([]matrix.Matrix[float64], 0, len(labels))
	for _, labelBatch := range labels {
		batchMatrix := matrix.NewZeroMatrix[float64](10, len(labelBatch))
		// batchMatrix := matrix.NewOnesMatrix(10, len(labelBatch))
		// batchMatrix = batchMatrix.MultiplyByScalar(0.001)
		for j, label := range labelBatch {
			batchMatrix.Set(int(label), j, 1)
		}
		matrixLabels = append(matrixLabels, batchMatrix)
	}

	matrixData := make([]matrix.Matrix[float64], 0, len(data))
	for _, dataBatch := range data {
		batchMatrix, _ := matrix.NewMatrix(dataBatch)
		batchMatrix = batchMatrix.T()
		matrix.ApplyByElement(batchMatrix, func(x float64) float64 { return x / 255.0 })
		matrixData = append(matrixData, batchMatrix)
	}

	return matrixData, matrixLabels
}
