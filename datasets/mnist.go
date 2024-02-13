package datasets

import (
	"encoding/csv"
	"os"
	"strconv"
)

// MnistDataset reads the CSV files for MNIST dataset.
//
// Accepts a path to the .csv file, which has to be in format:
// N rows, first column is the label, next 784 columns - pixels of the image in range from 0 to 255.
//
// Outputs a list of pixels, where each entry is a separate image and
// a slice of labels corresponding to the images.
//
// Each slice of X represents a whole 28x28 image with values ranging from 0 to 255, i.e. length of 784.
// In order to feed it to the neural network, outputs have to be transposed and labels must be one-hot encoded.
func MnistDataset(filename string) ([][]float64, []float64, error) {
	// Open the CSV file
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	// Create a CSV reader
	reader := csv.NewReader(file)

	var data [][]float64
	var currentLine []float64
	var labels []float64

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

		data = append(data, currentLine)
		currentLine = make([]float64, 0)
		labels = append(labels, label)
	}

	return data, labels, nil
}
