package datasets

import (
	"encoding/csv"
	"os"
	"strconv"
)

func ReadMnistCSV(filename string) ([]float64, [][]float64, error) {
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

	return labels, data, nil
}
