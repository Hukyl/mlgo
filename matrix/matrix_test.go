package matrix_test

import (
	"testing"

	m "github.com/Hukyl/mlgo/matrix"
)

func TestBroadcast(t *testing.T) {
	m1, _ := m.NewMatrix([][]int{
		{1, 2, 3},
		{4, 5, 6},
	})
	var err error
	err = m1.Broadcast(2, 4)
	if err == nil {
		t.Error("didn't produce column error")
	}
	err = m1.Broadcast(3, 3)
	if err == nil {
		t.Error("didn't produce row error")
	}
	want, _ := m.NewMatrix([][]int{
		{1, 2, 3, 1, 2, 3, 1, 2, 3},
		{4, 5, 6, 4, 5, 6, 4, 5, 6},
		{1, 2, 3, 1, 2, 3, 1, 2, 3},
		{4, 5, 6, 4, 5, 6, 4, 5, 6},
	})
	err = m1.Broadcast(4, 9)
	if err != nil {
		t.Error("broadcast false negative")

	}
	if !m1.Equals(want) {
		t.Error("invalid broadcasting result")
	}
}
