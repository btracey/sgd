package sgd

import "math/rand"

// Batcher returns the (mini) batch indices. The returned slice
// will not be modified.
type Batcher interface {
	Batch() []int
}

type RandomBatch struct {
	Size        int  // minibatch size
	Replacement bool // Can the random indices be generated with replacement.
	Source      *rand.Rand

	data []int
}

func (r *RandomBatch) Batch(max int) []int {
	if r.data == nil {
		r.data = make([]int, r.Size)
	}
	for i := range r.data {
		if r.Source == nil {
			r.data[i] = rand.Intn(max)
		} else {
			r.data[i] = r.Source.Intn(max)
		}
	}
	return r.data
}
