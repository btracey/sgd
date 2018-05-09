package sgd

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/sampleuv"
)

// Batcher generates minibatches from a dataset of a fixed size.
type Batcher interface {
	// Init initializes the Batcher for a dataset of a size specified by nSamples.
	Init(nSamples int)
	// Batch returns the indices of the next minibatch to evaluate. The returned
	// slice will not be modified.
	Batch() []int
}

// RandomBatch generates a minibatch of the specified size at random from the
// total dataset.
type RandomBatch struct {
	// Size is the minibatch size.
	Size int
	// Replacement sets if the minibatch can have the same sample multiple times
	// in the minibatch.
	Replacement bool
	// Source sets the random number source
	Source rand.Source

	nData int
	idxs  []int
}

func (r *RandomBatch) Init(nSamples int) {
	r.nData = nSamples
	r.idxs = make([]int, r.Size)
}

func (r *RandomBatch) Batch() []int {
	if r.Replacement {
		// Replacement okay.
		intn := rand.Intn
		if r.Source != nil {
			intn = rand.New(r.Source).Intn
		}
		for i := range r.idxs {
			r.idxs[i] = intn(r.nData)
		}
	} else {
		sampleuv.WithoutReplacement(r.idxs, r.nData, r.Source)
	}
	return r.idxs
}
