package sgd

import (
	"gonum.org/v1/gonum/mat"
)

// resizeZero takes x and returns a slice of length dim. It returns a resliced x
// if cap(x) >= dim, and a new slice otherwise.
// TODO(btracey): This zeros x before it returns it, but the version in optimize
// doesn't. These should be rectified.
func resizeZero(x []float64, dim int) []float64 {
	if dim > cap(x) {
		return make([]float64, dim)
	}
	x = x[:dim]
	for i := range x {
		x[i] = 0
	}
	return x
}

func resizeMat(x *mat.Dense, r, c int) *mat.Dense {
	var rCap, cCap int
	if x != nil {
		rCap, cCap = x.Caps()
	}
	if rCap < r || cCap < c {
		// Not enough space, allocate new.
		return mat.NewDense(r, c, nil)
	}
	return x.Slice(0, r, 0, c).(*mat.Dense)
}

/*
// SubsetDense is a mat.Matrix represented by a subset of the rows of a Dense matrix.
type SubsetDense struct {
	Data *mat.Dense // Underlying index
	Rows []int      // Rows considered to be a part of the matrix.
}

func (s SubsetDense) Dims() (r, c int) {
	_, c = s.Data.Dims()
	r = len(s.Rows)
	return r, c
}

func (s SubsetDense) At(i, j int) float64 {
	i = s.Rows[i]
	return s.Data.At(i, j)
}

func (s SubsetDense) T() mat.Matrix {
	return mat.Transpose{s}
}

func (s SubsetDense) RawRowView(i int) []float64 {
	i = s.Rows[i]
	return s.Data.RawRowView(i)
}
*/
