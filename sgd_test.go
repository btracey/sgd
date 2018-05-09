package sgd

import (
	"fmt"
	"testing"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/stat/distuv"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// LeastSquares is a type for solving least squares problems with sgd.
type LeastSquares struct {
	X *mat.Dense
	Y []float64
}

func (l LeastSquares) Problem() Problem {
	r, c := l.X.Dims()
	return Problem{
		Dim:  c,
		Size: r,
		Func: l.Func,
		Grad: l.Grad,
	}
}

func (l LeastSquares) Optimal() []float64 {
	r, c := l.X.Dims()
	if len(l.Y) != r {
		panic("size mismatch")
	}
	yVec := mat.NewVecDense(len(l.Y), l.Y)

	tmp := make([]float64, c)
	tmpVec := mat.NewVecDense(len(tmp), tmp)

	err := tmpVec.SolveVec(l.X, yVec)
	if err != nil {
		panic("singular")
	}
	return tmp
}

func (l LeastSquares) Func(dst, params []float64, batch []int) {
	l.funcGrad(nil, dst, params, batch)
}

func (l LeastSquares) Grad(dst *mat.Dense, params []float64, batch []int) {
	l.funcGrad(dst, nil, params, batch)
}

func (l LeastSquares) funcGrad(gradDst *mat.Dense, funDst, params []float64, batch []int) {
	for i, idx := range batch {
		x := l.X.RawRowView(idx)
		diff := floats.Dot(x, params) - l.Y[idx]
		if funDst != nil {
			funDst[i] = diff * diff
		}
		if gradDst != nil {
			g := gradDst.RawRowView(i)
			for j := range g {
				g[j] = 2 * diff * x[j]
			}
		}
	}
}

func constructLeastSquares(trueParam []float64, noise float64, offset bool, nData int, source rand.Source) *LeastSquares {
	norm := rand.New(source).NormFloat64
	dim := len(trueParam)
	xs := mat.NewDense(nData, len(trueParam), nil)
	ys := make([]float64, nData)
	for i := 0; i < nData; i++ {
		if offset {
			xs.Set(i, 0, 1)
		} else {
			xs.Set(i, 0, norm())
		}
		for j := 1; j < dim; j++ {
			xs.Set(i, j, norm())
		}

		x := xs.RawRowView(i)
		y := floats.Dot(trueParam, x) + distuv.Normal{Mu: 0, Sigma: noise, Src: source}.Rand()
		ys[i] = y
	}
	return &LeastSquares{
		X: xs,
		Y: ys,
	}
}

// Set of least squares problems to test the steppers on.
func leastSquaresList() []*LeastSquares {
	var lses []*LeastSquares
	src := rand.NewSource(1)
	for _, v := range []struct {
		param  []float64
		noise  float64
		offset bool
		nData  int
	}{
		// Simple linear fit.
		{
			param:  []float64{7, 8},
			noise:  1e-2,
			offset: true,
			nData:  10,
		},
		// Simple linear fit with no offset.
		{
			param:  []float64{7, 8},
			noise:  1e-2,
			offset: false,
			nData:  10,
		},
		// Simple linear fit with larger data.
		{
			param:  []float64{7, 8},
			noise:  1e-2,
			offset: false,
			nData:  1000,
		},
	} {
		ls := constructLeastSquares(v.param, v.noise, v.offset, v.nData, src)
		lses = append(lses, ls)
	}
	return lses
}

func randBatchersList() []*RandomBatch {
	src := rand.NewSource(2)
	return []*RandomBatch{
		{
			Size:        5,
			Replacement: true,
			Source:      src,
		},
		{
			Size:        5,
			Replacement: false,
			Source:      src,
		},
		{
			Size:        10,
			Replacement: false,
			Source:      src,
		},
	}
}

type SGDTest interface {
	Optimal() []float64
	Problem() Problem
}

var sgdProbs []SGDTest
var batchers []Batcher

func init() {
	lses := leastSquaresList()
	for i := range lses {
		sgdProbs = append(sgdProbs, lses[i])
	}
	randbatch := randBatchersList()
	for i := range randbatch {
		batchers = append(batchers, randbatch[i])
	}
}

func TestAnneal(t *testing.T) {
	// Test each stepper with the problems and batchers.
	for cas, stepper := range []Stepper{
		//&Adadelta{}, // no
		&Adagrad{}, // no
		// &Adam{}, // does'nt crash, but wrong
		//&Anneal{}, // yes
		//&Momentum{}, // fails one of them,
		//&Nesterov{}, // no
		//&RMSProp{}, // no
	} {
		for p, prob := range sgdProbs {
			for b, batcher := range batchers {
				optimal := prob.Optimal()
				problem := prob.Problem()

				testStr := fmt.Sprintf("cas =%v, p = %v, b = %v", cas, p, b)
				// TODO(btracey): add in a variety of steppers.
				result, err := SGD(problem, batcher, stepper, nil)
				if err != nil {
					t.Errorf("unexepected error: "+testStr+":", err)
				}
				if !floats.EqualApprox(result.X, optimal, 1e-2) {
					t.Errorf("Optimal mismatch:"+testStr+": got %v, want %v", result.X, optimal)
				}
			}
		}
	}
}
