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
			param:  []float64{0.7, 0.8},
			noise:  1e-2,
			offset: true,
			nData:  50,
		},
		// Simple linear fit with no offset.
		{
			param:  []float64{0.7, 0.8},
			noise:  1e-2,
			offset: false,
			nData:  50,
		},
		// Simple linear fit with larger data.
		{
			param:  []float64{0.7, 0.8},
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
	src := rand.NewSource(3)
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

func TestSteppers(t *testing.T) {
	// Test each stepper with the problems and batchers.
	for cas, step := range []struct {
		Name     string
		Stepper  Stepper
		Settings *Settings
		AnsTol   float64
	}{
		{
			Name:    "Adadelta",
			Stepper: &Adadelta{},
			Settings: &Settings{
				Iterations:    100000,
				StepTolerance: -1, // The steps get really small but crawl to the minimum.
			},
			AnsTol: 3e-3,
		},
		{
			Name:    "Adagrad",
			Stepper: &Adagrad{},
			Settings: &Settings{
				Iterations:    1000000,
				StepTolerance: -1, // Adagrad doesn't converge to a specific step size but oscilates around the true value.
			},
			AnsTol: 2e-2,
		},
		{
			Name:    "Adam",
			Stepper: &Adam{},
			Settings: &Settings{
				StepTolerance: 1e-6,
			},
			AnsTol: 3e-3,
		},
		{
			Name:    "Anneal",
			Stepper: &Anneal{},
			Settings: &Settings{
				StepTolerance: 1e-8,
			},
			AnsTol: 1e-3,
		},
		{
			Name:    "Momentum",
			Stepper: &Momentum{},
			Settings: &Settings{
				StepTolerance: 1e-8,
			},
			AnsTol: 3e-3,
		},
		{
			Name:    "Nesterov",
			Stepper: &Nesterov{},
			Settings: &Settings{
				StepTolerance: 1e-6,
			},
			AnsTol: 5e-3,
		},
		{
			Name:    "RMSProp",
			Stepper: &RMSProp{},
			Settings: &Settings{
				Iterations:    100000,
				StepTolerance: 1e-6,
			},
			AnsTol: 5e-3,
		},
	} {
		for p, prob := range sgdProbs {
			for b, batcher := range batchers {
				t.Run(fmt.Sprintf("Name: %v, cas =%v, p = %v, b = %v", step.Name, cas, p, b), func(t *testing.T) {
					optimal := prob.Optimal()
					problem := prob.Problem()
					settings := step.Settings
					stepper := step.Stepper
					result, err := SGD(problem, batcher, stepper, settings)
					if err != nil {
						t.Errorf("unexepected error: %v", err)
					}
					if !floats.EqualApprox(result.X, optimal, step.AnsTol) {
						t.Errorf("Optimal mismatch:: got %v, want %v", result.X, optimal)
					}
				})
			}
		}
	}
}
