// package sgdopt implements optimization routines for performing stochastic
// gradient descent. That is, for functions which are the sum of functions and
// can be optimized through minibatches.
package sgd

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

type Settings struct {
	// Iterations represents the maximum number of iterations (calls to stepper).
	// If Iterations is 0, then an unlimited iteration count is used by default.
	Iterations int

	// StepTolerance sets the stopping tolerance on the step size. If StepTolerance
	// is zero then it is defaulted to a value of 1e-8.
	StepTolerance float64
}

func defaultSettings(set *Settings) {
	if set.StepTolerance == 0 {
		set.StepTolerance = 1e-8
	}
}

type Result struct {
	X      []float64
	Status optimize.Status
}

func SGD(problem Problem, batcher Batcher, stepper Stepper, settings *Settings) (*Result, error) {
	var set Settings
	if settings != nil {
		set = *settings
	}
	dim := problem.Dim
	if problem.Dim == 0 {
		panic("sgd: problem dimension is 0")
	}
	size := problem.Size
	if problem.Size == 0 {
		panic("sgd: problem size is 0")
	}
	batcher.Init(size)
	stepper.Init(dim)
	defaultSettings(&set)
	status := optimize.NotTerminated
	var dstFun []float64
	var avgGrad []float64
	var dstGrads *mat.Dense
	parameters := make([]float64, dim)
	step := make([]float64, dim)
	for iter := 0; ; iter++ {
		if set.Iterations != 0 && iter > set.Iterations {
			status = optimize.IterationLimit
			break
		}
		batch := batcher.Batch()

		// Evaluate the function and gradient.
		// TODO(btracey): Will need the function for certain convergence measures
		// and recording things.
		nData := len(batch)
		dstFun = resizeZero(dstFun, nData)
		dstGrads = resizeMat(dstGrads, nData, dim)

		problem.Func(dstFun, parameters, batch)
		problem.Grad(dstGrads, parameters, batch)

		// Given the gradient, update the step.
		avgGrad = resizeZero(avgGrad, dim)
		for i := 0; i < nData; i++ {
			floats.Add(avgGrad, dstGrads.RawRowView(i))
		}
		floats.Scale(1/float64(nData), avgGrad)
		stepper.Step(step, avgGrad)
		//fmt.Println(step)

		stepNorm := floats.Norm(step, 2)
		//fmt.Println(parameters)
		//fmt.Println(stepNorm)
		if math.IsNaN(stepNorm) || math.IsInf(stepNorm, 0) {
			status = optimize.Failure
			break
		}
		if stepNorm < set.StepTolerance {
			status = optimize.StepConvergence
			break
		}
		floats.Add(parameters, step)
		//fmt.Println("parameters = ", parameters)
	}
	return &Result{
		X:      parameters,
		Status: status,
	}, nil
}

// Problem is a function for running stochastic gradient descent.
type Problem struct {
	// Dim is the dimension of the parameters of the problem.
	Dim int
	// Size specifies the total number of locations (to pass to the Batcher).
	Size int
	// Func computes the function value for the indices specified by idxs and
	// stores the values into dst.
	Func func(dst, param []float64, idxs []int)
	// Grad computes the gradient of the function value for all of the indices
	// specified by idxs and stores them in-place into dst.
	Grad func(dst *mat.Dense, param []float64, idxs []int)
}
