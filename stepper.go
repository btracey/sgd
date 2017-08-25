package sgd

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// resize takes x and returns a slice of length dim. It returns a resliced x
// if cap(x) >= dim, and a new slice otherwise.
// TODO(btracey): This zeros x before it returns it, but the version in optimize
// doesn't. These should be rectified.
func resize(x []float64, dim int) []float64 {
	if dim > cap(x) {
		return make([]float64, dim)
	}
	x = x[:dim]
	for i := range x {
		x[i] = 0
	}
	return x
}

// Stepper is an interface that sets the step size for the next update to
// the gradient descent. The calling algorithm should update
//  θ += step
// where θ are the parameters being optimized.
type Stepper interface {
	Init(grad []float64)
	Step(step, grad []float64)
}

// Anneal is a stepper that has a step size which is annealed over time.
// Anneal computes the step as
//  step = -η * df/dx
//  η *= α
// See the struct field documentation for more information.
type Anneal struct {
	// Size sets the initial size of the step parameter η. If Init is zero,
	// the default value of 1 is used.
	Size float64
	// Rate sets the annealing rate α. If Rate is zero, a default value of 0.99
	// is used.
	Rate float64

	curr float64
}

func (a *Anneal) Init(grad []float64) {
	a.curr = a.Size
	if a.curr == 0 {
		a.curr = 1
	}
	if a.Rate == 0 {
		a.Rate = 0.99
	}
}

func (a *Anneal) Step(step, grad []float64) {
	copy(step, grad)
	floats.Scale(-a.curr, step)
	a.curr *= a.Rate
}

// Momentum is a stepper that implements a momentum-based step direction.
// Specifically, Momentum sets
//  ν_t = γ * ν_{t-1} + η * df/dx
//  step = - ν_t
//  η *= α
type Momentum struct {
	// Size sets the initial size of the step parameter η. If Init is zero,
	// the default value of 1 is used.
	Size float64
	// Rate sets the annealing rate α. If Rate is zero, a default value of 0.99
	// is used.
	Rate float64
	// Momen sets the momentum term γ. If Momen is 0, a default value of 0.9 is used.
	Momen float64

	curr float64
	nu   []float64
}

func (m *Momentum) Init(grad []float64) {
	m.curr = m.Size
	if m.curr == 0 {
		m.curr = 1
	}
	if m.Rate == 0 {
		m.Rate = 0.99
	}
	if m.Momen == 0 {
		m.Momen = 0.9
	}
	m.nu = resize(m.nu, len(grad))
}

func (m *Momentum) Step(step, grad []float64) {
	// Compute ν_t
	copy(step, grad)
	floats.Scale(m.curr, step)
	floats.AddScaled(step, m.Momen, m.nu)

	// Set for next iteration.
	copy(m.nu, step)
	m.curr *= m.Rate

	// return the step.
	floats.Scale(-1, step)
}

// Adagrad is a stepper with a per-parameter step size that is adjusted
// automatically based on past gradient evaluations. Adagrad updates as
//  G_{t,i} += df/dx ⊙ df/dx
//  step = -η/\sqrt(G_{t,i} + ϵ) ⊙ df/dx
// where ⊙ is the element-wise product.
type Adagrad struct {
	// Size sets the value of the parameter η. If Size is 0, a default value of
	// 0.01 is used.
	Size float64
	// Smooth sets the value of the smoothing parameter ϵ. If Smooth is 0, a
	// default value of 1e-8 is used.
	Smooth float64

	g []float64
}

func (a *Adagrad) Init(grad []float64) {
	if a.Size == 0 {
		a.Size = 0.01
	}
	if a.Smooth == 0 {
		a.Smooth = 1e-8
	}
	a.g = resize(a.g, len(grad))
}

func (a *Adagrad) Step(step, grad []float64) {
	copy(step, grad)
	for i, v := range grad {
		a.g[i] += v * v
		step[i] *= -a.Size / math.Sqrt(a.g[i]+a.Smooth)
	}
}

// Adadelta is a stepper with a per-parameter step size that is adjusted
// automatically based on past gradient evaluations. Adadelta updates as
//  step = - \sqrt(E[s_{t-1}] + ϵ) / \sqrt(E[g_t] + ϵ) ⊙ df/dx
// where
//  E[g_t] = (1-γ) df/dx ⊙ df/dx + γ * g_{t-1}
//  E[s_t] = (1-γ) Δθ_t ⊙ Δθ_t + γ * s_{t-2}
type Adadelta struct {
	// Smooth sets the value of the smoothing parameter ϵ. If Smooth is 0, a
	// default value of 1e-8 is used.
	Smooth float64
	// Momen sets the momentum term γ. If Momen is 0, a default value of 0.9 is used.
	Momen float64

	s []float64
	g []float64
}

func (a *Adadelta) Init(grad []float64) {
	if a.Smooth == 0 {
		a.Smooth = 1e-8
	}
	if a.Momen == 0 {
		a.Momen = 0.9
	}
	a.s = resize(a.s, len(grad))
	a.g = resize(a.g, len(grad))
}

func (a *Adadelta) Step(step, grad []float64) {
	for i, v := range grad {
		a.g[i] = (1-a.Momen)*v*v + a.Momen*a.g[i]
		step[i] = -math.Sqrt(a.s[i]+a.Smooth) / math.Sqrt(a.g[i]+a.Smooth) * v
		a.s[i] = (1-a.Momen)*step[i]*step[i] + a.Momen*a.s[i]
	}
}

// Adam is a stepper with a per-parameter step size that uses both a decaying
// average of past gradients and a momentum term.
//  m_t = γ_1 * m_{t-1} + (1-γ_1) * g_t
//  m̂_t = m_t/(1-γ_1^t)
//  ν_t = γ_2 * ν_{t-1} + (1-γ_2) * g_t ⊙ g_t
//  ν̂_t = ν_t/(1-γ_2^t)
//  step = - η/(sqrt(ν̂_t)+ϵ) ⊙ m̂_t
type Adam struct {
	// Size sets the value of the parameter η. If Size is 0, a default value of
	// 0.001 is used.
	Size float64
	// MeanMomen sets the momentum term for the mean gradient γ_1. If MeanMomen
	// is 0, a default value of 0.9 is used.
	MeanMomen float64
	// VarMomen sets the momentum term for the variance of the gradient γ_2.
	// If VarMomen is 0, a default value of 0.999 is used.
	VarMomen float64
	// Smooth sets the value of the smoothing parameter ϵ. If Smooth is 0, a
	// default value of 1e-8 is used.
	Smooth float64

	time float64
	m    []float64
	nu   []float64
}

func (a *Adam) Init(grad []float64) {
	if a.Size == 0 {
		a.Size = 0.001
	}
	if a.MeanMomen == 0 {
		a.MeanMomen = 0.9
	}
	if a.VarMomen == 0 {
		a.VarMomen = 0.999
	}
	if a.Smooth == 0 {
		a.Smooth = 1e-8
	}
	a.time = 0
	a.m = resize(a.m, len(grad))
	a.nu = resize(a.nu, len(grad))
}

func (a *Adam) Step(step, grad []float64) {
	a.time++
	for i, v := range grad {
		a.m[i] = a.MeanMomen*a.m[i] + (1-a.MeanMomen)*v
		a.nu[i] = a.VarMomen*a.nu[i] + (1-a.VarMomen)*v*v
		mhat := math.Pow(a.m[i], a.time)
		nuhat := math.Pow(a.nu[i], a.time)
		step[i] = -a.Size / (math.Sqrt(nuhat) + a.Smooth) * mhat
	}
}
