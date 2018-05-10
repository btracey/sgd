package sgd

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// Good review article: http://ruder.io/optimizing-gradient-descent/
//  with arXiv https://arxiv.org/abs/1609.04747

// Stepper is an interface that sets the step size for the next update to
// the gradient descent. The calling algorithm should update
//  θ += step
// where θ are the parameters being optimized.
type Stepper interface {
	Init(dim int)
	Step(step, grad []float64)
}

var (
	_ Stepper = &Adadelta{}
	_ Stepper = &Adagrad{}
	_ Stepper = &Adam{}
	_ Stepper = &Anneal{}
	_ Stepper = &Momentum{}
	_ Stepper = &Nesterov{}
	_ Stepper = &RMSProp{}
)

// Adadelta is a stepper with a per-parameter step size that is adjusted
// automatically based on past gradient evaluations. Adadelta updates as
//  step = - \sqrt(E[s_{t-1}] + ϵ) / \sqrt(E[g_t] + ϵ) ⊙ df/dx
// where
//  E[g_t] = (1-γ) df/dx ⊙ df/dx + γ * g_{t-1}
//  E[s_t] = (1-γ) Δθ_t ⊙ Δθ_t + γ * s_{t-1}
type Adadelta struct {
	// Smooth sets the value of the smoothing parameter ϵ. If Smooth is 0, a
	// default value of 1e-8 is used.
	Smooth float64
	// Momen sets the momentum term γ. If Momen is 0, a default value of 0.9 is used.
	Momen float64

	s []float64
	g []float64
}

func (a *Adadelta) Init(dim int) {
	if a.Smooth == 0 {
		a.Smooth = 1e-8
	}
	if a.Momen == 0 {
		a.Momen = 0.9
	}
	a.s = resizeZero(a.s, dim)
	a.g = resizeZero(a.g, dim)
}

func (a *Adadelta) Step(step, grad []float64) {
	for i, v := range grad {
		a.g[i] = (1-a.Momen)*v*v + a.Momen*a.g[i]
		step[i] = -math.Sqrt(a.s[i]+a.Smooth) / math.Sqrt(a.g[i]+a.Smooth) * v
		a.s[i] = (1-a.Momen)*step[i]*step[i] + a.Momen*a.s[i]
	}
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

func (a *Adagrad) Init(dim int) {
	if a.Size == 0 {
		a.Size = 0.01
	}
	if a.Smooth == 0 {
		a.Smooth = 1e-8
	}
	a.g = resizeZero(a.g, dim)
}

func (a *Adagrad) Step(step, grad []float64) {
	for i, v := range grad {
		a.g[i] += v * v
		step[i] = -a.Size * grad[i] / math.Sqrt(a.g[i]+a.Smooth)
	}
}

// Adam is a stepper with a per-parameter step size that uses both a decaying
// average of past gradients and a momentum term.
//  m_t = γ_1 * m_{t-1} + (1-γ_1) * g_t
//  ν_t = γ_2 * ν_{t-1} + (1-γ_2) * g_t ⊙ g_t
//  m̂_t = m_t/(1-γ_1^t)
//  ν̂_t = ν_t/(1-γ_2^t)
//  step = - η/(sqrt(ν̂_t)+ϵ) ⊙ m̂_t
// For more information see https://arxiv.org/pdf/1412.6980.pdf .
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

func (a *Adam) Init(dim int) {
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
	a.m = resizeZero(a.m, dim)
	a.nu = resizeZero(a.nu, dim)
}

func (a *Adam) Step(step, grad []float64) {
	a.time++
	for i, v := range grad {
		a.m[i] = a.MeanMomen*a.m[i] + (1-a.MeanMomen)*v
		a.nu[i] = a.VarMomen*a.nu[i] + (1-a.VarMomen)*v*v
		mhat := a.m[i] / (1 - math.Pow(a.MeanMomen, a.time))
		nuhat := a.nu[i] / (1 - math.Pow(a.VarMomen, a.time))
		step[i] = -a.Size / (math.Sqrt(nuhat) + a.Smooth) * mhat
	}
}

// Anneal is a stepper that has a step size which is annealed over time.
// Anneal computes the step as
//  η = a / (b + t)
//  step = -η * df/dx
type Anneal struct {
	// Size sets the initial value of the step-size parameter a. If Size is zero,
	// the default value of 1 is used.
	Size float64
	// Offset sets the initial value of the step-size parameter a. If Offset is zero,
	// the default value of 1 is used.
	Offset float64

	time   float64
	size   float64
	offset float64
}

func (a *Anneal) Init(dim int) {
	a.time = 0
	a.size = a.Size
	if a.size == 0 {
		a.size = 1
	}
	a.offset = a.Offset
	if a.offset == 0 {
		a.offset = 1
	}
}

func (a *Anneal) Step(step, grad []float64) {
	eta := a.size / (a.offset + a.time)
	copy(step, grad)
	floats.Scale(-eta, step)
	a.time++
}

// Momentum is a stepper that implements a momentum-based step direction.
// Specifically, Momentum sets
//  η = a / (b + t)
//  step_t = γ * step_{t-1} - η * df/dx
type Momentum struct {
	// Size sets the initial value of the step-size parameter a. If Size is zero,
	// the default value of 1 is used.
	Size float64
	// Offset sets the initial value of the step-size parameter a. If Offset is zero,
	// the default value of 1 is used.
	Offset float64
	// Momen sets the momentum term γ. If Momen is 0, a default value of 0.9 is used.
	Momen float64

	time   float64
	size   float64
	offset float64
	nu     []float64
}

func (m *Momentum) Init(dim int) {
	m.time = 0
	m.size = m.Size
	if m.size == 0 {
		m.size = 1
	}
	m.offset = m.Offset
	if m.offset == 0 {
		m.offset = 1
	}
	if m.Momen == 0 {
		m.Momen = 0.9
	}
	m.nu = resizeZero(m.nu, dim)
}

func (m *Momentum) Step(step, grad []float64) {
	eta := m.size / (m.offset + m.time)
	for i, g := range grad {
		m.nu[i] = m.Momen*m.nu[i] - eta*g
	}
	copy(step, m.nu)
	m.time++
}

// Nesterov implements Nesterov's Accelerated Gradient Descent.
// Nesterov sets
//  μ = 1 - 3/(t+5)
//  v_t = μ*v_t - β*df/dx
//  step_k = -μ*v_{t-1} + (1+μ)*v_t
// See
//  https://arxiv.org/pdf/1503.01243.pdf eq. 1
// OR REALLY SEE
//  cs231n.github.io/neural-networks-3/#sgd
type Nesterov struct {
	// Beta sets the inital step scaling. If Beta is 0, a default value of 0.1 is used.
	Beta float64

	vPrev []float64
	v     []float64
	t     float64
}

func (n *Nesterov) Init(dim int) {
	if n.Beta == 0 {
		n.Beta = 0.01
	}
	n.vPrev = resizeZero(n.vPrev, dim)
	n.v = resizeZero(n.v, dim)
	n.t = 0
}

func (n *Nesterov) Step(step, grad []float64) {
	n.t++
	mu := 1 - 3/(n.t+5)
	rate := n.Beta
	copy(n.vPrev, n.v)
	for i, g := range grad {
		n.v[i] = mu*n.vPrev[i] - rate*g
		step[i] = -mu*n.vPrev[i] + (1+mu)*n.v[i]
	}
}

// RMSProb implements the RMSProp stepper algorithm.
//  step_{t+1} = step_t - (η / \sqrt(E[g_t] + ϵ)) ⊙ df/dx
//  E[g_t] = (1-γ) df/dx ⊙ df/dx + γ * g_{t-1}
type RMSProp struct {
	// Smooth sets the value of the smoothing parameter ϵ. If Smooth is 0, a
	// default value of 1e-8 is used.
	Smooth float64
	// Momen sets the momentum term γ. If Momen is 0, a default value of 0.9 is used.
	Momen float64
	// Rate sets the learning rate η. If Rate is 0, a default value of 0.001 is used.
	Rate float64

	s []float64
	g []float64
}

func (r *RMSProp) Init(dim int) {
	if r.Smooth == 0 {
		r.Smooth = 1e-8
	}
	if r.Momen == 0 {
		r.Momen = 0.9
	}
	if r.Rate == 0 {
		r.Rate = 0.001
	}
	r.s = resizeZero(r.s, dim)
	r.g = resizeZero(r.g, dim)
}

func (r *RMSProp) Step(step, grad []float64) {
	for i, g := range grad {
		r.g[i] = r.Momen*r.g[i] + (1-r.Momen)*g*g
		r.s[i] = -g * r.Rate / (math.Sqrt(r.g[i] + r.Smooth))
	}
	copy(step, r.s)
}
