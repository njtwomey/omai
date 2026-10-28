# Scientific machine learning (part 6, neural architectures)

Learning with physics: physics-informed neural networks (residual losses on an ODE or PDE, boundary terms, collocation points), operator learning (DeepONet and Fourier neural operators on function pairs), learning dynamical systems from data (neural ODEs, SINDy's sparse regression over a library, Koopman and DMD), and their applications, as sessions over `nn`, `dynamics`, `autodiff`, `symbolic` and the trainer base.

## Capabilities

- PINNs: the network's solution against the true one on a named equation, the residual loss and boundary loss on the tape, the collocation points as a knob.
- Neural ODEs: the vector field learned from trajectories, the adjoint's gradients, the learned flow drawn.
- SINDy: the library of candidate terms, the sparse coefficients appearing along the sequential thresholding, against the true equation (`symbolic`).
- DMD and Koopman: the modes of a flow field.
- Operators: a DeepONet mapping forcing functions to solutions, the error over function space.

## Examples to make

- A PINN on the harmonic oscillator: the solution and residual over time as training proceeds.
- SINDy recovering the Lorenz equations: coefficients against the truth, threshold as the knob.
- A neural ODE on a spiral: the learned field and the true field side by side.
- DMD modes of a travelling wave.

## Rendering

- Curves against truth, vector fields, coefficient tables (`coefficientTable`), TeX of the recovered equation (`MathView`), 3-D trajectories.
- Missing: an equation-against-truth panel (TeX with coefficient colour).

## Shared

- `shared/`: the named-equation slot (from `dynamics`, with exact or reference solutions), shared with `mathematics` and `control-theory`; the trainer base.

## Problems

- `scientific/` moves here. Reference solutions come from `dynamics` solvers at fine tolerance; mark the truth as numerical where it is.
