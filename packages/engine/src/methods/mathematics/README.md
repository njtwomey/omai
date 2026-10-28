# Mathematics (part 1, mathematical foundations)

Mathematics computed, drawn and stepped through: linear algebra, calculus, differential equations, splines, numerical methods, geometry and graphs, as sessions over `numerics`, `autodiff`, `dynamics`, `symbolic`, `graph` and `foundations`. The methods here are studies: a procedure watched (an eigenvalue iteration, Newton's method, an ODE solver choosing its steps) rather than a model fitted.

## Capabilities

- Iterations of linear algebra: power and QR iteration, Jacobi, Gram-Schmidt, conjugate gradient, Lanczos; options: the matrix (random symmetric, a graph Laplacian, a user matrix), the shift, the start; log the iterate, the Rayleigh quotient, the residual.
- Root finding and quadrature: bisection, secant, Newton, fixed point; trapezoid, Simpson, Gauss, adaptive; options: the function (a catalogue of named functions with symbolic derivatives), the tolerance.
- Differential equations: solvers (Euler, RK4, adaptive RK45, implicit) on named systems (`dynamics`), with step-size control, events and stiffness; the matrix exponential and linear systems; phase portraits and bifurcations.
- Calculus and automatic differentiation: Taylor series, finite differences against `autodiff`, forward against reverse mode on a small graph.
- Splines and interpolation: polynomial interpolation and Runge's phenomenon, cubic and B-splines, Bézier curves; options: nodes, degree, the function.
- Geometry: rotations, projections, geodesics on a sphere, hyperbolic embeddings of trees.
- Graphs: traversal, shortest paths, spanning trees, flows, stepped (`graph`).

## Examples to make

- Power iteration converging on a 2×2 matrix with the iterate drawn as an arrow, the ratio of eigenvalues as the knob.
- Newton's method on a function with basins of attraction coloured by the root reached.
- Adaptive RK45 on a stiff system with step sizes on the tape and the rejected steps shown.
- Runge's phenomenon: interpolant against degree, nodes equispaced or Chebyshev.
- Dijkstra and A* on the same graph side by side, the frontier as a board.

## Rendering

- Curves, trajectories in the plane and 3-D (`Scene3DView`), vector fields and phase portraits (`fields`), boards for graph algorithms (`boardSteps`), the structure view for graphs.
- Missing: a matrix as a heatmap with its action drawn (unit circle to ellipse); a panel for a function with its derivatives (`symbolic` output as TeX beside the plot).

## Shared

- `shared/`: a catalogue of named functions and matrices with their truths (roots, eigenvalues, exact integrals); the ODE problem slot shared with `control-theory` and `scientific-machine-learning`.
- From `methods/shared`: the procedure base, pictures.

## Problems

- Where a study ends and the library's own example begins: a `numerics` function already has an `@example`; the study adds the watching, and should not duplicate the docs.
- `linalg/` (eigen iterations) and `foundations/` move here; `dynamics/odeSystems` is shared with `control-theory`.
