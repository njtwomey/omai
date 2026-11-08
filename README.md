# Observable Methods in Machine Learning

A machine learning library written in TypeScript. It has tensors with automatic differentiation, numerical methods,
probability and Bayesian inference, classical learning methods, neural networks, reinforcement learning, logic
programming and datasets. A separate package draws the results.

The aim is to have all of these in one place, working in a similar way. That is the aim, not where it is yet: some areas
are much more complete than others. It is meant for trying out ideas and understanding how methods work, not for speed.

## Main ideas

**Runs are recorded.** Methods that work in steps, such as optimisers, samplers, training loops and agents, are written
as algorithms. When one runs, it records a log of what happened (a trace): the state at each step, some metrics, and how
it ended. A trace can be saved, replayed and drawn.

**Models are set up with options.** A model is a function with options, not a class to extend. For example, logistic
regression, ridge and lasso are all options of one linear model. A choice such as the solver is picked from a list.

**Tensors, and gradients when you ask.** Tensors behave like plain arrays. To get a gradient you call `grad`, as in JAX.
`vmap`, `jvp` and `hessian` work the same way.

```ts
import { grad } from 'omai-engine/autodiff'
import { mul, tensor, toArray, type Value } from 'omai-engine/tensor'

toArray(grad((x: Value) => mul(x, x))(tensor(3))) // => 6
```

**Random numbers are explicit.** You pass in a key, and the same key gives the same numbers. The library does not use the
clock or any hidden state.

**Results are plain data.** What a method finds comes back as plain typed records. The render package turns them into
figures.

**Checked against other libraries.** The numerical code is tested against results from numpy, scipy, torch and others.

**Documented in the code.** Every folder, file and export has a short description, and the examples in the descriptions
are run by the build.

**Easy for coding agents to find their way around.** The repository has an `llms.txt` at the root, and one in each main
package (`packages/engine/llms.txt` and `packages/render/llms.txt`). They list the main things in each area, one line each,
with the file and line and the import path. An agent (or a person) can start there, open the file for the area it needs,
and then read the source. The lists are generated from `@index` tags in the code's own descriptions, and a check fails if
they are out of date.

## Examples

The programs below are a few of the examples in [`examples/`](examples/), shown to give a taste. There are more there, with a
README, a figure and the tape of the run for each (listed in [`examples/README.md`](examples/README.md)). Each is a program
in `examples/<name>/main.tsx` that computes with the library, draws its figure with the render package and keeps the tape
of its run; the values in the comments are what the programs printed. `make readme-examples` runs them all again, and
`make readme-examples-check` fails if a figure is out of date.

### An optimiser's trace

Three optimisers run on the Rosenbrock function. Each run records its parameters at every step, and the figure is drawn
from those records.

```ts
import { adam, minimizer, sgd } from 'omai-engine/optim'
import { add, mul, square, sub, take, tensor, type Value } from 'omai-engine/tensor'
import { policies, recorder, runAlgorithm } from 'omai-engine/trace'

const rosenbrock = (p: Value) => add(square(sub(1, take(p, 0))), mul(10, square(sub(take(p, 1), square(take(p, 0))))))
const tape = recorder()
const run = runAlgorithm(
  minimizer((p) => rosenbrock(p as Value), tensor([-1.2, 1]), adam({ learningRate: 0.1 })),
  {
    seed: 0,
    maxSteps: 120,
    tracer: tape,
    policy: policies.teaching,
  },
)
run.state.value // => 0.0039 (gradient descent: 0.2657, with momentum: 0.0001); tape.run() holds every step
```

![Three optimisers on the Rosenbrock function: their paths over the contours](examples/optimisers/figure.svg)

### Pausing and going back

A session runs a method one step at a time. It can be stopped, run on, and asked for the model as it was at an earlier
step. The dataset records the best possible accuracy.

```ts
import { generate } from 'omai-engine/data'
import { moons } from 'omai-engine/datasets'
import { key } from 'omai-engine/random'
import { svc } from 'omai-engine/svm'

const { data, truth } = generate(moons, { knobs: { n: 120, noise: 0.25 }, key: key(0) })
const run = svc.with({ kernel: 'rbf', C: 1 }).session(data, { key: key(1) })
run.advance({ steps: 3 }) // pause after three steps of the solver
run.advance() // run on to the end: 45 steps
run.probe(3, 'model/decision-surface') // jump back to step 3 and ask for its decision surface
run.fittedAt(3).score(data) // => 0.533
run.fittedAt(run.t).score(data) // => 0.892, where 1 - truth.bayesError = 0.941 is the best anyone could do
```

![The decision surface of a support vector classifier at step 3 and at the end](examples/session/figure.svg)

### A Gaussian process and the gradient of its likelihood

How well a Gaussian process explains the data depends on the kernel's hyperparameters. Here the kernel is an amplitude times
an RBF with a length scale, so there are two, and the library's `negativeLogMarginalLikelihood` is a function of their
logs, `theta = [log amplitude, log length scale]`. `grad` differentiates it with respect to both.

```ts
import { grad } from 'omai-engine/autodiff'
import { negativeLogMarginalLikelihood } from 'omai-engine/gp'
import { constantKernel, productKernel, rbfKernel } from 'omai-engine/kernels'
import { tensor } from 'omai-engine/tensor'

const x = tensor([[0], [0.7], [1.5], [2.2]])
const y = tensor([0.1, 0.9, 0.8, -0.2])
const kernel = productKernel(constantKernel({ value: 1 }), rbfKernel({ lengthScale: 1 }))
const nll = negativeLogMarginalLikelihood(kernel, x, y, { alpha: 0.1 }) // a function of theta
grad(nll)(tensor([0, 0])) // => [0.6427, 0.1209]: amplitude 1 and length scale 1; a finite difference agrees
```

The gradient says which way each hyperparameter should move to explain the data better. `gpRegression` uses it to fit the
kernel; see [examples/gp](examples/gp).

### Second derivatives

`grad` can be applied to its own result, and `hessian` is built from it.

```ts
import { grad, hessian } from 'omai-engine/autodiff'
import { mul, take, tensor, type Value } from 'omai-engine/tensor'

const cube = (x: Value) => mul(mul(x, x), x)
grad(grad(cube))(tensor(2)) // => 12, since the second derivative of x^3 is 6x
hessian((p) => mul(mul(take(p, 0), take(p, 0)), take(p, 1)))(tensor([1, 2])) // => [[4, 2], [2, 0]], for x^2 y
```

### Bijections between distributions

A bijector is an invertible map with its log-Jacobian. `transformed` pushes a distribution through one and applies the
change of variables, so `logProb` stays exact.

```ts
import { bijectors, normal } from 'omai-engine/dists'
import { transformed } from 'omai-engine/dists/compose'
import { key } from 'omai-engine/random'
import { mul, square, sub, tensor } from 'omai-engine/tensor'

const lognormal = transformed(normal(0, 1), bijectors.exp())
lognormal.logProb(2) // => -1.852312, the closed form of the log-normal density at 2
lognormal.sample(key(0), [4000]) // 4000 draws, mean 1.631 (exp(1/2) = 1.649)

// a coupling layer bends a 2-D Gaussian into a banana
const bend = bijectors.affineCoupling(1, (xa) => ({ shift: sub(mul(0.8, square(xa)), 0.8), logScale: mul(xa, 0) }))
transformed(normal([0, 0], [1, 1]), bend).logProb(tensor([0, -0.8])) // => -1.8379
```

The maps from the real line onto each support (`bijectors.unconstrained`) are how NUTS, Laplace and variational
inference run on constrained parameters: they work on the reals, and the Jacobian keeps the density correct.

![A Normal, the map exp, the log-normal it gives, and a coupling layer turning a Gaussian into a banana](examples/bijectors/figure.svg)

### A decision tree

A decision tree classifies a flower by asking a few questions about its measurements, one after another. It is fitted
as a session, one node per step, and the library draws both the tree and the regions of the plane it cuts.

```ts
import { iris } from 'omai-engine/datasets'
import { key } from 'omai-engine/random'
import { decisionTreeClassifier } from 'omai-engine/trees'

// data: the 150 iris flowers with only petal length and petal width
const fit = decisionTreeClassifier({ maxDepth: 3, record: 5 }).session(data, { key: key(1) })
fit.advance() // 9 steps, one node decided in each
fit.probe(fit.t, 'model/structure') // the tree, drawn by the render package
fit.probe(fit.t, 'model/decision-surface') // the rectangles of the plane, drawn the same way
fit.fittedAt(fit.t).predict(data.x) // 146 of the 150 flowers right
```

![A decision tree on iris petal length and width, drawn as a tree of splits beside the regions of the plane it makes](examples/decision-tree/figure.svg)

### A support vector classifier

A support vector classifier draws a boundary with a margin on each side, and only the rows near the boundary (the support
vectors) decide where it goes. The penalty `C` sets how much the margin may be crossed.

```ts
import { generate } from 'omai-engine/data'
import { moons } from 'omai-engine/datasets'
import { key } from 'omai-engine/random'
import { svc } from 'omai-engine/svm'

const { data, truth } = generate(moons, { knobs: { n: 100, noise: 0.25 }, key: key(0) })
const loose = svc.with({ kernel: 'rbf', C: 0.2 }).session(data, { key: key(1) })
const tight = svc.with({ kernel: 'rbf', C: 50 }).session(data, { key: key(1) })
loose.advance() // 48 steps: 68 support vectors, accuracy 0.895 on 400 new points
tight.advance() // 119 steps: 17 support vectors, accuracy 0.945, where 1 - truth.bayesError = 0.940 is the best possible
```

![Two support vector classifiers on the two-moons data: the score as a heat map with the boundary and margins as contours, and the support vectors ringed](examples/svm/figure.svg)

## Packages

- `omai-engine`: the library of tensors, autodiff, numerics, distributions, optimisation, kernels, learning, neural networks, RL, graphs, datasets and the trace. Imports as `omai-engine/<domain>`.
- `omai-render`: figures from results and traces: charts, layouts, drawings, SVG and animation export, and React components.
- `omai-meta`: the documentation tooling: it reads the descriptions in the code, runs the examples and keeps the list of concepts.

The packages share a version number (see `CHANGELOG.md`).

## Where to start

- `llms.txt` lists the packages. It leads to `packages/engine/llms.txt` and `packages/render/llms.txt`, which list the main
  things in each area with their file and import path.
- `docs/structure.md` describes what is in the library and the rules it follows.
- `docs/methods-catalog/` lists the methods the library aims to cover.
- `docs/code-review/` is the record of a review of the core code. Start at `docs/code-review/summary.md`.

## Working on it

Node, pnpm, Python and uv are pinned in `mise.toml`. Everything goes through `make`:

```sh
make setup            # install the tools and dependencies
make verify           # check a change: quick checks on the whole repo, then only the tests the change touches
make check            # everything: every test and example
make readme-examples  # rerun the programs in examples/ and redraw their figures
make fixtures-check   # rebuild the reference results from the Python code and compare
make help             # list every target
```

The reference results for the numerical code come from standard Python libraries (numpy, scipy, torch and others), in
`oracle/`. The tests themselves do not run Python. The public API is checked against a saved list
(`make api-surface-check`), the layering by `make layers`, and the index files by `make agent-index-check`.

## Licence

MIT. See [LICENSE.md](LICENSE.md).
