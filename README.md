# mdml-js

All of machine learning in one TypeScript library.

Numerics, probability, classical learning, neural networks, reinforcement learning and datasets live together and work
the same way, so you can try an idea without gluing six libraries together. It is built for prototyping and for
understanding, not for being the fastest.

## The ideas

They build on each other, and there are more than these. `docs/structure.md` has the full set.

**1. One library for all of ML.** This is the goal. Things that are usually separate packages share one set of tools:
tensors, random numbers, datasets, optimisers and metrics. There is no separate layer of methods on top: a demonstration
of a method is a short page that combines an estimator with a dataset.

**2. Everything leaves a trace.** This is the key idea. Every routine that runs in steps (an optimiser, a sampler, a
trainer, an agent) records what it did as it went: the state, the metrics and how it ended. You can pause a run, replay it,
jump to any step, save it, and draw it. The figures, the players and the documentation all read the same record.

```ts
import { grad } from 'mdml-engine/autodiff'
import { mul, tensor, toArray, type Value } from 'mdml-engine/tensor'

toArray(grad((x: Value) => mul(x, x))(tensor(3))) // => 6
```

**3. Models are configurations.** A model is a small setting you can read, not a class to subclass. Logistic, ridge and
lasso regression are settings of one linear model. A choice like the solver is a slot you fill from a list of options, so
a model is never tied to one solver.

**4. Data knows the truth.** Every dataset generator carries the ground truth it was made from (the true parameters, the
true labels, the true shift), so you can check what a method found against what was really there.

**5. Results explain themselves.** What a method found comes back with plain, typed records of what is worth showing. The
render package turns those records into figures, and nothing in the library has to know about drawing.

**6. Tensors, with gradients when you ask.** Tensors are plain arrays until you want a gradient. Then `grad`, `vmap`,
`jvp` and `hessian` work on that call, as in JAX.

**7. Randomness is explicit.** You pass in a key, and the same key gives the same numbers. The library never uses a clock
or hidden state, so any run can be repeated exactly.

**8. Checked against the standard libraries.** The numerical code is tested against results from numpy, scipy, torch and
others, so you can trust that it agrees with what you already know.

**9. The code documents itself.** Every folder, file and export is described, and every example in the docs is run by the
build. A generated index of the main things (`llms.txt`) helps people and tools find their way.

## A closer look: tuning a Gaussian process

A Gaussian process fits a curve through noisy points using a kernel, and the kernel has hyperparameters such as a length
scale. You usually choose them by maximising the marginal likelihood, which needs its gradient. Here is how you would
write that gradient by hand from the textbook formula, `1/2 tr((a aᵀ − K⁻¹) ∂K/∂θ)` with `a = K⁻¹ y`:

```ts
import { gramGradient, rbfKernel } from 'mdml-engine/kernels'
import { choSolve, cholesky } from 'mdml-engine/numerics/linalg'
import { add, eye, matmul, mul, outer, reshape, sub, tensor } from 'mdml-engine/tensor'

const x = tensor([[0], [0.7], [1.5], [2.2]])
const y = tensor([0.1, 0.9, 0.8, -0.2])
const kernel = rbfKernel({ lengthScale: 1 })

const { gram, gradient: dK } = gramGradient(kernel, x) // K, and how K changes with each hyperparameter
const L = cholesky(add(gram, mul(eye(4), 0.1))) // add the noise, then factor
const a = choSolve(L, y)
const W = sub(outer(a, a), choSolve(L, eye(4)))
mul(matmul(reshape(W, [1, 16]), reshape(dK, [16, 1])), 0.5) // => [-0.1209288]
```

The library already has that, for any kernel and any number of hyperparameters, and it agrees to the last digit:

```ts
import { logMarginalLikelihood } from 'mdml-engine/gp'

logMarginalLikelihood(kernel, x, y, { alpha: 0.1 }).gradient // => [-0.1209288]
```

And because the gradient is a derivative rule of an ordinary function, `grad` can differentiate through it, so any
optimiser can tune the kernel:

```ts
import { grad } from 'mdml-engine/autodiff'
import { negativeLogMarginalLikelihood } from 'mdml-engine/gp'

grad(negativeLogMarginalLikelihood(kernel, x, y, { alpha: 0.1 }))(tensor([0])) // => [0.1209288]
```

A finite difference of the likelihood gives the same number, and a test in the repository checks all of them.

## Packages

| package | what it is |
| --- | --- |
| `mdml-engine` | the library: tensors, autodiff, numerics, distributions, optimisation, kernels, learning, neural networks, RL, graphs, datasets and the trace. Imports as `mdml-engine/<domain>`. |
| `mdml-render` | figures from results and traces: charts, layouts, drawings, SVG and animation export, and React components (hooks, players, panels). |
| `mdml-meta` | the documentation tooling: it reads the code's descriptions, runs the examples and keeps the concept list. |

The packages share a version number (see `CHANGELOG.md`).

## Where to start

- `llms.txt` is the map of the packages. It leads to `packages/engine/llms.txt` and `packages/render/llms.txt`, which list
  the main things in each area with their file and import path.
- `docs/structure.md` is the plan: what is in the library, the shared rules, and the order it is built in.
- `docs/methods-catalog/` lists the methods the library is meant to cover, area by area.
- `docs/code-review/` is the record of a structural review of the core. Start at `docs/code-review/summary.md`.

## Working on it

Node, pnpm, Python and uv are pinned in `mise.toml`. Everything goes through `make`:

```sh
make setup            # install the tools and dependencies
make verify           # check a change: quick whole-repo checks, then only the tests it touches
make check            # everything, every test and example (a periodic full run)
make fixtures-check   # rebuild the reference results from the Python oracle and compare
make help             # list every target
```

The reference results for the numerical code come from standard Python libraries (numpy, scipy, torch and others) in
`oracle/`. The tests themselves never run Python. The public surface is guarded by a digest (`make api-surface-check`),
the layering by `make layers`, and the agent index by `make agent-index-check`.

## Licence

MIT.
