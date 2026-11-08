# mdml-js

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
import { grad } from 'mdml-engine/autodiff'
import { mul, tensor, toArray, type Value } from 'mdml-engine/tensor'

toArray(grad((x: Value) => mul(x, x))(tensor(3))) // => 6
```

**Random numbers are explicit.** You pass in a key, and the same key gives the same numbers. The library does not use the
clock or any hidden state.

**Results are plain data.** What a method finds comes back as plain typed records. The render package turns them into
figures.

**Checked against other libraries.** The numerical code is tested against results from numpy, scipy, torch and others.

**Documented in the code.** Every folder, file and export has a short description, and the examples in the descriptions
are run by the build.

## Packages

| package | what it is |
| --- | --- |
| `mdml-engine` | the library: tensors, autodiff, numerics, distributions, optimisation, kernels, learning, neural networks, RL, graphs, datasets and the trace. Imports as `mdml-engine/<domain>`. |
| `mdml-render` | figures from results and traces: charts, layouts, drawings, SVG and animation export, and React components. |
| `mdml-meta` | the documentation tooling: it reads the descriptions in the code, runs the examples and keeps the list of concepts. |

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
make fixtures-check   # rebuild the reference results from the Python code and compare
make help             # list every target
```

The reference results for the numerical code come from standard Python libraries (numpy, scipy, torch and others), in
`oracle/`. The tests themselves do not run Python. The public API is checked against a saved list
(`make api-surface-check`), the layering by `make layers`, and the index files by `make agent-index-check`.

## Licence

MIT. See [LICENSE.md](LICENSE.md).
