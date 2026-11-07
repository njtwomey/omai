# mdml-js

All of machine learning in one TypeScript library.

Numerics, probability, classical learning, neural networks, reinforcement learning and datasets live together and work
the same way, so you can try an idea without gluing six libraries together. It is built for prototyping and for
understanding, not for being the fastest.

## The ideas

**1. One library for all of ML.** Things that are usually separate packages share one set of tools: tensors,
random numbers, datasets, optimisers and metrics. A model is a small configuration you can read. Logistic, ridge and
lasso regression are settings of one linear model, and a choice like the solver is a slot you fill from a list of
options, so a model is never tied to one solver.

**2. Everything leaves a trace.** This is the key idea. Every routine that runs in steps (an optimiser, a sampler, a
trainer, an agent) records what it did as it went: the state, the metrics and how it ended. You can pause a run, replay it,
jump to any step, save it, and draw it. The figures, the players and the documentation all read the same record.

```ts
import { grad } from 'mdml-engine/autodiff'
import { mul, tensor, toArray, type Value } from 'mdml-engine/tensor'

toArray(grad((x: Value) => mul(x, x))(tensor(3))) // => 6
```

**3. Simple, explicit foundations.**

- Tensors are plain arrays until you ask for a gradient. Then `grad`, `vmap`, `jvp` and `hessian` work on that call, as
  in JAX.
- Randomness is explicit. You pass in a key, and the same key gives the same numbers. The library never uses a clock or
  hidden state.
- Results explain themselves. They come back as plain typed records that the render package can draw.
- The code documents itself. Every folder, file and export is described, and every example in the docs is run by the build.

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
