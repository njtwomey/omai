# mdml-engine

> A TypeScript library of numerics, probability, learning, neural networks, RL and datasets. Every iterative routine is
> an `Algorithm` that records a trace. This index lists the main things in `src/lib`; it is short on purpose and grows
> as things are indexed. The source is the reference.

Rendering of its records, traces and sessions (figures, views, players, React bindings) lives in mdml-render: see the
`llms.txt` of mdml-render (`packages/render/llms.txt` in the repository, `node_modules/mdml-render/llms.txt` once
installed).

## How to use this index

Look here first, then open the file an entry points to, then follow its imports. Each entry is one line: kind, name,
what it is, `path:line` (relative to the package root) and the import path. Entries are not complete and carry no
signatures: open the source for those. A name can exist in several domains (`gradient`, `solve`, `normal`, `gamma`), so
the import path is part of an entry's identity. `index/index.json` has every entry for tools.

Kinds: `function` (also metrics, losses, tests, autodiff transforms), `estimator` (fit and predict objects),
`algorithm` (a stepped routine you drive or watch), `structure` (a type or contract you build or implement against),
`table` (named options filling a slot), `module` (a neural-network layer or network), `dataset` (a generator that
carries its truth), `distribution`, `kernel`, `agent`. A "(file)" entry is a hub: open the file first, then what it imports.

## Conventions you need

- Import from an entry point, not a file: `import { solve } from 'mdml-engine/numerics/linalg'`. Every folder under
  `src/lib` is an import path (`mdml-engine/<domain>`, and `mdml-engine/<domain>/<subfolder>` for a namespace). Inside the
  library, relative imports carry the `.ts` extension. Resolving packages to TypeScript source needs the `source`
  condition (`node --conditions=source`).
- Randomness is explicit: an immutable `Key` from `mdml-engine/random` (`key(seed)`, `split(k, n)`, `foldIn(k, i)`).
  There is no `Math.random`, no clock and no hidden state; the same key gives the same draw.
- Tensors are plain arrays outside a transform. Differentiation is opt-in per call, JAX style: `grad(f)(x)`,
  `vmap`, `jvp`, `hessian` from `mdml-engine/autodiff`.
- Estimators are configured, not subclassed. `linearModel({ family: 'binomial', alpha: 0.01 }).fit({ x, y })` returns a
  fitted value with its verbs (`predict`, `score`), `params` and `exposition` records. Logistic, ridge and lasso
  regression are configurations of `linearModel`, not separate exports. Hyperparameters can be changed with
  `dbscan.with({ eps: 0.3 })`.
- A slot is a hyperparameter filled from a table of options (`smoothSolvers`, `optimisers`, `schedules`,
  `regressionLosses`, `fieldEncodings`): the estimator takes the option by name, with that option's own knobs, such as
  `solver: { kind: 'lbfgs', gradTol: 1e-9 }`. An estimator fitted by minimising a smooth objective takes its solver
  from `smoothSolvers`; it is never fused to one solver. The tables list their options (`table.names`).

## Algorithms, runs and the log contract

- An `Algorithm` (`mdml-engine/trace`) is `{ name, config, init(key), step(state, ctx), done?, observe? }`.
  `runAlgorithm(algorithm, { seed, maxSteps, tracer })` runs it and records a trace; `iterate` and `runFor` step it
  lazily or in time slices. Step `t` gets the key `foldIn(stepsKey, t)`, so a run is the same however it was paused.
- A step reports through `ctx.log(path, value)` (metrics, tensors) and `ctx.emit(event)`. Paths such as
  `training/loss` form the log contract of the method. `defineContract({ 'solver/objective': scalar() })` declares it
  with literal path types; `defineMethod({ name, config, make, contract })` makes a `Method<S, C, Path>`, so `ctx.log`
  and `defaults` are checked against the paths.
- `session(method, { mode: 'eager' | 'live', steps, seed, logs })` runs a method and keeps the tape: `advance`,
  `stateAt(t)`, `fork(t)`, `probe`, and `metricSeries(session.run(), 'path')` to read a series. Estimators on the
  iterative, trainer, agent and sweep bases carry `.method` and `.session` for free; `fit` is a session run to completion.
- Explanations come back as typed exposition records (`mdml-engine/exposition`) beside the result, never as logging.
- Errors are `MdmlError` with a machine-readable code. Test files sit next to code (`foo.test.ts`); fixtures are generated
  from a Python oracle and tests never run Python.

## Where to look

1. This index: find the domain, then the entry.
2. The file at `path:line`: its opening paragraph says what it is; each export has TSDoc with an `@example`.
3. Its imports, for the pieces it composes. A domain folder's `index.ts` opens with a summary of the folder.
