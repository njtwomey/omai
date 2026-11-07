# Summary

Where the code review stands at the end of 2026-11-07, for the person deciding whether to push. Everything below is local:
311 commits ahead of `origin/main`, 1,022 files changed (29,308 insertions, 7,337 deletions). Nothing has been pushed since
`aaf3c87f` (decision D-014), so the CI runner has not seen any of it.

## State of the tree

|                                     |                                                                                                                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full local gate (last run)          | 26,624 tests passed (26,208 at the start of this stretch), 0 failed; 9,612 `@example`s passed; barrels, layers, lint, agent index, API digest and all six `tsc` runs clean |
| Suite wall time                     | 162 s (it was 222 s before the review: the kernel and reverse-mode speed-ups pay for the new tests)                                                                        |
| Work items (`progress.md`, Phase 3) | 39 of 44 complete; 5 partly done (P3.8, P3.20, P3.30, P3.32, P3.40): each is done except the follow-ups recorded in its row                                                |
| Register (`register.md`)            | 253 findings: 120 fixed, 66 partly fixed, 25 noted, 5 open, 1 wontfix, 36 still planned (reconciled against the tree; see "What is not done")                              |
| Completion by effort                | about 93%                                                                                                                                                                  |

## What changed, by wave

**Wave A, performance (behaviour-preserving, measured):** `scheduler.fired` 421 ms to 0.5 ms; the gamma and count samplers about
20x (samples bit-identical per key); `einsum` 30x; `eig` n=200 15.6 s to 0.26 s; `convolve` 4096x4096 2.0 s to 16 ms; `spmv` 25x;
stationary kernels' Gram matrix 7x to 20x at d=20 (values bit-identical); OMP 32x; LP default Dantzig (Klee-Minty 8: 67 to 15
pivots); elementwise and reduction kernels 2x to 10x; reverse mode with `needs` (`grad` of a chain 142 to 10 ms); render decision
surface 29 ms to 5 ms per frame.

**Wave B, certain bugs (each with a probe that failed first):** dozens of fixes. The ones with reach: `matmul` no longer turns
0 x NaN into 0 (and `einsum` with it); `lstsq` and `polyfit` return the minimum-norm solution on rank-deficient systems (were
garbage of size 1e15); `delaunay` on cocircular points; `svd` and `norm` at extreme scales; empty-axis reductions throw as numpy
does; samplers validate their parameters; NaN no longer reads as a tie in rank tests or as certainty in `chdtr`/`fdtr`;
`permutationTest` with `pairings` enumerated every permutation n! times; the nucleolus failed on denormal data; `FramesPlayer`
dropped the title of 14 of 61 record kinds; `parseNpy`/`untar`/`unzip` on damaged files; `erfinv`, `betainc`, `gammainc`, `tanhSinh`
at the edges. The gate found one regression of my own (`linprog` scaling overflowed for a denormal right-hand side): fixed.

**Wave C, conventions:** one `RunStatus` vocabulary with `status` and `converged` on results, invalid arguments throw
`MdmlError('value')` (`findings/failure-convention.md`); a NaN rule (elementwise propagate; anything that decides rejects
non-finite at its boundary); `make layers` now fails on tier violations and new file-level cycles; one trace stepper, one JSON
codec, one option-table builder; shared helpers for complex arithmetic, coercions and small functions; 59 generic type names that
meant different things (`Matrix`, `Box`, `Bounds`, `Constraint`, ...) renamed to what they are, with a `RESERVED` guard in `make
lint`; the dead backend abstraction removed.

**Wave D, tests (+198 `it` blocks, 4 removed, since the start of this session; many thousand table rows inside them):** core, numerics, dists/stats/metrics, data/kernels, render
and exposition; sampler goodness of fit and key independence; second-order `checkPrimitive`. They found about 25 more bugs, all fixed or recorded.

**Wave E, native arrays to tensors (D-009), five stages:** `random`, `data`, `stats`, `metrics`, `numerics`, `optim`, `losses` and
the result and state records of those domains.

## Public surface changes (all listed per merge in `gates.md`)

- **The deliberate break (native arrays to tensors):** about 180 changed signatures, plus the type renames below. Inputs mostly widen to `TensorLike` (old calls
  still compile); returns that were `number[]`, `number[][]` or `Float64Array` are now tensors; `GlmProblem.x` is the `[n, d]` matrix
  (`n` and `d` are gone); `constraintNullSpace` takes no `k`; `data.balancedWeights`, `noiseTransition` return tensors.
- **Renames (no deprecated aliases):** 59 type names; `metrics#matrix` is `shapedArray`.
- **Removed (proved unused):** `Context.backend` and `Context.dtype`, `BackendCapabilities`, `supportsDType`, `avalOf`,
  `Tolerance`, `defaultTolerance`, `resolveTolerance`, the `not`/`and`/`or`/`xor` kernels, `data#RowColumn`.
- **Added:** dozens (shared helpers in `base` and `space`, `tensor#toVector`, `tensor#fromVector`, `RecordBoundary`,
  `expositionSamples`, `Kernel.psd`, `Primitive.linear`, `status` and `runStatus` fields on about 30 result types, optional
  `CreateOptions` on eight constructors, `useTape` `onError`).
- Tape and checkpoint JSON: typed arrays are `{ $typed, values }` and a tensor is `{ $object: { $tensor } }`; non-finite numbers and
  -0 survive text. Tapes saved earlier still load (`stateFromJson` reads the old spellings).

## Behaviour a user could notice

Rank-deficient least squares answers differently (right now); `fixedPoint` returns a result at its cap instead of throwing;
`iterate` ends `converged` where it said `completed`; the factorisations, Krylov solvers and nearest-neighbour queries throw on
NaN or Infinity input; the scalar root finders, `fsolve`, `broyden` and the eigen iterations end `completed` at their cap on the
tape; `pairings` permutation p-values are now exact; `floorDiv(0, -1)` is -0 as in numpy; Dantzig pivots visit different vertices
on ties (one `@example` path changed).

## What is not done, and why

- **P3.32 step 2 (array-valued batch `poisson` and `binomial`):** changes which key draws each element, so seeds move. It needs
  your decision; nothing was changed.
- **Native arrays outside the reviewed core:** `learn` (415 fields), `timeseries`, `graph`, `nn`, `signal`, and `ComplexArray` in
  `systems` (about 190 reads). Not in the review's scope; a separate project.
- **Design questions recorded, not settled:** `vonMises` has two conventions (DSM-32) and `cdf` at concentration 1e6 takes 9.5 s;
  `trainTestSplit` shuffle conventions (KDES-7 and 26); F-039 and F-041e (which new public home for three duplicated helpers);
  ND-25 (about 80 `as Value` casts in `special`, a circular-type error stopped the typing); the `eigsh` non-convergence shape; the
  bare-value solvers that still throw `convergence` (rule 3 of the convention).
- **Browser verification:** the render work was verified through benchmarks and the test harness only. Worth a look by eye:
  the classifier animation and decision-surface export, focus order in the expanded figure, export menu and popovers, Space and
  Enter on the player in Chrome, Firefox and Safari, Alt+Left in the narrative, a screen reader on a chart, reduced motion,
  dashed comparison lines, a recording started while playing, GIF/WebM/JSON/PNG downloads.
- **The 36 register rows still reading `planned`** (reconciled row by row against the tree on 2026-11-07; 74 rows were read; 24 of the 60 moved to fixed, noted or partly fixed, and the 9 marked in progress became partly fixed): untouched or unchanged, not forgotten. In optim and losses: the failure of IRLS and the GLM modes (OL-10), the knob specs and casts (OL-12), the tensor coercions (OL-13), the two proximal types (OL-14), the tolerance vocabulary (OL-17), the two `sinkhorn` and `slicedWasserstein` (OL-21), tests of the minimise family (OL-26), TSDoc (OL-28), `sharpness` as an Algorithm (OL-29), bare `catch` (OL-30), copied path and history (OL-31). In dists, stats and metrics: tail precision (DSM-10), `streaming` (DSM-15), the 18 `spec` builders (DSM-16), the metrics bootstrap (DSM-17), error codes (DSM-20), the `unavailable` methods (DSM-22), TSDoc and notes (DSM-29, 30). In kernels, data, exposition and scene: camera (KDES-14), `clusterAgreement` (KDES-15), data helpers (KDES-27, 36), exposition duplicates (KDES-29), scene model (KDES-30), structured kernels (KDES-35). In render: memoisation and the marker (RD-12, 14), unused exports (RD-19), module state (RD-21), swallowed export errors (RD-22), test helpers in the package (RD-24), repeated hooks (RD-25), TSDoc (RD-27). In the foundations: the copies forced by the layer order (FD-28), and no CPU profile (F-033, P0.6). The rows reading `partly fixed` or `open` carry what is left in their notes.
- **`ci.yml` does not run `api-surface-check`** (F-031), and the repository has no `LICENSE` file.

## Before pushing

The CI runner found one thing the Mac gate missed today (platform float digits in the sampler pins, F-045). Three hundred commits
have not met it. The safest order is: push to a branch and open a PR so CI runs on all four shards, fix what it finds, then merge.
`docs/code-review/gates.md` has every gate with its numbers and surface diff, `log.md` has the narrative, `decisions.md` the
rulings (D-001 to D-015). The process notes that paid for themselves: pins and goldens on float samplers compare to a tolerance;
a property-test counterexample is a regression finder, so bisect it before calling it a flake; resolve doc-table merges by row key.
