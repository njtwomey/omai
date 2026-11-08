# Synthesis

Written 2026-11-07 after the six readings. Sources: `register.md` (236 rows: 44 from Phase 0 as `F-`, 192 from the readings as
`FD-` foundations 37, `ND-` numerics 29, `OL-` optim and losses 31, `DSM-` dists, stats and metrics 30, `KDES-` kernels, data,
exposition and scene 36, `RD-` render 29), `findings/*.md` for the detail and `progress.md` for the work items. Figures are the
readers' own measurements; estimates are marked. Nothing here has been changed in the code yet beyond the work items already
complete (P3.1, P3.2, P3.3 in part, P3.11).

## The headline

The core is sound in shape and weak in its corners. The ordinary paths are fast, consistent and well tested (statement coverage
93.9% in the core; the minimisers, the common distributions, the kernels and the data pipeline behave on ordinary input). What is
wrong is the same few things in every domain:
1. **No policy for degenerate input.** NaN, Infinity, empty input and denormals are handled differently in every function:
   `matmul` skips zeros so `0 * Infinity` is 1 (FD-3); `chi2Sf(1, NaN)` is 1 so every chi-square or F test gives `p = 1`
   for a NaN statistic (DSM); `spearman` returns 1 for NaN data; `cdf(NaN)` depends on the family; `trainTestSplit` with a NaN
   size gives an empty train set (KDES); `adaptiveSimpson` on a NaN integrand would take 4.5e15 evaluations (ND-3). These are the
   live twins of the property tests that failed on CI.
2. **Absolute tolerances where the scale matters.** `svd` is wrong above 1e77 and below 1e-155 (ND-1); `bisect` reports a
   tiny-scale function as converged (ND-2); `linprog` with `c` scaled by 1e-9 is wrong in 30 of 40 random programs; seven
   gradient methods return the start as `converged` in 0 steps at objective scale 1e-8; OMP on a 1e-9 dictionary returns zeros
   (OL). Wrong answers, silently.
3. **No single failure convention.** Non-convergence is thrown, returned as `converged: false`, set as a flag, or not reported
   at all; two status vocabularies (`RunStatus` against `optimal`/`success`); three error codes for one condition; `fixedPoint`
   and `fsolve` have the same return type and opposite behaviour (ND-10, ND-11, OL-5, OL-7, OL-16, KDES-17).
4. **Constructors and boundaries do not validate.** 59 of 61 exposition kinds throw a bare `TypeError` for a record with only its
   kind (RD-2); only 7 of 54 exposition files check finiteness; scene builders check nothing; kernels accept a zero length scale
   and give an all-NaN Gram; distributions accept `uniform(0, -1)`, `geometric(0)`.
5. **Duplicated machinery.** Three Algorithm drivers that already disagree on the end status (FD-8); `trace/serialise.ts`
   against `base/json.ts`; `modes()` against `optionTable()`; the option-table builder written by hand in five files; the SVM
   duals twice; the `recordOf` component four times in render.
6. **A few real performance bugs,** each a small fix: `scheduler.fired` quadratic (1.9 s for 20,000 firings), `gammaVariates` and
   every sampler built on it about 100 times slower than the batch path, `einsum` 22 times slower than `matmul`, `convolve` 1,000
   times slower than `fftConvolve`, general `eig` eigenvectors O(n^4), stationary kernels 40 times slower than a matmul distance,
   the decision-surface player frame 30 ms and 2 MB.
7. **Tests follow the generators.** The uncovered lines are almost exactly the degenerate branches; the property generators avoid
   exactly the points where the bugs are. The fix is a few table-driven contract tests, not more of the same.

What is healthy, as the readers found it: only `react/` imports React; no clocks, `Math.random` or `console` in the engine and
render outside justified frame budgets; layering has no cycles and the order is sound (the readers corrected the guessed order: `context`
is above `trace`, `scene` and `exposition` sit right above `base`, `optim` imports `dists` for a type only); TSDoc complete for
96 to 100% of exports (by domain); Philox known-answer tests pass; `grad` of a small MLP costs 3x the forward pass; 32 of 32 kernels symmetric.

## Ranked work, in waves

Each wave item is a register group and a progress row. Effort is the readers' estimate (S under an hour, M a day, L more).
"Changes behaviour" means a result that was wrong or undefined becomes right or an error; "public surface" means the digest changes.

### Wave A: behaviour-preserving performance (no decision needed; starting)

| work  | what                                                                                                                               | win                                      | effort | notes                                                                                           |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------ | ----------------------------------------------------------------------------------------------- |
| P3.12 | `scheduler.fired` copy, `gammaVariates` block per variate, `einsum` outer product, small tape and tree hot spots (FD-1, 2, 11, 37) | 10x to 250x on their paths               | S each | the sampler stream must stay bit-identical for a given key (fixtures and goldens must not move) |
| P3.32 | batch sampling for the gamma family and the count families (DSM)                                                                   | about 1.0 s per 1e5 draws to about 25 ms | M      | after FD-2; same stream rule                                                                    |
| P3.21 | `eig` eigenvectors, `convolve`, `spmv`, Bluestein chirp, `kNearest` (ND-7, 8, 9, 13, 14)                                           | `eig` 16 s to ms at n = 200              | M      | local rewrites, results unchanged                                                               |
| P3.38 | stationary kernels through a matmul squared distance (KDES-11)                                                                     | 2.65 s to about 64 ms at 1500 x 20       | S      |                                                                                                 |
| P3.40 | render decision surface: lazy PNG, separable bicubic (RD-1, 14)                                                                    | 2x to 3x on the classifier animation     | S to M | needs a browser to confirm the per-frame win                                                    |

### Wave B: certain bugs, each with a probe that fails today and a test (behaviour changes: wrong becomes right)

| work                | what                                                                                                                                                                                                      | effort | notes                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ----------------------------------------------------- |
| P3.13               | foundations degenerate inputs: `matmul` zeros, `tril`/`triu`, empty reductions, bool dtype wraps, scatter range checks, NaN derivative rules, `foldIn` aliasing (FD-3 to 7, 9, 10, 15, 20, 29, 30, F-042) | M      | FD-15 changes numpy-edge conventions (public surface) |
| P3.19               | numerics scale and edge guards: `svd`, `bisect`, `adaptiveSimpson`, special functions at the edges (ND-1 to 4, 12, 20 to 23)                                                                              | M      | needs the Python oracle for the reference values      |
| P3.20 (part)        | `lstsq` and `polyfit` rank-deficient to the minimum-norm solution; `delaunay` with cocircular points (ND-5, 6)                                                                                            | M      | `lstsq` has 20 callers outside numerics               |
| P3.26               | optim: the `Math.hypot(...grad)` stack overflow, the minibatch `converged`, relative tolerances and equilibration, OMP (OL-1, 2, 3, 6, 9)                                                                 | M      |                                                       |
| P3.29               | losses: NaN gradients at zero embeddings and p in {0, 1}, out-of-range hyperparameters, the CTC parameter property (OL-19, 20, 22, 23)                                                                    | S to M |                                                       |
| P3.33, P3.34        | small certain bugs in dists, stats and metrics; parameter validation in dists (DSM)                                                                                                                       | S each | one-line reproductions exist                          |
| P3.35, P3.36        | data boundary and kernel entry validation (KDES-1 to 8, 12, 18)                                                                                                                                           | S each | the `Kernel` PSD flag adds to the surface             |
| P3.37, P3.41        | exposition, scene and render validation: finite checks, `KINDS` derived from the type, an unknown-kind error, a `RecordBoundary`, one contract test over 61 kinds (KDES-9, 10, 13, 19, 22; RD-2, 4, 5)    | M      |                                                       |
| P3.42, P3.43, P3.44 | render accessibility quick wins, the 65 `Math.max(...x)` spreads, the export and tape path (RD)                                                                                                           | M      | a11y needs a browser to verify                        |

### Wave C: conventions (needs your decision first: see below)

| work                     | what                                                                                                                                 | effort |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| P3.20                    | one failure convention and one status vocabulary across solvers, tests and routines                                                  | L      |
| P3.30                    | one NaN, Infinity and empty-input rule for stats, metrics and distributions, and `sampleWeight` honoured or rejected by every metric | L      |
| P3.27                    | the LP default: Dantzig with Bland as the anti-cycling fallback and one verdict rule in place of the retry chain                     | M      |
| P3.14                    | trace: one stepper, one JSON codec, one option table, events written once (FD-8, 18, 19, 21)                                         | M to L |
| P3.15, P3.6              | tensor file-level cycles, the unused backend abstraction, the layering rank check in `make layers` with the corrected order          | M      |
| P3.4, P3.5, P3.22, P3.25 | the option-table builder, the coercion helpers, the SVM duals, complex arithmetic, the remaining duplicate clusters                  | M      |

### Wave D: tests (offset by merging redundant ones; the count stays about flat)

P3.7, P3.16, P3.23, P3.31, P3.39, P3.18: table-driven contract tests: a dtype by op-family matrix and about 40 numpy-referenced
degenerate rows (FD-31); a second-order option for `checkPrimitive` (FD-32); goodness-of-fit for the samplers (FD-33); one table of
(distribution, parameters) rows through generic consistency checks and about +24 cases for dists, stats and metrics (DSM); one
catalogue table of kernels, split, sampler and invalid-argument tables, one contract table over the 61 exposition kinds (KDES); about
290 `it.each` rows for numerics (5 to 15 tests if each table is one `it`); a smoke render of every record view (+3). **Estimated net
change: about +40 to +60 `it` blocks on 26,079 tests (about 0.2%), with the case count up by a few hundred rows.** An estimate from the
readers' own counts, to be measured.

### Wave E: the deliberate break (decision D-009)

Public numeric signatures with native arrays to tensors, staged by domain. The readers sorted the 310 upper-bound functions:
numerics 60 public vectors or matrices of 92 (12 kernels, 20 shapes), optim 34 of 73 (20 kernels, 19 shapes), stats 13 of 28,
metrics 11 of 21, `random` 6, data 3, exposition 1: about 128 functions in all, plus about 40 fields in input records the script
cannot see (`GlmProblem`, `AdmmOptions`, online learner state). The largest: `numerics/polynomial/power.ts` (10, callers in `timeseries`,
`systems`, `signal`), `numerics/geometry/transforms.ts` (6), `optim`'s `GlmProblem` and the online learners. Callbacks that take native
arrays (`AdmmProblem.xUpdate`, `ProximalStep`, `Lmo`, `Mirror`, `OnlineLoss`) cannot be differentiated through; the minimisers' objectives
are already tensor to tensor.

## Decisions I need from you

1. **Wave A: go.** Behaviour-preserving, each with a measured before and after. I am starting it now, one agent at a time. (Unless you say stop.)
2. **Wave B: approve the class of change.** Each item turns a wrong or undefined result into a right one or an error, with a failing probe
   and a test. The one with reach: `lstsq` and `polyfit` on a rank-deficient system return the minimum-norm solution (today garbage of size
   1e15); 20 callers outside numerics. I recommend yes to all of Wave B; say if any should wait.
3. **The failure convention (P3.20).** Options: (a) numeric routines return a result with one `status` vocabulary and never throw for
   non-convergence; invalid arguments throw `OmaiError('value')`; (b) throw on non-convergence everywhere; (c) leave it per domain.
   I recommend (a): the vector minimisers already do it, and `RunStatus` already exists; the mathematical-programming solvers map onto it.
4. **The NaN, Infinity and empty-input rule (P3.30).** Options: (a) elementwise operations propagate NaN like numpy, and everything that
   summarises or decides (tests, metrics, splits, solvers) rejects non-finite input at its boundary with `OmaiError('value')`; empty input
   returns the mathematically stated value or throws, per a table; (b) reject everywhere; (c) leave it. I recommend (a).
5. **The native-array migration (D-009 / Wave E).** Go, no-go, or only part (for example `random`, `data`, `stats`, `metrics` first)?
   It changes about 128 public signatures, staged by domain, and is the one deliberate break of the surface. I recommend starting with the
   six functions in `random` and the three in `data`, then deciding the rest from how that goes.
6. **The layering order.** Adopt the readers' corrected order for a rank check in `make layers`: `base < scene, exposition < context, tensor <
   random < space < trace < autodiff < numerics < dists < optim < losses < ...`? The readers disagree on small points (they cleared the
   `context` and `optim` to `dists` findings as rank mistakes, not code mistakes); I would adopt a check that forbids only the clear
   violations (core importing a constructed domain), not a total order.
7. **Test count.** An estimated +40 to +60 `it` blocks (about 0.2%) for the contract tables: acceptable? You asked for flat; I read that as
   the surface, not a hard count.
8. **`numerics/spatial` placement.** About 1,850 lines of approximate-nearest-neighbour code (graphs, hashing, quantisation) look like a
   retrieval domain, not numerics (ND-29). Move it, or leave it?
9. **`@example` coverage of the 22 exports with no `@example`** and the three TSDoc gaps in the foundations (`optionTable`, `logContext`,
   `contractOf`): fold into the work items as I go (my default).

## What happens next

Wave A starts one item at a time (the first is P3.12 and P3.32 together because both are about the sampler block); each merges after
the full local gate, a surface comparison and CI, as before, one heavy run at a time. Wave B starts as soon as you approve it (decision 2).
Waves C and E wait for decisions 3 to 5. The order inside each wave follows this file and can change at your word.
