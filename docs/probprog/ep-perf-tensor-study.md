# Tensorising the grid kernels of message passing: a feasibility study

**Verdict: NO-GO, as the tensor library stands.** A tensor version of the shared grid integrator (the pair profile, the whole algorithm pass
for pass) gives bit-identical results, but it is 3.5 to 4.0 times slower than the current scalar code at the current node counts and at two
and four times as many, and it allocates two to five times as much memory per call; the 1.5x speed criterion fails by more than a factor of two
and the gap does not close as the grid grows (it is a constant factor, not an overhead). The cause is measured and local: the tensor ops are
at raw speed for same-shape elementwise work (`exp`, `log`, `add`, `mul`, `sum`: within 5% of a typed-array loop from 1e4 elements up), but a
broadcast binary op is 6 to 8 times slower than the loop, `max` is 7 to 13 times, `take` (gather) 30 times, `logsumexp` 3 times, and a chain of
ops pays one pass and one allocation each where a hand loop fuses them. The verdict flips to GO if those are fixed (section "What would turn
it into a GO"), which is work in the tensor library, not in message passing. The criteria and their numbers are in the table below; the
baseline the numbers sit against is `ep-perf.md`.

> **Addendum, 2026-11-12: the verdict above is superseded.** The slow paths named below were fixed in the tensor library (broadcast, reductions,
> gather; section "Addendum 2026-11-12: after the fast paths"). The same study, rerun, gives tensor / scalar of 1.49, 1.2 and 1.15 at one, two and
> four times the nodes (was 4.1, 4.0 and 3.95), bit-identical: GO on the owner's rule, marginally at the current node counts. The text between
> here and the addendum is the original study, kept as it was written.

## The criteria and what was measured against each

The owner's rule: figure out whether it is worth it, and if it is, do it. The criteria are the four the owner gave; one measurement was added
to the first (allocation per call is stated as a number, not "does not blow up").

| criterion | threshold | measured | result |
| --- | --- | --- | --- |
| 1. speed: tensor integrator against the current one, pair profile, a Laplace likelihood with a random scale | at most 1.5x slower at the current node counts; no blow-up in memory per call | 4.0x at the current counts (4320 final nodes: 168 us scalar, 675 us tensor), 3.7x at 2x the panels (14400 nodes), 3.6x at 4x (57600 nodes); heap plus typed arrays per call 0.4 MB scalar against 1.9 MB tensor (4.7x) at 1x, 1.3 against 5.1 MB (4.0x) at 2x, 4.8 against 8.9 MB (1.9x) at 4x | FAIL |
| 2. correctness | within the digest's TOLERANT level (beliefs 1e-9 relative, evidence 1e-12), ideally bit-identical | maximum absolute difference 0 in the log mass, both means and both variances, at all three sizes: bit-identical (`Object.is` on every number) | PASS |
| 3. scope: how much of the per-node scalar integrand closures is a mechanical rewrite | a count and a rough effort | about 95 closures in the operators (about 50 table rows, about 44 inline call sites in 22 files), about 60 of them mechanical, about 20 needing care (special functions, `where` branches, a sum over classes inside each node), about 15 needing redesign; 8 to 12 agent-days with the digest at every step (section "What a full tensorisation involves") | a large but bounded job |
| 4. benefit | what the owner gains | the tensor library is exercised at grid sizes (done: it found the slow paths above); no raw buffers in the integrand code; **no speed gain** on this backend at any grid size measured (the ratio is flat at 3.5 to 4 from 4e3 to 6e4 nodes); an estimated 2.5 to 3 times slower EP runs end to end if every grid kernel were converted as it is today | no gain, a clear cost |

One criterion fails by a wide margin and the benefit is a validation exercise, not a performance or a correctness gain: NO-GO. Nothing below
needs a different criterion; had correctness been the problem (a summation order that moves digits) the verdict would have been about
the digest, and it is not.

## The question

The kernels of the numeric operators (the panelled Gauss-Legendre grids over one to three coordinates in `message-passing/numerics/_grid.ts`,
`tiltGrid`, `tiltedByQuadrature`, and the Gaussian and Bartlett quadratures) work on `Float64Array` buffers inside loops that call a scalar
integrand closure once per node. The owner parked converting them to tensors as close to a wholesale rewrite and asked for numbers first:
is the tensor version slower or faster, by how much, and what would it take?

## Method

- `scripts/perf/tensor-grid-study.ts` rewrites `integratePair` (profile `pair`: graded panels split at the kinks of the density, six passes at
  most, the zoom of each window onto the mass, the moments) with tensor operations only, following the library pass for pass with the same
  arithmetic in the same order: the nodes are built from `arange`, `pow`, `concat`, `flip`, `reshape` and a broadcast of panel starts against
  the Legendre offsets; the cavities' log densities are tensor expressions of the nodes; the integrand is one broadcast expression over
  the `n0 x n1` grid (the column form's "once per node of the second axis" is a `[1, n1]` factor); the reduction is `max`, `exp`, `sum` over the
  grid and over each axis. Control flow (which window next) reads the small marginal vectors. The integrand is the one of
  `scripts/perf/grid-cases.ts`: `-t - exp(-t) |y - x0|`, a Laplace outcome with a Gaussian location (kink at the outcome) and a gamma-distributed
  scale, in the column form the operators use.
- Both implementations are timed through `scripts/perf/timing.ts` (warm-up, calls batched to at least 30 ms, the median of nine timings), at the
  library's profile and with the panels multiplied by two and by four (the library's profile table is scaled for the call and restored; no
  library file is edited).
- Memory is the growth of `process.memoryUsage()` (heap and typed-array storage) over twenty calls in a child process with a semispace of 256 MB, so no
  collection runs during the measurement. It is a rough figure: the scalar side's typed-array number is within the noise of the measurement.
- `scripts/perf/tensor-ops-floor.ts` times single tensor ops at 1e3, 1e4 and 1e5 elements against the equivalent raw typed-array loop (which allocates
  its output as the tensor op does).
- The machine was shared with other test runs during all of this (1-minute load average between 15 and 37; the calibration workload stayed at
  20 to 21.7 ms, within 5% of its quiet value, because the process keeps a core, but the spreads of single timings are 5 to 60%). The ratios
  repeated across four runs of the study (3.8, 4.3, 3.8 and 4.0 at the current counts; 3.4 to 3.9 at four times the panels).

## The numbers

### The pair integrator, scalar against tensor

| panels | final nodes | scalar | tensor | tensor / scalar | max abs difference (log mass, means, variances) |
| --- | ---: | ---: | ---: | ---: | --- |
| as the library (1x) | 4320 | 168 us | 675 us | 4.0 | 0, bit-identical |
| 2x | 14400 | 470 us | 1728 us | 3.7 | 0, bit-identical |
| 4x | 57600 | 1866 us | 6622 us | 3.6 | 0, bit-identical |

Both runs make two passes here (a coarse pass over the cavity's window, then one fine pass over the window the mass was found in). The tensor time
splits, at 1x, into about 100 us to build the nodes and the cavity densities (some tens of small ops: `concat`, `reshape`, slices, `pow`), 550 us for the
grid expression and its reductions, and 15 us for the zoom.

### One pass of the grid expression, op by op (4320 nodes, a 60 x 72 grid)

| step | tensor | raw loop |
| --- | ---: | ---: |
| `lw0 + lw1` (broadcast `[n0,1] + [1,n1]`) | 81 us | 10 us |
| the integrand (`neg`, `exp`, `abs`, `sub`, `mul`, `sub`, with broadcasting) | 174 us | 33 us |
| `max` of the log weights | 25 us | 5 us |
| `exp(logw - top)` | 46 us | 34 us |
| `sum` of the weights | 5 us | 4 us |
| `weights / total` and the two marginals (`sum` over each axis) | 23 us | 16 us |
| total | 353 us | 102 us |

Three quarters of the gap is the broadcast ops and `max`; the rest is the one pass and one allocation per op.

### The tensor ops against raw loops (microseconds, ratio in brackets)

| op | 1e3 elements | 1e4 | 1e5 |
| --- | ---: | ---: | ---: |
| `exp` | 10.1 / 6.9 (1.5) | 67 / 69 (1.0) | 663 / 658 (1.0) |
| `log` | 7.3 / 6.7 (1.1) | 66 / 65 (1.0) | 632 / 624 (1.0) |
| `add`, same shape | 2.3 / 1.8 (1.3) | 20 / 19 (1.1) | 150 / 162 (0.9) |
| `mul`, same shape | 2.5 / 1.7 (1.4) | 19 / 19 (1.0) | 156 / 148 (1.1) |
| `sum` | 1.3 / 0.9 (1.4) | 10 / 9.7 (1.0) | 96 / 93 (1.0) |
| `sum` over one axis of a square | 2.1 / 0.9 (2.3) | 12 / 8.5 (1.5) | 103 / 90 (1.1) |
| `add`, broadcast `[s,1] + [1,s]` | 17 / 2.1 (8.3) | 116 / 18.5 (6.3) | 1036 / 167 (6.2) |
| `max` | 5.7 / 0.44 (13) | 55 / 8.1 (6.8) | 538 / 79 (6.8) |
| `logsumexp` | 14.7 / 4.5 (3.3) | 139 / 48 (2.9) | 1207 / 480 (2.5) |
| `take` (gather by an int32 index) | 51 / 1.7 (30) | 491 / 17 (29) | 5128 / 152 (34) |
| `exp(-0.5 (a-b)^2)`: four ops against one fused loop | 19 / 5.1 (3.8) | 146 / 50 (2.9) | 1454 / 493 (2.9) |

The overhead floor of a tensor op is about 0.6 microseconds (the `add` at 1e3 elements, 2.3 us against 1.8 us for the loop). Above 1e4 elements the
well-implemented ops are the loop. What is slow is specific: the broadcast kernel (about 25 ns an element against 4), `max` (5 ns against 0.8), the
gather (50 ns against 1.7), and the multi-pass composition of anything (`logsumexp`, a chain).

### Summation order

The tensor `sum` and `sum(axis)` add in storage order, as the scalar loops do, so the totals and the marginals are bit-identical here and the
shifted weights, the means and the variances follow. A tensor library that moved to pairwise or blocked summation would change the last digits
(about 1e-16 relative) but not leave the digest's TOLERANT level (1e-9, evidence 1e-12); a step that wants BITS would have to keep sequential sums.

## What a full tensorisation involves

The registry has 73 operators. The harness replays 57 of them from real runs (`ep-perf.md`); of those 39 take 5 microseconds or more per message call,
which is to say integrate numerically; the other 16 that no run of the suite reaches are named in `ep-perf.md`. Searching for what the work touches
(`grep` over `message-passing/`, counted by hand where a call site is a table row):

- **The shared kernels with a scalar integrand closure**: `integratePair`, `integrateAxes`, `integrateTriple`, `integrateLine` (all through one
  `integrateGrid`), `tiltGrid`, `tiltedByQuadrature`, `tiltedLogGrid`, `tiltedGrid`, `gaussianGrid`, `gaussianExpectation`, `productTilt`: eleven entry
  points, which contain the control flow (panels, zoom, windows) that stays scalar.
- **About 44 call sites in 22 files** pass an inline closure to those kernels (`links/links.ts` 9, `gaussian/gaussian.ts` 6, `links/truncation.ts` 4,
  `gaussian/log-scale.ts` 3, `gaussian/extremum.ts` 3, the multiclass auxiliary 2, concentration 2, triples 2, functions 2, and one each in about a dozen
  more).
- **About 50 table rows** are per-node densities: 12 scalar likelihoods (`likelihoods.ts`), 6 pair likelihoods and 3 triple ones (`pairs.ts`,
  `triples.ts`), about 30 unary functions of the `apply` operator (`functions.ts`), and the robust and prior-site rows. Together with the call sites that
  is the owner's "about 70" and a few more: about 95 closures.
- **About 190 raw `Float64Array` buffers in 27 files** (`new Float64Array` and `Float64Array.from`: `gaussian/chain.ts` 24, `covariance/_sides.ts` 20,
  `gaussian/transition.ts` 13, the multiclass variational kernel 12, `_bartlett.ts` 11, `lifts/plates.ts` 10, `_quadrature.ts` 10), most of them not
  integrand evaluations but dense linear algebra (Cholesky factors, Kalman sweeps, Bartlett nodes), which is a second job: batched small
  matrices need a tensor `cholesky`/`solve` that is batched and fast at 2 to 4 dimensions.

Classified by what the rewrite of the closure takes (an estimate from reading the call sites, not a count of lines):

| class | closures | what it is | effort each |
| --- | ---: | --- | --- |
| mechanical | about 60 | smooth, branch-free elementwise log densities (the Gaussian, gamma, beta, Laplace, Student-t, Weibull, Poisson rows; `exp`, `softplus`, `logistic`) | 10 to 20 minutes with its brute-force test |
| needs care | about 20 | special functions (`lgamma`, `logPhi`: tensor versions exist), `where` branches for a kink or a tail-stable form, a one-sided difference of two terms (ordinal outcomes) | an hour each |
| redesign | about 15 | a sum over the classes inside each node (`multiclass-site`, `-row`: a `[nodes, K]` tensor and a `logsumexp`), the gate (branch operators called per node), `max`/`min` of Gaussians, the truncation tails, `gaussian-chain` and `gaussian-transition` (sequential recursions), the `mvn-*` family (Bartlett quadrature on Cholesky coordinates, `d(d+1)/2` axes) | half a day to two days each |

An estimate: 8 to 12 agent-days with the digest (`make ep-digest-check` at the TOLERANT level, `--only` per operator) after every operator and the
brute-force test of that operator, in the order below, if the tensor library had no slow paths. The slow paths would add their own work (next section).

For the EP runs of the baseline (`ep-perf.md`), about 63% of the wall time is spent inside operators that call these grid kernels
(`gaussian-gamma-precision` 27%, `multiclass-site` 13%, `negative-binomial-dispersion` 9%, `likelihood-triple` 7%, `likelihood-site` 4%, the scale
operators 3%), another 8% in `gaussian-chain` (dense recursion, not a grid). A kernel 3.5 to 4 times slower would make those runs about 2.5 to 3
times slower overall, and the runs dominated by one kernel about 3.5 times: `softmax-two-groups` (93% in `multiclass-site`) from 9 s to about 30 s,
`circular-rows` (84% in `gaussian-gamma-precision`) from 18.6 s to about 58 s. These are extrapolations: the ratio was measured on the pair profile, not
on `tiltGrid` or `integrateLine`.

## Cost and benefit

- **Benefit.** The tensor library is validated at grid sizes (this study already did that, and found the slow paths); no raw buffers in the integrand code; the
  closures become expression trees that a future backend (WebGPU, a compiled fusion) could run. None of these is available today: there is no
  second backend, and on the CPU backend the tensor version is slower at every size measured (the ratio is flat, so a larger grid does not
  rescue it).
- **Cost.** 8 to 12 agent-days, a digest check at every step, and a 2.5 to 3 times slower EP. The performance cost is the part that is avoidable
  (below); the effort is not.
- **Risks.** Summation order (none seen here, see above). Allocation: the tensor version allocates 2 to 5 times as much per call and
  the GC pressure of the many small intermediates is not in the timings (a long run, many thousands of calls, would show it). Closure per node against vectorised
  expressions: an operator that precomputes node-only terms once per column (a gamma function of a shape) must keep doing so with `[1, n]`
  factors, and a density with data-dependent branches (a Poisson outcome of zero, a tail form) needs `where`, which evaluates both sides on all nodes,
  so a branch that guards against `log(0)` can produce NaN in the unselected side and needs care in the gradient-free path too. Finally, the tensor
  library's own special functions would be on the hot path (`lgamma`, `logNdtr`, `digamma`): their per-element cost is not measured here.

## What would turn it into a GO

The measured gap is in the tensor library, so these are the changes, each with the measurement that says whether it worked. The acceptance test is to rerun
`scripts/perf/tensor-grid-study.ts` (no change to it needed): tensor / scalar at most 1.5 at 1x, with bit-identical numbers and at most twice the memory.

1. **A fast path for broadcast binary ops** (`[n,1]` against `[1,m]`: 6 to 8 times slower than the loop today, about 25 ns an element). Worth about 200 of the 353 us of a pass.
2. **A fast path for `max` (and `min`, `argmax`) reductions**: 7 to 13 times slower than the loop. About 20 us a pass, and the same in every `logsumexp`.
3. **A fast gather (`take`, indexing with an int32 tensor)**: 30 times slower; not on the pair path, but the ordinal and multiclass kernels (and the tables) index.
4. **Fewer passes and allocations for chains**: either an in-place or `out=` form of the elementwise ops, or a fused primitive for `exp(a - top)` with its sum (`shiftedSum`), or
   expression fusion for a chain of elementwise ops (compile the expression to one loop). Without something of this kind a chain of `k` ops costs `k` passes, which
   by itself is 2.5 to 3 times a fused loop (the four-op `exp(-0.5 (a-b)^2)` row above, and `logsumexp` at 2.5 to 3.3).
5. **Cheaper node building**: some tens of small ops at 1 to 3 us each (about 100 us a call, where the scalar loop takes about 20): cache the panel rule and edge vectors for a
   window, or build all panels with one outer op.

An estimate, not a measurement: with 1 and 2 at raw speed, the grid-expression part of a pass falls from 353 to about 130 us (the same-shape ops
and the remaining allocation), which with the node building is about 2 times the scalar call; with 4 as well, near 1.3 to 1.5. That is the borderline of the criterion, so
the first step of a GO would be 1 and 2 and a remeasurement, before any operator is touched.

If those are met, the plan, so it can be executed without further design (each step ends with the ratio from `scripts/perf/tensor-grid-study.ts` or
`make ep-perf-check ARGS="--only grid/,op/<names>"` at most 1.5 against `ep-perf.json`, and `make ep-digest-check ARGS="--only <the items that use it>"` at the TOLERANT level or better):

1. Tensor-library prerequisites 1 to 3 above, each with a benchmark in `scripts/perf/tensor-ops-floor.ts` (the broadcast row, `max`, `take` at most 1.3x the loop at 1e4).
2. The shared integrator: `integrateGrid` with tensor nodes, log weights and reductions (the experiment is the starting code: the pair profile first, then `axes`, `triple` with
   the `indexed` form as an `[a,b]` table gathered to `[a,b,c]`, and `line`), keeping the scalar-closure entry points working beside it until the last operator moves. Check: `grid/*`
   rows of `ep-perf`, the `_grid.test.ts` suite, and the digest for the items that use the grid (`likelihood-pair`, `-triple`, `negative-binomial-dispersion`).
3. `tiltGrid`, `tiltedByQuadrature` and `findWindow` (which already takes a function from an array to an array): the one-dimensional kernels. Check: `kernel/*` rows,
   `gaussian-gamma-precision`, `gaussian-scale`, the scale operators, `likelihood-site`.
4. The mechanical table rows by folder: `likelihoods` (scalar, pair, triple rows), then `gaussian` (log-scale, scale, extremum), `scales`, `links` (probit, logistic,
   truncation). Per operator: the brute-force test, `ep-perf-check` for its `op/` rows, the digest `--only` its atoms.
5. The rows that need care: ordinal outcomes (pairs of terms), tails, the `apply` unary functions with their `where` branches.
6. The redesigns: `multiclass-site` and `-row` (`[nodes, K]` and `logsumexp`), the gate, `max`/`min`.
7. The dense-algebra operators last and only if a batched small-matrix tensor kernel exists: `gaussian-chain`, `-transition`, the `mvn-*` family.
8. Full digest at EXACT (BITS where the order of sums is kept), `make ep-perf` for a new baseline, and the ratios of the table in this document.

## Addendum 2026-11-12: after the fast paths

The owner decided to fix the slow paths with spot patches that keep the old generic code as the fallback (no new tensor type, no new API: the fast
kernels sit beside the generic ones and are chosen by layout or shape inside the existing ops) and to rerun the study. Done on the branch
`agent-a5ab09-tensorfast`, one commit per item, each with a property test that compares the fast kernel with the generic path bit for bit.

### What was changed

| slow path (section "What would turn it into a GO") | change | where |
| --- | --- | --- |
| 1. broadcast binary ops, 6 to 8x a raw loop (two output-sized index maps, a closure per element) | `planBroadcast` describes a broadcast as at most four merged axes with a 0 or 1 stride per operand (outer product, row or column against a matrix, trailing and leading axis, rank-2 and rank-3 grids, any mix); written-out row loops per op run it (`add sub mul div max min pow`, the six comparisons, `where`); `map2` and `broadcastTo` run through the plan too. More than four axes after merging keeps the index-map path (`map2Indexed`, `whereIndexed`, `broadcastToIndexed`, unchanged) | `tensor/_cpu/broadcast.ts`, `elementwise.ts`, `structure.ts` |
| 2. `max`, `min`, `argmax`, `logsumexp`, 7 to 15x | plain loops per operation over a block of adjacent reduced axes (no combine closure; `max` and `min` with four running values, which is order-free; sums and products keep the ascending storage order from the identity); `argmax` over the last axis as a direct loop; a fused `logsumexp` kernel (maximum pass, `exp(x - shift)` summed in storage order, `log(sum) + shift`) taken for concrete float64 values over adjacent axes through an optional `Backend.logsumexp`, the composition of operations kept for traced values, other dtypes, gapped axes and empty reductions | `tensor/_cpu/reduce.ts`, `backend.ts`, `ops/reduce.ts` |
| 3. `take`, 30x | the cost was mostly `wrapIndices`, which built its `Int32Array` with `Int32Array.from` and a callback (40 ns an index): a loop now that returns the indices untouched when they are in range; `take`, `takeAlongAxis`, `takeScatter`, `putAlongAxisAdd` get a loop of their own along the last axis; `oneHot` of concrete labels writes its ones directly | `tensor/ops/indexing.ts`, `_cpu/structure.ts` |
| 4. chains: fewer passes and allocations | **not done**: no elementwise op has an output buffer or an in-place form, and the design forbids one (every kernel allocates a fresh output and never mutates its inputs; reverse mode keeps inputs and outputs alive for the vjp, so an in-place op on a value a trace holds would corrupt gradients). Measured instead what it could buy (last three rows of the table) | none |

Results are bit-identical to the generic paths by construction (the same IEEE operation per element in the same order; every sum keeps its order) and by test: `broadcast.property.test.ts`,
`reduce.property.test.ts`, `gather.property.test.ts` (in `_cpu/`) and `logsumexp-fused.property.test.ts`, `onehot-fast.property.test.ts` (in `ops/`) compare every fast path with the generic
one (or with a reference written from the definition) with `Object.is` on random shapes, axes and values including NaN, infinities and signed zeros, over the shapes each path claims and
shapes it must hand back (one test builds a seven-axis alternating broadcast that has no plan). The existing tensor, autodiff, losses and linalg tests pass unchanged (tensor 595 tests,
autodiff 89, losses and `numerics/linalg` 753).

### The tensor ops against raw loops, after

Tensor microseconds before, then after, then the raw loop, with the ratio after in brackets. Every cell is the minimum over three runs of the median of seven timings (`scripts/perf/tensor-ops-floor.ts`;
"before" was measured with the same harness on the unmodified base `b8d3c8e9`). Load average 5 to 12 from another agent's tests, calibration workload 19.6 to 21.7 ms (20.0 in the baseline); the
ratios at 1e3 elements carry the 0.5 microsecond per-op floor and are noisy. The raw loops close over constant arrays, which the compiler specialises on.

| op | 1e3 elements | 1e4 | 1e5 |
| --- | ---: | ---: | ---: |
| add broadcast ([s,1] + [1,s]) | 9.0 to 2.7 / 2.4 (1.1) | 180 to 23 / 20 (1.2) | 1728 to 194 / 173 (1.1) |
| sub broadcast ([s,1] - [1,s]) | 18 to 3.0 / 2.3 (1.3) | 187 to 22 / 20 (1.1) | 1724 to 191 / 174 (1.1) |
| mul broadcast ([s,1] * [1,s]) | 18 to 2.8 / 2.3 (1.2) | 175 to 22 / 19 (1.1) | 1728 to 193 / 174 (1.1) |
| div broadcast ([s,1] / [1,s]) | 18 to 2.8 / 2.4 (1.2) | 180 to 23 / 20 (1.1) | 1763 to 197 / 174 (1.1) |
| maximum broadcast ([s,1], [1,s]) | 19 to 2.9 / 2.3 (1.2) | 195 to 22 / 21 (1.0) | 1825 to 197 / 173 (1.1) |
| minimum broadcast ([s,1], [1,s]) | 20 to 2.7 / 2.4 (1.1) | 186 to 23 / 20 (1.1) | 1783 to 191 / 172 (1.1) |
| pow broadcast ([s,1] ** [1,s]) | 35 to 13 / 13 (1.1) | 334 to 130 / 123 (1.1) | 3380 to 1228 / 1163 (1.1) |
| add row against matrix ([s,s] + [1,s]) | 18 to 3.1 / 2.1 (1.5) | 167 to 26 / 22 (1.2) | 1634 to 225 / 188 (1.2) |
| add column against matrix ([s,s] + [s,1]) | 17 to 2.8 / 1.9 (1.5) | 172 to 23 / 20 (1.1) | 1611 to 194 / 171 (1.1) |
| add trailing axis ([c,c,c] + [c]) | 18 to 3.6 / 2.2 (1.7) | 156 to 27 / 22 (1.2) | 1588 to 222 / 191 (1.2) |
| add leading axis ([c,c,c] + [c,1,1]) | 18 to 2.6 / 1.6 (1.6) | 165 to 21 / 18 (1.2) | 1553 to 183 / 163 (1.1) |
| add rank-3 grid ([c,1,1] + [1,1,c]) | 3.1 to 0.8 / 0.4 (2.0) | 8.7 to 1.9 / 1.1 (1.7) | 37 to 5.2 / 4.4 (1.2) |
| add rank-3 grid ([c,1,c] + [1,c,1]) | 20 to 3.3 / 2.2 (1.5) | 175 to 24 / 21 (1.1) | 1678 to 196 / 186 (1.1) |
| add rank-3 grid ([c,c,1] + [1,1,c]) | 22 to 4.9 / 2.0 (2.5) | 175 to 29 / 20 (1.5) | 1706 to 218 / 167 (1.3) |
| less broadcast ([s,1] < [1,s]) | 17 to 2.0 / 1.1 (1.8) | 168 to 16 / 14 (1.2) | 1563 to 145 / 127 (1.1) |
| where (cond [s,s], [s,1], [1,s]) | 12 to 4.4 / 2.9 (1.5) | 106 to 36 / 32 (1.1) | 983 to 319 / 306 (1.0) |
| sum over axis 1 of [s,s] | 2.0 to 1.8 / 0.7 (2.4) | 12 to 10 / 8.8 (1.2) | 100 to 97 / 93 (1.0) |
| sum over axis 0 of [s,s] | 2.0 to 2.0 / 1.0 (2.0) | 12 to 12 / 10 (1.2) | 111 to 103 / 100 (1.0) |
| sum over axis 1 of [c,c,c] | 9.3 to 2.5 / 1.3 (2.0) | 150 to 14 / 12 (1.1) | 1535 to 129 / 122 (1.1) |
| max | 6.0 to 1.0 / 0.7 (1.5) | 118 to 6.5 / 8.3 (0.8) | 1171 to 61 / 81 (0.8) |
| max over axis 1 of [s,s] | 6.3 to 1.6 / 0.7 (2.2) | 121 to 7.3 / 8.8 (0.8) | 1182 to 64 / 85 (0.7) |
| max over axis 0 of [s,s] | 3.5 to 2.1 / 1.0 (2.1) | 31 to 14 / 10 (1.3) | 296 to 117 / 100 (1.2) |
| min over axis 1 of [s,s] | 13 to 1.6 / 0.9 (1.7) | 121 to 7.4 / 11 (0.7) | 1158 to 63 / 101 (0.6) |
| max over axis 1 of [c,c,c] | 19 to 2.6 / 1.3 (2.0) | 163 to 14 / 12 (1.2) | 1628 to 130 / 116 (1.1) |
| argmax over axis 1 of [s,s] | 2.3 to 1.3 / 0.8 (1.5) | 20 to 9.1 / 10 (0.9) | 190 to 84 / 100 (0.8) |
| exp | 9.9 to 7.7 / 6.7 (1.1) | 68 to 68 / 69 (1.0) | 641 to 659 / 672 (1.0) |
| log | 6.8 to 6.9 / 6.7 (1.0) | 66 to 66 / 65 (1.0) | 620 to 637 / 634 (1.0) |
| add | 2.2 to 2.6 / 1.9 (1.4) | 18 to 20 / 18 (1.1) | 163 to 174 / 175 (1.0) |
| mul | 2.3 to 2.2 / 1.7 (1.3) | 19 to 20 / 19 (1.0) | 172 to 176 / 169 (1.0) |
| sum | 1.3 to 1.4 / 1.0 (1.5) | 9.8 to 10 / 9.8 (1.0) | 98 to 98 / 97 (1.0) |
| logsumexp | 20 to 5.8 / 4.9 (1.2) | 191 to 49 / 50 (1.0) | 1890 to 476 / 492 (1.0) |
| logsumexp over axis 1 of [s,s] | 40 to 6.1 / 5.9 (1.0) | 364 to 46 / 51 (0.9) | 3480 to 435 / 486 (0.9) |
| softmax over axis 1 of [s,s] | 63 to 16 / 14 (1.2) | 601 to 138 / 126 (1.1) | 5809 to 1278 / 1195 (1.1) |
| take (gather) | 49 to 4.4 / 1.6 (2.7) | 493 to 41 / 20 (2.0) | 5166 to 376 / 170 (2.2) |
| take rows of [s,s] (axis 0, s indices) | 4.1 to 2.6 / 2.2 (1.2) | 25 to 20 / 21 (1.0) | 180 to 182 / 189 (1.0) |
| take columns of [s,s] (axis 1, s indices) | 9.0 to 2.9 / 2.2 (1.3) | 61 to 23 / 23 (1.0) | 608 to 213 / 204 (1.0) |
| takeAlongAxis of [s,s] (axis 1) | 6.5 to 4.9 / 2.2 (2.3) | 49 to 36 / 23 (1.6) | 511 to 338 / 200 (1.7) |
| scatterAdd (n updates into [s]) | 48 to 2.7 / 1.0 (2.8) | 496 to 23 / 17 (1.4) | 5538 to 224 / 164 (1.4) |
| segmentSum (n values, s segments) | 50 to 2.5 / 1.0 (2.6) | 506 to 24 / 17 (1.4) | 5283 to 228 / 161 (1.4) |
| oneHot (s labels, s classes) | 24 to 1.4 / 1.2 (1.1) | 199 to 10 / 10 (1.0) | 1836 to 82 / 87 (0.9) |
| chain sub,mul,mul,exp vs one fused loop | 12 to 12 / 5.3 (2.3) | 103 to 107 / 53 (2.0) | 929 to 979 / 506 (1.9) |
| chain sub,mul,mul,exp vs 4 allocating passes | 12 to 12 / 12 (1.1) | 99 to 108 / 103 (1.0) | 953 to 953 / 934 (1.0) |
| chain sub,mul,mul,exp vs 4 passes, buffers reused | 12 to 13 / 4.8 (2.6) | 100 to 107 / 68 (1.6) | 916 to 1000 / 675 (1.5) |
| exp(-0.5 (a-b)^2) (4 ops vs 1 loop) | 19 to 20 / 5.2 (3.8) | 152 to 163 / 52 (3.2) | 1419 to 1475 / 516 (2.9) |

The last three rows bound item 4: for a chain of four elementwise ops (`sub`, square, scale, `exp`) at 1e4 elements the tensor library takes 107 us; one fused loop 53 us; four raw passes that each
allocate a fresh output 103 us (the tensor chain is as fast as that, which is the design cost of one allocation per op); four raw passes into buffers allocated once 68 us. An `out=` form would
save at most 37% of a chain, and fusion about half; neither is a spot patch (fusion is an expression compiler).

Acceptance targets at 1e4 elements, tensor against the raw loop:

| target (within 1.5x) | before | after | met |
| --- | ---: | ---: | --- |
| broadcast binary ops (add, sub, mul, div, max, min, pow, `[s,1]` against `[1,s]`) | 9.4 to 10.5 (pow 2.9, its raw loop is slow) | 1.0 to 1.2 | yes |
| the other broadcast layouts (row, column, trailing, leading, rank-3 grids), compare, `where` | 7.7 to 13 (`where` 3.5) | 1.1 to 1.5 | yes |
| `max` (flat, and along either axis of a square, and `min`) | 15 (flat), 14 (axis 1) | 0.8 | yes (faster than the raw compare loop, which does not propagate NaN) |
| `logsumexp`, `softmax` | 4.0, 4.9 | 1.0, 1.1 | yes |
| `take` (gather by an int32 index) | 26 | 2.0 (the kernel alone 1.6) | **no** |
| `takeAlongAxis` | 2.4 | 1.6 | no (just) |

`take` went from 493 to 41 us, a factor of 12, and `scatterAdd` and `segmentSum` from 500 to 23 us (22x); the 1.5x target was not reached for the gather itself. The kernel is a plain loop
(33 us of the 41; the rest is the op layer: the index scan, the dispatch and the shape checks) and the raw loop in the harness is 20 us. Micro-experiments with the same loop written as a function of
its arrays gave 17 to 57 us in separate processes, which points at alignment or compilation-mode noise and at the compiler specialising the harness's closure on its constant arrays, not at
anything in the kernel that a further patch would change. I could not find a formulation that got below 33 us for the kernel.

### The study, rerun

`scripts/perf/tensor-grid-study.ts`, unchanged, on the unmodified base and on the branch, the same afternoon, load 5 to 28 (the calibration workload stayed at 19.6 to 20.0 ms, so the process kept its core;
the single timings spread 5 to 20%):

| panels | final nodes | scalar | tensor, base | tensor, now | tensor / scalar, base | tensor / scalar, now (five runs) | difference |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| as the library (1x) | 4320 | 167 to 174 us | 666 us | 247 to 266 us | 4.08 | 1.48, 1.49, 1.49, 1.53, 1.48 (median 1.49) | 0, bit-identical |
| 2x | 14400 | 442 to 488 us | 1775 us | 539 to 680 us | 4.00 | 1.21, 1.22, 1.24, 1.39, 1.39 (median 1.24) | 0, bit-identical |
| 4x | 57600 | 1697 to 1742 us | 6711 us | 1967 to 2060 us | 3.95 | 1.14, 1.16, 1.16, 1.17, 1.18 (median 1.16) | 0, bit-identical |

The tensor time at 1x splits into about 65 us to build the nodes and the cavity densities (some tens of small ops at 0.5 to 1 us each), 170 us for the grid expression and reductions of both passes (355 before; one pass is
now 108 us against 93 us for the raw loops) and 13 us for the zoom. Memory per call (a rough figure: heap plus typed arrays growth over twenty calls, no collection) at 1x: scalar 0.42 MB, tensor 0.93 MB, 2.2 times; at 2x 1.2 and
2.0 MB; the 4x figures are not usable (the numbers are noisier than the effect).

### The verdict, re-evaluated against the owner's rule

The rule: GO if the tensorised integrator is no more than 1.5x slower, results are within the TOLERANT level, and the scope is bounded.

- Speed: 1.49 at the current node counts (five runs from 1.48 to 1.53, so on the line, not safely under it), 1.24 at 2x and 1.16 at 4x, against 4.0 before. **Pass, marginally at 1x.**
- Correctness: maximum absolute difference 0 in the log mass, both means and both variances at all three sizes, `Object.is` on every number: **bit-identical, pass.**
- Scope: unchanged by the fast paths, 8 to 12 agent-days with the digest at every step (about 95 closures, about 60 of them mechanical): **bounded, pass.**

**Verdict: GO, with one qualification: the 1.5 criterion is met at the current node counts only on the median.** What is left of the gap is no longer in the tensor kernels but in the per-op overhead of building the
nodes (65 of the 250 us at 1x, about a quarter) and in the fixed 0.5 us of every small op; the grid part is at the raw loops. Cheaper node building (item 5 of "What would turn it into a GO": cache the panel rule
and the edge vectors per window, or build the panels with one outer op) is the first step of a conversion and would move 1x to about 1.3. Two things the GO does not say: the tensor version is still not faster than the
scalar code at any size (the ratio is 1.15 to 1.5, never below one), so the benefit remains the validation of the tensor library at grid sizes and expression trees a future backend could run, not speed; and chains of
elementwise ops still cost twice a fused loop (2.0 at 1e4), which is what separates 1.16 from 1.0 at 4x.

The earlier extrapolation (2.5 to 3 times slower EP runs if every grid kernel were converted) becomes about 1.15 to 1.5 times on the kernel-dominated runs (`softmax-two-groups`, `circular-rows`), by the same arithmetic, still
an extrapolation from the pair profile.

### End to end, and the digest

`make ep-perf-check` on five items that are mostly array-built code, base against branch on the same afternoon, load 5 to 6, calibration 20.0 ms: `atom/gp-hyperparameters` 2.16 s to 2.04 s, `atom/chain-persistence-grid`
6.19 to 5.95 s, `atom/gate-poisson-negative-binomial` 3.47 to 3.54 s, `model/radon` 2.24 to 2.26 s, `atom/ordinal-probit-cuts-learned` 1.19 to 1.18 s. No effect beyond noise (the message-passing kernels do not use tensors, which
is the premise of the study); `gp-hyperparameters` and `chain-persistence-grid` are 4 to 6% faster, about the size of their spreads.

`make ep-digest-check ARGS="--only gp-,gate-poisson,chain-persistence,softmax-two-groups,radon,ordinal --bits"` (25 items): 22 identical at BITS. Three items (`gp-length-scale`, `ordinal-probit`, `ordinal-probit-learned`) differ from the committed
baseline in the last digits (1e-14, within TOLERANT), and they differ by exactly the same numbers on the unmodified base `b8d3c8e9`, so the drift is in the committed baseline, not in this change; the branch and the base are bit-identical on all of them.

## Alternatives to hand-written JS loops

The owner asked about the machine-code route, WebAssembly (WASM), and about a "halfway house", a specialised tensor type. An assessment for this library, written after the JS fast paths above, with their numbers as the evidence.
Nothing here is built.

### WebAssembly: what it could give

- **SIMD is two lanes for float64.** WASM SIMD is 128 bits wide: `f64x2`. On float64 arithmetic that is a 2x ceiling on a compute-bound loop, and the loops that matter here are not compute-bound. After the patches the
  same-shape and broadcast ops run at 1.0 to 1.2 times a scalar JS loop and are limited by memory traffic and allocation (an `add` of 1e4 elements is 20 us, of which about 9 is zeroing the output); `exp` and `log` cost 6.5 ns an element in
  libm, with no SIMD path in WASM. My estimate, not a measurement: a SIMD `add` might reach 1.3 to 1.5x on elements already in cache; nothing like the 6 to 30x the generic paths lost.
- **Data crosses the boundary by copy.** A WASM module reads its own linear memory; a JS `Float64Array` on the JS heap is not visible to it. Copying 1e4 doubles (80 KB) in and 80 KB out is about 10 us, as much as the whole `add`. So a
  kernel call per tensor op loses; WASM only pays if tensors live in WASM memory (a backend whose `data` is a view of the module's memory, with its own allocator and explicit lifetimes: the `Backend` interface allows a second backend, but
  `Tensor.data` is read as a plain typed array in about 5000 places in about 1100 files of `lib/`, and a memory that can grow invalidates every view), or the kernel is coarse enough to amortise the copy (a whole quadrature: 4320 nodes in, a few
  moments out).
- **No libm.** WASM has `sqrt`, `floor`, `abs`, `min`, `max` and the four operations (all exactly rounded, so bit-identical to JS), but no `exp`, `log`, `pow`, `sin`, `lgamma`. Either they are imported from JS (a call across the boundary per node costs
  20 to 50 ns, more than the `Math.exp` itself, which wipes out the gain) or they are compiled in. A compiled-in `exp` is not the engine's: The `exp` of AssemblyScript or of a musl-based C build can differ from `Math.exp` in the last bit for some inputs. V8's
  `Math.exp` and `Math.log` are ports of fdlibm, so a port of the same source, built without fused multiply-add, may match V8 bit for bit (to be proved by an exhaustive test on random doubles, not assumed), but not SpiderMonkey or JavaScriptCore, which use their own;
  and `lgamma`, `digamma` and the library's other special functions are written in TypeScript, so they would be ported too. Any difference moves digest items off BITS: the digest is recorded in one engine (V8), and a WASM libm would add a second thing the bits depend on.
- **Where it runs.** The library runs in Node, in browsers and workers (the pocs run EP in the browser) and in the test runner. WASM runs in all of them (with an asynchronous instantiation to arrange, which does not fit a synchronous
  `import`). A native addon (N-API) is Node only: it would not serve the browser and needs a prebuilt binary per platform. WebGPU is browser-side (Node only through experimental builds), its shading language has no float64, a dispatch and
  read-back costs on the order of 100 us, so a pair grid of 4000 nodes would be all overhead; it suits large batched work in float32, and can never be bit-identical.

### What it would cost

A second toolchain in a TypeScript monorepo: Rust, C or AssemblyScript (the last is TypeScript-like, with its own standard library and its own numerics), installed by `make setup` through mise, built in CI on every platform, with a binary
checked in or built on install, a loader that works in Node, the browsers and vitest, source maps and debugging that stop at the boundary, a bundle-size line for the pocs, lint and format rules for a language Biome does not cover, and the
fixtures and digest (a golden check that the module equals the JS kernel bit for bit on random inputs, on every platform that runs it). The repository's rules (no hand-built arrays in EP code, tensor ops validated by `checkPrimitive`, banned
globals) would need a counterpart for the module. It is a standing maintenance cost for a gain that, on the evidence above, is bounded for the tensor ops and uncertain for the integrators.

### Recommendation

Do the JS fast paths first, which is done: the tensor kernels are at 1.0 to 1.5 times a raw JS loop (except the gather, 2.0), the integrator is at 1.5 times the scalar one, and nothing about those kernels is limited by the language. Spike WASM only if,
after the first converted kernels (cheaper node building and a fused chain, the next steps of the plan), the integrator is still slower than the owner accepts. The most promising candidate is not a tensor op but a coarse scalar kernel whose cost is the
integrand: the Bartlett quadrature of the `mvn-*` family or `integratePair` with a fixed integrand, where one call crosses the boundary once and the loop inside is arithmetic plus `exp`/`log`.

Spike plan, one afternoon to two days, one kernel (`integratePair` with the Laplace-by-gamma integrand of `scripts/perf/grid-cases.ts`, 4320 nodes), in a throwaway directory outside `packages/`:

1. Build it in C or Rust to WASM with the fdlibm `exp` and `log` taken from the V8 sources, no fused multiply-add, no fast-math; load it from Node 22 and from the pocs build.
2. **Gate A, bit-identity:** `exp` and `log` equal `Math.exp` and `Math.log` bit for bit on 1e7 random doubles and the edge values (subnormals, overflow, 0, infinities, NaN); the integrator's log mass, means and variances equal the JS ones by `Object.is`
   on the three scales of the study and on the digest items that call it at BITS. If Gate A fails, stop: a WASM libm is not acceptable.
3. **Gate B, speed:** the kernel, including copying the inputs in and the moments out, at most half the time of the JS `integratePair` at 1x nodes (167 us, so at most 85 us), and at most 0.4 at 4x. A copy-free variant (inputs already in module memory) is
   measured separately to show the ceiling. If Gate B fails, stop.
4. **Gate C, cost:** the clean-machine setup time added to `make setup`, the CI minutes, the binary size (at most 50 KB), the loader working unchanged in Node, Chrome and vitest. Only then a design for a second backend or a kernel registry.

### The halfway house: a specialised tensor type

Would a contiguous 1-D or 2-D tensor type, or a stride and broadcast-view representation, give more than the layout-dispatched fast paths? On the numbers, not much, and at a high price.

- *What the fast paths already reach.* With no new type, every broadcast layout (outer product, row, column, trailing, leading, rank-3 grids) runs at 1.0 to 1.2 times the raw loop at 1e4 elements, comparisons and `where` at 1.1 to 1.2, the reductions at 0.8 to 1.3
  and `logsumexp` at 1.0. A dedicated low-rank type would buy the same loops with simpler dispatch (no plan, no four-axis merge); the plan costs 0.07 us a call, against an op floor of 0.5 us, so the dispatch is not where time goes.
- *What is left over the raw loop is not about rank.* The remaining excess in the integrator is the per-op overhead (coercion, `bind`, shape checks, one allocation: about 0.5 us an op, 65 of the 250 us at 1x) and the chains (2.0 times a fused loop, of which half is the
  allocation: four allocating raw passes take as long as the library does, buffers reused would save 37%). A 1-D or 2-D type has the same dispatch and the same allocation per op; it would shave perhaps 0.1 to 0.2 us an op, not the chains.
- *A view representation* (`broadcastTo` and `transpose` returning a view with strides, materialised on demand) would remove the materialised copies of explicit `broadcastTo`, `transpose` and `slice`, and would allow lazy expressions to be fused at the boundary. That is a
  real gain for code that broadcasts explicitly or fuses chains, and the only way in this list to attack the chain cost; inside the ops the plan already does the equivalent without copies. But it changes the contract `Tensor.data` is a flat row-major
  typed array (read in about 5000 places across about 1100 files, not all of them tensors), so every kernel (reduce, scan, matmul, take, sort, cast, the format and `toArray` functions, the backend interface, the autodiff rules that hold inputs) must accept strided input
  or call a `contiguous()` first, and every operation that is validated today (`checkPrimitive` for each primitive, the golden fixtures, the property tests, the digest) is rerun and several of its rules rewritten. That is weeks of work to touch the whole
  tensor layer, against the days the spot patches took.
- *Verdict on the halfway house:* not worth it now. If the integrator needs the chain cost removed after the first conversions, the cheaper experiment is an expression fusion for elementwise chains inside the existing types (a compiled loop for a chain recorded
  between two reductions), which the view representation would only make easier, not replace.

## Reproduce

```sh
nice -n 15 node --conditions=source --experimental-transform-types --no-warnings scripts/perf/tensor-grid-study.ts [--json out.json]
nice -n 15 node --conditions=source --experimental-transform-types --no-warnings scripts/perf/tensor-ops-floor.ts [--only <substring>[,<substring>...]] [--json out.json]
```

Both are experiments outside the library (`scripts/perf/`); they are not wired into any check. The typed-array loops they time against are the
reference being compared with, not a data structure for values (the CLAUDE.md rule on hand-built arrays names the hot loops of the grid kernels as
its exemption).
