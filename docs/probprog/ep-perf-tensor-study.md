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

## Reproduce

```sh
nice -n 15 node --conditions=source --experimental-transform-types --no-warnings scripts/perf/tensor-grid-study.ts [--json out.json]
nice -n 15 node --conditions=source --experimental-transform-types --no-warnings scripts/perf/tensor-ops-floor.ts [--json out.json]
```

Both are experiments outside the library (`scripts/perf/`); they are not wired into any check. The typed-array loops they time against are the
reference being compared with, not a data structure for values (the CLAUDE.md rule on hand-built arrays names the hot loops of the grid kernels as
its exemption).
