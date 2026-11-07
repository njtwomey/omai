# Optim and losses review: solvers, minimize, line searches, LP/QP/conic, SMO, stochastic, online, sparse; the loss library

Read-only review of `packages/engine/src/lib/optim` (88 non-test files, about 33,000 lines, 56 of them exporting an
`Algorithm`) and `packages/engine/src/lib/losses` (19 files). All paths below are relative to
`packages/engine/src/lib/`. Probes cited as "probe" are throwaway node scripts in
`/private/tmp/claude-501/-Users-nialltwomey-workspace-whitebox-js/d4034594-54d3-484e-a26a-331816a1f3f5/scratchpad/review/probe/`
(`p1.ts` ... `p22.ts`, `sigs.mjs`, `cov.mjs`, `cov2.mjs`, `doc.mjs`), run one at a time at `nice -n 15` on tiny sizes against
the source; nothing in the repository was changed and no test, tsc or gate was run. Scipy, torch and optax behaviour quoted
here is from memory, not from running them (no Python); treat it as an expectation to confirm with the oracle before a fix.
Coverage figures are Phase 0's (`coverage-final.json`, commit b09e46da); line numbers are from the tree at commit 94aeb98f.

## Summary

1. Healthy: the **vector-minimiser family is the most consistent code read so far**. All 13 `minimize/` methods probed
   (bfgs, lbfgs, newton, newtonCg, gradientDescent, conjugateGradient, nelderMead, powell, lbfgsb, trustRegion, cobyla,
   hookeJeeves, cmaes) agree on zero-gradient starts (`converged`, 0 steps), constant objectives, and a NaN objective
   (`diverged`, 1 evaluation); objectives and residuals are tensor-to-tensor (`Objective`, `ResidualFunction`,
   `BatchObjective`, `FiniteSum.term`, `Prox`), so `grad` reaches them; one `MinimizeResult` and one `runVector` own the result
   shape. The line searches are one implementation each (a direct search is the stepped one run to the end). No
   `Math.random`, clock, `console` or `// allow:` in 107 files; TSDoc is complete for 536 of 546 exported functions (F28).
   Solver overhead is negligible: L-BFGS costs 0.23 ms per gradient evaluation beyond autodiff at n = 1000 (probe p11),
   SMO about 250 us per iteration at n = 1000 (p18), the stochastic step 0.035 ms over a bare `valueAndGrad` (p19).
2. Not healthy, 1: **scale and degenerate inputs give silent wrong answers, which is the CI flake class**. `linprog` with
   the objective scaled by 1e-9 returns a wrong "optimal" 30 times in 40 random programs, and `c = [-1e-12, -2e-12]` returns
   `x = [0, 0]` where `[3, 1]` is optimal (F3); with the objective scaled by 1e-8 seven gradient methods return the start
   as `converged` in 0 steps, and `fitGlm` with `lbfgs`/`newton` does the same on a design scaled by 1e-8 while `irls` gets
   the right answer (F6); the stochastic `gradient` mode stops as `converged` at step 0 to 2 because it tests the gradient of
   the minibatch it just drew (F2); OMP on a dictionary scaled by 1e-9 returns zeros, `converged` (F9).
3. Not healthy, 2: **no policy for non-finite input and a second status vocabulary.** NaN or Infinity in an LP gives
   `optimal` (NaN `c`, NaN `b`, and `b = Infinity` returns `x = [Infinity, 0]`), a cryptic `choleskyJitter` error (interior
   point on `A = Infinity`), `converged` in 0 iterations (SMO on a NaN kernel, `nnls` with NaN), or 150 wasted iterations
   (QP) while the assignment and transport solvers throw `value` (F5). Mathematical-programming results
   (`linprog`, `quadprog`, `conicProgram`, `milp`) say `status: 'optimal' | 'infeasible' | ...` and `success`, vector methods
   say `status: RunStatus` and `converged`; the same field name has two vocabularies (F7).
4. Not healthy, 3: **the LP path is a patch chain on a poor default.** The default pivot rule is Bland's: 223 pivots at
   40 x 80 against 16 for Dantzig, and `numerical difficulties` on every 60 x 120 and 80 x 160 plain packing program tried, so
   `linprog` runs a retry chain of up to five solves (84 ms against 13 ms; F4). Row equilibration, a pivot floor, a retry
   without it, an interior-point cross-check and a Dantzig re-run are five mechanisms for one defect (no scaling of `c`, `b`
   or columns, absolute tolerances). The interior point's unbounded/infeasible call (`lp-interior-point.ts:273-277`) is still
   there; on a box-bounded program it never fired in 180 random cases, and `linprog` now cross-checks it.
5. Layering: no real violation. `optim/glm.ts` and `losses/{density,generative}.ts` import `dists` for types only (nothing
   at run time; `dists` does not import `optim` or `losses`); the sound order is `dists < optim < losses`, a rank mistake in
   the first-guess order, as foundations F17 found for `context` (F24).
6. Duplicates: the SVM duals are set up twice (`optim/smo.ts` and `svm/_dual.ts`, with the same exported names and different
   signatures; F8); two proximal-operator types (`Prox` on `Value`, `ProximalStep` on `Float64Array`; F14); `slicedWasserstein`
   and `sinkhorn` exist in both `losses` and `optim` with different meanings (F21); a knob's default is written three
   times and 53 `k['name'] as number` casts have one cause, the untyped `make(knobs)` of the option tables (F12); the line
   search outcome is read by a regex on a message in two places (F11).
7. Native arrays: 65 optim and 8 losses public functions; classified in F15: 20 kernels (allowed), 19 shape/index/combinatorial
   (not vectors), 34 public vector or matrix or callback (migrate, mostly small and leaf); about 40 more in input records the script
   cannot see (`GlmProblem`, `AdmmOptions`, `SmoOptions`, `OnlineLearner` state). **Objectives are `Value`-typed** in
   `minimize/`, `stochastic`, `variance-reduced`, `splitting`, `admm` (prox forms), `landscape` and `implicit`, so `grad` reaches
   every user objective; the native-array callbacks are `AdmmProblem.xUpdate/zUpdate`, `ProximalStep`, `Lmo`, `Mirror`,
   `OnlineLoss`, `finiteDifferenceHessian(gradient)` and `preconditionBy(fn)`, which cannot be differentiated through.
8. Losses are healthy where they are written once (reduction, weights, `LossOptions` uniform) and weak at the edges: NaN
   gradients at zero embeddings (`l2Normalise`, so `ntXent`, `infoNce`, `supervisedContrastive`, `clipLoss`) and at
   `binaryCrossEntropy` with `p` in {0, 1} (F20), hyperparameters accepted out of range (`huber` with `delta = -1` returns
   -2.5; F19), `crossEntropy` costing what `logSoftmax` costs, 247 ms for 256 x 8000 (F23), and one module that cannot be
   loaded by Node's native TypeScript (F22).
9. The three highest-value fixes: (a) F1 + F2 + F9 (three small certain bugs: `Math.hypot(...grad)` overflows the stack at
   130,000 parameters, the minibatch `converged`, OMP's absolute test; hours each); (b) F3 + F4 together (scale `c`, `b` and
   columns in `equilibrated`, relative tolerances, Dantzig by default with Bland as the anti-cycling fallback, one
   verdict rule instead of the chain); (c) F5 + F7 + F16 (one non-finite-input check in `_programming.ts`, one `SolveStatus`
   type with a mapping to `RunStatus`, one empty-problem convention), joined to numerics F10 and F11.
10. Not read or not verified: transport, conic, integer, nsga2, multiobjective internals, trust-region, cmaes,
    differential evolution, dual annealing, cobyla, slsqp, trf/dogbox/lm bodies, dictionary (K-SVD/MOD) bodies,
    total-variation, gromov, online experts bodies (read as signatures, status and coverage only); in `losses`, bodies other
    than classification, divergence, contrastive and regression were probed rather than read, and the CTC functions were not
    run (the module does not load under Node strip-types, F22). No test, tsc or gate was run; coverage is Phase 0's, so
    functions exercised only by an `@example` count as "never run" there (about 12 named ones, e.g. `trfAlgorithm`,
    `admm`, `sr1Algorithm`). About 36 source files opened.

Entries by kind: consistency 12, hot-path 5, duplicate 5, rule 4, test-gap 3, layering 1, native-array 1 (31 in all).

---

### F1: `Math.hypot(...grad)` overflows the stack at 130,000 parameters, in the solver logging path

kind: hot-path
evidence: optim/solvers.ts:329 (`ctx.log('solver/gradNorm', Math.hypot(...r.grad))`, inside `loggedSolverOf`, every step of every model that uses a solver slot) and optim/slot.ts:137 (`gradNorm: Math.hypot(...r.grad)` in `slotAlgorithm.observe`, every observation); probe p7: `loggedSolver` with n = 100,000 works, n = 130,000 and 200,000 throw `RangeError: Maximum call stack size exceeded`. `dnrm2` from `numerics/linalg/blas.ts` is the rule's helper and is already imported elsewhere in the folder.
what: spreading a `Float64Array` into call arguments fails past the engine's argument limit (about 125,000 here), so any model with a large parameter vector (an embedding table, a wide linear model) that goes through `loggedSolver` or `slotAlgorithm` dies on its first step with an error that does not name the cause. It is also a copy of the vector into an arguments list on every step.
fix: `dnrm2(r.grad)` in both places. While there: the metric `solver/gradNorm` is the 2-norm whereas the stopping tests of the same solvers use the infinity norm (`infNorm`, `slot.ts:131`); name it in the contract description (see F17).
size: S (minutes)
risk: low (`dnrm2` is scaled, so values equal `Math.hypot` to the last bit or two; a golden of a logged norm would move within tolerance)
preserves behaviour: yes (except that the failure disappears)
changes public surface: no

### F2: the stochastic `gradient` mode stops as `converged` because the gradient of the batch it just drew is small

kind: consistency
evidence: optim/stochastic.ts:105-106 (`stopped = (grad) => infNorm(grad) <= gtol ? { stop: converged }`), applied at :129 (`init`) and :148 (every step) to `grad` evaluated at :99-101 on the minibatch indices drawn from the step's key; `solvers.ts:197` (`gradTol` default 1e-5 for this mode too). Probe p9 (10 examples, squared hinge, one weight, start 0.5, optimum -0.065): full batch `converged@11` at w = -0.065; `batch: 1` six seeds `converged@2, @0, @1, @0, @0, @0` with w = 0.5 (never moved); `batch: 2` `converged@17, @25, @31, @4, @0, @16` with w from 0.5 to -0.24.
what: a minibatch whose examples all have zero loss (hinge, ReLU, a satisfied margin, a dead unit, an empty class) has an exactly zero gradient, so the run is declared `converged` at the first lucky batch, with a status that every viewer and every `slotAlgorithm` model reads as a result. The test is meaningful only on the full gradient. `variance-reduced.ts` tests the change over an epoch instead and does not have the problem.
fix: in the stochastic branch (`stochastic === true`) do not test `gradTol` on the batch gradient; either never (run to `maxSteps`, status `completed`) or test the full-batch gradient every `epoch = ceil(n / batch)` steps and at the end, or a moving average of the batch gradient norms. Keep the exact test for the full-batch case.
size: S (<1h)
risk: low (a stochastic run that used to stop early now runs to its step cap; `stochastic.test.ts` logistic test with `batch: 32` pins a final value, not the stop)
preserves behaviour: no (stochastic runs stop later, correctly)
changes public surface: no

### F3: LP tolerances are absolute: the answer depends on the scale of `c` and `b`, and the equilibration scales only the rows of `A`

kind: consistency
evidence: optim/lp-simplex.ts:96-102 (`tol = 1e-9`, `feasTol = tol * bScale`), :321-324 (the entering test `d < -tol` on reduced costs, so absolute in the scale of `c`); optim/lp-interior-point.ts:103 (`tol = 1e-8`) and :272 (`optimal = rhoP <= tol && rhoD <= tol && rhoA <= tol`, relative to `max(1, |.|)` so floored at absolute 1); optim/linear-programming.ts:205-244 (`equilibrated`: each row of `A_ub`, `A_eq` scaled to a largest entry of 1; nothing else). Probe p5: random 6-variable boxed programs with `c` multiplied by 1e-9: `linprog` objective wrong in 30 of 40 (by 1e-6: 0 of 40, by 1e-4: 0); probe p1: `c = [-1e-12, -2e-12]` on the textbook program returns `x = [0, 0]`, `optimal` (true `[3, 1]`); interior point with `b = 1e-12` returns `optimal`, `x = [6.5e-11, 6.5e-11]` (sum 1.3e-10, 130 times the bound) where the simplex returns the right `1e-12`.
what: the recent flake was this defect met on rows ("rows of 1e-6-sized coefficients beside rows of order 1 fall under the tolerances"); the fix scaled rows only. A cost vector or right-hand side at 1e-9 scale, or a program in other units, fails the same way, silently and with `status: 'optimal'`. scipy's HiGHS scales the problem and uses relative tolerances (from memory).
fix: extend `equilibrated` to the three missing scalings: scale `c` by `1 / max|c|` (and rescale `fun`, marginals, cost ranges on the way back, as it already does for rows), scale `b` by `1 / max(1, max|b|)` or by a power of two, scale columns of `A` the same way as rows (geometric or max scaling, with `x` and the bound rows rescaled back); make the simplex entering test relative (`d < -tol * max(1, cScale)` once `c` is scaled, equivalent to a fixed `tol` on the scaled problem) and the interior point's floor of 1 in `rp0`, `rd0`, `rg0` relative to the scaled data. Powers of two keep the scaling exact in floating point.
size: M (2-4h with the ranging and marginals un-scaling and tests)
risk: medium (changes pivots and so the vertex path shown in teaching examples; ranging and duals rescale must be exact; the existing near-tied tests are the check)
preserves behaviour: no (wrong answers become right; the path of vertices of badly scaled programs changes)
changes public surface: no

### F4: the default pivot rule is Bland's, which is 15 times slower and fails on plain programs; `linprog` hides it behind a five-solve retry chain

kind: hot-path
evidence: optim/lp-simplex.ts:98 (`rule = options.pivotRule ?? 'bland'`), :319-329 (the entering column: Bland takes the first negative reduced cost); optim/linear-programming.ts:332-355 (`linprog`: simplex, then simplex with `pivotFloor: 0`, then `linprogInteriorPoint`, then Dantzig, then Dantzig without the floor, then a three-way vote). Probe p6 (dense positive `A`, `b` in [10, 20], `c < 0`, so well conditioned and bounded): 20 x 40: Bland 67 pivots, Dantzig 8; 40 x 80: 223 pivots against 16; 60 x 120 (three random programs): Bland `numerical difficulties` after 474, 524 and 913 pivots, Dantzig `optimal` in 29, 26 and 20 pivots (12-18 ms); 80 x 160: 1022 pivots `numerical difficulties` against 30; `linprog` takes 84, 79, 149 and 283 ms for what a Dantzig run does in 13 to 34 ms. The `pivotFloor` on and off give identical outcomes on all of these.
what: Bland's rule is the anti-cycling guarantee, taken as the default. It makes the number of pivots, and so the accumulated rounding error in the dense tableau (re-inverted every 50 pivots, at most 8 times), grow until the "numerical difficulties" exit, after which `linprog` pays for up to four more solves. The chain is a patch in the sense of the question: each element was added for one failing program (a near-tied game) and the combination is tested by seed-pinned cases (F25); it is a consistent _policy_ only in its last step (return an infeasible or unbounded verdict only when two methods agree), which is the part worth keeping.
fix: make Dantzig (or Devex) the default entering rule and fall back to Bland's rule for the rest of the run after a run of degenerate pivots (ratio 0) longer than `rows`, the standard hybrid; keep `pivotRule: 'bland'` selectable for the teaching path. Then `linprog` becomes: simplex, interior point on `numerical difficulties`, cross-check infeasible/unbounded verdicts with the other method. Re-run the three seed-pinned near-tied tests as the acceptance check; remove the `pivotFloor: 0` and double-Dantzig rungs if they stay unneeded.
size: M (2-3h)
risk: medium (changes the vertex path of every `@example` that prints `path`, for instance `simplexAlgorithm`'s `[[0, 0], [4, 0], [3, 1]]`; golden-free but example-checked; degenerate programs need the Bland fallback to be right)
preserves behaviour: no (default path of vertices and iteration counts change; optimal values do not)
changes public surface: no (the default of an option)

### F5: non-finite input has no policy in the programming and dual solvers: they answer `optimal`, `converged`, or fail with an unrelated message

kind: consistency
evidence: probes p1, p8, p20 (all on the sources named): `linprogSimplex({ c: [NaN, 1], aUb: [[1, 1]], bUb: [1] })` -> `optimal`, `x = [0, 0]`; `linprog` with `bUb: [NaN]` -> `optimal`; `bUb: [Infinity]` -> `optimal`, `x = [Infinity, 0]`; `linprogInteriorPoint` with NaN `c` or `A = Infinity` -> `MdmlError value 'choleskyJitter: not positive definite even with relative jitter 0.01'`; `linprogSimplex(..., { tol: NaN })` -> `optimal`, `x = [0, 0]` (the textbook program's optimum is `[3, 1]`); `quadprog` with `P = [[1, 0], [0, NaN]]` -> 150 iterations, `iteration limit`, `x = [NaN, NaN]`; `oneClassDual` with a NaN or Infinity kernel -> `converged`, 0 iterations, alpha unchanged, objective NaN (optim/smo.ts:234 `pick.violation < eps || pick.j < 0`: a NaN violation makes `pick.j` stay -1 and the loop reads that as optimal); `nnls([[1, 0], [0, NaN]], [1, 1])` -> `converged`, `x = [0, 0]`; `admmLasso` with `lambda = -1` -> `converged`, objective -3; against `linearSumAssignment` ("the cost matrix contains NaN", assignment.ts:25), `auction`, `emd` ("marginals must be nonnegative"), `sinkhorn` ("reg must be positive"), which validate. `vectorFrom`/`matrixFrom` (optim/_programming.ts:16-31), the entry of LP, QP, SMO and conic, check shapes only.
what: the same finding as numerics F11 for this folder: a NaN comparison is false, so a loop guard written as `violation < eps` reads NaN as "not yet converged" or "nothing to do" depending on which way the test is written, and the five solvers that go through `_programming.ts` inherit no check. The failure the maintainer cares about (a flaky test on a nearly tied game) is the same shape.
fix: one `assertFinite(values, name)` in `_programming.ts`, called by `vectorFrom`/`matrixFrom` for data that must be finite (`c`, `A`, `b`, `P`, `q`, the kernel) with a parameter to allow infinite bounds; options validated by one helper (`tol` and `maxIterations` finite and positive). Throw `value` with the argument's name. In the loops themselves, treat a non-finite violation or step as `status` `'numerical difficulties'` (LP, QP) or `'diverged'` (Algorithms) rather than falling through.
size: S-M (2h)
risk: low (inputs that now throw returned meaningless results; the property tests generate finite data)
preserves behaviour: no (invalid input throws)
changes public surface: no

### F6: the absolute gradient test makes every generic smooth solver return the start on small-scale objectives, so the modes of one slot disagree

kind: consistency
evidence: optim/minimize/objective.ts:32-37 (`gradTol`: "the infinity norm of the gradient falls to this"), solvers.ts:78 (`gradTol = real(1e-12, 1e-2, default 1e-5)` shared by six modes), quasi-newton.ts / gradient.ts / newton.ts / lbfgsb.ts (`infNorm(grad) <= gtol`). Probe p14 (`f = scale * sum((x - 1)^2)`, start 0, `maxIterations: 300`): scale 1: all ten methods reach x = 1.00; scale 1e-8: bfgs, lbfgs, newton, gradientDescent, conjugateGradient, lbfgsb and trustRegion return x = 0.00 `converged` in 0 steps (the derivative-free nelderMead, powell, cmaes still reach 1.00); scale 1e-4: trustRegion stops at 0.71 `converged`. Probe p3 (`fitGlm`, six points, Gaussian and binomial, design scaled by `s`): at `s = 1e-8` and below, `lbfgs` and `newton` return `coef = 0`, `converged: true`, `steps: 0` while `irls` returns the exact slope `1.07e8`; at `s = 1e100` `lbfgs` and `newton` do not converge and `irls` does. glm.ts:5-9 (the module's promise): "the fitted coefficients are the same whichever mode runs".
what: the 1e-5 absolute gradient test is scipy's `gtol`, and scipy's users standardise; here the test sits behind a solver _slot_ whose point is that modes are interchangeable, and IRLS (relative coefficient change, glm.ts:459) uses a different, scale-free criterion. The same objective therefore ends at different points depending on the mode, with the same `converged` label. A model whose data are in other units (counts of 1e-9 per unit, a design in metres against millimetres) hits it. It is the IRLS-on-denormal-designs flake from the other side.
fix: add the scale-free test as a second criterion and a documented option: `gradTolRelative` (stop when `|g|_inf <= gradTolRelative * |g0|_inf`, PETSc's `rtol`), tested in the shared stop helper so all modes get it; set the slot's default to relative 1e-8 together with the absolute 1e-5 only when the objective is not tiny (or: default `gradTol` to 0 in the slot and rely on `ftol`/relative). At minimum state the absolute scale in the `gradTol` label and TSDoc (`'gradient tolerance (absolute)'`) and in `glm.ts`'s summary.
size: M (2h with the table of modes)
risk: medium (changes when runs stop; golden fixtures for the generic solvers at ordinary scale are unaffected if the absolute test stays)
preserves behaviour: no (tiny-scale problems now run)
changes public surface: yes (one added option)

### F7: two status vocabularies behind one field name, a result without `converged`, and a status that lies

kind: consistency
evidence: `status: RunStatus` (`'completed' | 'converged' | 'diverged' | 'stopped' | 'failed'`, trace/events.ts:12) with `converged: boolean`: MinimizeResult (minimize/objective.ts:293-311), AdmmResult (admm.ts:87-88), SparseResult (sparse.ts:64-65), dictionary, gromov, nsga2, splitting; `status: 'optimal' | 'infeasible' | 'unbounded' | 'iteration limit' | 'numerical difficulties'` with `success: boolean`: LinprogResult (linear-programming.ts:54-55), QuadprogResult (quadratic.ts:262-263), ConicResult (conic.ts:452-453, plus a state-only `'running'`), MilpResult (integer.ts:59-60, `'node limit'`); `converged` without `status`: SmoResult (smo.ts:80), DualCdResult (dual-cd.ts:306), transport (transport.ts:340), wasserstein, ScalarResult; `success` alone: line searches (line-search.ts:57), pattern search (pattern-search.ts:220). `LinprogResult` keeps only the mathematical status and drops the `RunStatus` that `runAlgorithm` computed (linear-programming.ts:255-261, `assembleResult(form, result.state.solution)`), so `converged` is absent from LP, QP, conic and MILP results. A status that is wrong in the vocabulary: orthogonal matching pursuit ends with `status: 'converged'` and the message "ended early: atom already selected (linear dependence)" or "selected atoms are linearly dependent" (sparse.ts:239-268), probe p8: a dictionary with a repeated column returns `converged` having selected 2 of the 3 requested atoms.
what: a caller cannot write `if (!result.converged)` across the folder, and a consumer of both families (a method that falls back from a QP to a vector method) needs two checks. `iteration limit` is `completed` in the other vocabulary, `infeasible`/`unbounded` are `failed`. Linear dependence is not convergence.
fix: state the convention in the folder summary and apply it: every result has `converged` (true iff the mathematical status is `optimal` or the run status is `converged`) and `status`; define `type SolveStatus = 'optimal' | 'infeasible' | 'unbounded' | 'iteration limit' | 'numerical difficulties' | 'node limit'` once in `_programming.ts`, with `runStatusOf(status): RunStatus` used by the six `done:` functions that currently map by hand (lp-simplex.ts:392, lp-interior-point.ts:282, conic.ts:434, qp-active-set.ts:262); the programming results add `converged` as a synonym of `success`; OMP's dependent-atoms exit becomes `stopped`. This is numerics F10's convention (`converged` plus `status`) extended to the other half of optim.
size: M (2h)
risk: low-medium (adds fields; the OMP status change alters `converged` of the degenerate case and the tests that pin its message)
preserves behaviour: yes except the OMP early-exit status
changes public surface: yes (added `converged`; a union member for OMP)

### F8: the SVM duals are set up twice, with the same names and different signatures

kind: duplicate
evidence: optim/smo.ts:391-527 (`svcDual`, `svrDual`, `oneClassDual`, `svddDual`: each builds `alpha0`, `Q`, `p`, `y`, `C` and calls `smo`) against svm/_dual.ts:81-330 (`cSvcDual`, `nuSvcDual`, `svrDual`, `nuSvrDual`, `oneClassDual`, `svddDual`: the same `alpha0` fill loop (`oneClassDual` at smo.ts:473-479 and _dual.ts:266-272, `svddDual` at smo.ts:515-520 and _dual.ts:304-310 are line for line), the same `Q = 2K` and `p = -diag K` for SVDD, then `smoAlgorithm` stepped instead of run to the end, plus `gap`/`margins`); `nuSvcDual` in optim/smo-nu.ts:343 and svm/_dual.ts:115. svm/classifier.ts:31 imports optim's `svcDual`/`nuSvcDual` for its fast path and `_dual.ts` for the stepped one, svm/one-class.ts and regressor.ts use `_dual.ts` only; outside tests nothing uses optim's `oneClassDual`/`svddDual` (grep). Two exported names (`svrDual`, `oneClassDual`, `svddDual`, `nuSvcDual`) mean different things in `optim` (run to the end, returns a result) and in `svm/_dual` (builds a steppable `Dual`).
what: the problem data of each dual (the initial feasible point, the quadratic, the linear term, the bound) is the model's statement, and it is written in two domains; a fix to one (the feasible start, the `C >= 1/n` rule) does not reach the other.
fix: in `optim`, export the problem builders (`svcProblem(K, y, C)`, `svrProblem(K, t, C, epsilon)`, `oneClassProblem(K, nu)`, `svddProblem(K, C)` returning the `SmoProblem`, with `alpha0`); `optim`'s run-to-end functions and `svm/_dual.ts` both call them, and `_dual.ts` keeps only what is its own (`gap`, `margins`, `expansion`). Rename svm's private ones so no two exports share a name.
size: M (2-3h; the nu forms have their own start in `smo-nu.ts`)
risk: low-medium (`svm` has golden and property tests; the builders return what the code already builds)
preserves behaviour: yes
changes public surface: yes (added builders)

### F9: OMP rebuilds the Gram matrix and its factorisation at every step (7 s for 100 atoms), and its linear-dependence test is absolute

kind: hot-path
evidence: optim/sparse.ts:224-285 (`orthogonalMatchingPursuitAlgorithm`), :245-258 (each step builds `columns(d, support)`, a `support x support` Gram with `ddot(column(d, i), column(d, j))` allocating two columns per entry, and calls `cholesky` + `choSolve` from scratch) against its own TSDoc ("an updated Cholesky factor of their Gram matrix", :201); :239 (`(alpha[lam] ?? 0) ** 2 < eps` with `eps = 2.22e-16`, absolute); :171, :280, :368 (`history: [...s.history, rn]` copies the history each step). Probe p13 (random Gaussian dictionary, `nNonzeroCoefs` = nnz): m = 100, k = 300, 20 atoms: 28 ms; m = 200, k = 600, 50 atoms: 600 ms; m = 300, k = 900, 100 atoms: **7,319 ms** (cubic in the support); probe p8: a dictionary `[[1e-9, 0], [0, 1e-9]]` with `y = [1e-9, 2e-9]` returns zeros, `status: 'converged'`, 1 step (the coefficient is 1 and 2); `matchingPursuit` with 16,000 iterations spends 216 ms against 47 ms at 8,000 (the history copy).
what: sklearn's OMP does this in a rank-one update per step (O(support^2)), and the doc claims the same. The scale test `alpha^2 < eps` treats a correlation of 1e-8 as zero, so a dictionary or a signal in small units selects nothing and reports success.
fix: keep the Cholesky factor in the state and extend it by one row per step (`L_new = [[L, 0], [w', sqrt(1 - w'w)]]` for unit-norm atoms, sklearn's `_cholesky_omp`), compute the new Gram row as one `dgemv` against the support columns; replace the absolute test by the sklearn test on the _normalised_ correlation (`alpha^2 < eps * |y|^2`, or the pivot of the new row against `eps * diag`); push to `history` (mutating a private array carried in the state) or store the residual norms in a preallocated typed array.
size: M (2h)
risk: low (`sparse.test.ts` and `information/dictionary-coding` tests pin the supports and coefficients; the update is exact in exact arithmetic)
preserves behaviour: yes (values to 1e-12) except the scale bug fixed
changes public surface: no

### F10: IRLS and the generic GLM modes hide failure: non-finite pieces are zeroed, a NaN objective is accepted after the halvings, and outside [1e-154, 1e154] the intercept-only fit is `converged`

kind: consistency
evidence: optim/glm.ts:226-227 (`gradientAt`: `u[i] = Number.isFinite(v) ? v : 0`, a NaN or infinite per-observation gradient term is dropped), glm.ts:435 (`for (i) if (!Number.isFinite(ww[i])) ww[i] = 0`, the same for the IRLS weights), :443-448 (`while (!first && !(value <= s.value + 1e-12 * |s.value|) && halvings < maxHalvings)`: a NaN `value` fails `<=` so it halves up to 30 times and then **accepts** the last NaN iterate), :367-376 (`solveNormal`: on a non-positive-definite system a Levenberg shift `1e-12 * max(diag, 1)` growing by 10 up to 30 times, then `singular`; the floor of 1 in `max(diag, 1)` makes the shift absolute). Probe p3 (`fitGlm`, Gaussian and binomial, `irls`): design scaled by 1e-170, 1e-200, 1e-310: coefficient 3e-158 and the intercept-only fit (the squares underflow to 0, the system is singular, the shift dominates), `converged: true`, steps 2; design scaled by 1e160 (squares overflow): coefficient 0, intercept 3.58 = the mean of `y`, `converged: true`. The real slope there is 1e170 or 1e-160: not representable honestly, but it is reported without a flag.
what: three separate places convert "this number is not finite" into a plausible number (0) and carry on, and the IRLS exit does not distinguish "step is zero because the solution was reached" from "step is zero because the normal equations were damped to nothing". This is the denormal flake in its open form. Also `GlmFit.converged` is only the run status, so there is no field for "shift used".
fix: count non-finite terms in `gradientAt`/`weights` and, if any appear, return `value: NaN` so the run ends `diverged` (the minimise family's convention) instead of zeroing; stop the halving loop on a non-finite `value` and report it; record the largest shift used in `GlmState` and end the run `failed` ('normal equations singular after damping') when the shift exceeded, say, 1e-6 of `diag`; scale the shift by `diag` without the floor of 1.
size: S-M (1-2h)
risk: medium (property tests in glm.property.test.ts exercise ordinary data; the zeroing was added to survive some case, find which by removing it)
preserves behaviour: no (degenerate fits now report failure)
changes public surface: no

### F11: the outcome of a line search is read by a regex on its message, in two places

kind: rule
evidence: optim/minimize/quasi-newton.ts:402-407 (`usable = !raw.success && Number.isFinite(raw.alpha) && raw.value <= st.value && /xtol|alphaMax|alphaMin|rounding/.test(raw.message)`) and optim/minimize/lbfgsb.ts:495 (`ls.success || (Number.isFinite(ls.alpha) && /xtol|alphaMax|alphaMin|rounding/.test(ls.message))`); `LineSearchResult.message: string` (line-search.ts:63) has no typed reason.
what: control flow on the text of a human message: rewording a message in line-search-interval.ts silently turns "usable warning" into "failure" in two solvers (and L-BFGS-B's, BFGS's and L-BFGS's behaviour differs only by what these two regexes see). The same list is repeated verbatim.
fix: add `readonly outcome: 'wolfe' | 'armijo' | 'warning' | 'failed'` (or `reason`) to `LineSearchResult`, set by the searches that already know which branch they took (`moreThuente`'s `dcsrch` task codes are exactly this), and test it in both places; the message stays for humans.
size: S (1h)
risk: low (a search returns the same message plus a field; the two tests' behaviour is unchanged)
preserves behaviour: yes
changes public surface: yes (an added field)

### F12: knob specs and defaults are written three times; 53 casts in `optim` and `losses` have one cause

kind: duplicate
evidence: the same knob in several places: `gradTol = real(1e-12, 1e-2, { log: true, default: 1e-5, label: 'gradient tolerance' })` in optim/solvers.ts:78 and optim/bounded-solvers.ts:103; the L-BFGS `ftol` spec and the literal `2.220446049250313e-9` in solvers.ts:147, bounded-solvers.ts:109, minimize/quasi-newton.ts:366, minimize/lbfgsb.ts:379; `memory` is `int(1, 50)` in the L-BFGS mode (solvers.ts:146) and `real(1, 50)` rounded by `Math.round` in the L-BFGS-B mode (bounded-solvers.ts:107, :118); a default is declared in the space and again in the `make`: `(k['tol'] as number | undefined) ?? 1e-10` beside `real(..., default: 1e-10)` (glm.ts:476-477), `(k['learningRate'] as number | undefined) ?? 1e-2` and `?? 1e-5` for `gradTol` in the `gradient` mode (solvers.ts:203, :207), and a dead branch `k['ftol'] === undefined ? {} : ...` (solvers.ts:153) because `ftol` always has its default; casts: `k['x'] as T` 20 in solvers.ts, 13 in schedule-table.ts (19 per 100 lines, the density Phase 0 flagged), 11 in losses/tables.ts, 5 in bounded-solvers.ts, 42 in `optim` and 11 in `losses` (53), 90 in the engine; the cause: `Option.make(knobs: ModeConfig)` and `ModeSpec.make(p, k)` (space/tables.ts:33) take a loose record although the `space: dict({...})` beside it declares each key's type.
what: three copies of one default drift (and the two `memory` specs already differ); the casts are not defects but each is a place where a renamed knob compiles and fails at run time. This is the "duplicated option tables" question: the tables themselves are not copied (each domain has its own, as numerics F15 found), but the knob specs inside them are.
fix: (a) type `make`'s knobs from the dict: `make(knobs: SpaceOf<S>)` with `S` inferred from `space` (the engine already infers `DictSpace` values in `defaults`/`SpaceValue`), which removes the casts and catches a renamed key; (b) hoist shared knob specs (`gradTol`, `lbfgsFtol`, `memory`, `learningRate`) into `solvers.ts` (or a `_knobs.ts`) and import them in `bounded-solvers.ts`; (c) drop the `??` fallbacks that duplicate the space default (the table validates the configuration before `make`, so the value is always there) and name `LBFGS_FTOL = 2.220446049250313e-9` once in `quasi-newton.ts`.
size: M (3h for (a), which is a `space` change; (b) and (c) are S)
risk: low-medium ((a) is type-level only; (b), (c) change no value; `solvers.test.ts` and `optimiser-table.test.ts` pin the tables)
preserves behaviour: yes
changes public surface: no ((a) changes the declared type of `make`'s parameter; every call site that casts keeps compiling)

### F13: local copies of the tensor-to-array coercions, five of them double-copying

kind: duplicate
evidence: `Float64Array.from(toFlat(primalOf(v)))` (a boxed `number[]`, then a typed copy) written as `flatOf` (minimize/objective.ts:95, called on every objective and gradient evaluation), `flat` (glm.ts:121; minimize/convex.ts:39; splitting.ts:290), `flatPrimal` (proximal.ts:48), `flatOf` (analytic.ts:71), `toVec` (landscape.ts:403); `vectorFrom`/`matrixFrom` exist in optim/_programming.ts:16-31 and serve LP, QP, SMO and conic only; the matrix coercion `{ data, rows, cols }` with a "must be a matrix" check is re-written as `matrix` (sparse.ts:23, dictionary.ts:24), `matrixOf` (assignment.ts:22, minimize/convex.ts:436), and as the type itself four times (`Matrix` in sparse.ts:15 and dictionary.ts, `Dense` in _programming.ts:9, an inline literal in convex.ts); `rowsOf` in linear-programming.ts:193 and multiobjective.ts:31 (also in 25 files across the engine, per Phase 0). duplicates-strict.txt lists `matrixOf` x4, `columnOf` x4, `rowsOf` x12, `flat` x2.
what: the rule is "no local copies of generic helpers" and "a missing helper goes into a shared module first"; the missing helper is a tensor-to-`Float64Array` that reads `data` once. Today every objective evaluation of every vector method allocates a `number[]` of n boxed numbers and a typed array.
fix: one `floatsOf(v: Value): Float64Array` (a single `Float64Array.from(concrete(v).data)`, or the data itself where the caller does not mutate) beside `toFlat` in `tensor`, one `Dense` and `denseOf(a, name)` in `optim/_programming.ts` that `sparse`, `dictionary`, `convex`, `assignment` use (assignment adds its NaN check through F5's `assertFinite`), and `_programming.ts`'s `rowsOf` for the two `rowsOf`. The duplicate-helper agent (P3.3) may already be near these; this is the optim list.
size: S (1-2h)
risk: low
preserves behaviour: yes
changes public surface: no (one added tensor export)

### F14: two proximal-operator types, and a family of native-array callbacks next to the tensor ones

kind: duplicate
evidence: `Prox = (x: Value, t: number) => Value` (proximal.ts:46), used by `ista`, `fista`, `admmProx`, `chambollePock`, `proxConjugate`, and all the prox functions of proximal.ts; `ProximalStep = (x: Float64Array, step: number) => Float64Array` (variance-reduced.ts:24), used by SAGA and SVRG (`prox` option, :33, :66) and wrapped by hand inside splitting.ts:87-89 (`proxFlat`). Other callbacks on native arrays: `AdmmProblem.xUpdate`/`zUpdate`/`objective` (admm.ts:35-44, `Vec`), `Lmo = (grad: Vec) => Atom` (frank-wolfe.ts:30), `Mirror` (convex.ts:207), `OnlineLoss` and `OnlineLearner` state (online.ts), `finiteDifferenceHessian(gradient: (x: Float64Array) => ArrayLike<number>)` (analytic.ts:47), `preconditionBy(fn: (grad: Float64Array) => ArrayLike<number>)` (preconditioned.ts:440), `OgdOptions.project` (online.ts:56), `ProjectedGradientOptions.project` (constrained.ts:104).
what: `proxL1` cannot be handed to SAGA without a wrapper, and the same concept exists in two shapes; the native-array callbacks are the places where a user's function cannot be differentiated through (an ADMM `xUpdate` cannot be a `grad`-able closure) and where `Value`-typed library functions (`proxL1`, a loss) cannot be passed without conversion. The `Value`-typed forms are the ones that compose with `grad`.
fix: SAGA/SVRG take `Prox` (the folder's one type) and convert to a flat view internally, as splitting.ts does; offer `admmProx` as the documented path and keep `admm`'s native callbacks as the kernel form; accept `TensorLike` in `finiteDifferenceHessian`'s gradient (or mark it a kernel). The rest stay as kernels (F15).
size: M (2h)
risk: medium (the `prox` option of two algorithms changes type; their tests and the golden for variance-reduced pin the values)
preserves behaviour: yes
changes public surface: yes (type of `VarianceReducedOptions.prox`)

### F15: native arrays in public signatures, classified (65 optim + 8 losses functions)

kind: native-array
evidence: `node scripts/review/signatures.mjs` read through a copy that prints every row (probe `sigs.mjs`, 73 rows) against the signatures; classification below. How callbacks are typed (decides differentiability): minimize/* take `Objective = (params: Tree) => Value`, `ResidualFunction`, `BatchObjective = (params: Tree, batch?) => Value` (stochastic.ts:28), `FiniteSum.term(params: Tree, i) => Value` (variance-reduced.ts:19), `Prox`, `LossFunction` (landscape.ts); `flatObjective` (objective.ts:104) is the one bridge to flat vectors, calling `valueAndGrad` on the tensor objective, so a user's objective is differentiable and the solvers themselves are not (their outputs are concrete; `implicitArgmin`, `implicitRoot`, `implicitFixedPoint` in implicit.ts differentiate through a solution).
- **kernel, allowed (20)**: minimize/quasi-newton.ts `bfgsInverseUpdate`, `twoLoopRecursion`; minimize/lbfgsb.ts `projectedGradientNorm`; minimize/newton.ts `modifiedCholesky`; minimize/owlqn.ts `pseudoGradient`; minimize/trust-subproblems.ts `quadraticModelValue`; minimize/objective.ts `clipToBox`, `infNorm` (and `Box`); smo.ts `smoRho`; smo-nu.ts `nuRho`, `nuSvcStart`, `nuSvrStart`; lp-standard-form.ts `fromStandard`; quadratic.ts `kktResiduals` (on `QpArrays`); line-search.ts `lineFunction` (the line function); minimize/frank-wolfe.ts `boxLmo`, `l2BallLmo` and minimize/convex.ts `euclideanMirror` (build `Lmo`/`Mirror` callbacks); solvers.ts `loggedSolverOf`, `loggedSolver` (the `extra(x: Float64Array, ...)` hook reads the iterate); the SMO, simplex and interior-point states. State this in `optim/index.ts`'s summary ("the kernels of the solvers work on `Float64Array`").
- **shape, axis, index or combinatorial (19)**: schedules.ts `multiStepDecay`, `piecewiseConstant`, `joinSchedules` (step boundaries); assignment.ts `galeShapleyAlgorithm`, `stableMatching`, `blockingPairs` (preference lists, ragged indices); integer.ts `setCover` (index sets), `knapsack`/`knapsackAlgorithm` (`values`, `weights` are real vectors but the problem is combinatorial; borderline); multiobjective.ts `nonDominatedSort`, `paretoFront` (return indices), `dominates` (two objective vectors, a point not a tensor); losses/large-output.ts `huffmanCoding` (counts), `sampleClasses` (class indices); losses/regularisers.ts `groupLasso` (`groups`), optim/proximal.ts `proxGroupLasso` (`groups`); losses/sequence.ts `ctcTrellis`, `ctcForcedAlignment`, `ctcGreedyDecode` (label sequences and the path).
- **public vector or matrix, migrate (34)**: landscape.ts `slice1d`, `slice2d`, `randomSlice`, `interpolationSlice`, `bezierSlice` (`alphas`, `betas`, `ts` grids; `readonly number[]`) and `sharpness` (`v0`); multiobjective.ts `crowdingDistance` (returns `Float64Array`), `hypervolume`, `hypervolumeContributions` (`reference`), `epsilonConstraint` (`epsilons`); online.ts `onlineGradientDescent`, `ftrl`, `hedge`, `exponentiatedGradient` and online-experts.ts `pNormMirrorDescent`, `onlineAdagrad`, `hindsightComparator` (learners whose `play`/`update` take and return `Float64Array`, see below), `bestSwitchingLoss`, `exp4Algorithm`, `adagradRegretBound` (loss/gradient matrices as `number[][]`); nsga2.ts `sbxCrossover`, `polynomialMutation` (individuals `ArrayLike<number>` in, `Float64Array` out); glm.ts `glmValue`, `glmGradient` (`beta`); bounded-solvers.ts `boxMap` (`lower`, `upper`); quadratic.ts `boxQp`, `boxQpAlgorithm` (`x0` option); line-search.ts `lineSearch` (`direction: ArrayLike<number> | Parameters`); analytic.ts `finiteDifferenceHessian`; preconditioned.ts `preconditionBy` (a callback on `Float64Array`); minimize/penalty.ts `augmentedLagrangian` and minimize/slsqp.ts `slsqp` (return `multipliers: Float64Array` next to `x: Tree`); losses/physics.ts `gradientNormWeights`, `ntkWeights` (term weights as `number[]`).
- **not visible to the script (about 40 fields)**: the input records `GlmProblem` (`x`, `y`, `weights`, `offset`, `start`: 5, glm.ts:65-81) and `GlmPenalty.matrix`, `AdmmProblem.c` and `AdmmOptions.x0/z0/u0` (admm.ts:40-49), `SmoOptions`' problem fields are `TensorLike` (good), `OnlineLearner`'s state types, `QuadraticProgram` arrays; `LinearProgram`, `QuadraticProgram`, `SmoProblem` are `TensorLike` (the rest of the programming code is already migrated).
  what: 34 real candidates, small and leaf-like (grids of scalars, loss matrices, individuals); the large structural ones are the two problem records (`GlmProblem` with row-major `x` and explicit `n`, `d`) and the online learners. Every other solver already takes `TensorLike` and returns `Tensor`.
  fix: staged by cost: (1) `GlmProblem` takes `x: TensorLike` shaped `[n, d]` (drops `n`, `d`) and `y`, `weights`, `offset`, `start` as `TensorLike`; `glmValue`/`glmGradient` take and return tensors; (2) `multiobjective.ts` and `landscape.ts` accept `TensorLike` for grids and references and return `Tensor`; (3) the multiplier fields become `Tensor`; (4) the online learners keep `Float64Array` state (they are `init/update/play` kernels) but their public constructors accept `TensorLike` options and `play` returns `Tensor` through a thin wrapper, or are declared kernels. Widening parameters is source compatible; return types are the break.
  size: L (1 day, spread over four steps; (1) is the one with callers: `linear/*`, `additive`, `stats` use `GlmProblem`)
  risk: medium
  preserves behaviour: yes (values), no (return types)
  changes public surface: yes (the deliberate break, D-009)

### F16: empty, one-element and bound-conflict conventions differ between solvers

kind: consistency
evidence: probe p2 (empty `x0 = []`): bfgs, lbfgs, newton, lbfgsb return `converged`, 0 steps; nelderMead, powell, cmaes throw `value 'no parameters'` (derivative-free.ts, powell.ts, cmaes.ts); a diverged run keeps the last finite iterate everywhere except powell, which returns `x = [NaN, NaN]`, `fun NaN`, `diverged` (`f = sum(sqrt(x))` at 0 and `-sum(log x)`); on an objective with an infinite gradient at the start (`sum(sqrt x)`, `x0 = 0`) the status is `failed` for bfgs, lbfgs, newton, newtonCg, gradientDescent, conjugateGradient, lbfgsb, `converged` for nelderMead and hookeJeeves (f = 0 is the minimum), `diverged` for powell and `completed` after 200 steps at the same point for trustRegion, cobyla and cmaes. Bound conflicts: `boxOf` (objective.ts:263-264), `boxMap` (bounded-solvers.ts:63) and `lsqLinear` throw `value`; `boxQp([[1]], [0], [1], [0])` (lower 1, upper 0) returns `x = [0]`, `converged` (quadratic.ts:491-505 never compares `lb` and `ub`); `linprog` with `bounds: [[2, 1]]` returns `infeasible` (the right mathematical answer, but a third convention); `fminbound`/`minimizeScalar` take a bracket.
what: the same degenerate input is a result, an exception or a wrong result depending on the solver; the trust-region `completed` after burning 200 steps is the least informative status (`failed` was available). Overlaps numerics F10/F11 for the iteration, and is the optim half of that policy.
fix: the convention to write down: empty parameter vector -> `converged` with zero steps for every vector method (the three that throw change); a run that meets a non-finite objective or gradient -> `diverged` or `failed` and returns the _last finite_ iterate (powell, trust region); conflicting bounds -> `value` thrown in the box helpers and the QP/LP front ends (LP may keep `infeasible`, documented). The probes above are the table for F26.
size: S-M (2h)
risk: low (degenerate inputs only)
preserves behaviour: no
changes public surface: no

### F17: tolerance and iteration vocabulary, defaults and units differ across the folder; `gradNorm` means three things

kind: consistency
evidence: stop test names: `gradTol` (vector methods, stochastic, lbfgsb), `ftol`/`xtol` (gradient.ts, lbfgsb, powell, trf, least-squares), `tol` (lp-simplex.ts:97 1e-9, lp-interior-point.ts:103 1e-8, smo.ts:114 1e-3, quadratic.ts:367 1e-10, conic.ts:284 1e-9, dual-cd.ts:153 1e-4, variance-reduced.ts:97 1e-6, sparse.ts:344 1e-10, transport.ts:404 1e-9, wasserstein.ts:225 1e-9 and :389 1e-7, total-variation.ts:238 1e-10, dictionary.ts 1e-8/1e-4, landscape.ts 1e-10), `absTol`/`relTol` (admm.ts:146, 1e-9), `epsilon` (assignment, online, line-search-interval); cap names: `maxIterations` (22 files), `maxEpochs` (variance-reduced.ts:140), `maxSweeps` (dual-cd.ts:333), `maxSteps` (minimizer.ts, `fitGlm` glm.ts:575), `maxEvaluations` (trf, cobyla), `iterations` (landscape.ts), and `maxIter` in losses/divergence.ts:316 (`sinkhorn`) against `maxIterations` in optim/transport.ts; units of "an iteration": a pivot, an outer round (penalty: 30), a sweep over n coordinates (coordinate descent), a generation (differential evolution, 1000), a pair update (SMO), an epoch; defaults of the cap: 30 (penalty) to 100,000 (cobyla, pattern search); `gradNorm` as a metric: the infinity norm in `vectorSchema` (minimize/_shared.ts:38 "infinity norm of the gradient"), the global 2-norm in `minimizer` (minimizer.ts, `globalNorm`), the 2-norm in `solver/gradNorm` and `slotAlgorithm` (F1) while those solvers _stop_ on the infinity norm.
what: mostly deliberate scipy mirroring (`gtol`, `ftol`, `xtol`, `maxiter` -> `maxIterations`), as numerics F21 judged; the cost is that two solvers cannot be swapped by renaming an option, and that one trace channel name carries two quantities.
fix: no renames. State the convention in `optim/index.ts` ("tolerances follow the method mirrored; `maxIterations` counts the method's own iteration, listed in its TSDoc; the vector methods stop on the infinity norm of the gradient"); accept `maxIterations` as an alias of `maxEpochs`/`maxSweeps` and `maxIter` in `sinkhorn`; name the 2-norm channel `gradNorm2` or the description "Euclidean norm" in the contract; keep the metric of `vectorSchema` as is.
size: S (1h)
risk: low
preserves behaviour: yes
changes public surface: yes (aliases, a channel description)

### F18: hyperparameters are not validated in the optimisers, QP and ADMM; non-convex QPs are called `optimal`; a NaN gradient poisons optimiser state

kind: consistency
evidence: probe p21: `adam({ eps: 0 })` on a zero gradient -> `[NaN, NaN]`; `adam({ b1: 1 })`, `adam({ b2: 2 })` -> NaN; `rmsprop({ eps: 0 })` zero gradient -> NaN; `sgd({ learningRate: -1 })` accepted; one NaN gradient entry leaves that coordinate NaN for every later step (`adam`: `[NaN, -0.1]` three steps running; the other coordinates keep working); `clipByGlobalNorm(1)` of `[Infinity, 1]` -> `[NaN, 0]` (the cap is 0 times Infinity); there is no `applyIfFinite`/`skipNonFinite` wrapper (optax has `apply_if_finite`, from memory). probe p8: `quadprog({ P: -I, bounds [-1, 1] })` -> `optimal` at `x = 0, fun = 0` (a stationary _maximum_; the true minimum is -1 at a corner), quadratic.ts:16-17 and qp-active-set.ts:46 document "convex" and nothing checks `P`; `admmLasso(..., lambda = -1)` -> `converged`. Counter-examples that validate: `glm.ts:145` (`l2 >= 0`), `gradientAlgorithm` (`batch`), `sinkhorn` (`reg`), `coordinateDescent` (`learningRate`).
what: the validation convention is per author. The non-convex QP is the one with a wrong label rather than a missing error.
fix: a `positive(name, v)`/`inRange` helper next to `assertFinite` (F5) used by the optimiser factories (`eps >= 0` strictly positive where divided by, `b1`, `b2` in [0, 1), `learningRate` finite), `admmLasso`/`ista` (`lambda >= 0`); `quadprog`/`qpActiveSet`/`qpInteriorPoint` check `P` positive semidefinite by one `cholesky` with relative jitter, or the Gershgorin/`eigvalsh` of a small `P`, and report `status: 'numerical difficulties'` with the message "P is not positive semidefinite" (or run and say so); an optax-style `applyIfFinite(tx)` wrapper in `transform.ts`; `clipByGlobalNorm` treats a non-finite norm as "scale to zero".
size: M (2-3h)
risk: low-medium (some user code passes `eps: 0` deliberately; an indefinite QP that now reports will change a test that uses one)
preserves behaviour: no
changes public surface: no (one added wrapper)

### F19: loss hyperparameters are checked unevenly, and out-of-range values give finite wrong numbers

kind: consistency
evidence: probe p15 (value, then gradient): `huber(x, 0, { delta: -1 })` -> value -2.5 (a negative loss); `pinball(x, 0, { quantile: 2 })` -> 1; `crossEntropy(..., { labelSmoothing: 2 })` -> 0.41; `crossEntropy` with a negative class weight -> a negative-weight loss; `focal(..., { gamma: -1 })` -> 0.79; `gaussianNll` with a negative variance -> -6.9 (torch raises, from memory); `throw new MdmlError` counts per file: classification 4, density 4, divergence 2, generative 3, invariance 5, large-output 3, physics 2, preference 1, ranking 1, regression 1 (only `tweedieDeviance`, regression.ts:279), regularisers 1, segmentation 1, sequence 3, **contrastive 0, margin 0** (38 throws in 141 functions). The same class as foundations F9 for the samplers.
what: a loss that returns a number for a nonsense hyperparameter turns a typo (`delta: -1`) into a training run that minimises towards minus infinity. Validation costs one line per loss and is absent exactly in the files with the most knobs (`contrastive.ts` has temperature, margin, normalise, alpha, scale; none checked).
fix: `positive(name, v)`/`unit(name, v)` helpers in `losses/_reduce.ts` (or the shared one from F18), applied to the 15 or so knobs that must be positive or in [0, 1]: `delta`, `quantile`, `labelSmoothing`, `gamma`, `temperature`, `margin`, `alpha`, `power`, the variance of `gaussianNll`, class weights (non-negative); one table-driven test (F26).
size: S-M (2h)
risk: low (only invalid values throw)
preserves behaviour: no (invalid values throw)
changes public surface: no

### F20: NaN gradients at degenerate points of the losses (zero embeddings, probabilities of exactly 0 or 1, a Tweedie mean of 0)

kind: consistency
evidence: probe p10/p15 (value and `grad`): `ntXent` of all-zero embeddings -> value 0, gradient `[NaN, NaN, NaN, NaN]` (losses/_vectors.ts:4-6, `l2Normalise = div(x, maximum(sqrt(sum(square(x))), eps))`: the gradient of `sqrt` at 0 is Infinity, times the 0 that `maximum` passes: NaN); `l2Normalise` is used by `infoNce` (contrastive.ts:86-96), `ntXent` (:128), `supervisedContrastive` (:280), `clipLoss` (:320-321), `arcFaceLogits`/`cosFaceLogits` (:340); `cosineEmbedding` of zero vectors is fine (its gradient is 0), so the two normalisations disagree; `binaryCrossEntropy([1, 0, 0.5], [0, 1, 1])` -> value 66.9 (the torch `-100` clamp, correct) with gradient `[NaN, NaN, -0.667]` (classification.ts:121, `maximum(log(v), -100)`: the gradient of `log` at 0 is Infinity times the clamp's 0); `tweedieDeviance(power 1.5)` at `mu = 0` -> value Infinity, gradient NaN. torch's `F.normalize` and `binary_cross_entropy` have finite backward at these points (from memory; the BCE backward clamps its denominator).
what: this is foundations F7 (norm, std, pow(x, 0)) arriving in the loss library: an all-zero embedding row (padding, a dead feature, a masked token) turns the whole batch's gradient into NaN, and saturated sigmoid outputs do the same through `binaryCrossEntropy`. The property tests use random non-degenerate inputs.
fix: fix `norm`'s vjp at zero in `tensor` (foundations F7) and the loss follows; independently, make `l2Normalise` use `x / sqrt(max(sum(x^2), eps^2))` (the gradient at 0 is then 0/eps, finite), and BCE `clamp` the _input_ of `log` (`log(max(p, eps))`, with `eps = exp(-100)`) so the gradient is `0` where clamped, as the value is.
size: S (1h each)
risk: low (a value at the degenerate points is unchanged; gradient at ordinary points unchanged to rounding)
preserves behaviour: yes for values, no for the NaN gradients
changes public surface: no

### F21: `slicedWasserstein` and `sinkhorn` exist in `losses` and in `optim` with different meanings

kind: duplicate
evidence: losses/divergence.ts:324 `sinkhorn(x, y, { epsilon, a, b, maxIter, tol, debiased })` (differentiable, unrolled, log domain, returns a `Value`; its TSDoc says it differs from `optim`'s) against optim/transport.ts:482 `sinkhorn(a, b, cost, reg, options)` (plain numbers, returns the plan; `maxIterations`); losses/divergence.ts:361 `slicedWasserstein(x, y, key, { projections: number, p, root })` returns the mean **power** `SW_p^p` by default (`root: false`), takes the key positionally, `projections` is a _count_; optim/wasserstein.ts:155 `slicedWasserstein(x, y, { a, b, projections: TensorLike, nProjections, key, p })` returns the **p-th root** always (`pow(mean(powers), 1 / p)`, :195), `projections` is the direction _matrix_ `[d, L]`, the count is `nProjections`, the key an option, weights supported, clouds of different sizes supported; both are exported under the same name from `mdml-engine/losses` and `mdml-engine/optim`, and both are `Value`-typed and differentiable. The loss-side `sinkhorn` TSDoc documents the difference; `slicedWasserstein` is not documented as a pair. Also `losses/_vectors.ts` `l2Normalise` and `pairwiseDistance` are `normalize` and `pairwise_distance` of nn/functional/normalisation.ts:345 (nn is above losses, so losses cannot import it).
what: a user who swaps one import for the other gets a different number (power versus root), a different `projections`, and a different key position, without a type error for the common case of omitted options. Two sort-based 1-D transport implementations (`wasserstein1d` in optim, the sort in the loss) are maintained for one idea.
fix: make the loss-side `slicedWasserstein` a thin call into the optim one (losses may import optim: `tables.ts` already does) with its own defaults (`root: false` -> `pow` the result), or rename it `slicedWassersteinLoss`; at minimum add the difference to both TSDocs and a `@see`. Move `normalize`/`pairwiseDistance` to `tensor` or `numerics` so `nn`, `losses` and `contrastive` share one.
size: S-M (1-2h)
risk: medium (a rename changes the public surface; a thin wrapper must reproduce the loss's values, which the golden fixtures check)
preserves behaviour: yes with the wrapper
changes public surface: yes (rename) or no (wrapper)

### F22: `losses/sequence.ts` uses a TypeScript parameter property and cannot be loaded by Node's native type stripping

kind: rule
evidence: losses/sequence.ts:171-175 (`class Lattice { constructor(readonly rows: number, readonly cols: number, fill: number) ... }`); running `import('.../losses/index.ts')` under `node` (v24, strip-only) fails with `ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX: TypeScript parameter property is not supported in strip-only mode`; it is the only parameter property and the only `enum`/`namespace`-like construct in the engine source (grep). `import('.../losses/classification.ts')` etc. load, so the probes imported the loss files one by one.
what: every script, probe and tool that runs engine source with `node file.ts` (the review scripts do, the examples runner may) fails as soon as it imports `mdml-engine/losses` as a whole; `import 'mdml-engine/losses'` also pulls `sequence.ts` into anything that touches the barrel. I could not run `ctc`, `ctcTrellis`, `ctcForcedAlignment` for that reason.
fix: write the two fields out (`readonly rows: number; readonly cols: number; constructor(rows, cols, fill) { this.rows = rows; ... }`); add `erasableSyntaxOnly: true` to the engine `tsconfig` so the next one is a compile error (TS 5.8+ option).
size: S (minutes)
risk: low
preserves behaviour: yes
changes public surface: no

### F23: `crossEntropy` and the classification losses cost what `logSoftmax` costs: 247 ms for 256 x 8000

kind: hot-path
evidence: probe p16/p17 (nice, one process): `logSoftmax` over `[256, 8000]` 213 ms, `logsumexp` 163 ms, `crossEntropy` 246 ms forward (87% of it is `logSoftmax`; losses/classification.ts:238 `logSoftmax(z, -1)`), 447 ms forward + gradient; `crossEntropy` with probability targets 535 ms; `infoNce` `[512, 128]` 88 ms forward, 223 ms with gradient; `huber` 343 ms forward + gradient at 1e6 elements against `mse` 102 ms (7 elementwise primitives against 3); `mse` forward 47 ms at 1e6 elements. The loss code is a clean composition; the cost is in the primitives.
what: foundations F13 (softmax, log-softmax, logsumexp, softplus as chains of 8-10 primitives) measured from the losses side: every classification and ranking loss that reaches a softmax pays it, and a 32,000-token vocabulary at batch 256 would take about a second per loss evaluation. Not a defect of this folder.
fix: fuse `logSoftmax`/`logsumexp` as primitives with analytic vjp/jvp rules (foundations F13); nothing to change in `losses` except to check, afterwards, `huber` and `smoothL1` (a `where` on `abs` is two primitives, not seven).
size: M (in `tensor`; see foundations F13)
risk: medium (new primitives must pass `checkPrimitive`)
preserves behaviour: yes
changes public surface: no

### F24: layering: the dependencies are `dists < optim < losses`, type-only upward imports, and one runtime import of a constructed domain

kind: layering
evidence: optim/glm.ts:14 `import type { DispersionFamily, Link } from '../dists/families/index.ts'` (type only; `family` and `link` are used through their interface at run time, `families.binomial()` is the caller's); losses/density.ts:11 and losses/generative.ts:12 `import type { Distribution } from '../dists/index.ts'`; nothing under `dists/` imports `optim` or `losses` (grep); losses/tables.ts:11 imports `../optim/proximal.ts` at run time (the prox functions of the penalties); losses/large-output.ts:8 imports `huffmanTree` from `information` (a constructed domain) and calls it in `huffmanCoding`. Phase 0 listed `optim/glm.ts -> dists` and `losses -> dists (2)` as "against the order".
what: the first-guess order `optim < ... < losses < dists` is wrong for these three: reading the real edges gives `dists < optim < losses`, with `glm.ts` needing only the _interface_ of a family and a link, and `losses/density.ts` only `logProb` of a `Distribution`. The one runtime constructed-domain import (`information`) is the Huffman tree for hierarchical softmax.
fix: change the order in `scripts/review/_shared.mjs` (`CORE_ORDER`) to put `dists` before `optim`, as for `context`/`scene` in foundations F17; optionally move the `DispersionFamily`/`Link` interfaces beside `Distribution`'s minimal structural type so a model library below `dists` could name them (not needed); for `information`, either move `huffmanTree` down to `base` or `numerics` (it is a priority-queue algorithm) or state the exception in the layers check.
size: S (30 min for the script order; M if `huffmanTree` moves)
risk: low
preserves behaviour: yes
changes public surface: no

### F25: test gap: the LP retry chain and the simplex's give-up branches are pinned by seed-pinned cases and either-or assertions

kind: test-gap
evidence: optim/lp-simplex.ts:285 (`if (current.finished) return current`) and :380 (`if (clean === undefined) return finish(s, 'numerical difficulties')`, the singular-basis exit on the unbounded confirmation) are never run; branch coverage of lp-simplex.ts 73%, lp-interior-point.ts 67% (lines 46-68, the `independentRows` elimination, 110-146, 180-258), lp-standard-form.ts 69%, qp-interior-point.ts 60%, quadratic.ts 64%, qp-active-set.ts 68%, `_pareto.ts` 57%, minimize/least-squares-linear.ts 54% (its `nnls` active-set loop lines 317-323 and 359-380 never run); linear-programming.test.ts:240-330 pins three near-tied programs by their data (one found by fast-check "seed 2075411073") and asserts `['optimal', 'numerical difficulties']` contains the simplex status (:319), `if (raw.status === 'optimal') ... else expect(raw.status).toBe('numerical difficulties')` (:256-261). There is no test of the order of the retry chain, of `pivotFloor`, of unboundedness cross-checks, of scale (F3), or of infeasible/unbounded programs through both methods on the same data.
what: the LP fix is covered where it was found and nowhere else; the table in F3/F4/F5 (scale of `c` and `b`, plain 60 x 120 packing program, NaN and Infinity inputs, empty, duplicate and inconsistent equality rows, bounds `lo > hi`) is what a property test cannot generate.
fix: one table-driven `it` for `linprog` (probes p1, p5, p6, p20 as rows: program, expected status, objective within a relative tolerance, both methods agreeing), replacing the either-or assertions with the expected status once F3/F4 are done; one for `nnls`/`lsqLinear` active-set paths (a rank-deficient, a bound-hit and a degenerate-tie row); one `qpActiveSet`/`qpInteriorPoint` table (infeasible, unbounded LP-like, equality-only, bound-only, indefinite P from F18). Offset: merge the three seed-pinned near-tied `it`s into the table (-2).
size: M (2h)
risk: low
preserves behaviour: yes
changes public surface: no

### F26: test gap: no direct test of the minimise family's degenerate-input contract, of mode agreement at scale, or of the stochastic stop

kind: test-gap
evidence: probe p2 shows the contract the family satisfies today (zero gradient, constant, NaN, infinite gradient, empty, one element, unbounded below) on 13 methods, and where it does not (F16); no test file under optim/minimize covers it as a table (minimize/*.test.ts are per family: constrained, convex, derivative-free, global, gradient, least-squares, quasi-newton, second-order); stochastic.test.ts uses logistic regression (no zero-loss examples, so F2 is unreachable); `glm.test.ts`/`glm.property.test.ts` compare modes on well-scaled designs only; `solvers.test.ts` checks the table's names and a few configurations. Files with no test file bearing their name (their coverage is incidental): admm.ts, dictionary.ts, gromov.ts, total-variation.ts, wasserstein.ts, qp-active-set.ts, qp-interior-point.ts, lp-simplex.ts, lp-interior-point.ts, lp-standard-form.ts, minimize/cobyla, cmaes, frank-wolfe, lbfgsb, trf, slsqp, penalty, nsga2 (coverage is high for most through family tests; the exceptions are dictionary.ts 63% branches with `worstSample` and three closures unrun, minimize/dual-annealing.ts 63%, slsqp.ts 62%, wasserstein.ts 64%).
fix: (a) one table-driven `it` in `minimize/minimizer.test.ts` over the 13 methods and the rows of p2 (expected status and `x` finite); (b) one `it` in `glm.test.ts` that fits the same problem at scales 1, 1e-8, 1e8 with every mode of `glmSolvers` and compares coefficients (fails today at 1e-8: F6); (c) one `it` in `stochastic.test.ts` with a hinge-type objective and `batch: 1` asserting the run does not stop `converged` before `maxSteps` unless the full gradient is small (fails today: F2); (d) a dictionary/OMP scale row in `sparse.test.ts` (F9). Net: +4 `it`s, offset by the -2 of F25 and by dropping the `solvers.test.ts` name-listing assertions that `smoothSolvers.names` already covers in its `@example` (-1).
size: M (2-3h)
risk: low
preserves behaviour: yes
changes public surface: no

### F27: test gap: losses have no test at the degenerate points of their gradients or of their hyperparameters

kind: test-gap
evidence: losses coverage is 97.9% statements / 84.1% branches (the best of the core), but every `*.golden.test.ts` and `*.test.ts` I opened uses ordinary inputs; the NaN gradients of F20 (`ntXent` at zero rows, `binaryCrossEntropy` at p in {0, 1}, Tweedie at 0) and the silent hyperparameters of F19 are untested; `sequence.ts` (CTC) branch coverage 79% with `ctcTrellis` etc. exercised, but the module fails to load outside vitest (F22); `physics.ts` and `representation`/`quantisation` have golden or small tests only.
fix: one table-driven `it` in a new `losses/degenerate.test.ts` (or in `tables.test.ts`): rows of `[loss, inputs, expect finite value and finite gradient]` for the 25 probes of p10/p15 that must hold (zero rows, saturated probabilities, tied scores, empty class, single sample) and `[loss, bad option, expect MdmlError]` for the F19 knobs; run `grad` and `checkPrimitive`-style finite-difference on the finite ones. +1 `it` (two tables in it); offset with F26's.
size: S-M (2h)
risk: low
preserves behaviour: yes
changes public surface: no

### F28: TSDoc: 10 exported functions have no `@example`, 6 no `@returns`; the hubs among them

kind: rule
evidence: probe `doc.mjs` over 546 exported functions of non-private files: no `@example`: `logTemperature`, `expTemperature` (losses/classification.ts:526, :532), `steppedLineSearch`, `backtrackingSearch`, `hagerZhangSearch` (optim/line-search-algorithms.ts:115, :339, :522), `newtonAt`, `slotAlgorithm` (optim/slot.ts:39, :108), `smoothSolverConfig`, `loggedSolverOf`, `loggedSolver` (optim/solvers.ts:229, :312, :346); no `@returns`: `logTemperature`, `expTemperature`, `backtrackingSearch`, `hagerZhangSearch`, `newtonAt`, `smoothSolverConfig`. The rule: every core export has summary, `@param`, `@returns` and a checked `@example`.
what: complete otherwise (summary and params for all 546); the gaps are on the slot hub (`loggedSolver`, `slotAlgorithm` are what every model calls) where an example would also have caught F1.
fix: add the ten examples (`loggedSolver` with a two-step run reading `solver/objective`; `slotAlgorithm` is longer, a short model suffices); `@returns` for six.
size: S (1h)
risk: low
preserves behaviour: yes
changes public surface: no

### F29: the logging contract is mixed inside optim, and one iterative routine is not an `Algorithm`

kind: rule
evidence: `defineContract` (typed paths) is used in two files, optim/solvers.ts (`solverContract`, `gradientContract`) and optim/line-search-algorithms.ts:41; the other 54 `Algorithm`-returning files declare `schema: channels({ metrics, tensors, messages })` and an `observe` (26 files); the CLAUDE.md rule asks the method's contract to be `defineContract` with a path type so `ctx.log`/`ctx.emit` are checked, which applies to estimators built on these algorithms rather than to the algorithms themselves; `sharpness` (optim/landscape.ts:350-402) is a power iteration with a tolerance (`options.tol ?? 1e-10`, `iterations ?? 200`) that returns `{ value, vector, products }` with no `converged`, so exhausting 200 iterations is indistinguishable from converging; `ripEstimate` (dictionary.ts:666) samples; `hindsightComparator` (online-experts.ts:839) loops on a comparator search.
what: two logging mechanisms coexist by age, not by design (`channels`/`observe` is the older, step-observation form; the contract is the newer, ctx-logged form); `sharpness` is one of the "iterative routines that are not Algorithms" of numerics F16, here with the status missing as well.
fix: record the convention in `optim/index.ts` (algorithms observe; a model that wraps one declares the contract and spreads `solverContract`); give `sharpness` a `converged` field (and `sharpnessAlgorithm` if the Lanczos/power state should be shown, as `eigsh` offers).
size: S (1h)
risk: low
preserves behaviour: yes
changes public surface: yes (an added field)

### F30: bare `catch {}` swallows every exception, including those of the user's objective

kind: consistency
evidence: optim/minimize/gradient.ts:482-485 (`coordinateDescent`, `update: 'exact'`: `try { brent(along, ...) } catch { next = xi }` where `along` calls the user's objective, "no bracket along this coordinate"), optim/sparse.ts:264 (`cholesky` of the Gram in OMP), optim/lp-simplex.ts:131 (`dgesv` singular), optim/conic.ts:135, optim/quadratic.ts:421, optim/minimize/cobyla.ts:126, minimize/_trstlp.ts:43, minimize/lbfgsb.ts:297 (8 sites; the other seven wrap a linear-algebra call and are intended).
what: the coordinate-descent case catches an error thrown by the user's function (a shape error, an out-of-memory, a bug) and reports "no bracket", the coordinate is skipped, and the run converges on whatever is left; the intended catch is `convergence` from `brent`.
fix: catch only `MdmlError` with `code === 'convergence'` (or have `brent` return its `flag`) in gradient.ts; narrow the other seven to `singular`/`value` codes.
size: S (30 min)
risk: low
preserves behaviour: yes except that user errors propagate
changes public surface: no

### F31: hot-path note: path and history are copied at every step (quadratic), small today

kind: hot-path
evidence: `path: [...s.path, vertex(next)]` per pivot in optim/lp-simplex.ts:318, :348, :390 and per iteration in lp-interior-point.ts:269; `history: [...s.history, rn]` in sparse.ts:171, :280, :368 and `particle-swarm.ts`; `pivotOn` allocates a new tableau per pivot (lp-simplex.ts:35); probe p22: `matchingPursuit` 16,000 steps 216 ms against 47 ms at 8,000 (quadratic); the simplex at 120 x 240 spends 411 ms for 1,132 pivots (about 0.36 ms each for a 29,000-entry tableau, comparable to the arithmetic), so the copy is not yet the cost.
what: the same pattern as foundations F1 (`scheduler.fired`): an immutable state that rebuilds its history with a spread. The sizes of the LP and the matching pursuit make it harmless; a long matching-pursuit run (10 k atoms for a 100,000-sample signal, `maxIterations` default `10 * cols`) and a visual run of the simplex that keeps the path will feel it.
fix: carry the history in a preallocated or push-only array owned by the state (as the engine does for traces), or store `residualNorm` only and rebuild the history from the tracer; leave the LP paths unless a program with several thousand pivots shows up.
size: S (1h)
risk: low
preserves behaviour: yes
changes public surface: no

---

## What I did not do or could not verify

- Python was not run: every value attributed to scipy (HiGHS scaling, `gtol`), torch (`F.normalize`, BCE backward, `gaussian_nll_loss` raising on negative variance), sklearn (`_cholesky_omp`) or optax (`apply_if_finite`) is from memory; confirm with the oracle before changing a default.
- Coverage is Phase 0's (commit b09e46da): the named "never run" functions (`trfAlgorithm`, `dogboxAlgorithm`, `sr1Algorithm`, `owlqnAlgorithm`, `newtonCgAlgorithm`, `trustRegionAlgorithm`, `projectedGradientAlgorithm`, `admm`, `clipByValue`, `projectNonnegative`, `worstSample`) are probably exercised by `@example`s that vitest coverage does not count; I did not check each.
- Not opened at all: transport.ts, conic.ts, integer.ts, nsga2.ts, multiobjective.ts bodies, minimize/{trust-region, cmaes, differential-evolution, dual-annealing, cobyla, slsqp, least-squares-*, levenberg-marquardt, penalty, powell, owlqn} bodies, dictionary.ts K-SVD/MOD bodies, total-variation.ts, gromov.ts, wasserstein.ts bodies beyond `slicedWasserstein`, online-experts.ts bodies, and in losses all but classification (part), divergence (part), contrastive (probed through its functions), regression (probed), `_reduce.ts`, `_vectors.ts`, `tables.ts` (part) and the CTC class. Findings about them rest on the Phase 0 scans, signatures and probes.
- The interior-point's unbounded verdict was checked on 240 random boxed programs (never fired) and on the textbook cases; I did not search for the original failing program (fast-check seed 2075411073 is in the test), so "still there, now cross-checked" is a statement about the code at `lp-interior-point.ts:273-277` and `linprog`'s vote, not a proof.
- Timings are single runs on a loaded laptop at `nice -n 15`; ratios (Bland 15x, OMP cubic, `crossEntropy` = `logSoftmax`) are reliable, absolute milliseconds are not.
