# Numerics review: approximation, convolution, differentiate, fourier, geometry, interpolate, inverse, linalg, polynomial, quadrature, roots, sparse, spatial, special

Read-only review of `packages/engine/src/lib/numerics` (28,832 lines in 112 non-test files; 97 of them without barrels). All
paths below are relative to `packages/engine/src/lib/`. Probes cited as "probe" are throwaway node scripts run from the
scratchpad (not the repo) against the source, at `nice -n 15`, one process at a time; nothing in the repository was changed
and no test, tsc or gate was run. Scipy and numpy values quoted as "scipy gives X" are from my knowledge of those libraries,
not from running them (Python was not run); treat them as expectations to confirm with the oracle before a fix.

## Summary

1. Healthy: the folder is the most disciplined of the core domains read so far. No `Math.random`, clock, `console` or
   `// allow:` in 112 files; imports follow the order (only `tensor`, `base`, `trace`, `autodiff`, `random`, `space`, plus a
   type-only `exposition`; nothing from `optim`, `dists`, `stats`); the subfolders form a DAG (`linalg` at the bottom);
   TSDoc is complete for 480 of 502 exported functions; local copies of generic helpers are few (F17); the roots, quadrature,
   chebfun, remez, Krylov and eigen-iteration routines are `Algorithm`s on the log contract; ties are broken the same way
   by brute force, KD, ball and VP trees (probe); `fft` is 3 ms for 2^16 and accurate to 1e-14 relative in Bluestein.
2. Not healthy, 1: **scale and Infinity/denormal inputs give silent wrong answers, which is the CI flake class**. The default
   `svd` (one-sided Jacobi) returns the column norms of the unrotated matrix once entries exceed 1e77 (or are below 1e-155),
   so `cond`, `matrixRank`, `pinv`, `nullSpace` follow (F1); `bisect` returns 1.0 and says `converged` for `f = 1e-200 (x - 0.3)`
   and `ridder` fails (F2); `adaptiveSimpson` on a NaN or infinite integrand needs 2^52 evaluations, i.e. never returns (F3);
   `erf(Infinity)`, `lgamma(Infinity)`, `logaddexp(-Infinity, -Infinity)`, `ndtri(5e-324)` are NaN (F4).
3. Not healthy, 2: **two silent wrong results on ordinary-looking inputs**: `lstsq`/`polyfit` of a rank-deficient system
   returns 2e15-sized garbage instead of the minimum-norm solution (20 callers, F5), and `delaunay` of 8 or more exactly
   cocircular points (a regular polygon) returns 32 overlapping triangles for 12 points (F6).
4. Not healthy, 3: **no single failure convention**: a non-converged iteration is a thrown `convergence` error in `fixedPoint`,
   `homotopy`, `sparseSolve`, `sqrtm`, `chebfun`, the kernels; a `converged: false` field in `fsolve`, `cg`, `quad`, `romberg`;
   a `flag` string in the scalar finders; nothing in `tanhSinh`, `adaptiveSimpson`, `kmeansCodebook` (F10). NaN input throws in
   `eigh`, is returned in `svd`, `eig`, `solve`, and corrupts the order in `kNearest` (F11).
5. Hot paths with big, cheap wins (measured): general `eig` is O(n^4) in the eigenvectors (1.0 s at n=100, 16 s at n=200, where
   `eigvals` takes 32 ms and 260 ms; F7); `convolve` builds an (n+m-1) x m window matrix (2.1 s and 270 MB for 4096 x 4096,
   the FFT version 2.6 ms; F8); `spmv` is three tensor ops and every Krylov step round-trips through tensors (70 ns per stored
   entry; F9); non-power-of-two FFT lengths all use Bluestein and recompute the chirp per line (1000-point rows cost 6x the
   1024-point rows; F13).
6. Native arrays: 92 public functions carry one. Classified: 12 kernel (allowed), 20 shape/axis/index/ragged (not a vector),
   **60 public vector or matrix (migrate)**, of which 10 in `polynomial/power.ts` (with 6+ callers in `timeseries`, `systems`,
   `signal`), 6 in `geometry/transforms.ts`, 6 in `interpolate/polynomial.ts`, 5 in `approximation/acceleration.ts` (F19).
7. The same copy-the-pattern problem as `trace`: the option-table builder is hand-written in five `*-options.ts` files of this
   folder (F15), complex-pair arithmetic exists four times (F18), a Lloyd loop duplicates `cluster/kmeans.ts` outside the
   `Algorithm` contract (F16).
8. Tests: statement coverage 94.0%, branch 71.1%; the uncovered lines are almost exactly the degenerate branches (`a <= 0`,
   `x === 0`, `x === Infinity`, non-convergence throws, tie cases), and 24 named functions never run (F26-F28).
9. The three highest-value fixes: (a) F1 + F2 + F3 + F4 (four one-to-ten-line guards, each with a probe that fails today);
   (b) F7 + F8 + F9 (three measured 20x to 6000x wins, each a local rewrite behind unchanged results); (c) F5 + F6 + F10 (the
   two silent wrong results and one stated failure convention).
10. Not read or not verified: about 50 of 112 files opened; not read at all: `special/_bessel.ts`, `_elliptic.ts`, `zeta.ts`,
    `bessel.ts`, `elliptic.ts` (probed only by value at edge points), `geometry/{contours,segments,distances,polygons}.ts`
    beyond signatures, `interpolate/{scattered,shape-preserving,grid}.ts` beyond probes, `spatial/{graphs,hashing,trees}.ts`,
    `sparse/{multigrid,stationary,lanczos-steps}.ts` bodies, `convolution/n-d.ts`, `fourier/{trig,nonuniform}.ts` bodies,
    `quadrature/{gauss,multidim,monte-carlo}.ts`, `linalg/{completion,randomized,subspace}.ts` bodies. Accuracy against scipy
    was not measured (no oracle run); only structure, degenerate behaviour and timing. Coverage is Phase 0's (b09e46da).

Entries by kind: consistency 13, hot-path 5, duplicate 3, rule 3, layering 1, test-gap 3, native-array 1 (29 in all).

---

### F1: the default `svd` is silently wrong above |a| ~ 1e77 and below ~ 1e-155 (and `cond`, `matrixRank`, `pinv`, `nullSpace` with it)

kind: consistency
evidence: linalg/_kernels.ts:457 (`Math.abs(gamma) <= EPS * Math.sqrt(alpha * beta)`: `alpha * beta` is a fourth power of the entries, so it overflows at 1e77 and underflows at 1e-155; `Infinity` makes every pair look converged and no rotation is applied) and :486-488 (column norms by `** 2`); default method is Jacobi (linalg/svd.ts:176); probe, singular values of `s * [[1,2],[3,4],[5,6]]` divided by `s`: 1e75 gives [9.5255, 0.5143] (right), 1e77 gives [7.4833, 5.9161] (the column norms of the input, wrong) with no error, 1e-162 gives [9.43, 0], 1e-170 gives [0, 0]; `[[1e200,0],[0,1e200]]` gives [Infinity, Infinity]; `[[1e-310,0],[0,1e-310]]` gives [0, 0]; consequences in the same probe: `cond` 1.265 instead of 18.52 and `matrixRank` unchanged at 2 for 1e77, `matrixRank` 1 at 1e-162 and 0 at 1e-170; the `golub-kahan` method throws `convergence` at 1e155, 1e-200 and above (linalg/_kernels.ts:660); `eigh` (QL) and `qr` are exact at 1e200 and 1e-310 in the same probe.
what: a silent wrong answer, not a loud one, in the factorisation that `pinv`, `lstsq` users, `cond`, `matrixRank`, `nullSpace`, `orth`, `polar`, `lowRankApproximation` and `inverse/spectralData` sit on. The denormal and huge-scale cases are exactly what the flaky tests probe.
fix: scale the matrix by an exact power of two (`2^-e` of the largest |entry|, via `Math.frexp`-style exponent from `Math.log2`/bit access) before the sweeps and rescale `s` after (U and V are unchanged); compute the convergence test as `Math.abs(gamma) <= EPS * Math.sqrt(alpha) * Math.sqrt(beta)`; do the same scaling in `svdGolubKahan`; return NaN (not a partly rotated matrix) when the input has a non-finite entry (see F11).
size: S (<1h)
risk: low (a power-of-two scale is exact; results for ordinary scale are bit-identical; fixtures use ordinary scale)
preserves behaviour: yes for finite inputs of ordinary scale; no for the broken range (now correct)
changes public surface: no

### F2: bracketing root finders: the sign test multiplies two function values (underflow) and a NaN `f` is reported as converged

kind: consistency
evidence: roots/scalar-algorithms.ts:107 (`checkBracket`: `fa * fb > 0`), :291 (`bisect` step: `if (fm * fa >= 0)`), :480-488 (`ridder`: `fm * fm - fa * fb`, `fn * fm < 0`), :606 (`toms748`: `A * fa > 0`); probe, `f = s (x - 0.3)` on [0, 1]: scale 1e-100 all four finders give 0.3; scale 1e-160 `bisect` gives 0.3125 and `ridder` stops with `converged: false`; scale 1e-200 and 1e-300 `bisect` returns 1.000000 (reported converged) and `ridder` returns 0 (not converged), `brentq` and `toms748` are right; scale 1e+300 all right; `bisect/brentq/toms748((x) => NaN, -1, 1)` give `converged: true, flag: 'converged'` with root -0.999999999998 / 0.3333 / -0.999999999996, `ridder` gives root NaN, `maximum iterations`; `bisect` with `f(-1) = NaN` and `f(1) = 1` returns root 0, converged.
what: the product of two function values underflows to 0 below 1e-160 (so `>= 0` picks the wrong half) and NaN passes `> 0` and every `<` test, so a diverged or mis-specified `f` produces a plausible "root". Four finders, four different behaviours on the same input.
fix: compare signs (`Math.sign(fm) === Math.sign(fa)` or `(fm > 0) === (fa > 0)`), never products; in `checkBracket` and after every evaluation, if `Number.isNaN(fx)` end the run with a new `flag: 'nan'` (status `failed`) and `converged: false`; scale `ridder`'s square-root term by `max(|fa|, |fb|, |fm|)`.
size: S (<1h)
risk: low (only the denormal-scale and NaN cases change; the golden fixtures use ordinary scale)
preserves behaviour: no (NaN and tiny-scale results change from wrong to correct or flagged)
changes public surface: yes (one more value in the `RootResult.flag` union)

### F3: `adaptiveSimpson` needs 2^52 evaluations for a NaN or infinite integrand (it does not return); `tanhSinh` and `adaptiveSimpson` have no status

kind: consistency
evidence: quadrature/adaptive.ts:339-341 (`if (depth <= 0 || Math.abs(delta) <= 15 * tol) return ...; return recurse(left) + recurse(right)`, with `maxDepth` default 50 at :350), `NaN <= x` is false so a NaN integrand recurses to the depth limit on both halves; probe, `adaptiveSimpson(x => x * NaN, 0, 1, { maxDepth })`: depth 10 4,097 evaluations, 14 65,537, 18 1,048,577 (0.6 s), 20 4,194,305 (2.6 s), so the default 50 is about 4.5e15 evaluations; `adaptiveSimpson(1/x, 0, 1)` hit the same wall (my first probe ran for over two minutes and was stopped); `tanhSinh(x => x * NaN)` returns NaN with `levels: 8` and `tanhSinh(1/x, 0, 1)` returns 38.0 with `levels: 8`, no flag; `romberg` and `quad` do return `converged`.
what: a function that returns NaN anywhere (or infinity at an endpoint, as 1/x at 0) hangs the caller. The other three integrators report `converged`; these two cannot say that they stopped at a limit.
fix: stop when `delta` is not finite (return NaN at once), cap the evaluations (`maxEvaluations` default e.g. 100,000 as `quad`'s `limit` does) and return `{ value, evaluations, converged }`; give `tanhSinh` the same `converged` (last two levels agree to `tol`) and make `maxDepth` default 50 into something below the point where `tol / 2^depth` falls under rounding (about 20).
size: S (<1h)
risk: low (additive field; the default depth change only affects runs that would not finish)
preserves behaviour: yes for runs that finish
changes public surface: yes (return types gain `converged`)

### F4: special functions at +-Infinity, at denormal tails and at zero parameters are NaN where scipy has a value

kind: consistency
evidence: probe (value returned; scipy value from memory in brackets): `erf(+-Infinity)` NaN (+-1) from special/_kernels.ts:64-69 (`erfcFraction(Infinity)` is Infinity / Infinity); `erfcx(Infinity)` NaN (0) from :84-88; `lgamma(+-Infinity)` NaN (Infinity) from :227-238; `logaddexp(-Infinity, -Infinity)` NaN (-Infinity) and `logaddexp(Infinity, Infinity)` NaN (Infinity) from special/logistic.ts:146-149 (`abs(sub(x, y))` is NaN); `ndtri(5e-324)` NaN (-38.47) from special/_kernels.ts:157 (`e * SQRT_2PI * Math.exp(x * x / 2)` is `0 * Infinity` once |x| > 37.6, the Halley step; `ndtri(1e-310)` is right); `gammainc(0, 1)` NaN (1) and `gammaincc(0, 1)` NaN (0) from special/_incomplete.ts:101-103 (`a <= 0` rejected); `gamma(0)`, `digamma(0)`, `polygamma(1, 0)` NaN (Infinity, -Infinity, Infinity); `zeta(Infinity)` and `hurwitzZeta(Infinity, 1)` NaN (1); `hurwitzZeta(2, 0)` NaN (Infinity); `lambertW(Infinity)` NaN (Infinity); `jv(0, Infinity)` NaN (0); `binom(-1, 2)` NaN (1); `expn(0, 0)` NaN (Infinity). Right at the same probe: `erfc(+-Infinity)`, `ndtr(+-Infinity)`, `gammainc(1, Infinity)`, `expit(+-1000)`, `softplus(1e308)`, `iv(0, 710)`, `kv(0, 1e-310)`, `ellipk(1)`.
what: the special functions are elementwise primitives, so the one NaN reaches a loss or a log-likelihood. `logaddexp(-Infinity, -Infinity)` is the usual log-space sum of two impossible events (HMM and CRF forward passes).
fix: add the explicit limit cases at the top of each kernel (a table of `[args, value]` per function is the test, F26); in `ndtri` do the Halley correction in log space or skip it when `|x| > 37` (the Acklam start is accurate to 1e-9 relative there); `logaddexp`: `where(isinf(max) & equal(x, y), x, ...)`; `gammainc(0, x)` is 1 for x > 0.
size: M (1-3h) for the roughly fifteen cases
risk: low (only NaN results change; derivatives at the limits are unchanged because they are separate rules)
preserves behaviour: no (NaN becomes the limit value)
changes public surface: no

### F5: `lstsq` of a rank-deficient system (and `polyfit`, `weightedFit`) returns garbage of size 1e15, not the minimum-norm solution

kind: consistency
evidence: linalg/qr.ts:154-160 (`lstsq`: `triangularSolve(r, qT b)` on the R of a Householder QR, no pivoting, no check of the diagonal; the TSDoc says "of full rank"); probes: `lstsq([[1,1,1],[1,1,1],[1,1,1]], [1,1,1])` gives [0, -2251799813685247, 2251799813685248]; `lstsq([[0,0],[0,0]], [1,1])` gives [NaN, Infinity]; `polyfit([1,1,1], [1,2,3], 1)` (a repeated abscissa, so the Vandermonde matrix has rank 1) gives [2251799813685249, -2251799813685247.2] (numpy returns the minimum-norm [2, 0.0...] with a `RankWarning`); `polynomial/power.ts:109-120` (`weightedFit` calls `lstsq`); callers of `lstsq` outside numerics: `linear/solvers.ts`, `optim/sparse.ts`, `optim/dictionary.ts`, `dynamics/identification.ts`, `dynamics/control/placement.ts`, `information/mdl.ts`, `timeseries/_ml.ts`, `nn/kan/grid.ts`, `vision/geometry.ts`, `fairness/*` (20 files); `pinv` (linalg/svd.ts:210) already does the right thing with `rtol = max(m, n) eps`.
what: the function is documented for full rank, but every caller that fits a model on data with a collinear column, a constant feature, or fewer distinct x than the degree gets a huge finite or non-finite vector with no error. An earlier probe on `[[1,2],[2,4],[3,6]]` happened to give a sensible-looking [0, 0.5] (R's rounding-level diagonal times a zero numerator), which hides the problem.
fix: after the QR, test `min |diag R| <= max(m, n) eps max |diag R|`; if rank deficient, fall back to the SVD solution `pinv(a) @ b` (differentiable already) so `lstsq` is the minimum-norm solution as numpy's; or add the `rcond` option numpy has and default it to the same value. Keep the QR path for full rank (it is the faster and tested one).
size: M (1-3h; the work is the test table and the jvp/vjp consistency of the fallback)
risk: medium (changes results for rank-deficient inputs only; a caller that depended on the garbage cannot exist, but a caller that catches a thrown error from a singular system now gets an answer; grep the 20 callers)
preserves behaviour: yes for full rank; no for rank-deficient (garbage becomes minimum-norm)
changes public surface: no

### F6: `delaunay` returns overlapping triangles for 8 or more exactly cocircular points

kind: consistency
evidence: geometry/triangulation.ts:127 (Bowyer-Watson cavity: `inCircle(...) > 0` on raw floating point, then the cavity boundary taken as the edges seen once); the predicate is the non-robust determinant of :35-47 (no adaptive precision); probe, a regular n-gon on the unit circle: n = 4, 5, 6 give n - 2 triangles and the right area; n = 8 gives 10 triangles (expected 6) and area 7.24 (expected 2.83); n = 12 gives 32 triangles, area 22.89 (expected 3.00); n = 16 gives 71 triangles, area 39.29 (expected 3.06); the same points perturbed by 1e-9 give 10 triangles and area 3.000000; the 6 x 6 integer grid (cocircular quadruples) is right (area 25); 200 random points: triangle area equals hull area. Also: `delaunay` of three duplicate points throws "the points are collinear", of huge coordinates (1e200) gives an infinite area (the normalised work coordinates are fine, `inCircle` on the originals in `voronoi`/`simplexTensor` is not checked); `convexHull` of a point set containing NaN returns [0, 2, 1, 2] (a repeated index).
what: with a tie in the circumcircle test the sign of the rounding noise decides whether a triangle is removed from the cavity, so the cavity is no longer star-shaped around the new point and triangles overlap. Regular polygons, circles of sample points and lattices of cocircular points are common test inputs and the output is wrong without any error. `interpolate/scattered.ts` and `interpolate/grid.ts` (`griddata`, natural neighbour) call `delaunay`; `voronoi` and `alphaShape` are in the same file.
fix: treat `|inCircle| <= 1e-12 * scale` as "not inside" (the cavity then excludes the cocircular triangle, which is a valid Delaunay choice), or use Shewchuk's `incircle` with the error bound and an exact fallback; add a post-condition test on the output (sum of triangle areas equals hull area, `n - 2 - h + ...` triangle count by Euler) so a regression is caught at once.
size: M (1-3h)
risk: medium (changes triangulations of degenerate inputs; ordinary inputs must keep identical simplices, which the golden fixtures pin)
preserves behaviour: yes for general position; no for cocircular sets (invalid becomes valid)
changes public surface: no

### F7: general `eig` extracts eigenvectors by a dense 2n x 2n LU per eigenvalue (O(n^4)): 1 s at n = 100, 16 s at n = 200

kind: hot-path
evidence: linalg/_schur.ts:179-232 (`eigenvectorKernel`: builds a (2n)^2 real system, perturbs the shift by `1e-10` and runs 3 rounds of `solveKernel`), called once per eigenvalue at linalg/eig.ts:122; probe: `eigvals` 7 ms / 32 ms / 260 ms and `eig` 80 ms / 1041 ms / 16142 ms for n = 50 / 100 / 200 (worst residual |A v - lambda v| is 4e-14 to 2e-13, so the answer is good; it is the cost); the Jordan block [[2,1,0],[0,2,1],[0,0,2]] gives three identical vectors [1, 0, 0] with no warning and a near-defective [[1, 1e-8], [0, 1+1e-8]] gives vectors 0.707 apart in direction as expected; 12 engine files call `eig`/`eigvals` (dynamics, timeseries, `nn/blocks/hippo.ts`, `stochastic/markov-chains.ts`).
what: the eigenvalues cost O(n^3) (Hessenberg + Francis QR), the eigenvectors O(n^4); at n = 200 the vectors cost 60x the values. The perturbed-shift inverse iteration also gives only about 10 digits for clustered eigenvalues.
fix: back-substitute the quasi-triangular Schur factor (LAPACK `dtrevc`: solve (T - lambda I) x = 0 by one triangular solve per eigenvalue, 2 x 2 blocks for complex pairs), then multiply by Z; O(n^3) in all; the existing residual test (|A v - lambda v|) is the oracle. Keep `eigenvectorKernel` as the reference in a test until the two agree on random and defective matrices.
size: M (1-3h)
risk: medium (eigenvectors of repeated eigenvalues may change sign/phase; normalisation rule "largest component real and positive" is kept)
preserves behaviour: yes (values to rounding; phase convention kept)
changes public surface: no

### F8: `convolve` materialises an (n + m - 1) x m window matrix: 2.1 s and 270 MB for two 4096-vectors

kind: hot-path
evidence: convolution/one-d.ts:76-91 (`idx = new Int32Array(count * m)`, `take(padded, idx)`, `reshape`, `matmul`); probe: `convolve` 4096 x 4096 2138 ms, `fftConvolve` 4096 x 4096 2.6 ms; memory is `count * m` doubles (2^25 entries, 268 MB, plus the 134 MB index array); 1e4 x 1e4 would allocate about 1.2 GB; users: `signal/*`, `foundations/probability.ts` (4 files import `fftConvolve`, the others `convolve`).
what: the direct convolution is quadratic in time (acceptable) and quadratic in memory (not); it is differentiable only because it is built from `take` and `matmul`.
fix: a `convolve` primitive with a loop kernel (`O(n m)` time, `O(n + m)` memory) whose vjp is two correlations (the same primitive with swapped arguments), or dispatch to `fftConvolve` above a size threshold as `scipy.signal.convolve(method='auto')` does (`n * m > 1e5` and both lengths > 64); keep the matmul form for tiny inputs.
size: M (1-3h)
risk: low (results agree to rounding; `checkPrimitive` pins the new primitive; the FFT dispatch changes the last digits above the threshold)
preserves behaviour: yes (values to rounding)
changes public surface: no

### F9: `spmv` is `take` + `mul` + `segmentSum`, and every Krylov step round-trips through tensors

kind: hot-path
evidence: sparse/csr.ts:186-192 (`segmentSum(mul(a.values, take(x, a.indices)), a.rows)`), sparse/csr.ts:313-331 (`asOperator`), sparse/iterative.ts:32-34 (`numeric`: `fromStorage` -> matvec -> `primalOf` -> `toFlat` -> `Float64Array.from` per iteration), duplicated at sparse/eigen.ts:25-29; probe: `spmv` of a 2000 x 2000 tridiagonal matrix (6,000 entries) costs 0.42 ms (70 ns per stored entry; a CSR loop is 2-5 ns), CG on it (1000 iterations to 1e-8) 496 ms; plain-array CSR loops already exist in sparse/_rows.ts (the stationary sweeps) and sparse/multigrid.ts:284 (`matvec`), so the same product is written three times; `segmentSum` also range-checks nothing (foundations F6).
what: for the iteration the tape and the gradient are not wanted (the differentiable solve is `sparseSolve`'s custom vjp), but the tensor path is paid on every matvec; CG, GMRES, MINRES, BiCGSTAB, Lanczos and Arnoldi all go through it.
fix: give `spmv` a kernel (one `definePrimitive` with a CSR loop kernel and the `take`/`segmentSum` pair as its vjp/jvp), and let `asOperator` return a `Float64Array -> Float64Array` fast path next to `matvec` that `numeric()` uses when present; delete the two other loops.
size: M (1-3h)
risk: low (kernel replaces ops; `checkPrimitive` and the existing sparse tests pin it)
preserves behaviour: yes
changes public surface: yes (an optional field on `LinearOperator`)

### F10: failure and non-convergence are reported five different ways

kind: consistency
evidence: throws `MdmlError('convergence')`: roots/systems.ts:542 (`fixedPoint`, whose `SystemResult` then always has `converged: true, message: 'converged'`, a vacuous field, while its sibling `fsolve`/`broyden` return `converged: false` with a message), roots/continuation.ts:58 (`homotopy`), sparse/iterative.ts:677 (`sparseSolve`, while `cg`/`gmres`/`minres`/`bicgstab` return `converged: false`), sparse/eigen.ts:458 (`eigsh`), linalg/functions.ts:177 (`sqrtm`), equations.ts:140 (`signm`), :170 (`solveCare`), :310 (`solveDare`), approximation/chebyshev.ts:329 (`chebfun`), approximation/pade.ts:90 (`continuedFraction`), approximation/best.ts:180 (`remez`), kernels linalg/_kernels.ts:369 (`eigh`), :660 (`svd` Golub-Kahan), _schur.ts:89 (`schur`); a `converged` boolean: quadrature/adaptive.ts (`quad`), newton-cotes.ts:179 (`romberg`), sparse/iterative.ts:55 (Krylov), linalg/eig.ts:142 (power iteration), subspace, completion, inverse; a `flag` string: roots/scalar-algorithms.ts:39 (`RootResult.flag`, with a `'sign error'` member that is never produced because `checkBracket` throws `value` at :107); an `info` code: linalg/lapack.ts:25 (`dpotrf`), while `dgesv`/`dsyev`/`dgesvd` of the same file throw (the file summary says they "report failure instead of throwing where LAPACK does"); nothing: `tanhSinh`, `adaptiveSimpson` (F3), `kmeansCodebook` (inertia history only), `derivative` (an `error` estimate only, `levels: 0` and `df: NaN` for a NaN function). Error codes in numerics: `value` 114, `shape` 68, `convergence` 14, `not-differentiable` 1.
what: a caller cannot write one `if (!result.converged)`; `fsolve` and `fixedPoint` have the same return type and opposite contracts. The iteration cap (`maxIterations`) is reached silently by the stationary methods and Krylov (they return `converged: false`) but by exception in `sparseSolve` on top of the same solvers.
fix: state the convention in the folder summary and apply it: **iterations run as `Algorithm`s and return a result with `converged` and a `status` (`RunStatus`); the wrappers that return a bare value (`sparseSolve`, `sqrtm`, `fixedPoint`, `chebfun`) throw `convergence` and say so in `@throws`; the kernels below `_` throw `convergence`.** Concretely: `fixedPoint` returns `converged: false` like `fsolve` (and `fixedPointAlgorithm` already exposes the state), `homotopy` returns the partial path with `converged: false`, `RootResult.flag` drops `'sign error'` or `checkBracket` produces it, and `lapack.ts`'s summary is corrected (or `dgesv` returns `info`).
size: M (1-3h; mostly small edits plus one summary paragraph)
risk: medium (callers that catch the `convergence` error of `fixedPoint`/`homotopy`; grep first: `fixedPoint` is used in `dynamics` and `learn`)
preserves behaviour: no (two functions change from throwing to returning)
changes public surface: yes (behaviour of `fixedPoint`, `homotopy`, a union member)

### F11: NaN and Infinity input has no policy: some routines throw, some return NaN, some return a wrong finite answer

kind: consistency
evidence: probes: `eigh([[NaN,1],[1,1]])` and `eigsh(NaN)` throw `convergence` "QL iteration did not converge" (linalg/_kernels.ts:369, after 60 sweeps); `svd`, `eigvals`, `qr`, `pinv`, `expm`, `solve`, `slogdet` return NaN (`slogdet` gives `sign: 1, logabsdet: NaN`); `cholesky([[NaN,0],[0,1]])` throws "matrix is not positive definite" (`!(d > 0)` at linalg/_kernels.ts:27; the message is wrong, the matrix is not finite); `isPositiveDefinite` NaN is false; `eigh([[Infinity,1],[1,1]])` returns [1, Infinity] with no error and `svd([[Infinity,1],[1,1]])` returns [Infinity, 1.414]; `solve([[Infinity,1],[1,1]], [1,2])` returns [0, 2]; `cg` with a NaN right-hand side returns x = [0,0], `converged: false` at iteration 0 (and `bicgstab` with a NaN matrix the same), `gmres`/`minres` on a singular system return NaN vectors, `cg` on an indefinite matrix returns [Infinity, Infinity] (no breakdown test: `alpha = rz / dot(p, ap)` at sparse/iterative.ts); `kNearest` with a NaN data row returns that row as the nearest (distance NaN, `values[a] - values[b]` is NaN so the comparator returns `0 || a - b`; spatial/brute.ts:71), and `kdTree.query` returns distances [NaN, 0] in that order, i.e. unsorted; `convexHull` with a NaN returns a repeated index; `compensatedSum([Infinity, 1, 2])` is NaN (Neumaier's compensation takes `Infinity - Infinity`; it delegates to `base/numbers.ts` `kahanSum`) while Python's `fsum` gives Infinity; `derivative` of a NaN function gives `df: NaN, error: Infinity, levels: 0`.
what: every flaky test of the "NaN/Infinity propagation" class has a live twin here, and there is no way to know from the name or the TSDoc which of the three behaviours a routine has.
fix: one rule, written in the folder summary and tested by a table (F26): **a routine whose result is a function of finite inputs returns NaN (never throws, never a finite value) when an input is non-finite; a factorisation that cannot proceed on finite input (not positive definite, singular) throws `value`; an iteration that fails to converge follows F10.** Implement as a one-line `if (!allFinite(a)) return nanLike(...)` at the top of each kernel (`eighKernel`, `eighQlKernel`, `svdKernel`, `qrKernel`, `choleskyKernel`'s message), and a NaN-last comparator in the neighbour sorts.
size: M (1-3h)
risk: medium (`eigh`/`eigsh`/`cholesky` on NaN change from throw to NaN or the reverse; `cholesky` callers that rely on the throw for `choleskyJitter` retries must keep it for non-PD finite input)
preserves behaviour: no (non-finite inputs only)
changes public surface: no

### F12: singular and zero-diagonal systems: `solve` throws, the triangular and Cholesky solves return Infinity, and the messages leak from the inner call

kind: consistency
evidence: linalg/_kernels.ts:114-118 (`solveKernel` throws `value` "solve: matrix is singular" only when a pivot is exactly 0; no condition test; probe `solve([[1,2],[2,4.000000000000001]], [1,2])` returns [1, 0] with no error, which is numpy's behaviour); linalg/_kernels.ts:47-65 (`triangularKernel` divides by the diagonal with no test): probe `solveTriangular([[0,0],[1,1]], [1,1], {lower:true})` and `choSolve` of the same give [Infinity, -Infinity]; `luSolve(luFactor(singular), b)` gives [NaN, NaN] (`luFactor` of a singular matrix succeeds, `luSolve` does not look at U); `solveTridiagonal` with a zero pivot is right (pivoting) but `solveBanded` of a singular band throws the dense "solve: matrix is singular"; `sqrtm([[0,0],[0,0]])`, `sqrtm([[-1,0],[0,1]])` and `logm([[-1,0],[0,1]])` throw "solve: matrix is singular" from their inner `inv` (the message names neither function nor cause; `sqrtm` of the zero matrix is the zero matrix in scipy); `det(zeros([0,0]))` is 0 and `slogdet` gives sign 0 (numpy: det 1, sign 1, logabsdet 0; linalg/_kernels.ts:126-137); `solve` with an empty 1-D system throws `shape` "expected a matrix" (the 0 x 0 `inv` works); `dgesv` (linalg/lapack.ts:105) throws where LAPACK returns `info > 0` and the file summary says it reports failure.
what: four routines for the same mathematical task (a triangular or triangular-like solve) have three failure behaviours, and two messages blame the wrong function.
fix: `triangularKernel` / `solveTriangular` / `choSolve` / `luSolve` throw `value` "singular triangular factor (zero on the diagonal at i)" the way `solve` does (or return NaN under the F11 rule for non-finite, but a zero diagonal is a singular matrix, so throw); `sqrtm`/`logm`/`signm` catch the inner `value` and rethrow with their own name; `det`/`slogdet` of a 0 x 0 matrix are 1 / (1, 0); `dgesv` returns `{ x, info }` or its summary changes.
size: S (<1h)
risk: low-medium (callers relying on Infinity from a zero diagonal are unlikely; `solveTriangular` is used by `stats`/`dists` GP code, grep first)
preserves behaviour: no (Infinity/NaN results become errors; 0 x 0 det changes to 1)
changes public surface: no

### F13: every non-power-of-two FFT length uses Bluestein, and the chirp and its transform are rebuilt for every line

kind: hot-path
evidence: fourier/_kernel.ts:104-125 (`transform`: `isPowerOfTwo(n) ? radix2 : bluestein`), :63-102 (`bluestein` builds `wr/wi` with `Math.cos/Math.sin` for all n points, the zero-padded chirp `b`, and transforms `b` with `radix2` on every call), :128-157 (`transformAxis` calls `transform` per line; only the radix-2 twiddles are cached, :3-20); `nextFastLen` exists (fourier/helpers.ts:134) but the transforms do not use 3-, 5-, 7-smooth factors; probe: `fft` of 1024 rows x 1000 points 178 ms against 29 ms for 1024 rows x 1024 points (6.2x slower for a smaller size); one 65537-point (prime) FFT 50 ms, 59049 = 3^10 points 23 ms, 2^20 points 62 ms; `rfft(65536)` 3.4 ms is the same as `fft` (no half-length real transform); `dct(4093)` 3.0 ms against `dct(4096)` 1.1 ms; accuracy of the Bluestein path (100003 points, five bins against a direct DFT) is 4e-12 absolute for a spectrum of norm 316, fine.
what: batched transforms of "ordinary" lengths (1000, 480, 960 samples) pay a factor of 4 to 6; the per-call chirp is a third of that cost and a cache removes it (as `twiddleCache` does for radix-2).
fix: (1) cache the chirp and the transform of `b` by `(n, sign)` next to `twiddleCache` (S, a third off every non-power-of-two line); (2) mixed-radix butterflies for factors 2, 3, 5 (and 7) when the length is smooth, Bluestein otherwise (M, the usual 5x to 6x); (3) `rfft` through a half-length complex transform (S).
size: S for (1) and (3), M for (2)
risk: low (the DFT result is unchanged to rounding; the fft tests and goldens pin it)
preserves behaviour: yes (values to rounding)
changes public surface: no

### F14: `kNearest` sorts every candidate for every query with a closure comparator

kind: hot-path
evidence: spatial/brute.ts:66-76 (`Array.from({ length: n }).filter(...).sort((a, b) => values[r*n+a] - values[r*n+b] || a - b)` per query row, then `take` of the flat distance matrix); probe: 200 queries x 2000 points x 8 dimensions 334 ms, x 8000 points 1403 ms (about 90 ns per distance; the distance matrix itself is a small part), the KD tree answers the same queries in 13 ms and 31 ms after a 5 ms and 17 ms build; `kNearest` is the reference the trees are tested against and also what `kNeighboursGraph`, `radiusNeighbours` and the `cluster`/`manifold` code call.
what: O(n log n) per query where O(n log k) selection suffices; the comparator also returns `NaN || a - b` for a NaN distance (F11).
fix: select the k smallest with a bounded max-heap or `quickselect` on a typed index array (ties by index, NaN last), sort only those k.
size: S (<1h)
risk: low (same indices and distances; the tie rule is kept and stated in the TSDoc)
preserves behaviour: yes for finite input
changes public surface: no

### F15: the option-table builder is copied into five `*-options.ts` files of this folder

kind: duplicate
evidence: `const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }` at approximation/approximation-options.ts:24-27 (a local `option()` of 15 lines, identical to `information/dictionary-coding-options.ts` and `logic/algorithm-options.ts`, docs/code-review/reports/duplicates-strict.txt), roots/scalar-options.ts:61-64 (`knobsOf`) and :66-73 (`finder()`), spatial/graphs-options.ts:45-52 (inline, with `as number`/`as Metric` casts on every field, 10 casts in 79 lines), sparse/eigen-options.ts:36-41, linalg/eig-options.ts:60-61; `knobsOf` appears in five more `*-options.ts` files outside numerics (duplicates-strict.txt).
what: the same three lines (merge `defaults(space)` with the config, read a knob) written seven times in the folder alone; it is what foundations F19 found in `trace` (`modes()` re-implementing `optionTable()`), so the helper is missing in `space`.
fix: one `optionTable`-side builder in `space/tables.ts`, `knobs(space, config): (name) => number` (plus `knob<T>(…)` for string-valued knobs such as the metric), and `option(description, space, build)` returning an `Option`; the seven files import it. Casts at the use sites disappear.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: yes (adds one export to `space`)

### F16: iterative routines that are not `Algorithm`s, and a Lloyd loop that duplicates `cluster/kmeans.ts`

kind: rule
evidence: CLAUDE.md "Every iterative routine ships as an `Algorithm`"; loops without the contract: linalg/functions.ts:162-177 (`sqrtm`, Denman-Beavers, `maxIterations`/`tol` options, two `inv` per step), linalg/equations.ts:128-140 (`signm`, the cap 100 and the tolerance 1e-14 hard-coded, no options), :161-170 (`solveCare` through `signm`), quadrature/newton-cotes.ts:174-203 (`romberg`), quadrature/adaptive.ts:315-358 (`adaptiveSimpson`) and :373 (`tanhSinh`), approximation/pade.ts:67-90 (`continuedFraction`), spatial/quantisation.ts:47-131 (`kmeansCodebook`: Lloyd iterations, history, tolerance), linalg/randomized.ts (power iterations); `solveDare` (equations.ts:233), `chebfun`, `remez`, the eigen iterations, the Krylov and root finders are `Algorithm`s; `kmeansCodebook` is used by `cluster/kmeans.ts:136` and :182 as a _step_ (`maxIterations: 0` and `1`) and by `learn/model-selection/blocked-splitters.ts:330`, while `cluster/kmeans.ts:158` has `lloydAlgorithm` and :231 `miniBatchKMeansAlgorithm`.
what: two Lloyd implementations (one an Algorithm with a tape, one a plain loop with its own history array) and eight loops with no tape, so the "show training over time" views cannot show `sqrtm`, `signm`, `romberg` or the codebook training; `kmeansCodebook`, `sqrtm` and `signm` also differ from their neighbours in option names (`maxIterations`/`tol` here, none there).
fix: `kmeansCodebook` becomes a thin call of `lloydAlgorithm` (move the shared nearest-centre kernel below `cluster`, into `spatial`, which `cluster` already imports); give `sqrtm`/`signm` a `sqrtmAlgorithm`/`signmAlgorithm` pair in the style of `solveDareAlgorithm` and a shared `MatrixIterationOptions`; `romberg`, `adaptiveSimpson`, `tanhSinh` lowest priority (they are short, fixed-structure quadrature rules; say so in the folder summary as an accepted exception).
size: M (1-3h) for the k-means and the two matrix functions; S for the summary
risk: low-medium (`kmeansCodebook` results must stay bit-identical for the product quantiser and IVF fixtures; keep the seeding order)
preserves behaviour: yes
changes public surface: yes (new exports; `kmeansCodebook` keeps its signature)

### F17: local copies of generic helpers inside the folder

kind: duplicate
evidence: dense row-major matvec written inline at linalg/eig.ts:147-155 (`matvec`), linalg/randomized.ts:158-159 and linalg/subspace.ts:54-55, where `dgemv(a, m, n, x)` exists in linalg/blas.ts:148; CSR matvec written at sparse/multigrid.ts:284-293 and in sparse/_rows.ts (the sweeps) besides `spmv` (F9); the `numeric(op)` wrapper at sparse/eigen.ts:25-29 and sparse/iterative.ts:32-34 (duplicates-strict.txt); `transpose` at fourier/nonuniform.ts:251-256 and interpolate/smoothing.ts:153-157 (also stats/_linear.ts, symbolic/matrices.ts, timeseries/_matrix.ts; the tensor op is `tensor/ops/shape.ts`); `norm1` at linalg/equations.ts:46 and linalg/functions.ts:35 (identical one-liners over `max(sum(abs()))`); `binomial` at quadrature/multidim.ts:199-203 against `binom` in special/_kernels.ts; `EPS = 2.220446049250313e-16` at roots/scalar-algorithms.ts:25, linalg/_kernels.ts:6, linalg/_schur.ts:5, differentiate/finite-differences.ts:26, special/_kernels.ts:7 and the bare literal in linalg/{svd,basics,functions}.ts, approximation/approximation-options.ts (10 places; 15 files in the core define it, findings.md), none uses `Number.EPSILON` (no occurrence in the folder); `nextAfter` (approximation/floating-point.ts:37) is the one numerics version, with two copies in `dynamics/ode/_vec.ts` and `optim/minimize/least-squares-trf.ts` that should import it; `identity` at inverse/tikhonov.ts:163 against linalg/_matrix.ts:95; `rowsOf` names two different things in sparse/csr.ts and sparse/_rows.ts.
what: small but exactly what the "no local copies of generic helpers" rule forbids, in the folder that owns `blas.ts`.
fix: import `dgemv`, `spmv`/the F9 kernel, the tensor `transpose`, one `norm1` in `linalg/_matrix.ts`, `special/_kernels.ts` `binom`; one `MACHINE_EPSILON` in `base/numbers.ts` (value `Number.EPSILON`) that the 15 files import.
size: S (<1h) for the numerics part
risk: low (identical values; the `dgemv` replacement keeps the summation order for the row-major case, check the power-iteration goldens)
preserves behaviour: yes
changes public surface: yes (one new `base` export)

### F18: complex-pair arithmetic exists in four places, one of them in `differentiate`

kind: duplicate
evidence: `cadd`/`cmul`/`cdiv` identical at polynomial/rational.ts:13-16 and symbolic/_numeric.ts:19-21 (duplicates-strict.txt); the same operations in special/_misc.ts:13-18 (`cmul`, `cdiv`, `cadd`); `complexOps` and `ComplexNumber` exported from differentiate/finite-differences.ts (about line 327; used by linalg/functions.ts:29 for `funm`); `Complex` from exposition/complex-plane.ts:11 (`readonly [re, im]`); the tensor package has `ComplexValue` (`{re, im}`); `cpow` locally at fourier/nonuniform.ts:257; three shapes for one thing (`C`, `Complex`, `ComplexNumber`, findings.md "Type definitions written more than once").
what: complex scalar arithmetic belongs next to the complex tensor type, not in the numerical-differentiation file; `linalg` importing from `differentiate` for it is a sideways dependency only because of that placement.
fix: one `ComplexNumber` type and one `complexOps` (add, sub, mul, div, pow, exp, log, sqrt, abs) in `base` or `tensor/complex.ts`; `differentiate` re-exports nothing; the four copies import it; `Complex` in `exposition` becomes an alias.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: yes (an export moves; `numerics/differentiate` loses `complexOps` unless re-exported)

### F19: native arrays in public signatures, classified (92 functions)

kind: native-array
evidence: `node scripts/review/signatures.mjs` (a copy that prints every row) read against the signatures; classification below; "callers" counts importers outside numerics found by `grep -w` on the name (indicative, not exact).
- **kernel, allowed (12)**: linalg/lapack.ts `dpotrf`, `dpotrs`, `dtrsv`, `dgesv`, `dsyev`, `dgeqrf`, `dgesvd` (documented plain-array LAPACK layer; callers `causal/*`, `nn/additive`, `linear/mixed.ts`: dpotrf 20 files, dpotrs 15, dsyev 14, dgesv 8, dgesvd 7); approximation/floating-point.ts `compensatedSum` (`ArrayLike<number>`, an error-free-transformation kernel); fourier/helpers.ts `dft` (the O(n^2) reference); linalg/qr.ts `householderReflector` (a LAPACK `dlarfg` kernel); linalg/eigh.ts `jacobiEigenSteps` and linalg/svd-steps.ts `jacobiSingularValues` (teaching routines whose result is the step trace). State the exception in `linalg/index.ts`'s summary (it already says "BLAS and LAPACK-style kernels on plain number arrays") and add `householderReflector`, `givensRotation`.
- **shape, axis, index or ragged data (20)**: convolution/n-d.ts `im2col`, `col2im` (kernel and output sizes); convolution/one-d.ts `correlationLags` (returns lag indices), `fftConvolve` (`axes`); fourier/fft.ts `fftn`, `ifftn` (`shape`, `axes`); fourier/helpers.ts `fftshift`, `ifftshift` (`axes`); fourier/nonuniform.ts `modes` (mode indices); geometry/polygons.ts `convexHull` (vertex indices); sparse/csr.ts `fromCoo`, `csr` (index arrays); sparse/multigrid.ts `prolongation` (grid sizes); spatial/brute.ts `radiusNeighbours` (ragged per-query lists), spatial/graphs.ts `nnDescentAlgorithm`, `nnDescent` (`truth` indices); spatial/hashing.ts `minHash`, `estimateJaccard`, `minHashBanding`, `probeSequence` (ragged integer sets and band pairs).
- **public vector or matrix, migrate (60)**: polynomial/power.ts 10 (`polytrim`, `polyder`, `polyint`, `polyadd`, `polysub`, `polymul`, `polydiv`, `companion`, `polyroots`, `polyFromRoots`; callers: `timeseries/{differencing,_arima-model,arma}.ts`, `foundations/series.ts`, `systems/{margins,stability,_shared}.ts`, `signal/parametric.ts`, `symbolic/series.ts`, a gallery recipe) while `polyval`, `polyvander`, `polyfit` in the same file already take `TensorLike`; geometry/transforms.ts 6 (`rotationFromQuat`, `rotationFromRotvec`, `rotationFromEuler`: fixed-size 3- and 4-vectors, `homography`, `fundamentalMatrix` return `number[][]`, `applyHomography(h: number[][])`; callers `vision/{calibration,geometry}.ts`); interpolate/polynomial.ts 6 (`barycentric` weights, `barycentricWeights`, `lagrange`, `chebyshevNodes`, `equispacedNodes`, `lebesgueConstant`; `barycentric(x: TensorLike, y: TensorLike, weights?: number[])` mixes both); approximation/acceleration.ts 5 (`richardson`, `aitken`, `wynnEpsilon`, `shanks`, `eulerTransform`: sequences of scalars; caller `foundations/_extrapolate.ts`); inverse/parameter-choice.ts 4 (`lambdaGrid`, `lCurve`, `gcv`, `quasiOptimality`: `lambdas` grids; caller `learn/model-selection/criteria.ts`) and inverse/tikhonov.ts `tikhonovPath`; differentiate/finite-differences.ts 3 (`jacobianFd`, `hessianFd`: the point `x`; the callbacks `(x: number[]) => number[]` are by design flat; `finiteDifferenceWeights(order, stencil: number[]): number[]`); interpolate/splines.ts 3 (`notAKnot`, `makeInterpSpline` `knotVector`, `makeLsqSpline` `t`, whose `w` is already `TensorLike`); polynomial/orthogonal.ts 3 (`orthoToPower`, `powerToOrtho`, `monicFromNodes`); polynomial/rational.ts 3 (`residue`, `residuez`, `invres`); sparse/eigen.ts 3 (`lanczos`, `arnoldi`, `arnoldiAlgorithm`: `v0: ArrayLike<number>`, `deflate: Float64Array[]`); approximation/chebyshev.ts 2 (`chebyshev`, `chebyshevCoefficients`); geometry/contours.ts 2 (`contourLevels` returns `number[]`, `isolines` takes `levels`); geometry/polygons.ts 2 (`windingNumber`, `pointInPolygon` return `number[]`); approximation/best.ts `chebyshevSeries`, approximation/floating-point.ts `propagateError`, approximation/pade.ts `pade`, interpolate/piecewise.ts `realRoots`, interpolate/smoothing.ts `makeSmoothingSpline` (`w`), sparse/csr.ts `primalValues` (returns the primal values; a kernel in practice), sparse/multigrid.ts `multigridCycle` (`x: ArrayLike`, returns three `number[]`).
  what: 60 of the 92 are real migration candidates (accept `TensorLike`, return `Tensor`/`Value`), 12 are the allowed kernel layer, 20 are not vectors. Seven callback or result types also say `Float64Array`: `Preconditioner = (r: Float64Array) => Float64Array` (sparse/iterative.ts:28), `KrylovState.x`, `LanczosResult.basis`, which are internal state of the iterations and can stay.
  fix: migration staged by file, most-used first. For each function: widen the parameter to `TensorLike`, read it once with `toFlat(concreteOrThrow(promoteFloat(x)[0]))` at entry (as `kmeansCodebook` already does), return `fromStorage(Float64Array, shape, 'float64')`. Widening accepts every `number[]` caller, so the parameter changes are source compatible; the _return_ changes (`number[]` to `Tensor`) are the breaking ones and touch about 25 call sites in `timeseries`, `systems`, `signal`, `vision`, `symbolic`: do `polynomial/power.ts` plus `rational.ts`, `orthogonal.ts` first (16 functions, one coefficient convention, see F20), then `geometry/transforms.ts` (the 3- and 4-vectors become `Vec3`/`Quat` tuples rather than tensors if the canonical small-vector types of findings.md are adopted), then `interpolate`, `approximation`, `inverse`. Leave the `jacobianFd` callbacks flat.
  size: L per file group (3 to 6 hours each for the three first groups, including the callers)
  risk: medium (return-type changes ripple; parameter widening is low)
  preserves behaviour: yes (values), no (return types)
  changes public surface: yes (the deliberate break; parameters widen, returns change)

### F20: polynomial coefficient order differs between modules, and there are two polynomial root finders

kind: consistency
evidence: lowest degree first: polynomial/power.ts:35 (numpy.polynomial), polynomial/orthogonal.ts:64, approximation/pade.ts:13; highest degree first: polynomial/rational.ts:11 (scipy.signal `residue`), interpolate/piecewise.ts:35 and :67 (`PPoly`, scipy), interpolate/polynomial.ts:108-113 (`lagrange`, scipy), interpolate/piecewise.ts:202-203 (`realRoots`: "sum_j a_j t^(n - j) (highest first)"); roots: polynomial/power.ts:328 `polyroots` (companion eigenvalues, `Complex`) and interpolate/piecewise.ts:203 `realRoots` (companion eigenvalues polished with Newton, real only).
what: each file copies the convention of the scipy/numpy function it mirrors, so `polyval(x, lagrange(xs, ys))` is wrong unless the caller reverses the array, and nothing in a type says which order a `number[]` is in. This is the single most likely cause of a wrong-by-reversal bug at a module boundary.
fix: introduce `type Polynomial = { readonly coefficients: Tensor }` (or a branded `LowestFirst`) for the `polynomial` module; make the scipy-mirroring functions return it (or document the reversal in their `@returns` and provide `reverse` in the one-line `@example`); have `realRoots` call `polyroots` and filter. Combine with F19 so the migration changes each signature once.
size: M (1-3h) on its own; folded into F19 otherwise
risk: low-medium
preserves behaviour: yes if the types are added without reordering
changes public surface: yes (types)

### F21: tolerance and iteration-cap vocabulary, and default tolerances, differ for the same job

kind: consistency
evidence: option names by subfolder: `tol` (approximation, linalg eigen iterations, subspace, completion, sparse/eigen, spatial/quantisation, open-ended roots: absolute), `rtol`/`atol` (Krylov, stationary, multigrid, `sparseSolve`), `xtol`/`rtol` (bracketing roots), `xtol`/`ftol` (`fsolve`), `epsabs`/`epsrel`/`limit` (`quad`), `tol`/`rtol`/`divmax` (`romberg`), `tol`/`maxDepth` (`adaptiveSimpson`), `tol`/`maxLevel` (`tanhSinh`), `initialStep`/`stepFactor`/`maxLevels` (`derivative`), `relativeStep` (`jacobianFd`), `rtol` (`pinv`, absolute `tol` in `matrixRank`, `rcond` in `nullSpace` and `orth`), `maxIterations` in 16 files; defaults for the same Krylov-type job: `cg`/`gmres` `rtol` 1e-5 (sparse/iterative.ts:74, :122, scipy), stationary 1e-8 (stationary.ts:32), `sparseSolve` 1e-12 (iterative.ts:662), `sqrtm` 1e-14 (functions.ts:167), `signm` 1e-14 hard-coded (equations.ts:139), `chebfun` `10 eps`; whether `tol` is relative or absolute is stated per function and differs (power iteration: relative to max(1, |lambda|); Remez: relative; k-means: relative change; open roots: absolute).
what: most of this is deliberate scipy mirroring (`epsabs`, `limit`, `divmax`, `rtol`/`atol`), which is a defensible choice; the cost is that no two functions can be swapped by renaming one option, and `rcond`/`rtol`/`tol` for the rank cutoff of three SVD-based functions differ.
fix: no renames. Document the convention in the folder summaries ("tolerances follow the scipy function mirrored; every iteration takes `maxIterations`, and `Algorithm`s expose it as `maxSteps`"); unify the SVD cutoff to `rtol` (`matrixRank(a, tol)` is the odd one: absolute); make `signm` take `MatrixIterationOptions` like `sqrtm`.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: yes (one added option; `matrixRank`'s `tol` kept, `rtol` added)

### F22: function-argument conventions differ between the scalar and the tensor APIs

kind: consistency
evidence: roots/scalar.ts `f: (x: number) => number`; differentiate `derivative(f: (x: number) => number)`; quadrature/adaptive.ts:316, newton-cotes.ts:166, gauss.ts `fixedQuad` `f: (x: Value) => Value` (vectorised over a 1-D Value); roots/systems.ts `VectorFunction` over `Value`; differentiate `jacobianFd(f: (x: number[]) => readonly number[])`; approximation `chebfun`/`remez` take number-valued functions; `adaptiveSimpson` and `romberg` unwrap `Value` results by `.data[0]` / `Array.from(...data)` (adaptive.ts:324, newton-cotes.ts:186-187; the second casts to `{ data: ArrayLike<number> }`).
what: the same user function `Math.cos` works with `brentq` and `derivative` and needs a wrapper (`x => cos(x)` on tensors) for `quad`, and the other way round for `fsolve`; a user cannot tell from the module name which.
fix: accept both in the scalar-function entry points of `quad`, `romberg`, `tanhSinh`, `adaptiveSimpson` (probe the callback once with a 1-element tensor; if it returns a number, vectorise by a loop) or document the rule once in `quadrature/index.ts`. Low priority; not a defect.
size: S
risk: low
preserves behaviour: yes
changes public surface: yes (parameter types widen)

### F23: small degenerate-input defects in `approximation`, `inverse`, `differentiate`

kind: consistency
evidence: probe: `ulp(Number.MAX_VALUE)` is Infinity (approximation/floating-point.ts:55-61, `nextAfter(a, Infinity) - a` is `Infinity - a`; numpy `spacing` gives 1.99e292); `shanks([1,1,1,1,1])` gives [NaN, NaN, NaN] while `aitken` of a constant sequence gives [1, 1] (approximation/acceleration.ts:57 guards a zero denominator, :118 does not); `wynnEpsilon([1,1,1,1])` has NaN columns though `value` is 1; `tikhonov(spectralData(zero matrix, b), 0)` and `tikhonov` at lambda 0 with a zero singular value give [NaN, NaN] (0/0 in `s / (s^2 + lambda^2)`, inverse/tikhonov.ts:48-70; the pseudo-inverse value is 0); `lambdaGrid` of zero data gives [0, NaN, NaN] (log of 0, inverse/parameter-choice.ts:16); `finiteDifferenceWeights(1, [0, 0, 1])` (a repeated node) gives [NaN, NaN, NaN] with no error (differentiate/finite-differences.ts:286) while `order >= n` throws; `derivative` uses an absolute initial step 0.5 (the scipy default): `derivative(x => x*x, 1e16)` returns `df: 0, error: 0` (the true value is 2e16; `f(x + 0.5) === f(x - 0.5)`), `derivative(Math.abs, 0)` returns `0` with `error: 0` at the kink; `monteCarlo` with `n: 0` fails with "data of length 0 does not fill shape [, 2]" (quadrature/monte-carlo.ts); `homography` of collinear points returns a matrix of NaN and Infinity (geometry/transforms.ts:461) while `delaunay` of collinear points throws; `interp1d`/`regularGridInterpolator` throw `value` outside the range (scipy parity) but give NaN for a NaN query.
what: each is a one-to-five-line guard; together they are the "degenerate inputs" class on the less-used routines.
fix: `ulp`: use `x - nextAfter(x, -Infinity)` for the largest finite double; `shanks`: the same zero-denominator guard as `aitken`; `tikhonov`: define 0/0 as 0 in the filter factor (`s > 0 ? s / (s*s + l*l) : 0`); `lambdaGrid`: reject all-zero spectra with `value`; `finiteDifferenceWeights`: throw `value` on repeated nodes; `derivative`: scale the initial step by `max(1, |x|)` when `initialStep` is not given and report `error: Infinity` when the first two central differences are identical but non-zero (or accept this as documented scipy behaviour); `homography`/`monteCarlo`: throw `value` with a message naming the cause.
size: S (<1h) in total
risk: low
preserves behaviour: no (NaN or 0 results change; `derivative`'s default step changes for |x| > 1)
changes public surface: no

### F24: TSDoc: 22 exports have only a summary line

kind: rule
evidence: `tsdoc.mjs` over the 502 exported functions of non-private modules: no `@param`, `@returns` or `@example`: fourier/nonuniform.ts:34 `modes`, fourier/trig.ts:276 `sineWindow`, geometry/polygons.ts:257 `isConvex`, geometry/triangulation.ts:30 `orient`, :35 `inCircle`, :49 `circumcentre`, :61 `toPoints`, :347 `simplexTensor`, interpolate/piecewise.ts:203 `realRoots`, interpolate/polynomial.ts:94 `barycentricWeights`, :166 `equispacedNodes`, interpolate/splines.ts:453 `notAKnot`, inverse/parameter-choice.ts:16 `lambdaGrid`, inverse/tikhonov.ts:155 `secondDifference`, and eight more of the same kind (the script lists the first 14); `fftshift`/`ifftshift` are flagged by the script only because of their overload signature (their docs are complete).
what: CLAUDE.md: "every core export has TSDoc: summary, `@param`, `@returns`, and an `@example` whose `// => value` lines are checked". The 22 are small helpers (several the `orient`/`inCircle` predicates that F6 is about).
fix: add the three tags and a one-line example to each.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: no

### F25: cast density in `special/` has one cause, the untyped partial-derivative tuple

kind: rule
evidence: special/_elementwise.ts:36-37 (`type Partial = (xs: readonly Value[], out: Value) => Value` and `special(name, kernel, partials)`) forces `xs[i]` to be `Value | undefined` under `noUncheckedIndexedAccess`, so every rule writes `([a, x]) => gammaincDa(a as Value, x as Value)`: `as Value` is 18 of 19 casts in special/incomplete.ts, 12/13 in bessel.ts, 15/18 in gamma.ts, 12/15 in elliptic.ts, 10/15 in zeta.ts, 7/12 in normal.ts, 6/14 in cdf.ts (about 80 casts, the densest files of findings.md: 21, 15, 15, 12 and 11 per 100 lines); the `as unknown as` JSON-state casts are a second group: linalg/equations.ts:277, subspace.ts:182, completion.ts:150 and :603, svd-steps.ts:170, approximation/chebyshev.ts:288, best.ts:200, interpolate/smoothing.ts:292, sparse/lanczos-steps.ts:181 (the same pattern as foundations F21).
what: the casts hide nothing wrong, but they bury the real ones and are a one-cause problem.
fix: type `special` by arity (`special<N extends 1 | 2 | 3>(name, kernel, partials: PartialTuple<N>)` with `Partial<N> = (xs: Tuple<Value, N>, out: Value) => Value`, or three overloads `special1`/`special2`/`special3`), which removes the `as Value` in 80 places without a runtime change; the deserialise casts go with foundations F21's typed state codec.
size: S (<1h) for the `special` typing
risk: low (types only)
preserves behaviour: yes
changes public surface: no

### F26: test gap: scale, Infinity, denormal and singular-input tables for `linalg`, `special`, `roots`

kind: test-gap
evidence: coverage (Phase 0, `numcov.mjs` on coverage-final.json): numerics 94.0% statements, 71.1% branches; lowest branch coverage in sizeable files: special/zeta.ts 25% (4 branches), approximation/acceleration.ts 50%, fourier/_kernel.ts 57% (122 branches), linalg/functions.ts 59%, linalg/qr.ts 61%, linalg/_kernels.ts 61% (383 branches), geometry/transforms.ts 61%, spatial/quantisation.ts 62%, special/_incomplete.ts 69% (265) and 78% of statements; the uncovered lines of special/_incomplete.ts are 107, 109, 116-118, 125-126, 133-134, 141-142, 153-158, 163-167, 174-191 (the `invalidGamma`, `x === 0`, `x === Infinity` returns of every function), of special/_elliptic.ts 7-9, 17-19, 141-155, of linalg/_kernels.ts 369 (eigh non-convergence), 384-387, 402, 632-650, 660 (Golub-Kahan zero-diagonal and non-convergence), of special/zeta.ts 38-46, 117-120; the generators in the property tests keep to ordinary scale (as foundations F31).
what: F1, F2, F4, F5 and F11 are all caught by a single table of degenerate inputs per domain, and nothing tests that table today.
fix: the smallest set, as `it.each` tables (no new fixtures; expected values from closed forms or from the oracle for the special functions):
1. `linalg/degenerate.test.ts`: for `svd`, `eigh`, `qr`, `norm`, `det`, `cond`, `matrixRank`, `pinv`: scales 1e-200, 1e-155, 1e-5, 1, 1e75, 1e80, 1e200 relative to the unscaled result (8 functions x 7 scales = 56 rows, or 8 `it`s with a loop); singular, near-singular, rank-deficient (`solve`, `inv`, `cholesky`, `lu`, `lstsq`, `pinv`, `solveTriangular`, `choSolve`: 8 x 3 = 24 rows); NaN and Infinity entries for the 10 factorisations (10 rows); 0 x 0 and 1 x 1 for 10 functions (10 rows).
2. `special/limits.test.ts`: one row per function at `-Infinity, -1, -0, 0, 5e-324, 1e-310, 1, 1e308, Infinity, NaN` against the value scipy documents for the 40 most-used functions (about 120 rows, expected values from `oracle/cases/special`).
3. `roots/scale.test.ts`: 4 finders x scales {1e-300, 1e-200, 1e-160, 1, 1e300} x {linear, cubic} and the NaN function (44 rows); `quadrature/degenerate.test.ts`: NaN, Infinity at an endpoint, reversed limits, empty interval for 5 integrators (20 rows, with a 1-second timeout so F3 fails instead of hanging).
   Total about 290 table rows in 5 files; as `it.each` they count as one test per row (+290), or as 5 to 15 tests if each table is one `it` with a loop; offset by merging redundant cases elsewhere, net count to be reported.
   size: M (1-3h) for the harness and the first two tables; S for the others
   risk: low (tests only; some rows fail until F1-F5 land, so land the tests with the fixes)
   preserves behaviour: yes
   changes public surface: no

### F27: test gap: geometry and spatial degenerate configurations, and output post-conditions

kind: test-gap
evidence: geometry/transforms.ts 61% branches, geometry/distances.ts 69%, geometry/triangulation.ts and polygons not listed below 70% but the F6 failure shows no test builds a regular polygon (a probe with n = 4, 5, 6 passes, n = 8 fails); spatial/quantisation.ts 62%, hashing.ts 63%, graphs.ts 69%; the tie test between brute force and the three trees exists (probe agrees) but the NaN, duplicate and one-point cases do not (`kdTree` with a NaN row is unsorted, F11); `convexHull` NaN (F11).
what: the geometry routines have exact properties that make cheap oracles (sum of simplex areas equals hull area; Euler's count `2n - 2 - h` triangles; every point lies in some simplex; Voronoi cells partition) and none is asserted.
fix: `geometry/triangulation.property.test.ts`: for random points, regular n-gons (n = 3..40), lattices, duplicated points and points on a circle with 1e-9 noise (6 generators x 4 properties = 24 properties); `spatial/neighbours.property.test.ts`: brute, KD, ball and VP trees agree on indices for random data with injected duplicates and ties (4 structures, 3 generators); a NaN row sorts last in all of them.
size: M (1-3h)
risk: low
preserves behaviour: yes
changes public surface: no

### F28: test gap: 24 named functions never run, and the non-convergence branches of the kernels

kind: test-gap
evidence: functions with zero calls in the engine suite (coverage-final.json, named ones only; 86 more are anonymous closures): approximation/acceleration.ts `shanks`, `eulerTransform`; approximation/best.ts `chebyshevSeries`; approximation/floating-point.ts `split`, `twoProduct`, `conditionNumber`, `propagateError`, `significantDigits`; geometry/contours.ts `isolines`; linalg/basics.ts `isSymmetric`, `isPositiveDefinite`; linalg/cholesky.ts `choFactor`; linalg/eigh.ts `jacobiEigenSteps`; linalg/qr.ts `householderReflector`, `givensRotation`; linalg/triangular.ts `invTriangular`; polynomial/orthogonal.ts `monicFromNodes`; polynomial/power.ts `polysub`; sparse/csr.ts `csrDiagonal`, `nnz`; special/_incomplete.ts `gammaincDx`, `logTailNewton`, `betaincDx`; special/bessel.ts `i1`; `@example` blocks run elsewhere (`make examples`), so some are exercised there; a function with neither is the gap. The non-convergence throws at linalg/_kernels.ts:369 and :660, _schur.ts:89, roots/systems.ts:542, sparse/iterative.ts:677 and equations.ts:140, :310 have tests only where a failing configuration is easy (check).
what: `polysub` unrun is the clearest: a one-line function with a sign. `i1` (a Bessel function, unrun) and `gammaincDx`/`betaincDx` (the derivative kernels the implicit-reparameterisation gradients use) are the ones that matter.
fix: one `it` per function, using the closed form or an identity (`i1`: against the Wronskian-free series at 3 points and `iv(1, x)`; `polysub(a, a) = [0]`; `isPositiveDefinite` of [[1,2],[2,1]] false; `householderReflector([3,4]).alpha = -5` already an `@example`): 24 tests; a non-convergence test per throw site by passing `maxIterations: 1` or a known-bad input (7 tests).
size: S (<1h) for the 24, M with the throw sites
risk: low
preserves behaviour: yes
changes public surface: no

### F29: layering: two type-only upward imports, one sideways placement, and approximate-NN code that is not numerics

kind: layering
evidence: fourier/nonuniform.ts:8 and polynomial/rational.ts:8 import `type Complex` from `exposition` (rank above numerics in the first-guess order; type-only, no runtime edge); linalg/functions.ts:29 imports `complexOps` from `differentiate` (F18); inter-folder edges otherwise follow a DAG (`sparse -> linalg` 6, `spatial -> geometry` 5, `interpolate -> linalg` 4, `geometry -> linalg` 3, `approximation -> linalg` 3, `roots -> linalg`, `inverse -> linalg`, `interpolate -> geometry`, `spatial -> special`, `spatial -> sparse`, `quadrature -> special`, `quadrature -> linalg`, `polynomial -> quadrature`, `polynomial -> linalg`, `convolution -> fourier`, `approximation -> polynomial`, `approximation -> fourier`; no edge into `special`, `sparse` or `fourier` from below them); no import of `optim`, `dists`, `losses`, `stats`, `metrics`, `kernels`, `data` (checked by grep over all 112 files); `spatial/{graphs,hashing,quantisation}.ts` (HNSW, NN-descent, LSH, MinHash, product quantisation, IVF with its own k-means) is about 1,850 lines of approximate-nearest-neighbour _indexes_ with `Key`s, trained codebooks and `Algorithm`s, used by `cluster`, `learn`, `manifold`.
what: nothing breaks a rule; the `exposition` import is the one that would break a stricter order (it is the `Complex` type, F18 removes it); the `spatial` indexes are a placement question for `docs/structure.md`, not a defect: they are retrieval structures that a `neighbors` or `retrieval` domain would own, and they pull `random`, `space` and `trace` into numerics.
fix: with F18 the `exposition` import disappears. Leave `spatial` where it is; if `docs/structure.md` section 12 grows a `retrieval` domain, move `graphs`, `hashing`, `quantisation` there and keep `brute`/`trees` (exact search, no randomness) in `numerics/spatial`.
size: S (<1h) for the import; M for the optional move
risk: low
preserves behaviour: yes
changes public surface: only if the optional move is made (entry points change; the surface check lists it)
