# omai: structure

This document is the table of contents of the whole library as designed: every package, folder and file, with what
each file contains. It is written for review before implementation. It does not describe any existing code. Read it top
to bottom: it starts at the bottom of the dependency graph (errors, tensors, random keys) and climbs one layer at a time
to neural networks, reinforcement learning, the named methods built on them, the render layer, the metadata layer and
the two apps.

Nothing here has to be built in one go. The point is that every concept has a known home before it is written, so the
library can grow for years without reorganising.

## 1. Shape of the repository

```
omai/
  packages/
    engine/      omai-engine   compute: tensors to reinforcement learning, plus datasets           (published)
    render/      omai-render   headless figures: ECharts options, layouts, scene graphs, SVG; React (published)
    meta/        omai-meta     docs metadata: schema, docstring extractor, example runner, worker  (published)
  apps/
    docs/        how to use omai-engine: how-tos, API reference, live examples, trace viewer      (private)
    gallery/     what omai-render can draw: galleries of charts, graphs, diagrams, environments    (private)
  oracle/        Python ground truth: generates golden fixtures and vendors static datasets (uv)
  scripts/       repository tooling in TypeScript (barrels, layering, docs build, packing, smoke test)
  docs/          structure.md (this file), decisions.md, extending.md, conventions.md
  Makefile, mk/  the single interface for all routine work
  CLAUDE.md, INIT.md, AUDIT.md, PORT.md
```

`docs` is the counterpart of `engine` and `gallery` is the counterpart of `render`. Outside projects (the encyclopedia,
the slide and blog tool, the separate documentation site) consume the three published packages as release tarballs.

## 2. Conventions every file follows

**Packages and entry points.** Every folder is an entry point. `packages/engine/src/lib/numerics/linalg/` is imported
as `omai-engine/numerics/linalg`; its parent `omai-engine/numerics` re-exports it as the namespace `linalg`.
Index files (`index.ts`) are generated from the folder contents and never edited, so adding a file never edits another
file.

**Core and applications.** `packages/engine/src/lib/` holds the core: general machinery (tensors, autodiff, solvers,
distributions, estimator protocol, layers, the environment interface). `packages/engine/src/methods/` holds applications
of that core: named models and algorithms (PCA, random forests, GPT, CartPole, TrueSkill), datasets and studies. Methods
import the core; the core never imports methods.

**Layering.** Each core domain has a `domain.json` with a tier. A domain imports only domains of a lower tier. Each
methods area has an `area.json` listing the other areas it may import. A lint (dependency-cruiser rules generated from
these files) enforces both.

**File size.** A file groups related exports around one idea: typically 3 to 12 exports and 100 to 400 lines. A file
past 500 lines is split by sub-idea; a lone tiny function joins its siblings. The file lists below are drawn to that
size.

**Procedural core, thin classes.** Functions take explicit data and return new data. Randomness comes from explicit
`Key` arguments. No module-level mutable state, no registration side effects at import. Familiar class APIs
(`new PCA({ nComponents: 2 }).fit(X).transform(X)`, `new nn.Linear(4, 8)`, `env.step(action)`) are thin wrappers over
the functions, at most one level of inheritance, and both forms are documented.

**Iterative algorithms.** Everything that iterates (optimisers, training, EM, samplers, message passing, solvers,
agents, tree growth) is written as an `Algorithm` (`init`, `step`, `done`, `metrics`, `params`) and comes in two
spellings: `kmeansAlgorithm(...)` returns the algorithm, `kmeans(...)` runs it and returns the result. Running an
algorithm through `runAlgorithm` emits typed trace events. Each algorithm also has an untraced fast path (the same step
function with a disabled tracer), so tracing never makes an algorithm asymptotically slower.

**Estimators.** Supervised and unsupervised models follow one protocol: `fit(hyper, data, ctx) -> model` where the
model is plain data, and pure functions `predict`, `predictProba`, `transform`, `score`, `sample` that take the model.
A model advertises what it can do by which of these exist. sklearn-style classes wrap this.

**Names.** Function and option names follow the most familiar library for the concept: numpy and scipy for numerics,
scikit-learn for estimators, PyTorch for layers and modules, optax for optimiser transformations, gymnasium for
environments, NetworkX for graphs, Pyro and NumPyro for probabilistic programming, pgmpy for graphical models. Spelling
is camelCase (`nComponents`, `predictProba`). Prose uses British English; identifiers keep the established spelling
(`normalize` stays `normalize` only where the familiar API spells it so).

**Docstrings.** Every export has a TSDoc block: summary, `@param` for each parameter, `@returns`, at least one
`@example` with `// => value` (or `// => figure`, `// => trace`) expectations that the build checks, and optionally
`@concept`, `@see` and `@reference`.

**Tests next to code.** For `foo.ts`: `foo.test.ts` (unit), `foo.property.test.ts` (fast-check properties such as
gradients against finite differences, distribution identities, decomposition reconstruction), `foo.golden.test.ts`
(against fixtures from the Python oracle) and `foo.test-d.ts` (type tests). Golden fixtures live in
`packages/engine/fixtures/<same path>/<file>.json`. Browser tests live beside the React code as `*.browser.test.tsx`.

**Python ground truth.** Wherever a reference implementation exists in Python (numpy, scipy, scikit-learn, PyTorch,
statsmodels, NetworkX, pgmpy, gymnasium and others), the oracle has a case file at the mirrored path,
`oracle/cases/<domain>/<file>.py`, that computes reference outputs for fixed inputs. `make fixtures` regenerates the
JSON fixtures; tests read them, so the test suite never needs Python. Each folder below names its oracle libraries.

## 3. omai-engine core, `packages/engine/src/lib/`

The core is 24 domains in 9 tiers. A domain may import only domains in lower tiers.

| tier | domains                                                            |
| ---: | ------------------------------------------------------------------ |
|    0 | `base`, `scene`                                                    |
|    1 | `tensor`                                                           |
|    2 | `random`                                                           |
|    3 | `space`, `trace`, `autodiff`                                       |
|    4 | `numerics`, `data`                                                 |
|    5 | `graph`, `dists`                                                   |
|    6 | `stats`, `optim`, `kernels`, `signal`, `text`, `logic`, `sampling` |
|    7 | `dynamics`, `pgm`, `probprog`, `learn`                             |
|    8 | `nn`, `rl`                                                         |

### 3.1 `base/` (tier 0): what every file may assume

- `errors.ts`: `OmaiError` (one class with a `code`: shape, dtype, value, domain, schema, convergence, singular,
  not-implemented) and `details`; `isOmaiError`, `invariant`, `assertShape`, `assertFinite`.
- `json.ts`: the `Json` type and `JsonSchema`; `toJson`/`fromJson` for values with non-finite numbers
  (`Infinity`, `NaN` encoded as strings) so traces and fixtures round-trip.
- `options.ts`: shared option shapes: `Tolerance` (`atol`, `rtol`, `maxIter`), `Reduction` (`mean`, `sum`, `none`),
  `resolveTolerance`, `defaultTolerance`.
- `version.ts`: `VERSION` (generated from `package.json`, recorded in every trace).
- `numbers.ts`: scalar helpers used everywhere: `isClose`, `allClose` (numpy semantics), `clamp`, `sum` and `kahanSum`,
  `maxOf`/`minOf`/`argmaxOf` (loops, never `Math.max(...spread)`), `range`.

### 3.2 `scene/` (tier 0): drawings as data

Environments, diagrams and graphical models describe what they look like as a scene graph; omai-render draws it.
Living in tier 0 lets the engine produce drawings without depending on the renderer.
- `types.ts`: `Scene`, `SceneNode` (`rect`, `circle`, `ellipse`, `line`, `path`, `polygon`, `text`, `image`, `group`),
  `Style`, `Paint` (theme tokens `fg`, `bg`, `surface`, `muted`, `grid`, `accent`, `tab0`..`tab9`, or literal colours),
  `Transform`.
- `build.ts`: constructors `scene`, `rect`, `circle`, `ellipse`, `line`, `arrow`, `path`, `polygon`, `text`, `group`,
  `image` (from a 2-D tensor or RGBA data).
- `geometry.ts`: `bounds`, `translate`, `rotate`, `scaleTo`, `fit` (fit a scene into a box), `grid` (tile scenes into
  a filmstrip), `isScene`.

### 3.3 `tensor/` (tier 1): n-dimensional arrays (numpy, torch.Tensor)

oracle: numpy
- `dtype.ts`: `DType` (`float64`, `float32`, `int32`, `bool`, `complex128`), `Storage<D>`, `allocate`, `isFloat`,
  `isComplex`, `resultType` (explicit promotion table, numpy rules), `finfo` (`eps`, `max`, `tiny` per dtype).
- `shape.ts`: `Shape`, `sizeOf`, `stridesOf`, `broadcastShapes`, `normaliseAxis`, `normaliseAxes`, `shapesEqual`,
  `ShapeOf<T>` type helpers (literal shapes where cheap).
- `tensor.ts`: the `Tensor<D, S>` type (dtype, shape, contiguous row-major storage, backend), `tensor` (from nested
  arrays), `fromStorage`, `isTensor`, `item`, `toArray`, `toFlat`, `clone`.
- `create.ts`: `zeros`, `ones`, `full`, `zerosLike`, `onesLike`, `fullLike`, `empty`, `arange`, `linspace`,
  `logspace`, `geomspace`, `eye`, `identity`, `diag`, `tri`, `tril`, `triu`, `meshgrid`.
- `backend.ts`: the `Backend` interface (the primitive kernel vocabulary: unary, binary, compare, reduce, scan, matmul,
  transpose, gather, scatter, where, cast, conv) and `BackendCapabilities`. New backends (WASM, WebGPU) implement this.
- `cpu/elementwise.ts`: CPU unary, binary, comparison and `where` kernels with broadcasting by index maps.
- `cpu/reduce.ts`: CPU reductions and scans (`sum`, `prod`, `max`, `min`, `argmax`, `argmin`, `cumsum`, `cumprod`)
  over arbitrary axes.
- `cpu/structure.ts`: CPU `transpose`, `gather`/`take`, `scatter`, `concat`, `pad`, `cast`, `matmul` (2-D and batched).
- `cpu/index.ts`: `cpu`, the reference backend assembled from the three kernel files.
- `ops/arithmetic.ts`: `add`, `sub`, `mul`, `div`, `floorDiv`, `mod`, `pow`, `neg`, `abs`, `sign`, `maximum`,
  `minimum`, `clip`, `reciprocal`.
- `ops/math.ts`: `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `cbrt`, `square`, `sin`, `cos`, `tan`,
  `arcsin`, `arccos`, `arctan`, `arctan2`, `sinh`, `cosh`, `tanh`, `sigmoid`, `relu`, `step`, `floor`, `ceil`, `round`.
- `ops/compare.ts`: `equal`, `notEqual`, `less`, `lessEqual`, `greater`, `greaterEqual`, `isnan`, `isfinite`,
  `isinf`, `logicalAnd`, `logicalOr`, `logicalNot`, `where`, `allclose`, `any`, `all`.
- `ops/reduce.ts`: `sum`, `prod`, `mean`, `var` (two-pass), `std`, `max`, `min`, `argmax`, `argmin`, `cumsum`,
  `cumprod`, `logsumexp` (max-shifted), `softmax`, `logSoftmax`, `norm` (vector and matrix norms).
- `ops/shape.ts`: `reshape`, `flatten`, `ravel`, `transpose`, `swapaxes`, `moveaxis`, `expandDims`, `squeeze`,
  `broadcastTo`, `sumToShape`, `concat`, `stack`, `split`, `chunk`, `flip`, `roll`, `tile`, `repeat`, `pad`.
- `ops/indexing.ts`: `slice` (start/stop/step per axis), `take`, `takeAlongAxis`, `gather`, `scatter`, `scatterAdd`,
  `put`, `maskedSelect`, `nonzero`, `argwhere`, `sort`, `argsort`, `topk`, `searchsorted`, `unique`, `bincount`,
  `oneHot`.
- `ops/linear.ts`: `matmul`, `dot`, `vdot`, `outer`, `inner`, `tensordot`, `einsum` (string spec), `kron`, `trace`,
  `diagonal`.
- `complex.ts`: complex128 helpers: `complex`, `real`, `imag`, `conj`, `angle`, `absComplex`, `polar`.
- `pytree.ts`: `Tree`, `MapLeaves`, `treeLeaves`, `treeMap`, `treePaths`, `treeFlatten`, `treeUnflatten`, `treeZeros`,
  `treeCount` (parameter counts), `ravel`/`unravel` (a pytree to one flat vector and back).
- `format.ts`: `formatTensor` (numpy-style `repr` with precision and summarisation), `toJSON`/`fromJSON` for tensors.

### 3.4 `random/` (tier 2): explicit, counter-based randomness (jax.random, numpy.random)

oracle: Random123 known-answer vectors; numpy for distribution moments
- `philox.ts`: `philox4x32` (Philox4x32-10 block function).
- `key.ts`: `Key`, `key(seed)`, `split(key, n)`, `foldIn(key, data)`, `keyFromPath(seed, ...names)` (named substreams),
  `randomBits`, `isKey`.
- `continuous.ts`: `uniform`, `normal`, `truncatedNormal`, `exponential`, `laplace`, `logistic`, `gumbel`, `cauchy`,
  `gamma` (Marsaglia-Tsang, log-space boost for shape below 1), `beta`, `chisquare`, `studentT`, `dirichlet`,
  `multivariateNormal`.
- `discrete.ts`: `randint`, `bernoulli`, `categorical` (probabilities or logits, Gumbel-max), `multinomial`, `poisson`
  (inversion and PTRS), `binomial`, `geometric`, `aliasTable`/`aliasSample` (Vose).
- `permutation.ts`: `permutation`, `shuffle`, `choice` (with and without replacement, weighted), `bootstrapIndices`,
  `subsample`, `randomSplit`.
- `sequences.ts`: low-discrepancy sequences: `sobol`, `halton`, `latinHypercube`, `scrambledSobol`.
- `generator.ts`: `Generator` class (numpy `default_rng` style, wraps a key and splits on every call), `defaultRng`.

### 3.5 `space/` (tier 3): typed spaces for parameters, observations and actions (gymnasium.spaces, hyperopt)

- `types.ts`: `Space` union: `real` (bounds, log scale, step), `int`, `choice`, `bool`, `box` (tensor bounds),
  `discrete`, `multiDiscrete`, `multiBinary`, `tuple`, `dict`; conditional dimensions (`when`).
- `operations.ts`: `sample(space, key)`, `contains`, `clip`, `defaults`, `flatDim`, `flatten`/`unflatten` values.
- `encoding.ts`: `encode`/`decode` to the unit cube (for Bayesian optimisation), `grid` enumeration, `activeDims`.

### 3.6 `trace/` (tier 3): typed events for every iterative algorithm

- `events.ts`: event types `run`, `step`, `metric`, `param`, `message`, `episode`, `checkpoint`, `end`; `Run`;
  zod schemas `TraceEventSchema`, `RunSchema`.
- `tracer.ts`: `Tracer` (`enabled`, `emit`), `nullTracer`, `recorder`, `tee`, `filter` (by event type or every n
  steps), `startRun` and `RunHandle`.
- `algorithm.ts`: the `Algorithm<S>` protocol, `StepContext` (step index and per-step key), `runAlgorithm`,
  `iterate` (a generator yielding states for live views), `runFor` (time-budgeted slices for the UI).
- `checkpoint.ts`: `checkpointEvery`, `resume` (continue a run from a stored state with the same step keys),
  `seek` (reconstruct step t from the nearest checkpoint).
- `replay.ts`: `exportRun`, `parseRun`, `replay` (feed events to handlers), `mergeRuns`, `isRun`.
- `query.ts`: `metricSeries`, `paramSeries`, `messages`, `episodes`, `runsOf`, `childRuns`, `summary` (final metrics,
  status, steps), `diffRuns` (compare two runs' configurations and outcomes).

### 3.7 `autodiff/` (tier 3): reverse- and forward-mode differentiation (jax.grad, torch.autograd)

oracle: torch.autograd, jax where useful; finite differences for property tests
- `var.ts`: `Var` (immutable graph node: value, op, inputs, vjp), `variable`, `constant`, `asVar`, `isVar`, `node` (the
  extension point for new differentiable ops), `stopGradient`.
- `ops/arithmetic.ts`: differentiable `add`, `sub`, `mul`, `div`, `pow`, `neg`, `abs`, `maximum`, `minimum`, `clip`,
  `where`.
- `ops/math.ts`: differentiable `exp`, `log`, `log1p`, `expm1`, `sqrt`, `square`, `sin`, `cos`, `tanh`, `sigmoid`,
  `relu`, `softplus`, `logSigmoid`, `erf`.
- `ops/reduce.ts`: differentiable `sum`, `mean`, `var`, `max`, `min`, `logsumexp`, `softmax`, `logSoftmax`, `norm`.
- `ops/shape.ts`: differentiable `reshape`, `transpose`, `expandDims`, `squeeze`, `broadcastTo`, `concat`, `stack`,
  `slice`, `take`, `gather`, `pad`.
- `ops/linear.ts`: differentiable `matmul`, `dot`, `outer`, `einsum`, `solve`, `choleskyFactor`, `triangularSolve`,
  `logdet`, `inv` (for small matrices only, documented).
- `reverse.ts`: `backward` (topological sweep, cotangent accumulation), `vjp`.
- `forward.ts`: forward mode with dual numbers: `jvp`, `linearize`.
- `transforms.ts`: `grad`, `valueAndGrad` (over pytrees, leaves labelled by path), `jacobian` (forward or reverse),
  `hessian`, `hvp`, `customVjp`, `checkpoint` (recompute instead of store).
- `vmap.ts`: `vmap` (batching by a per-example loop with a later batch-rule fast path).
- `graph.ts`: `computationGraph` (nodes and edges as data for diagrams, with shapes and optional gradients),
  `graphStats` (node counts by op, parameter counts).
- `check.ts`: `numericalGradient` (central differences), `gradCheck` (report of relative errors per leaf),
  `dotProductTest` (vjp/jvp consistency).

### 3.8 `numerics/` (tier 4): numerical methods (scipy, numpy.linalg)

oracle: numpy, scipy

`special/` (scipy.special)
- `erf.ts`: `erf`, `erfc`, `erfcx`, `erfinv`, `erfcinv`, `dawsn`.
- `gamma.ts`: `gamma`, `logGamma`, `digamma`, `trigamma`, `polygamma`, `beta`, `logBeta`, `factorial`,
  `logFactorial`, `binom`, `logBinom`.
- `incomplete.ts`: regularised incomplete gamma `gammainc`, `gammaincc` and inverses; incomplete beta `betainc` and
  `betaincinv`; log versions.
- `normal.ts`: `ndtr`, `logNdtr`, `ndtri`, `normalPdf`, `normalLogPdf`, `owensT`, `logIntervalProbability`.
- `logistic.ts`: `expit`, `logit`, `softplus`, `logSigmoid`, `log1mexp`, `log1pexp`, `logaddexp`, `xlogy`,
  `xlog1py`, `entr`, `relEntr`.
- `bessel.ts`: `i0`, `i1`, `i0e`, `i1e`, `iv`, `kv`, `j0`, `j1`, `logBesselI` (for von Mises and Matern kernels).
- `zeta.ts`: `zeta`, `hurwitzZeta`, `lambertW`, `expn`, `expi`.
- `cdf.ts`: CDFs and inverses for test statistics: `stdtr`, `stdtrit`, `chdtr`, `chdtrc`, `fdtr`, `kolmogorov`,
  `smirnov`.

`linalg/` (numpy.linalg, scipy.linalg)
- `basics.ts`: `norm`, `cond`, `matrixRank`, `matrixPower`, `trace`, `det`, `slogdet`, `isSymmetric`,
  `isPositiveDefinite`.
- `triangular.ts`: `solveTriangular`, `invTriangular`, back and forward substitution.
- `cholesky.ts`: `cholesky` (with an optional jitter ladder that reports the jitter used), `choFactor`, `choSolve`,
  `choleskyLogdet`, `choleskyUpdate`/`choleskyDowndate` (rank one).
- `lu.ts`: `lu` (partial pivoting), `luFactor`, `luSolve`, `solve`, `inv`.
- `qr.ts`: `qr` (Householder; reduced and complete), `gramSchmidt` (classical and modified, as a traced algorithm),
  `lstsq` (QR and SVD paths), `householderReflector`.
- `eigh.ts`: symmetric eigendecomposition: tridiagonal reduction plus implicit QL, and cyclic Jacobi (traced, for
  teaching); `eigh`, `eigvalsh`.
- `svd.ts`: `svd` (one-sided Jacobi and Golub-Kahan), `svdvals`, `pinv`, `nullSpace`, `orth`, `polar`.
- `eig.ts`: general eigenproblem: `hessenberg`, `schur` (Francis double shift), `eig`, `eigvals` (complex results).
- `functions.ts`: matrix functions `expm` (Pade with scaling and squaring, Frechet derivative), `logm`, `sqrtm`,
  `funm`.
- `equations.ts`: `solveSylvester`, `solveLyapunov` (continuous and discrete), `solveCare`, `solveDare`.
- `structured.ts`: `solveTridiagonal` (Thomas), `solveBanded`, `toeplitz`, `solveToeplitz` (Levinson), `circulant`,
  `hankel`, `blockDiag`.
- `randomized.ts`: `randomizedSvd`, `randomizedRangeFinder`, `randomProjectionMatrix` (Gaussian and sparse),
  `johnsonLindenstraussDim`.

`sparse/` (scipy.sparse)
- `csr.ts`: `CsrMatrix`, `csr`, `fromCoo`, `toDense`, `spmv`, `spmm`, `transposeCsr`, `diagCsr`.
- `iterative.ts`: `cg`, `bicgstab`, `gmres`, `minres` as traced algorithms; preconditioners (`jacobi`, `incompleteCholesky`).
- `eigen.ts`: `lanczos`, `arnoldi`, `eigsh` (top k), `svds`, `powerIteration`.

`polynomial/` (numpy.polynomial)
- `power.ts`: `polyval`, `polyfit`, `polyder`, `polyint`, `polyadd`, `polymul`, `polydiv`, `roots`, `polyFromRoots`,
  `companion`.
- `orthogonal.ts`: Legendre, Chebyshev, Hermite and Laguerre families: evaluation by recurrence, `*Fit`, `*Gauss`
  (nodes and weights), conversion to the power basis.
- `rational.ts`: `residue`, `residuez`, `invres`, partial fraction expansion.

`quadrature/` (scipy.integrate)
- `newton-cotes.ts`: `trapezoid`, `simpson`, `cumulativeTrapezoid`, `romberg` (traced Richardson table).
- `gauss.ts`: `gaussLegendre`, `gaussHermite`, `gaussLaguerre`, `gaussJacobi`, `fixedQuad`, `normalExpectation`.
- `adaptive.ts`: `quad` (Gauss-Kronrod G7-K15, globally adaptive, infinite limits, breakpoints, traced intervals),
  `adaptiveSimpson`.
- `multidim.ts`: `dblquad`, `tplquad`, `nquad`, product rules, `cubature` on boxes.
- `monte-carlo.ts`: `monteCarloIntegrate` (plain, quasi-random, importance-sampled) with standard errors.

`roots/` (scipy.optimize root finding)
- `scalar.ts`: `bisect`, `brentq`, `ridder`, `newton`, `secant`, `halley`, `toms748` as traced algorithms.
- `systems.ts`: `fsolve` (damped Newton with Armijo backtracking), `broyden`, `fixedPoint` (with Anderson
  acceleration), `root` dispatcher.
- `continuation.ts`: `homotopy` (Newton homotopy), `arcLengthContinuation`, bifurcation point detection.

`interpolate/` (scipy.interpolate)
- `piecewise.ts`: `PPoly` piecewise polynomials: evaluate, derivative, antiderivative, integrate, roots.
- `splines.ts`: `CubicSpline` (natural, clamped, not-a-knot, periodic), `BSpline`, `makeInterpSpline`, `bsplineBasis`.
- `shape-preserving.ts`: `PchipInterpolator`, `Akima1DInterpolator` (akima and makima).
- `smoothing.ts`: `makeSmoothingSpline` (Reinsch, GCV), `UnivariateSpline`.
- `polynomial.ts`: `BarycentricInterpolator`, `lagrange`, `chebyshevInterpolant`, `lebesgueConstant`.
- `grid.ts`: `interp` (numpy), `interp1d` (linear, nearest, previous, next), `RegularGridInterpolator`, `griddata`
  (linear on a Delaunay triangulation).
- `rbf.ts`: `RBFInterpolator` (thin plate, multiquadric, Gaussian; smoothing).

`differentiate/`
- `finite-differences.ts`: `derivative` (Richardson extrapolation), `gradient` (numpy.gradient), `jacobianFd`,
  `hessianFd`, `differenceMatrix`, `finiteDifferenceWeights` (Fornberg).

`fourier/` (numpy.fft, scipy.fft)
- `fft.ts`: `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `fftn`, `ifftn`, `fft2`, `ifft2` (radix 2 plus Bluestein for any
  length; `norm` modes).
- `helpers.ts`: `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`, `nextFastLen`, `dft` and `dftMatrix` (reference).
- `trig.ts`: `dct`, `idct`, `dst`, `idst` (types I to IV).

`convolution/` (numpy.convolve, scipy.signal, scipy.ndimage)
- `one-d.ts`: `convolve`, `correlate` (full, same, valid), `fftConvolve`, `oaConvolve`, `correlationLags`.
- `n-d.ts`: `convNd` (stride, dilation, groups, padding), `convTranspose`, `convOutputSize`, `im2col`, `col2im`.
- `padding.ts`: `pad` modes (constant, edge, reflect, symmetric, wrap), `ndimage` border modes.

`geometry/`
- `polygons.ts`: `convexHull`, `polygonArea`, `polygonCentroid`, `pointInPolygon`, `polygonClip`.
- `distances.ts`: `cdist`, `pdist`, `squareform`, metrics (euclidean, sqeuclidean, cityblock, chebyshev, minkowski,
  cosine, correlation, hamming, jaccard, mahalanobis).
- `triangulation.ts`: `delaunay`, `voronoi`, `triangleBarycentric`.
- `transforms.ts`: rotation matrices, affine transforms, `homography` (normalised DLT), `fundamentalMatrix` (8-point),
  `procrustes`, `normalisePoints`.
- `contours.ts`: `contourLines` (marching squares with saddle resolution), `contourLevels`, `isolines`,
  `covarianceEllipse`.

`spatial/` (scipy.spatial, sklearn.neighbors, faiss ideas)
- `brute.ts`: `kNearest`, `radiusNeighbours`, `kNeighboursGraph` (exact, any metric).
- `trees.ts`: `KDTree`, `BallTree`, `VPTree` with query logs (visited, pruned) for visualisation.
- `hashing.ts`: LSH families (random hyperplanes, p-stable), `MinHash` banding, candidate probability curves.
- `graphs.ts`: approximate neighbour graphs: `hnswIndex` and search, `nnDescent`.
- `quantisation.ts`: product quantisation (`pqTrain`, `pqEncode`, asymmetric distances), IVF coarse quantiser.

`inverse/` (ill-posed problems)
- `tikhonov.ts`: `tikhonov` (standard and general form via SVD and GSVD), filter factors, `tikhonovPath`.
- `iterative.ts`: `landweber`, `cgls`, `lsqr` as traced algorithms with semi-convergence plots.
- `parameter-choice.ts`: `lCurve` (corner by curvature), `gcv`, `discrepancyPrinciple`, `quasiOptimality`.

### 3.9 `data/` (tier 4): dataset schemas, loading and batching (torch.utils.data, sklearn.datasets)

One schema per task kind. A dataset of a given kind looks the same whether it was generated or vendored. Each schema
declares which fields are per-row, so one `take`/`split` works for all of them.
- `schema/common.ts`: `DatasetMeta` (name, description, task, source, url, licence, citation, sha256, generator recipe,
  key), `Provenance`, `perRowFields`, `DatasetKind`.
- `schema/tabular.ts`: `RegressionData` (`x`, `y`, `featureNames`, `targetName`), `ClassificationData` (adds
  `classNames`), `MultilabelData`, `TableData` (named mixed-type columns), zod schemas for each.
- `schema/ranking.ts`: `InteractionData` (users, items, values, timestamps, sessions, propensities), `QueryData`
  (queries, documents, relevance judgements).
- `schema/corpus.ts`: `CorpusData` (documents or token sequences, labels, vocabulary), `SequenceLabellingData` (tokens
  and per-token tags), `ParallelData` (source/target pairs).
- `schema/timeseries.ts`: `TimeSeriesData` (times, values `[n, channels]`, sampling rate, events, segment labels).
- `schema/graph.ts`: `GraphData` (one or many graphs, node and edge features, node labels, masks), `TripleData`
  (knowledge-graph triples).
- `schema/image.ts`: `ImageData` (`[n, h, w, c]`, labels, optional boxes and masks).
- `schema/trajectories.ts`: `TrajectoryData` (observations, actions, rewards, terminated, truncated, behaviour
  propensities) for offline RL and logged bandits.
- `schema/truth.ts`: `Truth` attached by generators (the true function, Bayes posterior, log density, changepoints,
  planted structure) so methods can be scored against the ground truth.
- `rows.ts`: `take`, `split` (`trainTestSplit`, stratified), `bootstrap`, `concatRows`, `shuffleRows`: generic over every
  schema through its per-row fields.
- `tensors.ts`: `toTensors` (x and y as tensors), `fromTensors`, `describe` (shapes, class counts, summary).
- `loader.ts`: `DatasetRef` (embedded or remote), `load` (resolve, fetch, verify sha256 with Web Crypto, decode),
  `AssetCache` interface with a memory cache and an IndexedDB cache.
- `batching.ts`: `batches` (DataLoader: batch size, shuffle by key, drop last), `epochs`, `collate`, `padSequences`.
- `recipes.ts`: `Recipe` (a generator plus a list of modifiers with their keys, as data), `makeFromRecipe`,
  `describeRecipe`, `encodeRecipe` (URL-safe), `parseRecipe`.
- `modifiers/labels.ts`: `withLabelNoise` (symmetric, pairwise, transition matrix), `withPrevalence`, `withLabelShift`.
- `modifiers/features.ts`: `withMissing` (MCAR, MAR, MNAR), `withOutliers`, `withNuisanceFeatures`, `withTransform`
  (rotation, shear, scaling, keeping the density truth consistent).
- `modifiers/shift.ts`: `withCovariateShift` (logistic tilt), `withConceptDrift`, `shiftedSplit`,
  `semiSupervisedSplit`.
### 3.10 `graph/` (tier 5): graphs and their algorithms (NetworkX)

oracle: networkx, scipy.sparse.csgraph
- `graph.ts`: `Graph` (nodes, weighted edges, directed flag, attributes), `fromEdges`, `fromAdjacency`, `fromMatrix`,
  `neighbours`, `successors`, `predecessors`, `degree`, `inDegree`, `outDegree`, `subgraph`, `reverse`, `toUndirected`.
- `structures.ts`: `BinaryHeap` (functional priority queue), `UnionFind`, `IndexedQueue`.
- `traversal.ts`: `bfs`, `dfs` (pre, post and edge classification), `iterativeDeepening`, `topologicalSort` (Kahn),
  `isDag`, `findCycle`, `isBipartite`, `kHopNeighbourhood`; all as traced algorithms.
- `components.ts`: `connectedComponents`, `stronglyConnectedComponents` (Tarjan, Kosaraju), `condensation`,
  `articulationPoints`, `bridges`, `biconnectedComponents`.
- `paths.ts`: `dijkstra`, `bellmanFord` (negative cycles), `aStar`, `floydWarshall`, `johnson`, `shortestPath`
  (from predecessors), `kShortestPaths` (Yen).
- `trees.ts`: `minimumSpanningTree` (Kruskal, Prim, Boruvka); rooted trees: `fromParents`, `depths`, `lca`,
  `subtreeSizes`, `preOrder`, `postOrder`, `levelOrder`, `eulerTour`.
- `flows.ts`: `maxFlow` (Edmonds-Karp, Dinic), `minCut`, `minCostFlow` (successive shortest paths), `bipartiteMatching`
  (Hopcroft-Karp).
- `centrality.ts`: `degreeCentrality`, `closeness`, `betweenness` (Brandes), `eigenvectorCentrality`, `katz`,
  `pagerank`, `hits`.
- `community.ts`: `modularity`, `louvain`, `labelPropagationCommunities`, `girvanNewman`, `spectralBisection`.
- `matrices.ts`: `adjacencyMatrix`, `incidenceMatrix`, `degreeMatrix`, `laplacian` (combinatorial, symmetric,
  random-walk), `transitionMatrix`, `laplacianPositionalEncoding`, `randomWalkEncoding`.
- `measures.ts`: `density`, `diameter`, `eccentricity`, `clusteringCoefficient`, `transitivity`, `assortativity`,
  `homophily` (edge, node, adjusted), `cheegerConstant`, `spectralGap`.
- `generators.ts`: deterministic (`path`, `cycle`, `star`, `complete`, `grid`, `balancedTree`, `barbell`, `hypercube`)
  and random (`erdosRenyi`, `wattsStrogatz`, `barabasiAlbert`, `stochasticBlockModel`, `randomDag`,
  `randomGeometric`, `kNearestGraph`).
- `isomorphism.ts`: `weisfeilerLehman` (1-WL colour refinement, traced), `wlHash`, `isIsomorphic` (VF2),
  `subgraphIsomorphisms`.
- `colouring.ts`: `greedyColouring`, `dsatur`, `chromaticBounds`, `maximalIndependentSet`, `maximumClique`
  (Bron-Kerbosch).

### 3.11 `dists/` (tier 5): probability distributions (scipy.stats, torch.distributions)

oracle: scipy.stats, torch.distributions

Every distribution is a plain object from a factory: `name`, `params`, `support`, `batchShape`, `eventShape` and pure
methods `logProb`, `prob`, `cdf`, `logCdf`, `sf`, `logSf`, `ppf` (quantile), `isf`, `sample(key, shape)`, `rsample`
(reparameterised, through autodiff), `mean`, `variance`, `stddev`, `mode`, `entropy`. Methods accept tensors or
autodiff `Var`s, so log densities are differentiable.
- `distribution.ts`: the `Distribution` interface, `Support`, `ContinuousDistribution`, `DiscreteDistribution`,
  `batchShapeOf`, parameter validation helpers.
- `continuous/normal.ts`: `normal`, `halfNormal`, `truncatedNormal`, `logNormal`, `skewNormal`.
- `continuous/heavy-tailed.ts`: `studentT`, `cauchy`, `halfCauchy`, `laplace`, `logistic`, `pareto`.
- `continuous/gamma.ts`: `gamma`, `inverseGamma`, `exponential`, `chi2`, `weibull`, `erlang`.
- `continuous/bounded.ts`: `uniform`, `beta`, `kumaraswamy`, `triangular`, `vonMises`.
- `continuous/extreme.ts`: `gumbel`, `generalisedExtremeValue`, `generalisedPareto`, `frechet`.
- `continuous/sampling-distributions.ts`: `fDistribution`, `noncentralChi2`, `noncentralT`.
- `discrete/binary.ts`: `bernoulli`, `binomial`, `betaBinomial`.
- `discrete/categorical.ts`: `categorical` (probabilities or logits), `oneHotCategorical`, `discreteUniform`.
- `discrete/counts.ts`: `poisson`, `negativeBinomial`, `geometric`, `hypergeometric`, `zeroInflatedPoisson`,
  `zeroInflatedNegativeBinomial`.
- `multivariate/normal.ts`: `multivariateNormal` (covariance, precision or Cholesky parameterisation), `condition`,
  `marginal`, `lowRankMultivariateNormal`.
- `multivariate/simplex.ts`: `dirichlet`, `multinomial`, `dirichletMultinomial`, `logisticNormal`.
- `multivariate/matrix.ts`: `wishart`, `inverseWishart`, `lkjCholesky`, `matrixNormal`.
- `compose/mixture.ts`: `mixture` (log-sum-exp), `mixtureSameFamily`, component responsibilities.
- `compose/structure.ts`: `independent` (reinterpret batch as event dimensions), `truncated`, `censored`,
  `transformed` (push forward through a bijector), `empirical`.
- `bijectors/bijector.ts`: `Bijector` (forward, inverse, `logDetJacobian`), `chain`, `invert`.
- `bijectors/elementwise.ts`: `exp`, `softplus`, `sigmoid`, `tanh`, `affine`, `power`, `normalCdf`.
- `bijectors/structured.ts`: `ordered`, `stickBreaking` (simplex), `correlationCholesky`, `affineCoupling` (RealNVP),
  `permute`.
- `bijectors/unconstrain.ts`: `bijectorForSupport` (any support to the real line), `transformLogDensity` (with the
  Jacobian correction).
- `exponential-family.ts`: natural parameters, sufficient statistics and log partition for the families that have them;
  `conjugatePrior` table (Beta-Bernoulli, Gamma-Poisson, Normal-Gamma, Dirichlet-Categorical, NIW-MVN).
- `divergences.ts`: analytic `kl` for pairs of families (normal, MVN, Bernoulli, categorical, gamma, beta, Dirichlet),
  with a Monte Carlo fallback; `hellingerDistance`, `wasserstein2Gaussian`.
- `families/links.ts`: GLM link functions with inverse and derivative: identity, log, logit, probit, cloglog,
  inverse, inverse-squared, sqrt (each numerically guarded with `log1p`/`expm1`).
- `families/dispersion.ts`: exponential dispersion families for GLMs: gaussian, binomial, poisson, gamma,
  inverseGaussian, negativeBinomial, tweedie: variance function, unit deviance, log-likelihood, starting values.
- `families/distributional.ts`: GAMLSS-style families with per-parameter links: normal, studentT, gamma, BCCG,
  zero-inflated Poisson; cumulative-link ordinal families.

### 3.12 `stats/` (tier 6): statistics (scipy.stats, statsmodels)

oracle: scipy.stats, statsmodels, numpy
- `descriptive.ts`: `mean`, `variance` (two-pass, `ddof`), `std`, `skewness`, `kurtosis`, `moment`, `weightedMean`,
  `weightedVariance`, `describe`, `zscore`, `sem`, `mode`.
- `running.ts`: Welford running moments (`push`, `merge`), running quantiles (P-squared), exponentially weighted
  moments.
- `quantiles.ts`: `quantile` (the nine Hyndman-Fan methods, numpy names), `median`, `iqr`, `percentileOfScore`,
  `ecdf`, `medianAbsoluteDeviation`.
- `histograms.ts`: `histogram`, `histogramBinEdges` (auto, Freedman-Diaconis, Scott, Sturges, Doane), `histogram2d`,
  `binnedStatistic`.
- `correlation.ts`: `pearson`, `spearman`, `kendallTau`, `cov`, `corrcoef`, `partialCorrelation`, `pointBiserial`,
  `distanceCorrelation`.
- `covariance.ts`: `empiricalCovariance`, `ledoitWolf`, `oas`, `shrunkCovariance` (robust and sparse estimators are in
  `methods/covariance`).
- `density.ts`: `gaussianKde` (univariate and multivariate, Scott and Silverman bandwidths), `kdeEvaluate`,
  `bandwidthCv`.
- `ranks.ts`: `rankdata` (tie methods), `argsortStable`, `tieCorrection`.
- `transforms.ts`: `boxcox` and `yeojohnson` with maximum-likelihood lambda and inverses, `logit`/`expit` transforms,
  `standardise`.
- `tests/means.ts`: `ttest1samp`, `ttestInd` (pooled and Welch), `ttestRel`, `ztest`, `onewayAnova`, `welchAnova`,
  `tukeyHsd`.
- `tests/contingency.ts`: `chi2Contingency`, `gTest`, `fisherExact`, `mcnemar`, `oddsRatio`, `cochranQ`.
- `tests/nonparametric.ts`: `mannWhitneyU`, `wilcoxon`, `kruskal`, `friedman`, `nemenyi`, `signTest`.
- `tests/distribution.ts`: `ksOneSample`, `ksTwoSample`, `andersonDarling`, `shapiroWilk`, `jarqueBera`,
  `chiSquareGoodnessOfFit`.
- `tests/proportions.ts`: `binomTest`, `proportionsZtest`, `proportionConfint` (Wald, Wilson, Agresti-Coull,
  Clopper-Pearson, Jeffreys).
- `tests/timeseries.ts`: `ljungBox`, `durbinWatson`, `adfuller`, `kpss`.
- `tests/multiple.ts`: `multipletests` (Bonferroni, Holm, Hochberg, Benjamini-Hochberg, Benjamini-Yekutieli),
  `combinePvalues` (Fisher, Stouffer).
- `effect-size.ts`: `cohensD`, `hedgesG`, `glassDelta`, `cohensH`, `cramersV`, `etaSquared`, `cliffsDelta`.
- `power.ts`: power and sample size for t tests, proportions and ANOVA (`solvePower`).
- `resampling.ts`: `bootstrap` (percentile, basic, BCa intervals), `permutationTest`, `jackknife`, `monteCarloTest`.
- `information.ts`: `entropy`, `crossEntropy`, `jointEntropy`, `conditionalEntropy`, `mutualInformation`,
  `pointwiseMutualInformation`, `klDivergence`, `jensenShannon`, `totalVariation`, `fDivergence`,
  `differentialEntropy`, `ksgMutualInformation`, variational bounds (`infoNce`, `nwj`, `donskerVaradhan`).
- `extremes.ts`: `fitGeneralisedPareto`, `peaksOverThreshold`, `returnLevel`, `meanExcess`, `fitGev` (block maxima).
- `robust.ts`: `trimmedMean`, `winsorize`, `huberLocation`, `tukeyBiweight`, `ransac` (traced, generic model).
- `smoothing/bases.ts`: B-spline basis, P-spline difference penalties, cyclic cubic, natural cubic, thin-plate
  regression splines, tensor-product bases, sum-to-zero constraints.
- `smoothing/smoothers.ts`: `loess`/`lowess`, `kernelSmoother` (Nadaraya-Watson, local linear), `runningMean`,
  `splineSmoother`.
- `fit.ts`: closed-form and moment estimators for the families in `dists` (`fitNormal`, `fitGamma` by Newton,
  `fitBeta`, `fitPoisson`), standard errors from the observed information.

### 3.13 `optim/` (tier 6): optimisation (optax, torch.optim, scipy.optimize)

oracle: torch.optim, optax (via jax), scipy.optimize, cvxpy for small LPs and QPs
- `transform.ts`: `GradientTransformation` (`init`, `update`), `applyUpdates`, `chain`, `scaleByLearningRate`,
  `clipByGlobalNorm`, `clipByValue`, `addDecayedWeights`, `maskedTransform`, `globalNorm`.
- `first-order.ts`: `sgd` (momentum, Nesterov), `adagrad`, `rmsprop`, `adadelta`.
- `adaptive.ts`: `adam`, `adamw`, `adamax`, `nadam`, `radam`, `amsgrad`, `lion`, `lamb`, `adafactor`.
- `schedules.ts`: `constantSchedule`, `stepDecay`, `exponentialDecay`, `inverseTimeDecay`, `cosineDecay`,
  `warmupCosine`, `oneCycle`, `piecewiseConstant`, `polynomialDecay`.
- `line-search.ts`: `backtracking` (Armijo), `strongWolfe` (bracket and zoom), `moreThuente`, each returning trial
  points for display.
- `minimize/minimizer.ts`: `minimizer` (any transformation as a traced algorithm), `minimize(f, x0, { method })` where
  `method` is a function value (no name registry), `MinimizeResult`.
- `minimize/gradient.ts`: `gradientDescent` (fixed step or line search), `conjugateGradient` (Fletcher-Reeves,
  Polak-Ribiere with restarts), `coordinateDescent`.
- `minimize/quasi-newton.ts`: `bfgs`, `lbfgs` (two-loop), `lbfgsb` (box bounds), `owlqn` (L1).
- `minimize/newton.ts`: `newton` (modified Cholesky), `trustRegion` (dogleg, Steihaug CG), `newtonCg`.
- `minimize/least-squares.ts`: `gaussNewton`, `levenbergMarquardt`, `leastSquares` (with bounds via trust region
  reflective), `curveFit`.
- `minimize/derivative-free.ts`: `nelderMead`, `powell`, `patternSearch`, `cobyla`.
- `minimize/global.ts`: `simulatedAnnealing`, `differentialEvolution`, `cmaEs`, `basinHopping`, `particleSwarm`,
  `dualAnnealing`.
- `minimize/constrained.ts`: `projectedGradient`, `penaltyMethod`, `augmentedLagrangian`, `slsqp`, `barrierMethod`.
- `scalar.ts`: `minimizeScalar` (golden section, Brent, bounded), `bracket`.
- `proximal.ts`: proximal operators (L1, L2, group, nuclear, box, simplex, total variation 1-D), `ista`, `fista`,
  `admm`, `projections` (box, ball, simplex, nonnegative).
- `linear-programming.ts`: `linprog` (two-phase simplex with Bland's rule, Mehrotra interior point), duals, sensitivity.
- `quadratic.ts`: `quadprog` (active set, interior point), `boxQp`, KKT residual report.
- `integer.ts`: `milp` (branch and bound, Gomory cuts), `knapsack`, `setCover` (greedy and LP rounding).
- `assignment.ts`: `linearSumAssignment` (Hungarian), `auction`, `stableMatching` (Gale-Shapley).
- `online.ts`: `onlineGradientDescent`, `ftrl`, `hedge`, `exponentiatedGradient`, `onlineNewtonStep`, regret
  tracking.
- `transport.ts`: `emd` (exact optimal transport), `sinkhorn` (log domain), `wasserstein1d`, `slicedWasserstein`,
  `gromovWasserstein`, `barycentre`.
- `multiobjective.ts`: `paretoFront`, `dominates`, `hypervolume`, `hypervolumeContribution`, scalarisations (weighted
  sum, Chebyshev), `nsga2`.
- `implicit.ts`: implicit differentiation through solutions: `implicitRoot`, `implicitFixedPoint`, `implicitArgmin`
  (custom vjp via the implicit function theorem).
- `search.ts`: combinatorial search over refinement operators: `beamSearch`, `bestFirst`, `localSearch`, `tabuSearch`.

### 3.14 `kernels/` (tier 6): kernel functions and kernel approximations (sklearn.gaussian_process.kernels, sklearn.kernel_approximation, GPyTorch)

oracle: scikit-learn
- `kernel.ts`: `Kernel` (`k(x, y)`, `diag`, hyperparameters as a pytree, differentiable), `gram`, `crossGram`,
  `sumKernel`, `productKernel`, `scaleKernel`, `ardKernel`, `fromLogParams`.
- `stationary.ts`: `rbf`, `matern` (1/2, 3/2, 5/2, general nu), `rationalQuadratic`, `exponential`, `periodic`
  (exp-sine-squared), `locallyPeriodic`, `spectralMixture`, `cosineKernel`.
- `dot-product.ts`: `linear`, `polynomial`, `sigmoidKernel`, `arcCosine` (infinitely wide networks), `ntk` (neural
  tangent kernel of an MLP).
- `structured.ts`: `whiteNoise`, `constantKernel`, string kernels (`spectrumKernel`, `subsequenceKernel`), set kernels,
  graph kernels (`diffusionKernel`, `regularisedLaplacian`, `wlSubtreeKernel`), `chi2Kernel`, `histogramIntersection`.
- `expansion.ts`: explicit feature maps: `randomFourierFeatures` (RBFSampler), `orthogonalRandomFeatures`, `nystroem`,
  `additiveChi2Sampler`, `polynomialCountSketch`, `tensorSketch`, approximation error helpers.
- `matrices.ts`: `centerKernel`, `kernelAlignment`, `hsic`, `mmd` (biased and unbiased, with permutation test),
  `effectiveDimension`, `kernelPcaProjection`.

### 3.15 `signal/` (tier 6): signal and image processing (scipy.signal, scipy.ndimage, pywt, librosa basics)

oracle: scipy.signal, scipy.ndimage, pywt, librosa
- `signal.ts`: `Signal` (samples, sampling rate, start time, units, channels), `signal`, `sampleTimes`, `resampleTo`,
  `Spectrum` and `TimeFrequency` records, `magnitudeDb`, `unwrap`.
- `windows.ts`: `getWindow` (boxcar, hann, hamming, blackman, blackmanharris, nuttall, flattop, bartlett, kaiser,
  gaussian, tukey, cosine, dpss), periodic and symmetric.
- `filters/fir.ts`: `firwin`, `firwin2`, `remez`, `kaiserord`, `kaiserBeta`, `minimumPhase`.
- `filters/iir.ts`: `butter`, `cheby1`, `cheby2`, `ellip`, `bessel`, `iirfilter`, `iirnotch`, `iirpeak`; outputs in
  `ba`, `zpk` and `sos` forms.
- `filters/apply.ts`: `lfilter`, `sosfilt`, `filtfilt`, `sosfiltfilt`, `lfilterZi`, `sosfiltZi`, `medfilt`,
  `savgolFilter`, `wiener`.
- `filters/response.ts`: `freqz`, `sosfreqz`, `freqs`, `groupDelay`, filter conversions (`tf2zpk`, `zpk2sos`, ...).
- `lti/systems.ts`: `TransferFunction`, `ZerosPolesGain`, `StateSpace` (continuous and discrete), conversions, poles,
  zeros, `series`, `parallel`, `feedback`.
- `lti/responses.ts`: `step`, `impulse`, `lsim` (traced), `bode`, `nyquist`, `margins` (gain, phase, delay).
- `lti/discretise.ts`: `contToDiscrete` (zero-order hold, first-order hold, Tustin with prewarping, Euler), `c2d`.
- `spectral.ts`: `periodogram`, `welch`, `csd`, `coherence`, `spectrogram`, `stft`, `istft`, `checkCola`,
  `multitaper` (DPSS, adaptive weights), `lombscargle`.
- `time-frequency.ts`: `cwt` (Morlet, Ricker), `cqt`, `hilbert` (analytic signal, instantaneous frequency),
  `wignerVille`, `reassignedSpectrogram`, `synchrosqueeze`.
- `wavelets.ts`: `dwt`, `idwt`, `wavedec`, `waverec`, `swt`, wavelet families (Haar, Daubechies, Symlets, Coiflets),
  boundary modes, `waveletDenoise` (VisuShrink, SURE).
- `parametric.ts`: `yuleWalker`, `burg`, `arPsd`, `levinsonDurbin`, `music`, `esprit`, `prony`.
- `adaptive.ts`: `lms`, `nlms`, `rls`, `kalmanTracker` (simple), as traced algorithms.
- `multirate.ts`: `resample` (FFT), `resamplePoly`, `decimate`, `upfirdn`, polyphase filter banks.
- `peaks.ts`: `findPeaks` (height, prominence, distance, width), `peakProminences`, `peakWidths`, `argrelextrema`,
  `zeroCrossings`.
- `similarity.ts`: `dtw` (windowed, with path), `lbKeogh`, `mass` (distance profile), `matrixProfile` (STOMP and
  SCRIMP), `paa`, `sax`, `crossCorrelationLag`.
- `decompositions.ts`: `emd`, `eemd`, `ceemdan`, `vmd`, `ssa` (singular spectrum analysis).
- `cepstrum.ts`: `realCepstrum`, `complexCepstrum`, `lifter`, `yin` pitch tracking.
- `audio.ts`: `melScale`, `melFilterbank`, `melSpectrogram`, `mfcc`, `chroma`, `onsetStrength`.
- `sparse.ts`: `matchingPursuit`, `orthogonalMatchingPursuit`, `basisPursuit`, `iterativeHardThresholding`,
  `sparseEncode`, `ksvd`, `dictionaryLearning` (MOD).
- `image/filters.ts`: `gaussianFilter`, `uniformFilter`, `medianFilter`, `sobel`, `scharr`, `prewitt`, `laplace`,
  `gaussianLaplace`, `bilateral`.
- `image/features.ts`: `canny`, `cornerHarris`, `cornerShiTomasi`, `peakLocalMax`, `blobLog`, `blobDog`, `houghLine`,
  `houghCircle`, `hog`, `structureTensor`.
- `image/morphology.ts`: `erosion`, `dilation`, `opening`, `closing`, `morphologicalGradient`, `whiteTophat`,
  `label` (connected components), `distanceTransform`, `skeletonize`.
- `image/geometry.ts`: `resize` (nearest, bilinear, bicubic), `rotate`, `warpAffine`, `warpPerspective`,
  `gaussianPyramid`, `laplacianPyramid`, `integralImage`.
- `image/segmentation.ts`: `otsuThreshold`, `watershed`, `slic` superpixels, `felzenszwalb`, `graphCut` (two-label,
  using `graph/flows`).

### 3.16 `text/` (tier 6): text processing (HF tokenizers, nltk, sklearn.feature_extraction.text)

oracle: HF tokenizers, tiktoken, nltk, scikit-learn
- `normalise.ts`: `unicodeNormalise` (NFC, NFD, NFKC, NFKD), `caseFold`, `stripAccents`, `collapseWhitespace`,
  `AlignedText` (normalised text with offsets back to the original).
- `tokenise.ts`: `regexTokenise`, `wordTokenise` (sklearn token pattern, Treebank), `tweetTokenise`, `characters`
  (code points or graphemes), `sentences` (Punkt-style), `detokenise`, pre-tokeniser patterns (GPT-2, cl100k, BERT).
- `stem.ts`: `porterStem` (with a rule trace), `snowballEnglish`, `lemmatiseRules`, stop-word lists (`nltkEnglish`,
  `sklearnEnglish`).
- `vocabulary.ts`: `Vocabulary`, `buildVocabulary` (min count, max size, specials, ordering), `encodeTokens`,
  `decodeTokens`, `tokenCounts`.
- `subword/bpe.ts`: `trainBpe` (character and byte level, traced merges), `bpeEncode` (priority-queue merges, dropout),
  `byteAlphabet` (GPT-2 bytes-to-unicode).
- `subword/wordpiece.ts`: `trainWordPiece`, `wordPieceEncode` (greedy longest match).
- `subword/unigram.ts`: `trainUnigram` (EM with pruning), `unigramEncode` (Viterbi), n-best and sampling
  (subword regularisation).
- `tokenizer.ts`: an HF-tokenizers-style pipeline: `Tokenizer` (normaliser, pre-tokeniser, model, post-processor,
  decoder), `encode` with offsets, `decode`, `encodeBatch` with padding and truncation, `fromJson` for HF files.
- `features.ts`: `countVectorizer`, `tfidfTransformer`, `tfidfVectorizer`, `hashingVectorizer` (MurmurHash3), word and
  character n-grams, `bm25` (Okapi, BM25+).
- `representations.ts`: `cooccurrence` (windowed, weighted), `pmi`, `ppmi`, `lsa`, `svdWordVectors`,
  `randomIndexing`, `analogy` (3CosAdd, 3CosMul), `nearestWords`.
- `similarity.ts`: `levenshtein`, `damerauLevenshtein`, `jaroWinkler`, `lcs`, `needlemanWunsch`, `smithWaterman`,
  `jaccard`, `minhashSignature`, `shingles`.
- `hyphenation.ts`: Liang patterns: `parsePatterns`, `hyphenate` (with a matching trace), TeX pattern format.
- `statistics.ts`: `zipfFit`, `heapsLaw`, `typeTokenRatio`, `fertility` (tokens per word), `compressionRatio`,
  topic coherence (`npmi`, `umass`).

### 3.17 `logic/` (tier 6): logic programming, satisfiability and ILP

oracle: SWI-Prolog outputs recorded as fixtures; pysat for SAT
- `terms.ts`: `atom`, `variable`, `compound`, `list`, `num`, `isGround`, `termVariables`, `termsEqual`,
  `printTerm`.
- `parse.ts`: Prolog syntax parser for clauses and queries (operators, lists, comments), `parseProgram`,
  `parseQuery`.
- `unify.ts`: `unify` (iterative, optional occurs check), `Substitution`, `applySubstitution`, `compose`,
  `renameApart`, `unificationTrace` (Martelli-Montanari steps).
- `resolution.ts`: `solve` (SLD resolution with backtracking and depth limits, traced), built-ins (`is`, comparison,
  `=`, `\=`, `findall`, negation as failure), standard library predicates, `sldTree`.
- `datalog.ts`: bottom-up evaluation: `naiveEvaluation`, `seminaiveEvaluation`, `leastHerbrandModel`, stratified
  negation, derivation trees.
- `propositional.ts`: formula constructors, `evaluate`, `truthTable`, `models`, `entails`, `toCnf` (Tseitin),
  `toDnf`, `simplify`.
- `sat.ts`: `dpll`, `cdcl` (clause learning, watched literals, traced decisions), `walkSat`, DIMACS read and write.
- `first-order.ts`: first-order clausal form (`skolemise`, `toClauses`), `resolutionProver` (given-clause loop),
  `subsumes`.
- `ilp/generalisation.ts`: theta-subsumption, `lgg` and relative `rlgg` (Plotkin), clause reduction.
- `ilp/version-spaces.ts`: `findS`, `candidateElimination`, version-space boundaries and voting.
- `ilp/foil.ts`: FOIL (`foilGain`, refinement, coverage) as a traced algorithm.
- `ilp/progol.ts`: mode declarations, bottom clauses, inverse entailment search.
- `ilp/meta-interpretive.ts`: Metagol-style learning with metarules and iterative deepening; learning from failures.

### 3.18 `sampling/` (tier 6): Monte Carlo sampling (NumPyro/BlackJAX kernels, ArviZ diagnostics)

oracle: numpy statistical checks against analytic targets; arviz for diagnostics
- `target.ts`: `LogDensity` (log density with gradient by autodiff, dimension, optional constraints),
  `logDensityFromDistribution`, `unconstrainedTarget`.
- `metropolis.ts`: `randomWalkMetropolis`, `independenceMetropolis`, `metropolisHastings` (custom proposals), adaptive
  Metropolis.
- `hamiltonian.ts`: `leapfrog`, `hmc`, `nuts` (multinomial, generalised U-turn, divergences), `stepSizeAdaptation`
  (dual averaging), `massMatrixAdaptation` (diagonal and dense, windowed).
- `langevin.ts`: `mala`, `ula`, `sgld`, `svgd` (Stein variational gradient descent).
- `gibbs.ts`: `gibbs` (blocks of conditional samplers), `sliceSampler` (stepping out, shrinkage), `ellipticalSlice`.
- `importance.ts`: `importanceSampling`, `selfNormalised`, `psis` (Pareto smoothed weights, k-hat), effective sample
  size of weights.
- `smc.ts`: `smcSampler` (tempering with adaptive schedule), resampling schemes (multinomial, systematic, stratified,
  residual), `annealedImportanceSampling`.
- `rejection.ts`: `rejectionSampling`, `adaptiveRejection` (log-concave), `ratioOfUniforms`.
- `chains.ts`: `runChains` (several chains, independent keys), `Chains` (draws `[chains, draws, dim]`), `thin`,
  `burn`.
- `diagnostics.ts`: `rhat` (rank-normalised split R-hat), `ess` (bulk and tail), `mcse`, `autocorrelation`,
  `summary` (ArviZ-style table), divergence and tree-depth reports.
### 3.19 `dynamics/` (tier 7): dynamical systems, control and filtering (scipy.integrate, python-control, filterpy)

oracle: scipy.integrate, python-control, filterpy, statsmodels
- `ode/explicit.ts`: fixed-step `euler`, `heun`, `midpoint`, `rk4`, any explicit Butcher tableau (`butcherTableau`).
- `ode/adaptive.ts`: embedded pairs `rk23`, `rk45` (Dormand-Prince, dense output), `dop853`; error control and step
  rejection recorded as trace events.
- `ode/implicit.ts`: `backwardEuler`, `trapezoid`, `bdf` (variable order 1 to 5), `radau5` for stiff problems.
- `ode/solve-ivp.ts`: `solveIvp` (scipy semantics: method, `tEval`, dense output, events with direction and
  termination), `odeint`.
- `ode/boundary.ts`: `solveBvp` (collocation), shooting method.
- `sde.ts`: `eulerMaruyama`, `milstein`, `stochasticRungeKutta`, `brownianMotion`, `brownianBridge`, named processes
  with exact transitions (`ornsteinUhlenbeck`, `geometricBrownianMotion`).
- `fields.ts`: vector-field analysis: `jacobianAt`, `divergence`, `curl`, `fixedPoints` (seeded Newton), `classifyFixedPoint`
  (node, spiral, saddle, centre), `nullclines`, `streamlines`, `flowMap`, `invariantManifolds`.
- `maps.ts`: discrete-time maps: `iterateMap`, `orbit`, `cobweb`, `bifurcationDiagram`, `lyapunovExponent`,
  `lyapunovSpectrum` (QR method).
- `pde.ts`: finite-difference solvers for model PDEs: heat (explicit, implicit, Crank-Nicolson), advection (upwind,
  Lax-Wendroff), wave (leapfrog), reaction-diffusion; boundary conditions; CFL checks; `fokkerPlanck`; pseudo-spectral
  periodic solver.
- `markov.ts`: discrete Markov chains: `stationaryDistribution`, `classifyStates`, `absorptionProbabilities`,
  `hittingTimes`, `mixingTime`, `spectralGap`, `isReversible`, `simulateChain`; continuous-time chains
  (`generatorMatrix`, `transitionAt`).
- `control/feedback.ts`: `controllability`, `observability`, Gramians, `routhHurwitz`, `rootLocus`,
  `nyquistEncirclements`.
- `control/optimal.ts`: `lqr`, `dlqr`, `lqe` (Kalman gain), `lqg`, `placePoles` (KNV), `ackermann`.
- `control/predictive.ts`: `mpc` (condensed QP with input and state constraints, receding horizon as a traced
  algorithm), `pid` (discrete PID with anti-windup and derivative filtering).
- `filtering/kalman.ts`: `kalmanFilter` (Joseph form, missing data), `rtsSmoother`, `steadyStateKalman`,
  `kalmanLogLikelihood`, NIS and NEES consistency checks.
- `filtering/nonlinear.ts`: `extendedKalmanFilter`, `unscentedKalmanFilter` (sigma points), `ensembleKalmanFilter`.
- `filtering/particle.ts`: `bootstrapParticleFilter`, `auxiliaryParticleFilter`, particle smoother; resampling from
  `sampling/smc`.
- `filtering/changepoint.ts`: `bocpd` (Bayesian online changepoint detection with conjugate predictives), run-length
  posterior, `pelt`, `binarySegmentation`.
- `filtering/em.ts`: `linearGaussianEm` (Shumway-Stoffer EM for state-space parameters, traced).
- `identification.ts`: `arx`, `armax`, `predictionErrorMethod`, `n4sid` (subspace), `dmd` (dynamic mode decomposition),
  `sindy` (sparse identification of dynamics).

### 3.20 `pgm/` (tier 7): probabilistic graphical models (pgmpy, libDAI ideas)

oracle: pgmpy, networkx for structure; brute-force enumeration for exact checks

A model is plain data. Every model type has `modelGraph(model)`, a structure description (nodes with roles, edges,
plates, factor nodes) that omai-render draws, and inference algorithms emit `message` events so the drawing can be
animated.
- `factor.ts`: `DiscreteFactor` (scope, cardinalities, values in log space), `factorProduct`, `marginalise` (sum and
  max), `reduce` (evidence), `normalise`, `factorDivide`, `toTable`.
- `gaussian-factor.ts`: canonical (information) form Gaussian factors: product, marginalise, condition; conversion to
  and from moment form.
- `models/bayes-net.ts`: `bayesNet` (nodes, states, parents, CPTs, validation), `dSeparated`, `markovBlanket`,
  `ancestralSample`, `jointProbability`, `toFactors`.
- `models/markov-network.ts`: `markovNetwork` (cliques and potentials), `fromBayesNet` (moralisation), `isingModel`,
  `pottsModel`, `gibbsDistribution`.
- `models/factor-graph.ts`: `factorGraph` (variables and factors, bipartite), conversion from Bayes nets and Markov
  networks.
- `models/linear-gaussian.ts`: linear-Gaussian Bayes networks and Gaussian Markov random fields; conversion to a joint
  `multivariateNormal`.
- `models/dynamic.ts`: dynamic Bayes networks (two-slice templates), `unroll`, hidden Markov models as DBNs.
- `models/plates.ts`: plate notation: `plate`, `templateModel`, `ground` (unroll plates against sizes and data).
- `structure.ts`: `modelGraph` (for every model type), `moralise`, `triangulate`, `eliminationOrder` (min-degree,
  min-fill, weighted), `junctionTree` (cliques and separators), `treewidth`.
- `inference/enumerate.ts`: brute-force joint enumeration (reference for tests and teaching).
- `inference/elimination.ts`: `variableElimination` (sum-product and max-product, order choice, intermediate factor
  sizes as events).
- `inference/junction-tree.ts`: `junctionTreeInference` (Shafer-Shenoy and HUGIN propagation), all marginals at once.
- `inference/belief-propagation.ts`: `beliefPropagation` (sum and max product; tree, flooding and residual schedules;
  damping; loopy convergence), `betheFreeEnergy`.
- `inference/gaussian-bp.ts`: Gaussian belief propagation in information form, convergence diagnostics.
- `inference/chain.ts`: `forwardBackward` (scaled and log space), `viterbi`, `posteriorDecode`, forward filtering
  backward sampling.
- `inference/sampling.ts`: `forwardSampling`, `rejectionSampling`, `likelihoodWeighting`, `gibbsOnFactorGraph`.
- `inference/variational.ts`: `meanField` (naive mean field, traced), `variationalMessagePassing` for conjugate
  exponential models.
- `inference/ep.ts`: `expectationPropagation` on factor graphs (Gaussian and discrete sites, damping, power EP).
- `learning/parameters.ts`: maximum likelihood and Bayesian (Dirichlet) CPT estimation, `emParameters` (missing data,
  traced).
- `learning/structure.ts`: `chowLiu`, `pcAlgorithm` (constraint based, with independence tests from `stats`),
  `hillClimb` (BIC and BDeu scores), `scoreStructure`.
- `causal.ts`: `doIntervention` (graph surgery), `backdoorSets`, `frontdoorAdjustment`, `identifyEffect` (simple do
  calculus cases), `counterfactual` (abduction, action, prediction) for discrete structural models.

### 3.21 `probprog/` (tier 7): probabilistic programming (NumPyro, Pyro)

oracle: numpyro and pyro on small models (posterior moments within Monte Carlo error)

A model is an ordinary function that calls `sample`, `observe`, `deterministic`, `plate` and `factor` through an
effect-handler context. Handlers (`trace`, `condition`, `substitute`, `seed`, `block`, `replay`) are the extension
mechanism, exactly as in NumPyro.
- `primitives.ts`: `sample`, `observe`, `deterministic`, `param`, `plate`, `factor`, `Model` type.
- `handlers.ts`: `trace`, `seed`, `condition`, `substitute`, `replay`, `block`, `scale`, `mask`, `doHandler`
  (interventions).
- `density.ts`: `logJoint`, `potentialEnergy` (model to an unconstrained `LogDensity` with Jacobian corrections),
  `initToMedian`, `initToSample`, `constrain`/`unconstrain` parameter maps.
- `graph.ts`: `modelGraph` for a probabilistic program (trace the model once, record dependencies and plates) so it
  can be drawn like any PGM.
- `infer/mcmc.ts`: `mcmc(model, { kernel: nuts, numWarmup, numSamples, numChains })` on top of `sampling`.
- `infer/svi.ts`: `svi` (stochastic variational inference as a traced algorithm), ELBO estimators (`traceElbo`
  reparameterised and score function, `traceMeanFieldElbo`, `renyiElbo`), baselines and control variates.
- `infer/guides.ts`: automatic guides: `autoDelta`, `autoNormal` (mean field), `autoMultivariateNormal`,
  `autoLowRankMultivariateNormal`, `autoLaplace`.
- `infer/exact.ts`: discrete enumeration (`enumerateSites`) handing off to `pgm` variable elimination; conjugate
  updates when the model is conjugate.
- `infer/importance.ts`: likelihood weighting and importance sampling over programs, marginal likelihood estimates.
- `predictive.ts`: `priorPredictive`, `posteriorPredictive`, `logLikelihoodPerObservation`.
- `criticism.ts`: `waic`, `psisLoo` (elpd with Pareto k diagnostics), posterior predictive checks, `compareModels`.

### 3.22 `learn/` (tier 7): the machine-learning toolkit around models (sklearn's non-model modules)

oracle: scikit-learn
- `estimator.ts`: the estimator protocol: `Estimator<H, M>` (`name`, `hyper`, `fit(data, ctx)`), capability guards
  (`canPredict`, `canPredictProba`, `canTransform`, `canScore`, `canSample`), `FitContext` (key, tracer), and
  `EstimatorClass` (the base for sklearn-style class wrappers: `fit`, `predict`, `predictProba`, `transform`,
  `fitTransform`, `score`, `getParams`, `setParams`).
- `validation.ts`: `checkXy`, `checkArray`, `checkClassification`, `checkSampleWeight`, `asFloatMatrix` with clear shape
  errors.
- `losses/regression.ts`: `mse`, `mae`, `huber`, `logCosh`, `pinball` (quantile), `expectile`, `poissonNll`,
  `gaussianNll`, `tweedieDeviance` (Var-based, differentiable).
- `losses/classification.ts`: `binaryCrossEntropy` (with logits), `crossEntropy` (label smoothing, class weights),
  `focal`, `nllLoss`.
- `losses/margin.ts`: `hinge`, `squaredHinge`, `logisticLoss`, `exponentialLoss`, `modifiedHuber`, `crammerSinger`,
  `westonWatkins`.
- `losses/ranking.ts`: `rankNet`, `pairwiseHinge`, `bpr`, `lambdaRank`, `listNet`, `listMle`, `approxNdcg`,
  `sampledSoftmax`, `infoNce`, `tripletLoss`, `contrastiveLoss`.
- `losses/divergence.ts`: `klLoss`, `jsLoss`, `distillationLoss`, `mmdLoss`.
- `metrics/classification.ts`: `accuracy`, `balancedAccuracy`, `precision`, `recall`, `f1`, `fbeta`,
  `precisionRecallFscoreSupport`, `confusionMatrix`, `matthewsCorrcoef`, `cohenKappa`, `hammingLoss`, `jaccardScore`,
  `classificationReport`, averaging (micro, macro, weighted, samples).
- `metrics/curves.ts`: `rocCurve`, `rocAuc` (binary, one-vs-rest, one-vs-one), `precisionRecallCurve`,
  `averagePrecision`, `detCurve`, `calibrationCurve`, `liftCurve`.
- `metrics/probabilistic.ts`: `logLoss`, `brierScore` (with decomposition), `expectedCalibrationError`,
  `crps`, `intervalScore`, `coverage`, `pitHistogram`.
- `metrics/regression.ts`: `meanSquaredError`, `rootMeanSquaredError`, `meanAbsoluteError`, `medianAbsoluteError`,
  `r2`, `explainedVariance`, `meanAbsolutePercentageError`, `meanPinballLoss`, `d2Score`, `maxError`.
- `metrics/ranking.ts`: `ndcg`, `dcg`, `meanAveragePrecision`, `mrr`, `recallAtK`, `precisionAtK`, `hitRate`,
  `coverage`, `novelty`, `diversity`.
- `metrics/clustering.ts`: `adjustedRandIndex`, `randIndex`, `normalizedMutualInfo`, `adjustedMutualInfo`,
  `homogeneityCompletenessV`, `fowlkesMallows`, `silhouetteScore`, `silhouetteSamples`, `daviesBouldin`,
  `calinskiHarabasz`, `contingencyMatrix`.
- `metrics/fairness.ts`: group rates, `demographicParityDifference`, `equalisedOddsDifference`,
  `equalOpportunityDifference`, `predictiveParity`, disaggregated reports.
- `preprocessing/scalers.ts`: `standardScaler`, `minMaxScaler`, `maxAbsScaler`, `robustScaler`, `normalizer`,
  `quantileTransformer`, `powerTransformer` (Box-Cox, Yeo-Johnson).
- `preprocessing/encoders.ts`: `oneHotEncoder`, `ordinalEncoder`, `labelEncoder`, `labelBinarizer`, `targetEncoder`
  (with cross-fitting), `multiLabelBinarizer`.
- `preprocessing/features.ts`: `polynomialFeatures`, `splineTransformer`, `kBinsDiscretizer`, `functionTransformer`,
  `binarizer`.
- `preprocessing/imputation.ts`: `simpleImputer`, `knnImputer`, `iterativeImputer`, `missingIndicator`.
- `preprocessing/resampling.ts`: `randomOverSampler`, `randomUnderSampler`, `smote`, `borderlineSmote`, `adasyn`,
  `tomekLinks`, `nearMiss`.
- `feature-selection.ts`: `varianceThreshold`, `selectKBest` and `selectPercentile` (f-test, chi2, mutual
  information), `rfe`, `rfecv`, `sequentialFeatureSelector`, `selectFromModel`.
- `compose.ts`: `pipeline`, `makePipeline`, `columnTransformer`, `featureUnion`, `transformedTargetRegressor`, step
  inspection (`namedSteps`, intermediate outputs as trace events).
- `model-selection/splitters.ts`: `kFold`, `stratifiedKFold`, `groupKFold`, `stratifiedGroupKFold`, `shuffleSplit`,
  `leaveOneOut`, `leavePOut`, `repeatedKFold`, `timeSeriesSplit`, `predefinedSplit`, spatial block splits.
- `model-selection/cross-validation.ts`: `crossValidate`, `crossValScore`, `crossValPredict`, `learningCurve`,
  `validationCurve`, `permutationTestScore`.
- `model-selection/search.ts`: `gridSearchCv`, `randomizedSearchCv`, `halvingGridSearchCv`, nested cross-validation,
  each search a traced algorithm.
- `model-selection/criteria.ts`: `aic`, `aicc`, `bic`, `mallowsCp`, `gcv`, `looShortcut` (linear smoothers),
  `oneStandardErrorRule`, `bootstrap632`.
- `calibration.ts`: `plattScaling`, `isotonicCalibration`, `temperatureScaling`, `betaCalibration`,
  `histogramBinning`, `calibratedClassifier` (cross-fitted).
- `conformal.ts`: `splitConformalRegressor`, `conformalisedQuantileRegression`, `conformalClassifier` (LAC, APS,
  RAPS), `mondrianConformal`, `jackknifePlus`, coverage reports.
- `inspection.ts`: `permutationImportance`, `partialDependence`, `ice`, `accumulatedLocalEffects`, `hStatistic`,
  `functionalAnova`.
- `explain/gradients.ts`: `saliency`, `inputXGradient`, `integratedGradients`, `smoothGrad`, `deepLift`,
  `expectedGradients`.
- `explain/shapley.ts`: `exactShapley`, `kernelShap`, `samplingShapley`, `treeShap` (and interaction values),
  `shapleyInteractions`.
- `explain/surrogates.ts`: `lime`, `anchors` (KL-LUCB), global surrogate trees, fidelity.
- `explain/counterfactuals.ts`: `wachterCounterfactual`, `diverseCounterfactuals`, `face` (feasible paths), actionable
  recourse constraints.
- `privacy.ts`: `laplaceMechanism`, `gaussianMechanism` (classic and analytic), `exponentialMechanism`,
  `randomisedResponse`, composition theorems, RDP accounting for the subsampled Gaussian (`dpSgdEpsilon`).

### 3.23 `nn/` (tier 8): neural networks (torch.nn, torch.nn.functional, Flax/Haiku)

oracle: PyTorch (forward values and gradients in float64 for fixed weights)

Two equal forms. The functional form is `init(key, ...) -> params` and `apply(params, x, ctx) -> y` over pytrees of
tensors (Flax/Haiku style), used by training. The module form is PyTorch-shaped (`new nn.Linear(4, 8)`,
`nn.Sequential`, `module.forward(x)`, `module.parameters()`, `stateDict`) and is a thin wrapper that holds params and
calls the functional form. `ctx` carries the train flag, the key for dropout, activation taps and buffer updates
(batch-norm statistics), all as returned data rather than mutation.

How a network is trained, end to end: parameters are a pytree; `apply` builds an autodiff graph; a loss from
`learn/losses` produces a scalar `Var`; `valueAndGrad` gives the gradient pytree; an `optim` transformation turns
gradients into updates; `fitAlgorithm` wraps this as an `Algorithm`, so `runAlgorithm` emits `loss`, metric and
parameter-norm events per step; `nn/describe` turns the model into an architecture spec that omai-render draws.
- `functional/activations.ts`: `relu`, `leakyRelu`, `elu`, `selu`, `gelu` (exact and tanh), `silu`, `mish`, `softplus`,
  `hardTanh`, `hardSigmoid`, `glu`, `swiglu`, `activation(name)`.
- `functional/linear.ts`: `linear`, `bilinear`, `embedding`, `embeddingBag`.
- `functional/conv.ts`: `conv1d`, `conv2d`, `conv3d`, `convTranspose1d`, `convTranspose2d`, depthwise and grouped
  convolution, `unfold`/`fold`.
- `functional/pooling.ts`: `maxPool1d`, `maxPool2d`, `avgPool1d`, `avgPool2d`, `adaptiveAvgPool`, `globalPool`,
  `lpPool`.
- `functional/normalisation.ts`: `batchNorm` (returning updated running statistics), `layerNorm`, `rmsNorm`,
  `groupNorm`, `instanceNorm`, `weightNorm`, `spectralNorm`.
- `functional/regularisation.ts`: `dropout`, `dropPath`, `alphaDropout`, `gaussianNoise`, `cutout`.
- `functional/attention.ts`: `scaledDotProductAttention` (masks, causal, sliding window, bias, soft cap),
  `multiHeadAttention`, grouped and multi-query attention, `attentionWeights` tap, mask builders.
- `functional/positional.ts`: `sinusoidalPositions`, `learnedPositions`, `rotaryEmbedding` (RoPE), `alibiBias`.
- `functional/recurrent.ts`: `rnnCell`, `gruCell`, `lstmCell`, `scan` (unroll over time), bidirectional helpers,
  `packSequences` with masks.
- `functional/state-space.ts`: `discretiseSsm`, `ssmConvolution` (S4 kernel), `selectiveScan` (Mamba),
  `linearAttention`, `deltaRule`, `lru`.
- `functional/sets-graphs.ts`: `setPool` (sum, mean, max with masks), `messagePassing`, `gcnConv`, `gatConv`,
  `sageConv`, `ginConv`, `edgeSoftmax`, `graphPool`.
- `init.ts`: `xavierUniform`, `xavierNormal`, `kaimingUniform`, `kaimingNormal`, `lecunNormal`, `orthogonal`,
  `truncatedNormal`, `zeros`, `ones`, `calculateGain`, fan computation.
- `module/module.ts`: `Module` (base class: `forward`, `parameters`, `namedParameters`, `buffers`, `stateDict`,
  `loadStateDict`, `train`, `eval`, `apply` to the functional form), `Parameter`.
- `module/containers.ts`: `Sequential`, `ModuleList`, `ModuleDict`, `Residual`, `Parallel`.
- `module/layers.ts`: `Linear`, `Bilinear`, `Embedding`, `Conv1d`, `Conv2d`, `ConvTranspose2d`, `MaxPool2d`,
  `AvgPool2d`, `AdaptiveAvgPool2d`, `Flatten`, `Unflatten`, `Dropout`, activation modules (`ReLU`, `GELU`, `Tanh`, ...),
  `Identity`.
- `module/normalisation.ts`: `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm2d`.
- `module/recurrent.ts`: `RNN`, `GRU`, `LSTM` (stacked, bidirectional), cells.
- `module/attention.ts`: `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerDecoderLayer`,
  `TransformerEncoder`, `TransformerDecoder`, `Transformer` (PyTorch semantics).
- `blocks/transformer.ts`: functional transformer pieces: pre-norm and post-norm blocks, feed-forward and gated
  feed-forward, causal decoder stack, encoder stack, cross-attention block, KV cache.
- `blocks/convolutional.ts`: residual block, bottleneck block, depthwise-separable block, squeeze-excitation, U-Net
  down and up blocks.
- `blocks/experts.ts`: routing (softmax, top-k, noisy top-k, switch, expert choice), `mixtureOfExperts` layer,
  load-balancing and router-z losses, routing statistics.
- `describe.ts`: `describeArchitecture` (functional model or module to an `ArchitectureSpec`: blocks, tensor shapes,
  repeats, skip connections, parameter counts), `summary` (torchinfo-style table), FLOP and memory estimates.
- `training/state.ts`: `TrainState` (params, optimiser state, buffers, step, key), `createTrainState`.
- `training/fit.ts`: `fitAlgorithm` and `fit` (epochs of minibatches from `data/batching`, gradient clipping,
  gradient accumulation, early stopping, validation metrics, divergence detection; events: loss, metrics,
  per-layer gradient norms, learning rate), `evaluate`, `predictBatches`.
- `training/callbacks.ts`: event-driven hooks expressed as trace consumers: `earlyStopping`, `reduceLrOnPlateau`,
  `checkpointBest`, `activationTaps`.
- `training/serialise.ts`: `saveParams`/`loadParams` (JSON with dtype and shape, safetensors-like layout),
  `paramsFromTorch` (load fixture weights exported by the oracle).
- `decoding.ts`: `greedyDecode`, `sampleDecode` (temperature, top-k, top-p, repetition penalty), `beamSearch`,
  `speculativeDecode`; each a traced algorithm, one token per step.
- `quantise.ts`: `quantise`/`dequantise` (symmetric, affine, per-channel), `fakeQuantise` (straight-through),
  `gptq`, `awq`, quantisation error reports.
- `equivariant.ts`: group convolutions on p4 (`p4Lift`, `p4Conv`), orientation pooling, rotation tests.

### 3.24 `rl/` (tier 8): reinforcement learning infrastructure (gymnasium, stable-baselines3 building blocks)

oracle: gymnasium (environment dynamics step for step), stable-baselines3 for return-computation helpers

Environments are functional at heart, `reset(key) -> { state, obs }` and `step(state, action, key) -> { state, obs,
reward, terminated, truncated, info }`, plus `render(state) -> Scene`. A gymnasium-shaped class wraps that
(`env.reset({ seed })`, `env.step(action)`, `env.render()`). Adding an environment is adding one file in
`methods/rl/environments/` that exports an `EnvDefinition`.
- `env.ts`: `Env<S, O, A>` (spec, observation and action spaces, reset, step, render, optional `model` for planning
  and `oracle` for regret), `EnvSpec` (id, max episode steps, reward range, render size), `EnvDefinition` (factory
  with typed options), `isEnvDefinition`.
- `gym.ts`: `GymEnv` (gymnasium-style wrapper holding state and key), `make(definition, options)`,
  `listEnvironments(namespace)` (finds definitions in an imported module, so no registry is edited).
- `wrappers.ts`: `timeLimit`, `recordEpisodeStatistics`, `normaliseObservation`, `normaliseReward`, `clipAction`,
  `rescaleAction`, `frameStack`, `actionRepeat`, `transformObservation`, `transformReward`, `flattenObservation`,
  `oneHotObservation`.
- `vector.ts`: `vectorEnv` (n copies with independent keys, autoreset), batched step.
- `agent.ts`: `Agent<G, O, A>` protocol (`init`, `act(state, obs, key)`, `learn(state, transition)`, `greedy`), `Policy`,
  `Transition`.
- `rollout.ts`: `rollout` (one environment, one agent, as a traced algorithm with `episode` events), `runEpisode`,
  `evaluatePolicy`, `compareAgents` (common random numbers, confidence bands).
- `buffers.ts`: `replayBuffer` (persistent, chunked), `prioritisedReplay` (sum tree), `nStepBuffer`,
  `rolloutBuffer` (on-policy), sampling by key.
- `returns.ts`: `discountedReturns`, `gae` (generalised advantage estimation), `tdTargets`, `lambdaReturns`,
  `nStepReturns`, `vtrace`.
- `policies.ts`: `epsilonGreedy`, `boltzmann`, `gaussianPolicy` (with tanh squashing and log-prob correction),
  `categoricalPolicy`, `deterministicPolicy`, exploration noise (Gaussian, Ornstein-Uhlenbeck).
- `mdp.ts`: `TabularMdp` (transition and reward tables, terminal states), `tabularMdp`, `valueIteration`,
  `policyIteration`, `policyEvaluation` (iterative and exact), `qFromV`, `greedyPolicy`, `occupancyMeasure`.
- `planning.ts`: `mcts` (UCT with a generative model, traced tree), `rolloutPlanner`, `cem` (cross-entropy method
  over action sequences).
- `off-policy.ts`: `ips`, `snips`, `clippedIps`, `directMethod`, `doublyRobust`, `fittedQEvaluation`, propensity
  estimation.
- `bandits.ts`: bandit problem interface (arms, contexts, expected rewards), `regret`, `laiRobbinsBound`,
  `klBernoulli`.
- `training.ts`: `trainAgent` (episodes or steps as a traced algorithm, checkpoints with doubling spacing, greedy
  evaluation episodes), `replayEpisode` (recompute any episode from the nearest checkpoint and its key).
## 4. omai-engine methods, `packages/engine/src/methods/`

Methods are named models, algorithms, environments, datasets and studies built from the core. Areas are grouped below
the way a reader looks for them; on disk they are flat folders (`methods/svm/`, `methods/cluster/`). Each area has an
`area.json` naming the other areas it may import (most import none). Every estimator here comes in both forms:
`svc(hyper).fit(data, ctx)` functions and an sklearn-style class (`new SVC({ kernel: rbf({ lengthScale: 1 }) })`).

### 4.1 Supervised learning (sklearn's model modules, statsmodels)

oracle: scikit-learn, statsmodels, mgcv and pygam outputs recorded once, lifelines

`linear/` (sklearn.linear_model)
- `least-squares.ts`: `linearRegression` (QR or SVD), `ridge` (Cholesky, SVD, with sample weights), `ridgeCv` (GCV and
  LOO shortcut), `ridgePath`, effective degrees of freedom.
- `sparse.ts`: `lasso` and `elasticNet` (coordinate descent with duality gap, traced), `lassoPath`, `lassoCv`,
  `lars` and `lassoLars`, `orthogonalMatchingPursuitRegressor`, `groupLasso`.
- `logistic.ts`: `logisticRegression` (binary and multinomial; L1, L2, elastic net; L-BFGS, Newton and SAGA solvers),
  `logisticRegressionCv`.
- `bayesian.ts`: `bayesianRidge` (evidence maximisation), `ard`, `bayesianLinearRegression` (closed-form posterior and
  predictive).
- `robust.ts`: `huberRegressor`, `ransacRegressor`, `theilSen`, `quantileRegressor` (LP).
- `online.ts`: `sgdClassifier`, `sgdRegressor`, `perceptron`, `passiveAggressive` (partial fit as a traced
  algorithm).
- `isotonic.ts`: `isotonicRegression` (pool adjacent violators, traced), increasing or decreasing, interpolation.
- `subset.ts`: `bestSubset`, `forwardStepwise`, `backwardStepwise`, `forwardStagewise`.

`glm/` (statsmodels GLM, sklearn PoissonRegressor and friends)
- `irls.ts`: penalised iteratively reweighted least squares for any family and link (step halving, traced).
- `glm.ts`: `glm` (fit, coefficients with standard errors, Wald tests, deviance, dispersion, AIC), `glmPredict`
  (link and response scale, intervals), residuals (response, Pearson, deviance, working).
- `count.ts`: `poissonRegression`, `negativeBinomialRegression` (theta by ML), `zeroInflatedRegression`,
  `tweedieRegression`.
- `ordinal.ts`: `ordinalRegression` (cumulative logit and probit, continuation ratio, adjacent category), threshold
  methods.

`gam/` (mgcv, pygam)
- `terms.ts`: term constructors `s` (P-spline smooth with optional shape constraint and `by`), `te` and `ti` (tensor
  products), `cyclic`, `thinPlate`, `linearTerm`, `factorTerm`, `randomEffect`.
- `design.ts`: model matrix and penalty assembly, identifiability (sum-to-zero) constraints, null spaces.
- `fit.ts`: `gam` with six interchangeable traced fitters from one objective: penalised IRLS, backfitting, gradient
  descent, Adam, L-BFGS, SGD.
- `smoothness.ts`: smoothing parameter selection by REML, ML, GCV and UBRE (outer Newton or Nelder-Mead).
- `inference.ts`: effective degrees of freedom per term, Bayesian credible bands, approximate p-values, `partialEffect`,
  `gamSummary`, `gamCheck`.
- `gamlss.ts`: distributional regression (location, scale and shape each with their own additive predictor).

`svm/` (sklearn.svm, LIBSVM, LIBLINEAR)
- `smo.ts`: sequential minimal optimisation for the kernel dual (maximal violating pair and second-order working set,
  shrinking), as a traced algorithm with the active set per step.
- `classifier.ts`: `svc` (kernel SVM, one-vs-one multiclass, Platt probabilities), `nuSvc`, decision function, support
  vectors.
- `regressor.ts`: `svr`, `nuSvr` (epsilon-insensitive dual).
- `linear.ts`: `linearSvc` (dual coordinate descent), `pegasos`, `crammerSinger` multiclass SVM.
- `one-class.ts`: `oneClassSvm`, `svdd` (support vector data description).

`trees/` (sklearn.tree)
- `tree.ts`: the `Tree` data structure (nodes, splits, values, impurities, counts), `predictTree`, `applyTree`,
  `decisionPath`, `nodeRegions`, `treeToGraph` (for the tree diagram), `exportText`.
- `criteria.ts`: impurity criteria (gini, entropy, log loss, squared error with a two-pass update, absolute error,
  Poisson), `bestSplit` (exact and histogram-binned), surrogate splits.
- `grow.ts`: `growTree` (depth-first, breadth-first and best-first, one node per step as a traced algorithm),
  `decisionTreeClassifier`, `decisionTreeRegressor`, stopping rules.
- `prune.ts`: `costComplexityPath`, `prune`, reduced-error pruning, `featureImportances`.

`ensemble/` (sklearn.ensemble, XGBoost and LightGBM ideas)
- `bagging.ts`: `baggingClassifier`, `baggingRegressor`, out-of-bag estimates, `inBagMatrix`.
- `forest.ts`: `randomForestClassifier`, `randomForestRegressor`, `extraTrees`, `randomTreesEmbedding`, impurity and
  permutation importances.
- `adaboost.ts`: `adaBoostClassifier` (SAMME), `adaBoostRegressor` (R2), staged predictions.
- `gradient-boosting.ts`: `gradientBoostingClassifier` and `gradientBoostingRegressor` (squared, absolute, Huber,
  quantile, log losses; Newton leaf values; subsampling), staged predictions.
- `histogram-boosting.ts`: `histGradientBoosting` (binned features, leaf-wise growth, missing values, early stopping).
- `stacking.ts`: `votingClassifier`, `votingRegressor`, `stackingClassifier`, `stackingRegressor`.

`neighbours/` (sklearn.neighbors)
- `knn.ts`: `kNeighborsClassifier`, `kNeighborsRegressor`, `radiusNeighbors*` (uniform and distance weights, any
  metric, tree or brute force).
- `centroid.ts`: `nearestCentroid` (with shrinkage), `kernelDensityClassifier`.
- `metric-learning.ts`: `nca` (neighbourhood components analysis), `lmnn`.

`naive-bayes/` and `discriminant/` (sklearn.naive_bayes, sklearn.discriminant_analysis)
- `naive-bayes/naive-bayes.ts`: `gaussianNB`, `multinomialNB`, `bernoulliNB`, `complementNB`, `categoricalNB`
  (smoothing, partial fit).
- `discriminant/discriminant.ts`: `linearDiscriminantAnalysis` (SVD, eigen and least-squares solvers, shrinkage,
  transform), `quadraticDiscriminantAnalysis` (regularised).

`gp/` (sklearn.gaussian_process, GPyTorch)
- `regression.ts`: `gpPrior`, `gpPosterior` (Cholesky with reported jitter), `logMarginalLikelihood` and its gradient,
  `gaussianProcessRegressor` (hyperparameters by L-BFGS with restarts), posterior samples.
- `classification.ts`: `gaussianProcessClassifier` with Laplace and EP approximations, predictive probabilities.
- `sparse.ts`: inducing-point approximations (SoR, DTC, FITC, VFE), `svgp` (stochastic variational GP).
- `latent.ts`: `gplvm`, `bayesianGplvm`.
- `relevance-vector.ts`: `rvmRegressor`, `rvmClassifier`.

`multiclass/` (sklearn.multiclass, sklearn.multioutput)
- `reductions.ts`: `oneVsRest`, `oneVsOne`, `outputCode` (exhaustive, random, error-correcting), `nestedDichotomies`.
- `multilabel.ts`: `binaryRelevance`, `classifierChain`, `labelPowerset`.
- `multioutput.ts`: `multiOutputRegressor`, `regressorChain`.

`survival/` (lifelines, scikit-survival)
- `nonparametric.ts`: `kaplanMeier`, `nelsonAalen`, `logRankTest`, median survival with confidence intervals.
- `cox.ts`: `coxPh` (Breslow and Efron ties, Newton with a solve, not an inverse), baseline hazard, survival curves,
  `concordanceIndex`.
- `parametric.ts`: accelerated failure time models (Weibull, log-normal, log-logistic), time ratios.
- `metrics.ts`: `brierScoreSurvival`, `integratedBrierScore`, `timeDependentAuc`.

`rules/` (Orange, wittgenstein, mlxtend)
- `covering.ts`: `cn2`, `ripper` (IREP*, MDL pruning, traced), `sequentialCovering`, rule quality measures.
- `c45-rules.ts`: rules from trees with pessimistic pruning.
- `association.ts`: `apriori`, `fpGrowth`, `associationRules` (support, confidence, lift, conviction), `cba` and
  `cmar` classifiers.
- `rule-lists.ts`: `bayesianRuleList`, `fallingRuleList`, rule-list prediction.
- `subgroups.ts`: `subgroupDiscovery` (beam and exhaustive with optimistic pruning; WRAcc, lift, binomial quality),
  exceptional model mining.

### 4.2 Unsupervised learning (sklearn.cluster, sklearn.decomposition, sklearn.manifold, sklearn.mixture, sklearn.covariance)

oracle: scikit-learn, umap-learn, hdbscan

`cluster/`
- `kmeans.ts`: `kmeans` (k-means++ initialisation, Lloyd and Elkan, traced), `miniBatchKmeans`, `bisectingKmeans`,
  `kmeansElbow`.
- `medoids.ts`: `kMedoids` (PAM build and swap, traced), `clara`.
- `hierarchical.ts`: `agglomerative` (single, complete, average, Ward; merge events), `linkage` (scipy format),
  `cutTree`, `dendrogram` data, `cophenetic`.
- `density.ts`: `dbscan`, `hdbscan`, `optics`, `meanShift` (traced).
- `spectral.ts`: `spectralClustering` (affinities, normalised Laplacian, k-means or discretisation),
  `spectralCoclustering`.
- `exemplar.ts`: `affinityPropagation`, `birch`.
- `constrained.ts`: must-link and cannot-link k-means (COP-KMeans).

`mixture/`
- `gaussian.ts`: `gaussianMixture` (EM; full, tied, diagonal and spherical covariances; traced responsibilities),
  BIC and AIC model selection.
- `bayesian.ts`: `bayesianGaussianMixture` (variational, Dirichlet and Dirichlet-process priors).
- `other.ts`: mixtures of Bernoullis, multinomials and Student-t (EM).

`decomposition/`
- `pca.ts`: `pca` (SVD, randomized, explained variance, whitening, inverse transform), `incrementalPca`, `sparsePca`,
  `probabilisticPca` (EM, traced).
- `kernel-pca.ts`: `kernelPca` (centring, pre-images).
- `factor-analysis.ts`: `factorAnalysis` (EM, solves rather than inverses), varimax and quartimax rotations.
- `ica.ts`: `fastIca` (parallel and deflation, logcosh, exp and cube contrasts, traced).
- `nmf.ts`: `nmf` (multiplicative updates for Frobenius and KL, HALS, coordinate descent; traced),
  `latentDirichletAllocationOnline` (sklearn's LDA estimator; the Bayesian versions are in `models/topic`).
- `svd.ts`: `truncatedSvd` (LSA), `dictionaryLearning` and `sparseCoder` (wrapping `signal/sparse`).
- `cross-decomposition.ts`: `cca`, `plsRegression`, `plsCanonical`, `plsSvd`.
- `random-projection.ts`: `gaussianRandomProjection`, `sparseRandomProjection`, distortion reports.

`manifold/`
- `mds.ts`: `classicalMds`, `metricMds` (SMACOF, traced stress), `nonMetricMds`.
- `isomap.ts`: `isomap` (neighbour graph, geodesics, MDS).
- `lle.ts`: `locallyLinearEmbedding` (standard, modified, Hessian, LTSA).
- `spectral.ts`: `spectralEmbedding` (Laplacian eigenmaps), `diffusionMap`.
- `tsne.ts`: `tsne` (perplexity calibration, exact and Barnes-Hut gradients, early exaggeration; traced embeddings).
- `umap.ts`: `umap` (fuzzy simplicial set, spectral initialisation, SGD with negative sampling; traced).
- `pacmap.ts`: `pacmap` (near, mid-near and far pairs, three-phase weights).
- `som.ts`: `selfOrganisingMap` (traced neighbourhood shrinkage), U-matrix.
- `trustworthiness.ts`: `trustworthiness`, `continuity`, neighbourhood preservation metrics.

`embedding/` (gensim, node2vec, PyKEEN)
- `word.ts`: `word2vec` (skip-gram and CBOW with negative sampling, traced), `glove`, `fastText` subword vectors.
- `graph.ts`: `deepWalk`, `node2vec` (biased walks), `netmf`, `spectralNodeEmbedding`.
- `knowledge-graph.ts`: `transE`, `distMult`, `complEx`, `rotatE`, link-prediction ranking metrics.

`anomaly/` (sklearn outlier detection, PyOD)
- `isolation.ts`: `isolationForest`, `extendedIsolationForest`, path-length scores.
- `neighbours.ts`: `localOutlierFactor` (novelty and outlier modes), `knnScore`, `cof`.
- `statistical.ts`: `ellipticEnvelope` (minimum covariance determinant), `mahalanobisScore`, `pcaReconstructionScore`,
  `hbos`.
- `combine.ts`: score normalisation (rank, z, unify), score combination, thresholds (contamination, EVT, quantile).

`covariance/` (sklearn.covariance)
- `estimators.ts`: `empiricalCovariance`, `shrunkCovariance`, `ledoitWolf`, `oas` (wrapping `stats/covariance`), and
  `minCovDet` (fast MCD, traced).
- `graphical-lasso.ts`: `graphicalLasso` (block coordinate descent, traced), `graphicalLassoCv`, precision-matrix
  graphs for drawing.

### 4.3 Learning paradigms

`semi-supervised/` (sklearn.semi_supervised)
- `self-training.ts`: `selfTraining` (thresholds, k best), `coTraining`, `triTraining`.
- `graph-based.ts`: `labelPropagation`, `labelSpreading`, `harmonicFunction`.
- `margin-based.ts`: `transductiveSvm`, `laplacianRls`, `laplacianSvm`, `entropyMinimisation`.

`weak-supervision/` (Snorkel, cleanlab)
- `label-models.ts`: `majorityVote`, `dawidSkene` (EM, traced), `snorkelLabelModel`, crowd worker confusion estimates.
- `noisy-labels.ts`: `confidentLearning` (joint noise matrix, label issues), forward and backward loss correction.
- `positive-unlabelled.ts`: `elkanNoto`, `nnPu`, class-prior estimation.
- `proportions.ts`: learning from label proportions (`proportionLoss`, `alterSvm`, `meanMap`).
- `multiple-instance.ts`: `miSvm`, `attentionMil`.

`transfer/`
- `domain-adaptation.ts`: `mmdAdaptation`, `coral`, `dann` (gradient reversal), importance-weighted training.
- `label-shift.ts`: `bbse` (black-box shift estimation), `emPriorShift`, posterior reweighting.
- `continual.ts`: naive fine-tuning, `ewc` (elastic weight consolidation), experience replay, forgetting metrics.
- `meta-learning.ts`: `maml` (second order through autodiff), `prototypicalNetworks`, few-shot episodes.

### 4.4 Probabilistic models (`models/`)

oracle: hmmlearn, sklearn-crfsuite outputs, gensim LDA (statistical agreement), trueskill, numpyro
- `hmm.ts`: hidden Markov models with categorical, Gaussian and GMM emissions: `baumWelch` (traced), `viterbi`,
  `posteriorStates`, `sampleHmm`; the dishonest casino.
- `crf.ts`: linear-chain CRF (`crfFit` by L-BFGS, marginals, Viterbi), template features (CRF++ style).
- `topic.ts`: `lda` (collapsed Gibbs and mean-field VI, traced), labelled LDA, `topWords`, perplexity.
- `topic-extensions.ts`: `plsa`, `hdp`, `correlatedTopicModel`, `dynamicTopicModel`.
- `mixtures.ts`: Bayesian mixtures, Dirichlet-process mixtures (collapsed Gibbs, Chinese restaurant process traces).
- `conjugate.ts`: conjugate updates and posterior predictives: beta-binomial, gamma-Poisson, normal-normal,
  normal-gamma, Dirichlet-multinomial, normal-inverse-Wishart; CAVI for normal-gamma against the exact posterior.
- `rating.ts`: `elo`, `glicko`, `glicko2`, `trueSkill` (EP on the factor graph), TrueSkill through time,
  `bradleyTerry` (MM), `plackettLuce`.
- `irt.ts`: item response theory (1PL, 2PL, 3PL), learner models, item information.
- `lattice.ts`: Ising and Potts models on lattices: exact on chains, belief propagation, Gibbs and Swendsen-Wang
  sampling, magnetisation curves.
- `bayesian-regression.ts`: Bayesian linear, logistic and hierarchical regression as probabilistic programs, eight
  schools, radon.
- `bayesian-classifiers.ts`: `bayesPointMachine` (EP), `adPredictor` (online probit regression).
- `state-space.ts`: Bayesian structural time series (local level, local linear trend, seasonal) via Kalman filtering
  and sampling.

### 4.5 Causal inference (`causal/`) (DoWhy, EconML, causal-learn)

oracle: dowhy and econml on fixed synthetic data, causal-learn for discovery
- `scm.ts`: structural causal models with explicit noise (`scm`, `sampleScm`, `intervene`, `counterfactual`), drawn as
  graphs.
- `identification.ts`: backdoor and frontdoor criteria, instrumental variables, `identify` (adjustment sets).
- `estimation.ts`: average treatment effects: regression adjustment, IPW, AIPW (doubly robust), matching (propensity,
  nearest neighbour), stratification, difference in differences, two-stage least squares, regression discontinuity.
- `heterogeneous.ts`: CATE estimators: S, T, X and DR learners, causal forests.
- `discovery.ts`: `pc`, `fci`, `ges`, `lingam` (wrapping `pgm/learning`).
- `sensitivity.ts`: E-values, placebo and refutation tests.
### 4.6 Domains

`timeseries/` (statsmodels.tsa, sktime basics)
oracle: statsmodels
- `features.ts`: `acf`, `pacf`, `ccf`, `periodogramFeatures`, `seasonalStrength`, differencing and lag helpers.
- `arima.ts`: `armaProcess` (roots, impulse response, theoretical ACF), `arima` and `sarimax` (CSS and exact likelihood
  via Kalman filtering, traced fit), `forecast` with intervals, `autoArima` (order search by AIC).
- `smoothing.ts`: `simpleExponentialSmoothing`, `holt` (damped trend), `holtWinters`, `ets` (error-trend-season state
  space).
- `decomposition.ts`: `seasonalDecompose` (classical), `stl` (robust), `mstl`.
- `volatility.ts`: `garch`, `egarch`, `gjrGarch` fits and forecasts.
- `changepoints.ts`: changepoint studies combining `dynamics/filtering/changepoint` with segment models.
- `forecasting.ts`: baselines (`naive`, `seasonalNaive`, `drift`, `mean`), `backtest` (rolling origin), forecast
  metrics (`mase`, `smape`, `msis`).

`text/` (nltk, spaCy-like pipelines)
oracle: nltk, kenlm on small corpora
- `language-models.ts`: n-gram language models with add-k, Kneser-Ney and Witten-Bell smoothing, perplexity,
  generation.
- `tagging.ts`: part-of-speech and named-entity tagging with HMMs and CRFs; chunking.
- `classification.ts`: text classification pipelines (TF-IDF with naive Bayes, logistic regression, linear SVM).
- `keywords.ts`: `textRank`, `rake`, TF-IDF keywords, extractive summarisation.
- `hyphenation.ts`: PATGEN pattern induction, CRF and neural hyphenators, evaluation.
- `tokenisers.ts`: a preset suite of trained tokenisers (character, byte, word, BPE, byte-level BPE, WordPiece,
  Unigram) with comparison statistics.

`vision/` (OpenCV, torchvision.ops)
oracle: torchvision.ops, OpenCV
- `boxes.ts`: box formats and conversions, `boxIou` (IoU, GIoU, DIoU, CIoU), `nms`, `softNms`, `batchedNms`.
- `detection.ts`: anchor generation, box encoding and decoding, anchor matching, FCOS targets, FPN level assignment.
- `two-view.ts`: homography and fundamental-matrix estimation with RANSAC, triangulation, epipolar geometry.
- `segmentation.ts`: connected components, region growing, superpixels and graph-cut segmentation studies.
- `descriptors.ts`: HOG, simple keypoint descriptors and matching.

`dynamics/` (named systems)
oracle: scipy.integrate
- `systems.ts`: named continuous systems with parameters and known behaviour: Lorenz, Rossler, Van der Pol, Duffing,
  simple, damped and double pendulum, Lotka-Volterra, SIR and SEIR, Kuramoto oscillators, Hopf normal form.
- `maps.ts`: logistic, tent, sine, Henon and standard maps with bifurcation and Lyapunov studies.
- `nonlinear.ts`: Poincare sections, limit cycles and Floquet multipliers, Lyapunov-function checks.
- `pdes.ts`: named PDE problems with exact solutions (heat, wave, Burgers, Fisher-KPP, Allen-Cahn) for the solvers.
- `control-studies.ts`: PID loops, inverted-pendulum stabilisation, LQG and MPC on example plants.

### 4.7 Reinforcement learning applications (`rl/`)

oracle: gymnasium (step-for-step dynamics), stable-baselines3 learning curves recorded as loose references

`rl/environments/`: each file exports `EnvDefinition`s; each environment has `render(state) -> Scene`.
- `bandits.ts`: `bernoulliBandit`, `gaussianBandit`, `contextualLinearBandit`, `adversarialBandit`.
- `gridworlds.ts`: `gridWorld` (Russell-Norvig), `cliffWalking`, `frozenLake` (4x4, 8x8, slippery), `windyGridWorld`,
  `maze` (ASCII layouts with traps and keys).
- `classic-control.ts`: `cartPole` (CartPole-v1), `pendulum` (Pendulum-v1), `mountainCar` (discrete and continuous),
  `acrobot`.
- `toy-text.ts`: `blackjack`, `taxi`, `nChain`, `riverSwim`.
- `games.ts`: two-player games for search and self-play: `ticTacToe`, `connectFour`, `nim`.
- `planning-models.ts`: differentiable dynamics models of the control environments (for LQR, MPC and model-based RL).

`rl/agents/`: each file exports agents implementing the `Agent` protocol.
- `random.ts`: `randomAgent`, `constantAgent`.
- `bandits.ts`: `epsilonGreedy`, `ucb1`, `klUcb`, `thompsonBernoulli`, `thompsonGaussian`, `exp3`, `linUcb`,
  `linearThompson`, `gradientBandit`.
- `tabular.ts`: `qLearning`, `sarsa`, `expectedSarsa`, `nStepSarsa`, `monteCarloControl`, `td0`, `tdLambda`,
  `dynaQ`, `doubleQLearning`.
- `planning.ts`: `valueIterationAgent`, `policyIterationAgent`, `mctsAgent`, `minimaxAgent` (alpha-beta).
- `value-based.ts`: `dqn` (replay, target network, Huber, double and dueling variants), `c51` (distributional).
- `policy-gradient.ts`: `reinforce` (with baseline), `a2c`, `ppo` (clipped surrogate, GAE, approximate KL).
- `actor-critic.ts`: `ddpg`, `td3`, `sac` (temperature tuning).
- `control.ts`: `lqrAgent`, `swingUpAgent` (energy shaping plus LQR), `cemAgent`, `mpcAgent`.
- `offline.ts`: `behaviourCloning`, `cql` (discrete), `fittedQIteration`.

### 4.8 Generative models (`generative/`)

oracle: PyTorch (losses and single steps with fixed weights), diffusers schedules
- `autoencoders.ts`: `autoencoder`, `vae` (beta-VAE, Gaussian and Bernoulli likelihoods), `conditionalVae`, ELBO terms
  traced separately.
- `vector-quantised.ts`: `vqVae` (straight-through, codebook and commitment losses, EMA codebooks), residual VQ,
  codebook usage metrics.
- `gan.ts`: `gan` (minimax and non-saturating), `wganGp`, hinge GAN, conditional GAN, mode-coverage diagnostics.
- `flows.ts`: `planarFlow`, `realNvp`, `maf`, `glowLite` (actnorm, invertible 1x1, coupling), log-density and sampling.
- `diffusion/schedules.ts`: linear, cosine and sigmoid noise schedules, VP, sub-VP and VE SDEs, one time axis for
  discrete and continuous.
- `diffusion/ddpm.ts`: forward noising, epsilon, x0 and v parameterisations, training loss, ancestral sampling, DDIM.
- `diffusion/score.ts`: denoising score matching, reverse SDE samplers (Euler-Maruyama, predictor-corrector),
  probability flow ODE with log-likelihoods.
- `diffusion/guidance.ts`: classifier guidance, classifier-free guidance, guidance-scale sweeps.
- `energy.ts`: energy-based models, contrastive divergence with Langevin sampling and replay buffers, JEM.
- `boltzmann.ts`: restricted Boltzmann machines (CD-k, persistent CD, exact log partition for small models), deep
  belief networks.
- `autoregressive.ts`: `made`, `nade`, `pixelCnnLite`, autoregressive sampling.
- `densities.ts`: shared helpers: mixture log densities on grids, mode maps, sample-quality metrics for 2-D toy
  densities.

### 4.9 Neural network models (`neural/`)

oracle: PyTorch (forward and backward of each model with exported weights); Hugging Face for tokenizer-level checks

Each file defines a model with `init`, `apply`, `describe` (its architecture spec, which omai-render draws), a
training recipe as a traced algorithm, and an sklearn-style estimator where that makes sense.
- `mlp.ts`: `mlp`, `mlpClassifier`, `mlpRegressor` (sklearn.neural_network semantics).
- `cnn.ts`: `leNet`, `vggLite`, `resNetLite` (CIFAR-style), feature-map taps.
- `rnn.ts`: character RNN and LSTM language models, `seq2seq` with Bahdanau and Luong attention.
- `transformer.ts`: the original encoder-decoder Transformer (Vaswani et al.) for sequence tasks, with attention maps
  traced per head.
- `gpt.ts`: decoder-only GPT (learned, sinusoidal, RoPE or ALiBi positions; tied embeddings; RMS or layer norm),
  next-token training, generation through `nn/decoding`, KV cache.
- `bert.ts`: encoder-only masked language model, `[CLS]` classification head.
- `vit.ts`: vision transformer (patch embedding, class token), attention rollout.
- `state-space.ts`: S4, Mamba and linear-attention language models, associative recall studies.
- `gnn.ts`: node classification (GCN, GAT, GraphSAGE), graph classification (GIN with pooling), link prediction (GAE),
  oversmoothing studies.
- `neural-ode.ts`: neural ODEs (NODE, ANODE, second order), continuous normalising flows, adjoint versus backprop.
- `physics-informed.ts`: PINNs for Burgers, heat and Poisson problems, hard constraints, inverse problems.
- `operators.ts`: DeepONet, Fourier neural operator, comparisons with dense operators.
- `contrastive.ts`: SimCLR, CLIP-style two-tower models, alignment and uniformity, zero-shot classification.
- `self-supervised.ts`: BYOL, DINO (EMA teacher, centring), VICReg, collapse diagnostics.
- `semi-supervised.ts`: Pi-model, mean teacher, VAT, MixMatch, FixMatch (FlexMatch, FreeMatch thresholds).
- `adaptation.ts`: LoRA, rsLoRA, PiSSA, adapters, prefix tuning on small models.
- `alignment.ts`: reward models (Bradley-Terry), RLHF with PPO, GRPO and RLOO advantages, DPO, best-of-n.
- `mixture-density.ts`: mixture density networks.
- `mixture-of-experts.ts`: mixture-of-experts models (EM for linear experts, gated networks with load balancing).
- `phenomena.ts`: grokking on modular arithmetic, double descent in width, lottery tickets, edge of stability.
- `interpretability.ts`: linear probes, logit lens, activation patching, attention-head analysis, sparse autoencoders.
- `efficiency.ts`: parameter, FLOP and memory estimators, scaling-law fits, quantisation studies (with `nn/quantise`),
  knowledge distillation.

### 4.10 Retrieval and recommendation (`retrieval/`)

oracle: faiss, rank_bm25, implicit, scikit-learn
- `search.ts`: `bm25Index` and search, dense retrieval (dual encoders), hybrid fusion (reciprocal rank fusion),
  re-ranking.
- `indexes.ts`: approximate nearest-neighbour indexes (IVF, PQ, IVF-PQ, HNSW, LSH) built on `numerics/spatial`, with
  recall-versus-speed benchmarks.
- `learning-to-rank.ts`: `rankNetModel`, `lambdaMart`, `listNetModel`, feature-based rankers.
- `recommenders/neighbourhood.ts`: `popularity`, `userKnn`, `itemKnn` (cosine, shrinkage), `slim`.
- `recommenders/factorisation.ts`: `matrixFactorisation` (SGD, biases), `als`, `implicitAls`, `bpr`, `svdPlusPlus`.
- `recommenders/neural.ts`: `ncf`, factorisation machines (`fm`, `ffm`), `deepFm`, `wideAndDeep`, two-tower models.
- `recommenders/sequential.ts`: session and sequence recommenders (Markov chains, GRU4Rec, SASRec).
- `evaluation.ts`: offline evaluation protocols (leave-last-out, temporal splits), interleaving, counterfactual
  evaluation with logged propensities.
- `simulation.ts`: feedback-loop simulations (popularity bias, exposure, filter bubbles).

### 4.11 Evaluation suites (`evaluation/`)

oracle: sacrebleu, rouge-score, jiwer, pycocotools, torchmetrics, scikit-image
- `text.ts`: `bleu`, `chrf`, `rouge` (N and L), `meteorLite`, `ter`, `wer`, `cer`, `exactMatch`, token F1,
  `bertScore` from embeddings.
- `detection.ts`: COCO and VOC `meanAveragePrecision`, `panopticQuality`, mask IoU, Dice, boundary F1.
- `images.ts`: `psnr`, `ssim`, `msSsim`, `lpipsLite`, `sam` (spectral angle), `ergas`.
- `audio.ts`: `snr`, `siSdr`, permutation-invariant scoring.
- `generative.ts`: `fid`, `kid`, `inceptionScore`, generative precision and recall, density and coverage.
- `comparison.ts`: comparing models: McNemar, 5x2cv tests, corrected resampled t-test, Friedman and Nemenyi with
  critical-difference data, Bayesian signed-rank.
- `valuation.ts`: data valuation: influence functions, TracIn, data Shapley (truncated Monte Carlo), KNN-Shapley.

### 4.12 Information theory (`information/`)

oracle: closed forms; numpy
- `coding.ts`: Huffman (D-ary, traced tree building), Shannon-Fano, canonical codes, arithmetic coding (integer
  range coder), LZ77 and LZW, Kraft inequality.
- `channels.ts`: channel capacity and rate-distortion by Blahut-Arimoto (traced), binary symmetric and erasure
  channels, Gaussian channel.
- `error-correction.ts`: parity, Hamming codes, repetition codes, Reed-Solomon over small fields, convolutional codes
  with Viterbi decoding, bounds (Hamming, Singleton, Plotkin).
- `projections.ts`: information projections (moment matching, reverse KL fits), maximum-entropy distributions.

### 4.13 Classic algorithms (`algorithms/`)

- `dynamic-programming.ts`: knapsack, longest common subsequence, edit distance, matrix-chain order, optimal binary
  search trees, coin change; each with its table and traceback as trace events.
- `search.ts`: uninformed and informed search on puzzles (8-puzzle, maze), minimax with alpha-beta, iterative deepening.
- `constraints.ts`: constraint satisfaction (backtracking, forward checking, AC-3), n-queens, Sudoku, graph colouring.
- `sorting.ts`: classic sorting algorithms with comparison and swap traces.
- `program-synthesis.ts`: FlashFill-style string programs with version-space algebra.

### 4.14 Learning theory and studies (`theory/`)

Each file is a reproducible study that returns data and traces for figures.
- `bias-variance.ts`: bias, variance and noise by resampling training sets, complexity sweeps.
- `double-descent.ts`: random-feature regression across the interpolation threshold.
- `capacity.ts`: shattering, VC dimension demonstrations, Rademacher complexity.
- `concentration.ts`: Markov, Chebyshev, Hoeffding, Bernstein and Chernoff bounds against simulation; LLN and CLT.
- `model-selection.ts`: selection optimism, nested cross-validation, one-standard-error rule.
- `optimism.ts`: training-error optimism, covariance penalties, effective degrees of freedom.
- `shrinkage.ts`: James-Stein estimators and risk.
- `adaptive-data-analysis.ts`: Thresholdout and the Ladder against leaderboard attacks.
- `algorithm-comparison.ts`: type I error of classifier-comparison tests.
- `sample-complexity.ts`: PAC bounds against learning curves.

### 4.15 Datasets (`datasets/`)

Every dataset is a function returning a schema from `data/schema` (generated datasets take a key and options; static
ones take nothing). Small static data is embedded as generated `*.data.ts` files vendored by the oracle with source,
licence and checksum; larger data is a release asset loaded through `data/loader`. Generators attach `Truth` where it
is known.
- `regression/synthetic.ts`: `makeRegression`, `friedman1`, `friedman2`, `friedman3`, `regression1d` (sine, step,
  sinc, doppler, heteroscedastic), `additive` (GAM test data), `curve1d` with true quantiles.
- `regression/real.ts`: `diabetes`, `anscombe`, `mtcars`, `bostonFree` (a licence-clean housing set), `growthChart`.
- `classification/points.ts`: `blobs`, `moons`, `circles`, `xor`, `spirals`, `checkerboard`, `gaussianQuantiles`,
  `makeClassification` (informative, redundant and noise features).
- `classification/real.ts`: `iris`, `wine`, `breastCancer`, `digits8x8`, `titanic`, `penguins`.
- `clustering.ts`: `rings`, `anisotropicBlobs`, `varianceBlobs`, `oldFaithful`.
- `manifolds.ts`: `swissRoll`, `sCurve`, `severedSphere`, `torus`.
- `densities.ts`: 2-D densities with exact log density (`gaussianRing`, `pinwheel`, `banana`, `funnel`, `mixtures`) for
  samplers, flows and diffusion.
- `objectives.ts`: optimisation test functions with gradients, Hessians and known minima (Rosenbrock, Himmelblau,
  Beale, Rastrigin, Ackley, Branin, quadratic bowls), tuning surfaces.
- `ranking.ts`: `ratings` (low-rank explicit), `clickLogs` (position and cascade models with propensities),
  `implicitFeedback`, `zipfCatalogue`, `movieLensSmall` (remote asset).
- `corpus.ts`: `toyCorpus`, `topicCorpus`, `driftingTopics`, named worked-example corpora, `nurseryRhymes`,
  `tinyShakespeare` (remote asset).
- `sequence-tasks.ts`: copy, reverse, sort, Dyck languages, addition, associative recall, parity, modular arithmetic
  (grokking).
- `timeseries.ts`: AR and ARMA processes, sinusoids in noise, uneven sampling, regime switches, changepoint series,
  `airlinePassengers`, `sunspots`, `coalMining`, `mitBihEcg` (remote asset).
- `graphs.ts`: `karateClub`, `lesMiserables`, contextual stochastic block models, graph families, small molecules,
  knowledge-graph triples.
- `images.ts`: shapes, 5x7 digit glyphs, bars and stripes, geometric scenes with true edges and corners,
  `fashionTiny` (remote asset).
- `trajectories.ts`: logged bandit data, offline RL datasets from fixed behaviour policies.
- `weak-labels.ts`: labelling functions with known accuracy, crowd labels, positive-unlabelled sets, label-proportion
  bags.
- `operators.ts`: PDE operator-learning pairs (heat, Burgers, antiderivative).
- `predictions.ts`: model outputs with known miscalibration and survival data with known hazards, for metrics.

### 4.16 Hyperparameter optimisation (`tuning/`) (Optuna, scikit-optimize, BoTorch ideas)

oracle: scikit-optimize and optuna on fixed seeds (statistical agreement), closed-form acquisitions
- `bayesian.ts`: the Bayesian optimisation loop (GP or forest surrogate, initial designs, hyperparameter refits) as a
  traced algorithm with posterior and acquisition snapshots.
- `acquisitions.ts`: `probabilityOfImprovement`, `expectedImprovement`, `logExpectedImprovement`,
  `upperConfidenceBound`, `thompsonSampling`, `maxValueEntropySearch`, `qExpectedImprovement`, local penalisation.
- `tpe.ts`: tree-structured Parzen estimator over typed spaces.
- `multi-fidelity.ts`: successive halving, Hyperband, ASHA, BOHB.
- `population.ts`: population-based training with exploit and explore events.
- `portfolios.ts`: greedy portfolios, warm starting across tasks.
- `importance.ts`: fANOVA hyperparameter importance.
- `sweeps.ts`: grid and random sweeps, learning-rate range tests.
## 5. omai-render, `packages/render/src/`

Headless first: everything outside `react/` returns data (ECharts option objects, layouts, `Scene` graphs, SVG strings)
and runs in Node, so the slide and blog tools can render figures without a browser. `react/` binds that data to React
19. The render package depends on omai-engine for types (`Scene`, `Run`, `ComputationGraph`, `ArchitectureSpec`,
graphical-model structures, `Graph`) and never on its algorithms.

Entry points: `omai-render` (all headless parts), `omai-render/<part>`, and `omai-render/react`.

oracle: none (rendering is checked by snapshot tests of options and scenes, and browser tests in Chromium)

`theme/`
- `tokens.ts`: design tokens for light and dark (`fg`, `bg`, `surface`, `muted`, `grid`, `accent`, semantic colours,
  `tab0`..`tab9`), `tokens(mode)`, `cssVariables(mode)` so consuming apps use the same values.
- `palettes.ts`: `tab10`, sequential and diverging ramps, `interpolateColour`, `colourFor(index)` (wraps with a
  documented rule), colour-blind-safe checks.
- `echarts.ts`: `echartsTheme(mode)` (registered light and dark themes using tab10), `baseOption`.
- `format.ts`: number formatting (significant figures, engineering notation), nice tick steps, log ticks, time
  formatting.

`charts/` (ECharts option builders; all return `EChartsOption`)
- `lines.ts`: `lineChart` (curves, steps, areas, bands, fill between), `seriesFromTensor`, multi-series legends.
- `points.ts`: `scatterChart` (labels, sizes, categorical colours), `bubbleChart`, `rug`.
- `bars.ts`: `barChart` (grouped, stacked, horizontal), `histogramChart`, `paretoChart`.
- `distributions.ts`: `densityChart` (from a distribution or KDE), `ecdfChart`, `boxPlot`, `violinPlot`, `qqPlot`,
  `pmfChart`.
- `matrices.ts`: `heatmap`, `confusionMatrixChart`, `correlationMatrixChart`, `imageChart` (a 2-D tensor as an image).
- `fields.ts`: `contourChart`, `decisionRegions`, `vectorFieldChart`, `streamlineChart`.
- `evaluation.ts`: `rocChart`, `precisionRecallChart`, `calibrationChart`, `learningCurveChart`, `residualPlot`.
- `multivariate.ts`: `parallelCoordinates`, `radarChart`, `scatterMatrix`.
- `compose.ts`: `grid` (subplots with shared axes), `linkAxes`, `withDataZoom`, `withMarkLines`, `withAnnotations`.
- `downsample.ts`: `lttb` and min-max decimation for long series.

`layout/` (positions only; no drawing)
- `layered.ts`: `layeredLayout` (Sugiyama-style via dagre: ranks, crossing reduction, edge routes, compound nodes for
  plates and repeated blocks).
- `trees.ts`: `tidyTree` (Reingold-Tilford/Buchheim), `radialTree`, `dendrogramLayout`.
- `force.ts`: `forceLayout` (seeded and deterministic, stopping by energy change), `circularLayout`, `spectralLayout`.
- `grids.ts`: `gridLayout`, `bipartiteLayout`, `latticeLayout` (for Ising models and grid worlds).

`diagram/` (a general diagram model and its conversion to scenes)
- `spec.ts`: `DiagramSpec` (nodes with shape, label, role and ports; edges with style and arrowheads; groups such as
  plates and repeats; annotations), `validateDiagram`.
- `shapes.ts`: node shapes (circle, ellipse, box, rounded box, factor square, deterministic diamond, stacked tensor),
  observed shading, labels with automatic sizing.
- `edges.ts`: edge routing (straight, orthogonal, curved splines), arrowheads, edge labels, parallel-edge offsets.
- `to-scene.ts`: `diagramScene(spec, layout, options)` producing a `Scene`.
- `highlight.ts`: state overlays for animation: active nodes and edges, message arrows, value badges.

`graphs/` (domain structures to diagram specs)
- `graphical-models.ts`: `pgmDiagram` for Bayes nets, Markov networks, factor graphs, dynamic models and plate models
  (from `pgm.modelGraph` and `probprog.modelGraph`), with evidence shading and marginals as badges.
- `computation.ts`: `computationDiagram` (autodiff graphs with op labels, shapes, values and gradients).
- `networks.ts`: `graphDiagram` for any `Graph` (weights, communities, centrality sizing), shortest-path and flow
  overlays.
- `trees.ts`: decision trees, search trees (MCTS, minimax), SLD resolution trees, dendrograms, Huffman trees.
- `automata.ts`: Markov chains and MDPs as state diagrams, HMM trellises.

`architecture/`
- `to-diagram.ts`: `architectureDiagram` (an `ArchitectureSpec` from `nn/describe` to a diagram: blocks, repeated
  groups such as "N x", residual connections, tensor shapes on edges).
- `styles.ts`: block kinds and their visual language (attention, feed-forward, normalisation, embedding, convolution,
  pooling, recurrent cell, output head), colour per kind.
- `views.ts`: alternative views of the same spec: a compact stack, an expanded block, a tensor-shape flow, a
  parameter-count treemap.

`trace/` (figures from runs)
- `metrics.ts`: `metricChart` (one or more metrics over steps, several runs overlaid), smoothing, log scales.
- `params.ts`: parameter trajectories, weight histograms over time, gradient-norm heatmaps per layer.
- `messages.ts`: message-passing timelines, convergence plots, frames for animating diagrams from `message` events.
- `episodes.ts`: return and length curves, regret curves, agent comparisons with bands.
- `tables.ts`: run summaries (configuration, seed, version, final metrics, status) as table data.
- `optimisation.ts`: optimisation paths over contour plots (2-D objectives), line-search trials.

`gym/`
- `frames.ts`: `episodeFrames` (states rendered through the environment's `render`), `filmstrip`, frame interpolation.
- `overlays.ts`: value-function heatmaps, policy arrows and visit counts on grid-world scenes, action distributions.

`svg/`
- `svg.ts`: `sceneToSvg` (a `Scene` to an SVG string with tokens resolved for a theme), `diagramToSvg`.
- `export.ts`: `toDataUrl`, `toPngInBrowser` (canvas), sizing for slides.

`react/` (React 19; React and ECharts are peer dependencies)
- `Chart.tsx`: `Chart` (an ECharts option to a live chart; theme from mode, resize with the container, dispose on
  unmount, event callbacks).
- `SceneView.tsx`: `SceneView` (a `Scene` to SVG elements, token-resolved, accessible title), `useSceneHover`.
- `DiagramView.tsx`: `DiagramView` (a diagram spec with pan, zoom, hover details, highlight state).
- `TraceView.tsx`: `TraceView` (a `Run` to its standard figures), `useRunStream` (subscribe to a live tracer).
- `EpisodePlayer.tsx`: `EpisodePlayer` (environment frames with play, step and scrub).
- `Player.tsx`: `Player` and `usePlayhead` (generic playback controls for any per-step figure).
- `controls.tsx`: minimal controls for figures (`Slider`, `NumberInput`, `Select`, `Toggle`, `SeedInput`) bound to
  `space` definitions.
- `theme.tsx`: `ThemeProvider`, `useColourMode` (light, dark, system; persisted by the host app).

## 6. omai-meta, `packages/meta/src/`

Turns the source into documentation data and runs examples. Entry points: `omai-meta` (schema and runner, browser
safe), `omai-meta/extract` (Node only), `omai-meta/modules` (generated loader map), `omai-meta/worker`,
`omai-meta/react`, and `omai-meta/catalogue.json` (generated at build).

`schema/`
- `catalogue.ts`: zod schemas and types: `SymbolDoc` (id, package, module path, name, kind, signature, summary,
  description, params, returns, throws, examples, concepts, references, source location), `ModuleDoc`, `Catalogue`.
- `examples.ts`: `Example` (code, title, expectations), `Expectation` (line, expression, expected source, kind: value,
  figure, trace, throws), `ExampleResult`.
- `content.ts`: `HowTo` (title, domain, blocks of prose and runnable code), `GalleryEntry` (title, part, code,
  thumbnail), `Concept` (id, title, summary, related symbols).

`extract/` (Node only; uses ts-morph)
- `project.ts`: load the packages' TypeScript projects, enumerate entry points from `exports` and barrels.
- `symbols.ts`: exported declarations to `SymbolDoc` (signatures with resolved types, overloads, classes and their
  methods).
- `tsdoc.ts`: parse TSDoc blocks: summary, `@param`, `@returns`, `@throws`, `@example`, `@concept`, `@see`,
  `@reference`.
- `content.ts`: parse how-to Markdown and gallery files into `HowTo` and `GalleryEntry` records.
- `coverage.ts`: documentation coverage per module (missing summary, params, returns or example) as a report that fails
  the build.

`runner/` (browser and Node)
- `parse.ts`: split an example into statements, find `// => value`, `// => figure`, `// => trace` and
  `// => throws Code` markers.
- `transform.ts`: strip types and rewrite imports to the module loader (sucrase), instrument expectation lines.
- `execute.ts`: run transformed code with a module resolver, capture console output, time limits, errors mapped back to
  example lines.
- `compare.ts`: tolerant comparison (numbers to the precision stated, tensors and typed arrays as nested arrays, deep
  equality for objects), figure and trace recognition.
- `format.ts`: display of values next to code (tensors numpy-style, objects compact, scenes and runs as summaries).
- `run.ts`: `runExample`, `runExamples` (a whole catalogue, with a report).

`modules/`
- `index.ts`: `moduleLoaders` (every public entry point of engine and render to a lazy import; generated).

`worker/`
- `protocol.ts`: messages between a page and a compute worker (run example, call export by address, stream events,
  cancel).
- `worker.ts`: the worker entry (resolves modules, runs code, forwards trace events as they happen).
- `client.ts`: `createComputeClient` (latest-wins jobs, cancellation by terminating the worker, async iteration over
  streamed events).

`react/`
- `LiveExample.tsx`: an editable example with outputs beside each `// =>` line, re-run on edit, figures and traces
  rendered with omai-render.
- `CodeEditor.tsx`: CodeMirror 6 editor with TypeScript highlighting, completion from the catalogue, run on Mod-Enter.
- `ValueView.tsx`: renders any example output (scalars, tensors as tables, scenes, charts, runs).
- `ApiSymbol.tsx`: a symbol's documentation (signature, parameters, returns, examples, related concepts).
- `ModuleIndex.tsx`, `Search.tsx`: browsing and searching the catalogue.

## 7. Apps

Both apps are private Vite and React 19 applications with a navigation bar, a persistent light/dark toggle (system
default, stored choice, no flash), theme tokens from `omai-render/theme` and Tailwind CSS. They consume the
packages exactly as an outside project would.

`apps/docs/` (the counterpart of omai-engine: how to use it)
- `src/main.tsx`, `src/routes.tsx`: entry and routes.
- `src/shell/`: `AppShell.tsx`, `NavBar.tsx`, `ThemeToggle.tsx`, `Sidebar.tsx` (domains and modules).
- `src/pages/`: `Home.tsx`, `HowTo.tsx` (renders a how-to with live examples), `Module.tsx` (a module's API),
  `Symbol.tsx`, `Concepts.tsx`, `Playground.tsx` (free editor), `Traces.tsx` (open, replay and compare exported runs).
- `content/howto/`: one Markdown how-to per engine domain and per major method area (tensors, randomness, tracing,
  autodiff, optimisation, distributions, sampling, graphical models, probabilistic programming, estimators, neural
  networks, reinforcement learning, datasets, and so on). Their code blocks are runnable examples checked by the build.

`apps/gallery/` (the counterpart of omai-render: what it can draw)
- `src/main.tsx`, `src/routes.tsx`, `src/shell/`: as in docs.
- `src/pages/`: `Home.tsx`, `Section.tsx` (a grid of entries), `Entry.tsx` (one figure with its code, editable).
- `src/entries/`: one file per gallery section, each exporting entries whose code returns a figure: `charts.ts`,
  `distributions.ts`, `fields.ts`, `graphical-models.ts`, `computation-graphs.ts`, `architectures.ts`, `networks.ts`,
  `trees.ts`, `traces.ts`, `environments.ts`, `layouts.ts`, `themes.ts`.

## 8. Python oracle, `oracle/`

A uv project (root `pyproject.toml` with dependency groups `oracle` and `dev`) using pydantic and pydantic-settings.
It has two jobs: produce golden fixtures from reference libraries, and vendor static datasets.
- `oracle/omai_oracle/settings.py`: `OracleSettings` (pydantic-settings, reads the root `.env`; output directories,
  which cases to run).
- `oracle/omai_oracle/fixture.py`: `Case` and `Fixture` pydantic models (inputs, outputs, tolerances, reference
  library and version, seed), JSON encoding of non-finite floats and complex numbers, deterministic writing.
- `oracle/omai_oracle/cli.py`: `CliApp` commands: `generate` (all or a path prefix), `check` (regenerate to a
  temporary directory and diff against committed fixtures), `list`.
- `oracle/cases/<domain>/<file>.py`: one case file per engine file that has a Python counterpart, mirroring the engine
  path (for example `oracle/cases/numerics/special/erf.py` uses `scipy.special`; `oracle/cases/nn/module/layers.py`
  uses `torch.nn` in float64 with exported weights). Each defines `cases() -> list[Case]`.
- `oracle/datasets/<name>.py`: vendoring scripts (fetch a pinned source, verify its checksum, write
  `packages/engine/src/methods/datasets/**/<name>.data.ts` or a release asset with a manifest entry).
- `packages/engine/fixtures/`: the generated JSON fixtures, committed, mirroring the case paths.
- `packages/engine/src/testing/golden.ts` (excluded from the published build): `loadFixture`, `expectClose` (per-case
  tolerances), the fixture zod schema matching the pydantic model.

Python never runs during `make test`. `make fixtures` regenerates; `make fixtures-check` runs in a scheduled CI job
(it needs the heavy oracle group with PyTorch).

## 9. Repository tooling

- `Makefile` and `mk/*.mk`: `help`, `setup`, `dev` (docs and gallery), `dev-docs`, `dev-gallery`, `build`, `lint`,
  `format`, `typecheck`, `test`, `test-browser`, `coverage`, `docs` (extract and run every example), `barrels`,
  `layers`, `fixtures`, `fixtures-check`, `datasets`, `pack`, `smoke`, `check` (everything a commit needs), `clean`,
  `release`.
- `mise.toml`: Node, pnpm, Python and uv versions.
- `scripts/barrels.ts`: generate every `index.ts` and the module loader map; `--check`.
- `scripts/layers.ts`: generate dependency-cruiser rules from `domain.json` and `area.json` files and run them.
- `scripts/docs.ts`: extract the catalogue, run every example and how-to, enforce documentation coverage.
- `scripts/version.ts`: write `base/version.ts` and keep all package versions equal (changesets fixed group).
- `scripts/pack.ts`: build and `pnpm pack` the three packages; collect release assets (large datasets) with a manifest.
- `scripts/smoke.ts`: install the tarballs into a throwaway Vite React app; import every entry point; run an example;
  render a scene to SVG and a chart option in Node; build the app.
- `.github/workflows/ci.yml` (`make check` and `make smoke`), `release.yml` (changesets version pull request; on
  release, tarballs and assets attached to a GitHub release), `oracle.yml` (scheduled `make fixtures-check`),
  `dependabot.yml`.
- `docs/`: `structure.md` (this file), `decisions.md` (library and design decisions with reasons), `extending.md` (how to
  add a domain, file, method, environment, dataset, distribution, layer or gallery entry), `conventions.md`
  (docstrings, examples, naming, file size).
- `CLAUDE.md`, `INIT.md`, `AUDIT.md`, `PORT.md`: working instructions for Claude.
## 10. Where familiar things live

For a reviewer checking coverage, and for users arriving from Python.

### scikit-learn

| sklearn                                                     | omai                                                          |
| ----------------------------------------------------------- | ------------------------------------------------------------- |
| `base`, estimator API                                       | `engine/lib/learn/estimator.ts`                               |
| `linear_model`                                              | `methods/linear/`, `methods/glm/`                             |
| `svm`                                                       | `methods/svm/`                                                |
| `tree`                                                      | `methods/trees/`                                              |
| `ensemble`                                                  | `methods/ensemble/`                                           |
| `neighbors`                                                 | `methods/neighbours/`, `lib/numerics/spatial/`                |
| `naive_bayes`, `discriminant_analysis`                      | `methods/naive-bayes/`, `methods/discriminant/`               |
| `gaussian_process`                                          | `methods/gp/`, `lib/kernels/`                                 |
| `kernel_approximation`, `kernel_ridge`                      | `lib/kernels/expansion.ts`, `methods/linear/least-squares.ts` |
| `isotonic`                                                  | `methods/linear/isotonic.ts`                                  |
| `multiclass`, `multioutput`                                 | `methods/multiclass/`                                         |
| `semi_supervised`                                           | `methods/semi-supervised/`                                    |
| `neural_network`                                            | `methods/neural/mlp.ts`                                       |
| `cluster`                                                   | `methods/cluster/`                                            |
| `mixture`                                                   | `methods/mixture/`                                            |
| `decomposition`, `cross_decomposition`, `random_projection` | `methods/decomposition/`                                      |
| `manifold`                                                  | `methods/manifold/`                                           |
| `covariance`                                                | `methods/covariance/`, `lib/stats/covariance.ts`              |
| `preprocessing`, `impute`                                   | `lib/learn/preprocessing/`                                    |
| `feature_extraction.text`                                   | `lib/text/features.ts`                                        |
| `feature_selection`                                         | `lib/learn/feature-selection.ts`                              |
| `pipeline`, `compose`                                       | `lib/learn/compose.ts`                                        |
| `model_selection`                                           | `lib/learn/model-selection/`                                  |
| `metrics`                                                   | `lib/learn/metrics/`                                          |
| `calibration`                                               | `lib/learn/calibration.ts`                                    |
| `inspection`                                                | `lib/learn/inspection.ts`                                     |
| `datasets`                                                  | `methods/datasets/`, `lib/data/`                              |

### PyTorch

| torch                                 | omai                                          |
| ------------------------------------- | --------------------------------------------- |
| `torch.Tensor`, tensor ops            | `lib/tensor/`                                 |
| `torch.autograd`, `torch.func`        | `lib/autodiff/`                               |
| `torch.nn.functional`                 | `lib/nn/functional/`                          |
| `torch.nn` modules                    | `lib/nn/module/`                              |
| `torch.nn.init`                       | `lib/nn/init.ts`                              |
| `torch.optim`, `lr_scheduler`         | `lib/optim/`                                  |
| `torch.utils.data`                    | `lib/data/batching.ts`                        |
| `torch.distributions`                 | `lib/dists/`                                  |
| `torchvision.ops`                     | `methods/vision/boxes.ts`                     |
| training loops (Lightning-style)      | `lib/nn/training/`                            |
| `torchinfo.summary`                   | `lib/nn/describe.ts`                          |
| Hugging Face `generate`, `tokenizers` | `lib/nn/decoding.ts`, `lib/text/tokenizer.ts` |

### NumPy and SciPy

| numpy / scipy                         | omai                                              |
| ------------------------------------- | ------------------------------------------------- |
| `numpy` arrays, ufuncs, reductions    | `lib/tensor/`                                     |
| `numpy.random`                        | `lib/random/`                                     |
| `numpy.linalg`, `scipy.linalg`        | `lib/numerics/linalg/`                            |
| `scipy.sparse`, `scipy.sparse.linalg` | `lib/numerics/sparse/`                            |
| `numpy.fft`, `scipy.fft`              | `lib/numerics/fourier/`                           |
| `numpy.polynomial`                    | `lib/numerics/polynomial/`                        |
| `scipy.special`                       | `lib/numerics/special/`                           |
| `scipy.integrate` (quadrature)        | `lib/numerics/quadrature/`                        |
| `scipy.integrate` (ODEs)              | `lib/dynamics/ode/`                               |
| `scipy.optimize`                      | `lib/optim/`, `lib/numerics/roots/`               |
| `scipy.interpolate`                   | `lib/numerics/interpolate/`                       |
| `scipy.signal`                        | `lib/signal/`                                     |
| `scipy.ndimage`                       | `lib/signal/image/`, `lib/numerics/convolution/`  |
| `scipy.spatial`                       | `lib/numerics/geometry/`, `lib/numerics/spatial/` |
| `scipy.stats`                         | `lib/dists/`, `lib/stats/`                        |
| `scipy.cluster.hierarchy`             | `methods/cluster/hierarchical.ts`                 |

### Other familiar libraries

| library                       | omai                                                                               |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| statsmodels (GLM, tsa, tests) | `methods/glm/`, `methods/timeseries/`, `lib/stats/tests/`                          |
| mgcv, pygam                   | `methods/gam/`, `lib/stats/smoothing/`                                             |
| NumPyro, Pyro                 | `lib/probprog/`, `lib/sampling/`                                                   |
| ArviZ                         | `lib/sampling/diagnostics.ts`, `lib/probprog/criticism.ts`                         |
| pgmpy                         | `lib/pgm/`                                                                         |
| NetworkX                      | `lib/graph/`                                                                       |
| gymnasium                     | `lib/rl/env.ts`, `lib/rl/gym.ts`, `lib/rl/wrappers.ts`, `methods/rl/environments/` |
| stable-baselines3             | `methods/rl/agents/`, `lib/rl/buffers.ts`, `lib/rl/returns.ts`                     |
| python-control, filterpy      | `lib/dynamics/control/`, `lib/dynamics/filtering/`                                 |
| pywt, librosa                 | `lib/signal/wavelets.ts`, `lib/signal/audio.ts`                                    |
| optax                         | `lib/optim/transform.ts`, `lib/optim/adaptive.ts`                                  |
| POT (optimal transport)       | `lib/optim/transport.ts`                                                           |
| SWI-Prolog, Metagol, Aleph    | `lib/logic/`                                                                       |
| DoWhy, EconML                 | `methods/causal/`                                                                  |
| Optuna, scikit-optimize       | `methods/tuning/`                                                                  |
| faiss                         | `lib/numerics/spatial/`, `methods/retrieval/indexes.ts`                            |
| gensim                        | `methods/embedding/word.ts`, `methods/models/topic.ts`                             |
| lifelines                     | `methods/survival/`                                                                |

## 11. Order of construction

The library is built bottom up, one tier at a time, and each tier is finished (documented, tested against the oracle,
traced) before the next relies on it. The first pass is a thin vertical slice through every layer, so the whole
pipeline (tensor to trace to figure to docs) is proven early:
1. `base`, `scene`, `tensor` (core ops), `random` (keys, uniform, normal, permutation, categorical).
2. `trace` (events, recorder, `Algorithm`, `runAlgorithm`, export and replay), `autodiff` (reverse mode, `valueAndGrad`,
   `computationGraph`, gradient checks).
3. `numerics/special/erf.ts`, `numerics/linalg/cholesky.ts`; `optim` (`sgd`, `adam`, `minimize`); `dists` (`normal`,
   `categorical`).
4. `pgm` (`factor.ts`, `models/bayes-net.ts`, `inference/elimination.ts`, `structure.ts`); `nn` (functional `linear`,
   activations, `Sequential`, `fit`, `describe`); `rl` (`env.ts`, `gym.ts`, `rollout.ts`) with
   `methods/rl/environments/classic-control.ts` (CartPole and its render).
5. `data` schemas for regression with `methods/datasets/regression` (`friedman1` generated, `diabetes` vendored).
6. `omai-render`: theme, `charts/lines.ts`, `layout/layered.ts`, `diagram/`, `graphs/graphical-models.ts`,
   `graphs/computation.ts`, `architecture/` (a transformer spec), `trace/metrics.ts`, `svg/`, `react/` (Chart,
   SceneView, TraceView).
7. `omai-meta`: schema, extractor, runner with `// =>` checks, module map; the docs and gallery apps showing the
   slice; the oracle producing the first fixtures; CI, packing and the smoke test.

After the slice, each remaining file in this document becomes a `port` or `build` task, taken in tier order.

## 12. Questions for the reviewer

These are the structural choices most worth challenging:
1. **Granularity of entry points.** Every folder is an entry point and subfolders are namespaces. Is that too many
   import paths, or should only domains be entry points?
2. **Core versus methods boundary.** Rules for where something goes: general machinery in `lib`, named models in
   `methods`. Borderline cases: GLM families live in `lib/dists/families` but GLMs in `methods/glm`; kernels in `lib`
   but SVMs in `methods`; HMM inference in `lib/pgm/inference/chain.ts` but HMM models in `methods/models/hmm.ts`.
3. **Two forms everywhere.** Functional and class forms for estimators, layers and environments double the API
   surface. Is the class form worth it for every estimator, or only for nn modules and environments?
4. **Tier placement.** `space` at tier 3, `data` at tier 4, `stats` above `dists`, `learn` at tier 7 below `nn`. Any
   cycle these hide?
5. **Scene graph in the engine.** `scene` lives in the engine (tier 0) so environments and models can describe their own
   pictures without depending on the renderer. Alternative: a tiny fourth package `omai-scene`.
6. **Datasets in the engine package.** Static datasets inflate the engine tarball; the alternative is a fifth package
   `omai-datasets`.
7. **Coverage gaps.** Anything a comprehensive ML, statistics and numerics library should have that has no file here.
