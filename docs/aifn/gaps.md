# Gaps: what the existing notes need that omai does not have

What the 1274 notes that use the AIFN engine hit today, after matching by name, by hand-kept alias and by reading
(`README.md` says how). Of the 1264 values the notes use, 393 have no omai counterpart. Six of them are the convolutional
sparse coding and dictionary-learning module and the synthetic ECG beat, which another agent is porting: not gaps here.
The other 387 fold into 20 capabilities, listed below. Almost all of them are used by one note; 337 of the 393 by exactly
one.

Sizes are rough: S is a day or less, M a few days, L a week or more. Priority follows how many notes depend on the gap:
high is 40 notes or more, medium 10 to 39, low under 10. "Only this" is the number of notes whose every gap lies in this
capability (not counting `Readout`, which nearly every note needs): what building it alone would free.

## What costs the most to port is not a gap

Three differences are not missing features: omai has an answer for each, but not the same call. They touch far more
notes than any gap below, and a decision on each should come before the port.

### 1. The figure layer (1274 notes)

AIFN draws a figure as a React tree of layer components: `<Figure><Plot axis={useAxis(...)}><Curve/><Points/><Handle/>
</Plot><Readout/></Figure>`, with the controls kept in one typed `useFigureState` (`slider`, `int`, `float`, `choice`,
`setting`, `toggle`, `row`, `variants`, `when`). omai-render builds a chart as data (the functions of
`omai-render/charts` return an ECharts option or layers that combine on shared axes), draws it with `Chart` inside
`FigureFrame`, takes controls from an engine `Space` (`SpaceControls`, `NumberField`, `Choice`), and offers dragging as
overlays (`DraggablePoints`, `DraggableLine`, `Brush`). Every figure in the notes must be rewritten; no import swaps.

| AIFN (notes using it)                                                                                                                                                                   | omai                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Figure` (1274), `FigureSize`, `Dashboard*`, `ControlRow` (48), `ControlGroup` (31)                                                                                                     | `FigureFrame`, `figure()`, `FigureGrid`, `ControlPanel`                                                                                                 |
| `Plot` (1157), `Plots` (159), `useAxis` (1157)                                                                                                                                          | `Chart` with `cartesian`, `subplots`; axes shared by the layers of one option                                                                           |
| `Curve` (826), `Points` (675), `Segments` (183), `Bars` (146), `Area` (63), `Raster` (212), `Vectors` (64), `Contours` (25), `Rug`, `Shapes`, `Annotation` (71), `Histogram`, `Density` | `lines`/`fn`, `points`, `lines`, `bars`, `band`, `heatmap`/`image`, `fieldArrows`, `contours`, `rug`, `polygons`, `annotations`, `histogram`, `density` |
| `Handle` (533)                                                                                                                                                                          | `DraggablePoints`, `DraggableLine`, `Brush` with constraints (`onGraph`, `inBox`, `onCircle`)                                                           |
| `useFigureState` (1142), `slider` (554), `int` (682), `float` (527), `choice` (620), `setting` (142), `row`, `variants`, `when`                                                         | `SpaceControls` over `omai-engine/space` (`real`, `int`, `choice`, `union`), `NumberField`, `Choice`, `Toggle`                                          |
| `useComputed` (131), `call` (20)                                                                                                                                                        | `workerProducer`, `useSession`                                                                                                                          |
| `Diagram` (170), `variable`, `factor`, `link`, `op`, `projector`, `reparam`, diagram specs (`lstm`, `vae`, ...)                                                                         | `diagram()`, `architecture()`, `bayesNetCards`, `StructureView`: drawn from the engine model, not from a hand-built spec                                |
| `Player` (191), `usePlayhead`                                                                                                                                                           | `Player`, `usePlayback`                                                                                                                                 |
| `MathText` (57), `EquationSteps`                                                                                                                                                        | `Tex`, `Derivation`                                                                                                                                     |
| `Readout` (1188)                                                                                                                                                                        | nothing: the first gap below                                                                                                                            |

Two ways out: rewrite the figures, or put a thin declarative layer over `omai-render/charts` (a `Plot` with layer
components and a `useFigureState` that builds a `Space`) so that notes keep their shape. The second is an L item and
would live in `omai-render/react`; it is a decision for the owner, not something this survey can settle.

### 2. Random numbers (648 notes use `stream`, 126 `child`)

AIFN draws from a mutable named stream: `stream(seed)`, `child(s, 'chain', 3)`, `normal(s, mean, sd, { shape })`. omai uses
immutable keys (`key(seed)`, `foldIn`, `keyFromPath`, `normal(key, shape, { loc, scale })`) or a `Generator`. Both are
Philox-based and deterministic, but the same seed does not give the same numbers, so a figure that quotes a drawn value
(a sample mean, a fitted slope, "the error is 0.043") changes, and so do worker calls such as
`call('foundation/random/stream', ...)`. A port has to re-read the numbers a note states. A compatibility shim that
reproduced AIFN's stream bit for bit would be an M item in `omai-engine/random`, and only worth it if the numbers in the
prose matter more than the effort.

### 3. The estimator protocol and the methods layer

`dataset(x, y)` (81 notes), capability-typed models (`Fitted`, `Predicts`, `Decides`, `hasPredictive`), and the named
method runs of `aifn-methods` (`...Run`, `...Steps`, `...Trace`, `...Problem`: 32 of its 233 values without a counterpart,
and many of the 233 that have one). omai has no methods layer:
estimators are sklearn-style objects of `omai-engine/learn` and the families (`linearModel`, `svc`, `selfTraining`, ...),
iterative methods are `Algorithm`s run by a `Session`, and a demonstration is a page that composes them
(`docs/methods-catalog/` is the catalogue). A note that calls `linearRegression(...).fit(dataset(x, y))` becomes
`linearModel` with the Gaussian family on a `Dataset`; the 358 renamed values (`mapping.md`) are mostly this. Where a
`...Run` has no underlying library method the group below says so.

## The gaps, by priority

<!-- generated:groups -->
| gap                                                                                                                                                                    | area     | notes | only this | symbols | size | priority | most used                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ----- | --------- | ------- | ---- | -------- | ------------------------------------------------------------------------------------------------------ |
| Readout: a labelled value under a figure                                                                                                                               | render   | 1188  | 953       | 1       | S    | high     | Readout (1188)                                                                                         |
| Teaching dataset generators with their truth (full-covariance Gaussians, learning curves, Swiss-roll slice, shapes, signals, tuning surfaces)                          | datasets | 61    | 33        | 49      | M    | high     | gaussians (9), svmTuningSurface (6), learningCurves (4), swissRoll2d (4)                               |
| Small numerics and utilities: power-of-two tests, DCT matrix, 1-D optimal transport plan, relative L2 error, star discrepancy, scans, partial derivatives              | numerics | 48    | 34        | 40      | S    | high     | isPowerOfTwo (6), monotonePlan (6), dctMatrix (5), relativeL2Error (3)                                 |
| Efficient-attention and state-space helpers: sparse and block masks, mLSTM, Linformer, Nystrom and LSH attention, parity and recall probes                             | nn       | 25    | 14        | 47      | M    | medium   | ActivationLayer (3), methodTraining (3), positionMask (2), positionRange (2)                           |
| Semi-supervised helpers: consistency training, Laplacian-regularised fits, entropy minimisation, semi-supervised GAN, VAE and mixtures                                 | learn    | 24    | 23        | 33      | M    | medium   | rowArgmax (7), consistencyTraining (7), predictProbabilities (7), laplacianRegularisedLeastSquares (3) |
| Graph neural network helpers: batching, attention and pooling variants, temporal reach, k-WL, rewiring                                                                 | graph    | 15    | 14        | 25      | M    | medium   | batchGraphs (2), propagationStack (2), folkloreWeisfeilerLeman2 (1), subgraphWeisfeilerLeman (1)       |
| Other method runs: multi-fidelity and sparse GPs, GAMLSS, ridge hypergradients, forward stagewise, Cox likelihood, recommenders, constant-velocity model, fuzzy graphs | learn    | 15    | 10        | 16      | M    | medium   | dualDecision (2), fitMultiFidelityGp (1), samplePrior (1), gamlssProblem (1)                           |
| Rule lists and explanation helpers: CBA and CMAR, boolean rule column generation, rule simplification, subgroup descriptions, RuleFit                                  | rules    | 14    | 10        | 28      | M    | medium   | simplifyRule (9), applyRuleList (1), denseFunction (1), denseOutput (1)                                |
| Scientific machine learning: physics-informed networks, neural operators (FNO, DeepONet), neural ODE runs                                                              | nn       | 13    | 9         | 17      | L    | medium   | pinnVariantRun (4), heatProblem (2), pinnRun (2), lyapunovCurve (1)                                    |
| Generative model runs: autoencoder and VAE, energy-based (JEM), RBM and DBN, toy diffusion mixtures, flows trainers                                                    | nn       | 9     | 8         | 17      | L    | low      | gaussianMixtureData (3), autoencoderRun (2), mixtureLogDensity (2), mixtureNoisePredictor (2)          |
| Logic programming and induction helpers: propositionalisation, concept lattices, generality search, string-program synthesis                                           | logic    | 9     | 7         | 32      | M    | low      | allInstances (2), aggregateFeatures (1), allConjunctions (1), attributeFeatures (1)                    |
| Language-model and post-training toys: Kneser-Ney, character corpora, GRPO, best-of-N, reward-model data                                                               | text     | 9     | 4         | 19      | M    | low      | embed (2), charCorpus (2), decodeChars (2), encodeChars (2)                                            |
| Hyperparameter-search extras: max-value entropy search, q-EI, EHVI, BOHB, population-based training, portfolios                                                        | tune     | 8     | 5         | 13      | M    | low      | maxValueEntropySearch (2), multiPointExpectedImprovement (1), bohbSteps (1), fitPowerLaw (1)           |
| Learning-theory studies: concentration study, double descent, adaptive-holdout maximum, James-Stein risk                                                               | learn    | 6     | 6         | 4       | S    | low      | concentrationStudy (3), expectedGaussianMaximum (1), doubleDescent (1), jamesSteinExactRisk (1)        |
| Gym helpers: grid views, maze sets, action names, a training row                                                                                                       | rl       | 5     | 3         | 8       | S    | low      | GridView (5), GRID_ACTION_NAMES (2), train (2), greedyActions (1)                                      |
| Inference helpers: innovation statistics, run-length rows, clutter EP, learner models, factor graphs                                                                   | pgm      | 5     | 2         | 12      | S    | low      | laggedObservations (1), normalisedInnovationSquared (1), regressionNormalGamma (1), runLengthMass (1)  |
| Cost leftovers: convolution algorithm costs, dilated tap coverage, decode throughput, fine-tuning compute, reference LLM configurations                                | nn       | 4     | 4         | 7       | M    | low      | dilatedTapCoverage (1), convolutionAlgorithmCosts (1), winogradRelativeError (1), LLAMA_3_1_70B (1)    |
| Detection anchors and targets, capsule routing, deformable convolution                                                                                                 | vision   | 4     | 4         | 6       | M    | low      | anchorGrid (2), capsulePredictions (1), deformableConvolutionAt (1), dynamicRoutingTrace (1)           |
| Small render parts: status text, code editor                                                                                                                           | render   | 3     | 0         | 2       | S    | low      | StatusText (3), CodeEditor (1)                                                                         |
| Rating systems: Elo, Glicko and TrueSkill through time with the federations' scales                                                                                    | pgm      | 3     | 1         | 11      | M    | low      | rateStream (2), LICHESS_PROVISIONAL_RD (1), THURSTONE_BETA (1), chessComSpec (1)                       |
<!-- /generated:groups -->

The three biggest first, with what porting them involves.

### 1. `Readout`: a labelled value under a figure (S, 1188 notes, 953 need nothing else)

AIFN: `render/viz/Readout.tsx`, 13 lines. A label, a value set in tabular mono so it does not jitter, and an optional
colour swatch (the current loss, the fitted slope), plus `Readouts` (a wrapping row) and `ReadoutGroup`; a `Figure` given a
`readouts` record draws one per entry. Of the 99 render values the notes use, six have no omai counterpart and this is the one
that matters: the layer components are covered by a different shape (above), this one is not covered at all. `StatusText`
(3 notes) is its sibling.
Home: `packages/render/src/react/Readout.tsx` beside `NumberField` and `Equation`, exported from `omai-render/react`.
Builds on `omai-render/theme#formatNumber` and the theme tokens, and can take a `Tex` label or a live value of
`omai-render/maths#live`. An hour or two, and it removes the last missing symbol of 953 notes, whose only other differences are
the three above.
`CodeEditor` (1 note, M) is a CodeMirror editor and probably not worth porting.

### 2. Teaching dataset generators with their truth (M, 61 notes, 33 need nothing else)

AIFN: `methods/data/synthetic/*` (and `objectives`, `signals`, `targets`): generators that carry the exact truth
(a Bayes error, a likelihood, a true changepoint) and are seeded by a stream. 49 values of them have no omai counterpart.
The ones that matter:
- `gaussians` (9 notes: `margin-bounds`, `nearest-neighbour-consistency`, ...; `points.ts:1007`): k Gaussian classes with
  full covariances and a closed-form Bayes error. omai has isotropic `blobs` and `anisotropicBlobs`, not this.
- `svmTuningSurface` (6: `grid-search`, `tuning-a-support-vector-machine-by-bayesian-optimisation`;
  `objectives/svm.ts:88`), `learningCurves` (4: `hyperband`, `asynchronous-successive-halving`; `learning-curves.ts:114`) and
  `learningCurveSurface`: synthetic validation-loss surfaces and learning curves for tuning notes. omai has
  `learningRateSurface`, which is a different surface.
- `swissRoll2d` (4: `ball-tree`, `k-d-tree`; `densities.ts:418`): the 2-D slice of the Swiss roll as a density.
- `shapeTextureDataset` (3: `class-activation-mapping`, `feature-visualisation`; `shape-texture.ts:219`), `pairedShapes` (2),
  `blockModelProbabilities` (2), `shuffleDataset` (2), and about 40 generators each used by one note (learner responses,
  latent factors, molecules, rating populations, bandit logs, number sets, periodic Gaussian fields, ...).

Several have a near omai relative that a port can use at the price of a different figure (`regression1d` maps to
`heteroscedastic`, `pinwheel` to `pinwheelDensity`, `gaussianRing` to `gaussianMixtureDensity`; see `mapping.md`).
Home: `omai-engine/datasets` (`classification.ts` for `gaussians`, `objectives.ts` and `regression.ts` for the surfaces and
curves, `densities.ts`, `images.ts`, `graphs.ts`), registered with tags in the datasets registry like the others.
Builds on `omai-engine/random` keys, `omai-engine/dists` (`MultivariateNormal` for the truth) and the `DatasetGenerator`
shape of the existing generators. `gaussians`, the two tuning surfaces and the learning curves are an S each and cover
about 20 notes between them.

### 3. Small numerics and utilities (S in total, 48 notes, 34 need nothing else)

Forty small functions, many used by one note, the biggest by use:

| function (notes)                                                                                                                                              | AIFN file                               | omai home                                                                                                            | builds on           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `isPowerOfTwo` (6)                                                                                                                                            | `foundation/fourier/fft.ts:65`          | `numerics/fourier`                                                                                                   | `nextFastLen`       |
| `monotonePlan` (6): the 1-D optimal transport plan                                                                                                            | `transport/oneD.ts:117`                 | `optim/transport.ts` beside `wasserstein1d`                                                                          | sorting, `emd`      |
| `dctMatrix` (5)                                                                                                                                               | `foundation/fourier/dct.ts:26`          | `numerics/fourier` beside `dct`                                                                                      | `dct`, `eye`        |
| `relativeL2Error` (3)                                                                                                                                         | `learning/metrics/regression.ts:220`    | `metrics/regression.ts`                                                                                              | `norm`              |
| `partials` (2): partial derivatives of a pointwise field at collocation points, the derivatives of a PINN loss                                                | `dynamics/fields/partials.ts:71`        | `dynamics/pde`                                                                                                       | `autodiff#jacobian` |
| `evaluatePiecewise`, `linearInterpolant` (2)                                                                                                                  | `numerics/interpolate/piecewise.ts:107` | `numerics/interpolate`                                                                                               | `makeInterpSpline`  |
| `get` (2): element at a multi-index                                                                                                                           | `foundation/tensor/structure.ts:489`    | `tensor`                                                                                                             | `slice`, `item`     |
| `lqgSimulation`, `dic`, `nestedCvInterval`, `bootstrapPredictionError`, `softThresholdSure`, parallel-scan step functions, `starDiscrepancy`, `toComplexFlat` | one note each                           | next to their siblings (`dynamics/control#lqg`, `probprog#waic`, `learn/model-selection#nestedCrossValidation`, ...) | the sibling         |

A note can inline an `isPowerOfTwo`, so these are the cheapest gaps to close and the notes that need them are not
blocked by anything else.

### The rest, by priority

Each row below is a capability, its notes, the AIFN source, a home and a size. The home follows `docs/structure.md`
(a file under `packages/engine/src/lib/<domain>/`).

**Medium priority (10 to 39 notes)**
- **Efficient-attention and state-space helpers** (M, 25 notes, 14 only this). AIFN `compute/nn/attention/{masks,sparse,block-sparse}.ts`
  (position and sparse masks, `positionMask`, `sparseAttentionMask`, `blockSparseMask`, `attentionReach`, T5 buckets),
  `nn/sequence/*` (mLSTM parallel and recurrent forms, `implicitFilter`, positive random features, scan orders, the
  semiseparable mask), `methods/neural/sequence/approximate.ts` (Linformer, Nystrom and LSH attention matrices, 9 notes),
  `nn/layers` `ActivationLayer` (3 notes: `class-activation-mapping`, `feature-visualisation`), `nn/sets` `padSets` (deep sets).
  omai has `causalMask`, `alibiBias`, `linearAttention`, `lru`, `ssd`, Hopfield. Home: `nn` (`attention-masks.ts`) and
  `nn/blocks` (`linear-attention.ts`, `sequence.ts`). Builds on the `nn` attention functions and `tensor#where`.
- **Semi-supervised helpers** (M, 24 notes, 23 only this). `methods/neural/semi-supervised/consistency.ts` (the Pi-model, temporal
  ensembling, mean teacher, VAT and UDA with `sharpen` and confidence thresholds, 7 notes:
  `consistency-regularisation`, `adaptive-confidence-thresholds`), `learning/semi-supervised/laplacian-rls.ts`
  (Laplacian-regularised least squares, 3), the symmetric hinge and sigmoid surrogates of the transductive SVM, entropy
  minimisation, density-sensitive distances, semi-supervised GAN and VAE (`gan/semi-supervised.ts`, `autoencoders/m2.ts`),
  semi-supervised mixtures. omai has `selfTraining`, `coTraining`, `triTraining`, `s3vm`, `labelSpreading`. Home:
  `learn/semi-supervised.ts` for the classical ones, a new `learn/consistency.ts` for the networks. Builds on `nn`,
  `losses`, `data#withMixup`.
- **Graph neural network helpers** (M, 15 notes, 14 only this). `compute/nn/graph/{pooling,smoothing,transformer}.ts` (graph batching,
  DiffPool, attention pooling, Graphormer attention and spatial bias, PNA aggregation, relational GCN terms, hypergraph
  mean pass), `methods/neural/graph/{depth,temporal}.ts` (propagation stacks, time-respecting reach, link-prediction
  splits), `compute/graph/isomorphism` (folklore 2-WL, subgraph WL), `graph/rewiring` (FoSR, virtual node).
  omai has `messagePassing`, `GCNConv`, `ginConv`, `globalPool`, `weisfeilerLehman`. Home: `nn/blocks/graph.ts`,
  `graph/isomorphism.ts`, a new `graph/rewiring.ts` (the existing `rewiringAlgorithm` is Watts-Strogatz, not GNN rewiring).
- **Other method runs** (M, 15 notes, 10 only this): multi-fidelity and sparse GPs, GAMLSS problems, ridge
  hypergradients, forward stagewise, the Cox partial likelihood, a recommender run, constant-velocity model, fuzzy graph
  (UMAP). Mostly 1 note each; the library part exists for most (`gam`, `survival`, `linear`); the missing part is the
  small function.
- **Scientific machine learning** (L, 13 notes, 9 only this). `methods/neural/pinn/*` (a physics-informed network on a named
  equation: collocation, loss balancing, hard constraints, variants; 9 notes: `physics-informed-network-for-burgers-equation`,
  `hard-constraints-for-physics-informed-networks`, ...), neural ODE data, Lyapunov and sine-map demos. omai has
  `DeepONet`, `FourierLayer`, `SpectralConv1d/2d`, `ODEBlock`, `dynamics/pde`, `autodiff`; it lacks the PINN residual and
  collocation machinery. Home: `nn/pinn.ts` (new) and a `pdeResidual` in `losses`. Builds on `autodiff`, `dynamics/pde`, `random`
  (collocation points).
- **Rule lists and explanation helpers** (M, 14 notes, 10 only this). `learning/explain/rules.ts` (`simplifyRule` 9 notes, predicates,
  `applyRuleList`, `ruleCoverage`), `learning/rules/{associative,boolean-rules}.ts` (CBA and CMAR classification by
  association, boolean rule column generation), `learning/subgroups/language.ts` (selector languages, descriptions),
  RuleFit. omai has `rules` (RIPPER, CORELS, Bayesian rule lists), `patterns` (itemsets, rules, subgroups) and
  `explain/rules.ts` but keeps `simplify` private. Home: `explain/rules.ts` (export), `rules`, `patterns/subgroups.ts`.

**Low priority (under 10 notes)**
- **Logic and induction helpers** (M, 9 notes): propositionalisation, concept lattices (`hasseEdges`, conjunction level and
  coverage), generality search, string-program synthesis; omai has the `logic` and `logic/ilp` cores. Home: `logic/ilp`.
- **Generative model runs** (L, 9 notes): VAE and autoencoder runs (`autoencoders/run.ts`, `m2.ts`), toy diffusion mixtures with the
  exact score (`diffusion/mixture.ts`, 3 notes), DBN, JEM, flow trainers. omai has `negativeElbo`, `EncoderDecoder`, RBM, the
  coupling layers, the diffusion U-Net blocks and samplers; it has no VAE estimator and no exact mixture-score toy. Home:
  `nn/blocks/generative.ts` and `stochastic` (diffusion).
- **Language-model and post-training toys** (M, 9 notes): Kneser-Ney, character corpora, GRPO, best-of-N, reward-model
  data, CLIP training, full-batch comparison, grokking runs. Home: `text` and `losses`.
- **Hyperparameter-search extras** (M, 8 notes): max-value entropy search, q-EI, EHVI, BOHB, population-based training,
  portfolios. omai's `tune` has GP, TPE, hyperband, ASHA, fANOVA. Home: `tune/bayesian.ts`, `tune/samplers.ts`.
- **Learning-theory studies** (S, 6 notes): `concentrationStudy`, the expected maximum of Gaussians, double descent,
  James-Stein risk curves; omai has the bounds and `biasVariance`. Home: `learn` and `stats`.
- **Gym helpers** (S, 5 notes): `GridView` (5), action names, mazes, a training row. omai-render/gym has `worldView`; the grid view
  with value and policy overlays is the gap. Home: `omai-render/gym`.
- **Inference helpers** (S, 5 notes): innovation statistics, run-length rows, clutter-EP, learner models. Home: `dynamics/filtering`, `pgm`.
- **Vision and capsules** (M, 4 notes): detection anchors and targets (`anchorGrid`), capsule routing, deformable convolution. Home: `vision`, `nn/blocks/vision.ts`.
- **Rating systems** (M, 3 notes): Elo, Glicko and TrueSkill updates exist in `pgm/ratings.ts`; TrueSkill through time, rating
  streams and the federations' scales (FIDE, Lichess, chess.com) do not. Home: `pgm/ratings.ts`.
- **Cost leftovers** (S, 4 notes): convolution algorithm costs and Winograd error, dilated tap coverage, decode throughput,
  fine-tuning compute, reference LLM configurations. omai estimates FLOPs and memory from a built module
  (`describe`, `functionFlops`, `memoryEstimate`); the closed-form calculators by shape are what is missing.
- **Small render parts** (S, 3 notes): `StatusText`, `CodeEditor`.

## Where AIFN and omai differ on purpose

A few mismatches are choices omai made, not omissions; a port should not "fix" them:
- `nextPowerOfTwo` maps to `nextFastLen`, which gives a fast FFT length (a product of small primes), not a power of two.
  A note that explains radix-2 padding states the power of two itself.
- A model is an objective and the solver is a slot: `logisticRegression` is not IRLS. A note about IRLS names the solver.
- Tensors expose `.data`; `float64Data` and `readonlyData` are not functions in omai.
- omai has no UI kit (the 95 shadcn re-exports of `aifn-render/ui`): the site brings its own.
