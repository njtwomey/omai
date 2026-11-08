# Likely future gaps

`gaps.md` is what the existing notes hit now. This file is what they are likely to hit next, from four sources: the notes
not yet written, the engine requests and API sketches the note authors left in `.scratch/`, numerics the notes wrote for
themselves because the engine lacked them, and AIFN surface that no note uses yet. All of it is weaker evidence than the
imports, and each section says how weak. `scripts/aifn-usage/scan-wants.ts` reads the sources; the data is in
`data/wants.json`.

## 1. Notes planned but not written

360 of the 2486 notes are stubs (`status: stub`), 14%. They are spread over 82 chapters; the ones with most stubs:

| chapter                                                            | stubs |
| ------------------------------------------------------------------ | ----- |
| part 5, unsupervised learning, clustering                          | 15    |
| part 5, supervised learning, regression                            | 14    |
| part 5, supervised learning, trees and ensembles                   | 11    |
| part 5, representation learning, embeddings of structured data     | 9     |
| part 5, representation learning, theory of representation learning | 9     |
| part 5, logic and rule learning, neuro-symbolic learning           | 8     |
| part 5, reinforcement learning, imitation and reward learning      | 8     |
| part 5, reinforcement learning, policy methods                     | 8     |
| part 5, representation learning, self-supervision                  | 8     |
| part 5, supervised learning, survival and time-to-event learning   | 8     |
| part 6, feedforward networks, training                             | 8     |
| part 6, generative models, flow matching and fast sampling         | 8     |
| part 6, large language models, reasoning and test-time compute     | 8     |
| part 6, scientific machine learning, structured dynamics           | 8     |
| part 6, transformers, interpreting transformers                    | 8     |

Parts 5 and 6 are where the stubs are, and part 5 is where the methods layer of AIFN is thickest, so the gap list of
`gaps.md` will grow there first (semi-supervised, SciML, rules and logic, generative runs).

### What the unwritten notes need

The note authors listed, per unwritten note, the engine functions it will need (`.scratch/implementations/GAPS.md`, from
2026-10-17, written against AIFN 0.2.9): 218 needs, 197 of them wanted only by stubs, sized S (120), M (68) and L (9).
Looked up by name and by shared words in omai:
- **148 of the 197 already have a counterpart or a near one.** omai has MCTS, CFR, SAC, TD3, DQN variants, DAgger, MaxEnt
  IRL, GAIL, TD(lambda), double Q-learning, Dyna-Q, fuzzy c-means, kernel k-means, affinity propagation, BIRCH,
  incremental and sparse PCA, quantile and Huber regression, PLS and reduced-rank regression, KAN, EGNN, SINDy, flow
  matching, EDM preconditioning, mixture of experts, CTC, Nystrom features, subsequence kernels, T-norms, MLN and ProbLog,
  and more. The match is by name, so a few are wrong; the whole list was read and most hits are right.
- **49 have no hint, and 34 of those were checked by hand and are absent:**

| need                                                                                      | size       | stub notes | AIFN area                                      | first note                                                                                                              |
| ----------------------------------------------------------------------------------------- | ---------- | ---------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `scalingLawFit`                                                                           | M          | 3          | `compute/learning/estimators`                  | neural-scaling-laws                                                                                                     |
| `similarUnlabelledRisk`                                                                   | S          | 1          | `compute/learning/losses/weak`                 | learning-from-similar-and-unlabelled-pairs                                                                              |
| `disentanglementMetrics`                                                                  | M          | 1          | `compute/learning/metrics/representation`      | disentanglement-metrics                                                                                                 |
| `survivalNetworkLosses`                                                                   | S          | 1          | `compute/nn/losses`                            | neural-survival-models                                                                                                  |
| `exposureBiasSimulation`                                                                  | S          | 1          | `compute/nn/sequence`                          | teacher-forcing-and-exposure-bias                                                                                       |
| `coinBettingLearner`                                                                      | S          | 1          | `compute/optim/online`                         | parameter-free-online-learning                                                                                          |
| `marchenkoPasturDensity`                                                                  | S          | 1          | `compute/probability/distributions`            | sample-covariance-in-high-dimensions                                                                                    |
| `dpmSolver`, `reflow`                                                                     | S          | 1 each     | `methods/generative/{diffusion,flow-matching}` | higher-order-diffusion-samplers, rectified-flow                                                                         |
| `tigerPomdp`                                                                              | S          | 1          | `methods/gym/environments`                     | partially-observable-markov-decision-process                                                                            |
| `probabilisticSoftLogicMap`                                                               | M          | 1          | `methods/inference`                            | probabilistic-soft-logic                                                                                                |
| `deepGpPrior`                                                                             | S          | 1          | `methods/learning/gaussian-processes`          | deep-gaussian-processes                                                                                                 |
| `shrinkageProfiles`                                                                       | S          | 1          | `methods/learning/linear`                      | sparse-bayesian-priors                                                                                                  |
| `condensedNearestNeighbour`                                                               | S          | 1          | `methods/learning/neighbours`                  | nearest-neighbour-regression-and-condensing                                                                             |
| `orderedTargetStatistics`                                                                 | S          | 1          | `methods/learning/preprocessing`               | ordered-boosting-and-categorical-features                                                                               |
| `personPeriod`, `cumulativeIncidence`                                                     | S          | 1 each     | `methods/learning/survival`                    | discrete-time-hazard-models, competing-risks                                                                            |
| `continualNearestMean`, `partialDomainAdaptationRun`, `reptileRun`                        | S          | 1, 1, 2    | `methods/learning/transfer`                    | continual-learning-with-pretrained-models, reptile                                                                      |
| `missingValueSplits`                                                                      | M          | 1          | `methods/learning/trees-and-ensembles`         | missing-values-and-categorical-splits-in-trees                                                                          |
| `bart`                                                                                    | L          | 1          | same                                           | bayesian-additive-regression-trees                                                                                      |
| `monotoneConstraints`, `ngboost`                                                          | S, M       | 1 each     | `.../boosting`                                 | monotone-constraints-in-boosting, probabilistic-boosting                                                                |
| `taskVectorMerge`                                                                         | S          | 1          | `methods/neural/adaptation`                    | model-merging                                                                                                           |
| `conditionalNeuralProcess`                                                                | M          | 1          | `methods/neural/meta`                          | bayesian-meta-learning                                                                                                  |
| `neuralSdeFit`                                                                            | M          | 1          | `methods/neural/ode`                           | neural-stochastic-and-controlled-differential-equations                                                                 |
| `testTimeComputeCurves`                                                                   | S          | 2          | `methods/neural/post-training`                 | test-time-compute-scaling                                                                                               |
| `histogramOutlierScore`                                                                   | S          | 1          | `methods/unsupervised/anomaly`                 | histogram-based-and-copula-outlier-scores                                                                               |
| `sparseSubspaceClustering`, `deepEmbeddedClustering`, `coAssociation`, `hopkinsStatistic` | M, M, S, S | 1 each     | `methods/unsupervised/clustering`              | subspace-clustering-and-co-clustering, deep-clustering, consensus-clustering, clustering-tendency-and-cluster-stability |
| `principalComponentPursuit`                                                               | M          | 1          | `methods/unsupervised/reduction`               | robust-principal-component-analysis                                                                                     |

Where each would go in omai, by the same rules as `gaps.md`: scaling laws, `similarUnlabelledRisk`, `disentanglementMetrics` and
`hopkinsStatistic` in `learn` and `metrics`; `coinBettingLearner` in `optim/online.ts`; `marchenkoPasturDensity` in
`dists`; `dpmSolver` and `reflow` in `stochastic` (beside `ddimSampler`, `probabilityFlow`) and `losses#flowMatching`;
`tigerPomdp` in `environments`; `personPeriod`, `cumulativeIncidence` and `survivalNetworkLosses` in `survival`;
`bart`, `missingValueSplits`, `monotoneConstraints` and `ngboost` in `trees` and `ensembles` (omai already has Newton
leaves in boosting); the clustering and PCA items in `cluster` and `decomposition`; `reptileRun` and the
transfer runs in `learn` with `nn`. Sizes are the authors' sketches.

Nine of the needs are L: ProbLog, Markov logic networks, Newton boosting, survival forests, BART, knowledge-graph models,
DeepProbLog, differentiable ILP and Lagrangian networks. omai already has ProbLog (`logic/problog.ts`), MLN, `DiffIlp`,
knowledge-graph embeddings (`embedding#trainKg`), Newton leaves for boosting and the Lagrangian block (`nn/blocks`), so
the L needs omai lacks are `survivalForest`, `deepProblogRun` and `bart`.

The 21 needs in section B of that file are "probably met already" (some user is written); they were not rechecked.

## 2. Requests the note authors left for the engine

Three sets of sketches in `.scratch/`, written as asks to the engine owner. Their names are sketches, not exports.
- **Learning theory** (`learning-theory/engine-request.md`, for a chapter of 63 notes, all but one written): about 70
  distinct callable names in three tiers (a few are placeholders in the pseudo-code): bound calculators (`hoeffdingDeviation`, `bernsteinDeviation`, `binomialTailInversion`,
  `klInverse`, `finiteClassBound`, `occamBound`, `pacBayesBound`, `pacBayesLinear`, `rademacherBound`,
  `localRademacher`, `sampleComplexity`, `vcLowerBound`, ...) and simulators (`stabilityStudy`, `noisyMeanStudy`,
  `randomLabelStudy`, `minNormInterpolationStudy`, `adaptiveHoldoutStudy`, `maximalInequalityStudy`, ...). 14 of the names
  exist in omai (Hoeffding, Bernstein, Chernoff and Rademacher bounds are in `stats` and `learn`); the PAC-Bayes, Occam,
  fast-rate, sample-complexity and VC-lower-bound calculators and the studies were not found. This is
  the largest set of functions a chapter would put in a library: generalisation bounds as functions, with their
  inversions. Home: `stats/concentration.ts` and `learn` (a new `learn/bounds.ts`); size M for the calculators, M for the
  simulators. The notes of the chapter carry local code in the meantime.
- **Model selection** (`model-selection/engine-request{,-2}.md`): written as prose, no code blocks. AIFN's commits after the requests
  ("Model selection follow-up: CV intervals, spatial CV, BO variants, multi-objective, BOHB, simulation studies") built them there.
  omai has some of them (spatial blocks, nested CV, GP and TPE samplers, ASHA); BOHB, q-EI, EHVI and the studies are
  in `gaps.md` under hyperparameter-search extras.
- **Data setup and feature engineering** (`part-iii/engine-api-sketches.md`, collected against AIFN 0.2.6; of its 16 callable names,
  one exists in AIFN 0.2.10 and five in omai):
  `groupAggregate`, `asOfJoin`, `calendarFeatures`, `resampleIrregular`, `slidingWindows`, `estimateClockOffset`,
  `instanceNormaliser`, `cyclicalEncoder`, `multiHotEncoder`, `boxCoxLogLikelihood`. omai has `learn/preprocessing` and
  `timeseries`; point-in-time joins and group aggregates are the likely gaps (S each, home `data` and `learn/preprocessing`).

## 3. Numerics the notes wrote for themselves

103 `.ts` files of 120 lines or more under `content/notes` are reached by at least one note (28,000 lines in all; they are the
`_shared/` folders and the figure data modules). A note writes code when the engine lacks the thing, so each is a hint of a
future gap. Many are data, plots or demos and not libraries (group-sequential tables 4861 lines, shapes and paths of the
epicycles, a black-hole ray tracer, game rendering). Of the ones that look like library code, checked against omai by
name:

| local code (notes reaching it)                                                                                                                                                                                                                                                                            | in omai?                                                                                                                                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GP-LVM (`gplvm.ts`, 336 lines, 3 notes)                                                                                                                                                                                                                                                                   | not found: M, `gp`                                                                                                                                                                                                                                                           |
| correlated, dynamic, author and supervised topic models (`plates.ts`, 321 lines, 11 notes)                                                                                                                                                                                                                | not found as models; LDA is a probprog program: M, `probprog`                                                                                                                                                                                                                |
| spin lattices and the Ising model (`spins.ts`, 270, 5)                                                                                                                                                                                                                                                    | not found: M, `dynamics/automata` or `sampling`                                                                                                                                                                                                                              |
| deep one-class methods (`deepOneClass.ts`, 267, 3)                                                                                                                                                                                                                                                        | not found: M, `anomaly`                                                                                                                                                                                                                                                      |
| MMD two-sample test (`twoSample.ts`, 222, 7), flagged "a candidate for the engine"                                                                                                                                                                                                                        | `kernel-methods` has the witness, not the test: S                                                                                                                                                                                                                            |
| background check for out-of-distribution scores (`background.ts`, 1)                                                                                                                                                                                                                                      | not found: S, `anomaly`                                                                                                                                                                                                                                                      |
| crowd labels with BCC and communities (`crowd.ts`, 236, 1)                                                                                                                                                                                                                                                | `learn` has Dawid-Skene, not BCC: S                                                                                                                                                                                                                                          |
| empirical mode decomposition (`emd.ts`, 612, 4), matrix profile (241, 7), spectral estimation (244, 8), splines (822, 13), ordinal regression (1326, 10), density-ratio estimation (257, 5: `kliep`), seam carving, Hough transform, skill rating, group-sequential designs, Fourier-neural-operator data | yes: omai has `ceemdan`/`eemd`, `timeseries` matrix profile, `signal`, `numerics/interpolate`, `linear#ordinal`, `learn` KLIEP, `vision` seams and Hough, `pgm/ratings`, `stats/tests#GroupSequentialDesign`, `datasets#burgersOperator`. A port can delete the local copies |

So the notes' own code is mostly redundant with omai, and the genuine future gaps among it are the first six rows.

## 4. AIFN surface that nobody uses yet and omai lacks

2208 AIFN values have no counterpart by name and no use in a note (`data/unused-unmatched.json`). Most are internals
(registries named `*Algorithms`, `*Functions`), helpers and renames; the list is a place to look, not a gap list. Reading a
sample of 60 and then looking up the families that came up, these capabilities exist in AIFN, have no omai counterpart
found, and sit in areas the notes are likely to reach:
- **Language and text**: Kneser-Ney n-gram models and character GPT corpora (`methods/neural/language-models`, two notes
  already use `kneserNey`), the CRF and BiRNN hyphenation taggers (`methods/text/hyphenation`; omai has Liang's patterns only),
  the tokeniser suites and corpora (`methods/text/{tokenisers,corpora}`).
- **Transfer and meta-learning**: MAML (`mamlRun`), domain-adversarial training and gradient reversal (`domainAdaptationRun`,
  `gradientReversal`), few-shot episodes (`fewShotEpisode`), continual runs (`continualRun`); omai has `coral`.
- **Classifier models**: AdPredictor and the Bayes point machine (`methods/inference/classifier-models`).
- **Optimisation**: Pegasos and the dual coordinate steps of the SVM family (`methods/learning/kernel-methods`; omai has SMO
  and a dual coordinate solver in `optim`), multi-fidelity GP (`fitMultiFidelityGp`), `ladderParameterFree`.
- **Self-supervised**: self-distillation training and linear predictive codes (`methods/neural/self-supervised`); omai has
  VICReg and Barlow Twins losses.
- **Recommenders**: matchbox and the biased matrix-factorisation scorers (`methods/retrieval/recommenders`, 27 values;
  omai has implicit ALS and the `nn/blocks` recommenders).
- **Topic models**: dynamic and correlated topic steps (`methods/inference/topic-models`, 27 values).

Where the AIFN name is the only difference (metrics 88, probability tests 36, GLM and ordinal 59, text pipeline 60), a
count of unmatched names overstates the gap; those areas were not reviewed.

`lab/` also imports 574 values that have no omai counterpart by name and no note: mostly synthetic data (41),
distribution classes (22), subgroup-discovery internals (16), render controls (16), metrics (15), text features (15),
bijectors (14). The lab explores the whole engine, so this is breadth, not demand.

## 5. Render-side features of aifn-render that omai-render does not have

The big difference (a React layer-component API against option builders and frames) is in `gaps.md`. Beyond it,
AIFN's render surface that no note uses, and omai-render has no counterpart for:
- **A UI kit**: 95 shadcn re-exports (`aifn-render/ui/*`). Deliberate; the site brings its own.
- **Code display and editing**: `CodeBlock`, `CodeEditor` (CodeMirror), a Prolog language mode, `ProgramStatus`; used by the
  lab. Not planned for omai-render.
- **Control kit beyond the basics**: `Combobox`, `MultiCombobox`, `RampPicker`, `SwatchPicker`, `ThemeToggle`, `RevealToggle`,
  `Segmented`, `ParamControls`/`NumericControl` generated from a spec. omai-render has `Choice`, `Select`, `NumberField`,
  `Toggle`, `SpaceControls` and `ModelSelector`; a combobox for long lists is the visible gap (S).
- **Pinned parameters** (`usePinned`, `pinField`; 14 lab uses): bind a figure's parameter to a value pinned from elsewhere on the
  page. Not in omai-render (the session preferences and `useLinkState` cover related ground). M.
- **Layout helpers of the lab**: `Columns`, `PanelSlot`, `layeredLayout`, `FigurePage`. omai has `FigureGrid` and `bento`.
- **Plot extras**: `AxisToolbar`, `ScaleBar`, `ReadoutGroup`, `SupportBand`, `SignedArea`, `defineLayer` (a user-defined layer).
  omai has `ZoomPanel`, `band` and `Layer`; a scale bar and a signed-area layer are S each.
- **Gym renderers**: `CartPoleRenderer`, `PendulumRenderer`, `BanditRenderer`, `GridRenderer`. omai draws environments from
  their own `render` (`environments#renderCartPole`), so these are covered by a different route; only `GridView` with value
  and policy overlays is in `gaps.md`.
- **Maths macros and figure ids** (`registerMathMacros`, `useFigureId`, `FigureScope`): omai has `extendMacros` and its own
  figure ids; no gap.

Likely future demand, by the stubs: part 6 chapters on interpreting transformers, SciML and generative models will want
attention-map and token-strip views (omai has `TokenStripView`, `AttributionView`), vector-field and landscape views
(has `VectorFieldView`, `LandscapePanel`) and operator or PINN field views (partly: `FieldPanel`); no render gap is expected there
beyond the declarative layer itself.
