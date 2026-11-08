# What the notes use

Counts come from `scripts/aifn-usage/scan-usage.ts` (imports) and `match.ts` (the match against omai); see `README.md`
for the rules. A note "uses" a symbol when a file of the note, or a file it imports, imports it. The tables between the
`generated` markers are written by `match.ts`; the prose is hand-written.

## Headline numbers

<!-- generated:totals -->
- Notes scanned: 2486; notes that reach the aifn packages: 1274.
- Distinct aifn declarations used by notes: 1543 (1264 values, the rest types).
- Values with a counterpart in omai (same name, renamed, or a differently shaped API): 871 of 1264 (69%).
<!-- /generated:totals -->

By package, the values (functions, classes, constants) the notes use and what omai has:

| package        | values used | same name | renamed or moved | other shape (design) | not a capability | no counterpart |
| -------------- | ----------- | --------- | ---------------- | -------------------- | ---------------- | -------------- |
| `aifn-compute` | 699         | 286       | 240              | 15                   | 4                | 154            |
| `aifn-methods` | 466         | 105       | 118              | 10                   | 0                | 233            |
| `aifn-render`  | 99          | 7         | 0                | 71                   | 15               | 6              |

Three readings of the same data, from the kindest to the harshest:
- **By use.** 17966 of the 19664 (note, value) pairs, 91%, have an omai counterpart. The pairs that do not are mostly
  `Readout` (1188 notes).
- **By note.** Of the 1274 notes that reach the engine, 1039 (82%) have a counterpart in omai for every symbol except the
  render ones. Only 86 have one for every symbol including the render ones, because `Readout` is in nearly every figure.
  953 notes are blocked by `Readout` alone.
- **By symbol.** 871 of the 1264 values (69%). The rest is a long tail: 967 of the 1543 used declarations are used by one
  note, and 337 of the 393 values without a counterpart are used by one note.

"Has a counterpart" does not mean "drop-in". Of the 871 covered values, 398 have the same name, 358 were renamed or
moved, and 96 are covered by a differently shaped API (the render layer, random streams, the estimator protocol): a
note that uses them has to be rewritten, not re-imported. `gaps.md` starts with what that costs.

## What the notes use, by area

The area is the first two folders of the AIFN source path. "gap" counts values with no omai counterpart (it includes the
convolutional sparse coding symbols, which are in progress).

<!-- generated:usage-by-area -->
| area                 | notes | values used | same or renamed | other shape | not a capability | gap | top symbols (notes)                                                                                                          |
| -------------------- | ----- | ----------- | --------------- | ----------- | ---------------- | --- | ---------------------------------------------------------------------------------------------------------------------------- |
| render/layout        | 1274  | 8           | 1               | 7           | 0                | 0   | Figure (1274), ControlRow (48), ControlGroup (31), Dashboard (7), DashboardCell (7)                                          |
| render/viz           | 1190  | 25          | 2               | 22          | 0                | 1   | Readout (1188), Plot (1157), useAxis (1157), formatNumber (898), Curve (826)                                                 |
| render/state         | 1143  | 13          | 0               | 13          | 0                | 0   | useFigureState (1142), int (682), choice (620), slider (554), float (527)                                                    |
| compute/foundation   | 826   | 106         | 91              | 5           | 3                | 7   | stream (648), toFlat (577), normal (316), linspace (286), uniform (244)                                                      |
| render/controls      | 251   | 10          | 4               | 4           | 0                | 2   | Player (191), Select (52), Slider (41), NumberSelector (35), usePlayhead (15)                                                |
| compute/numerics     | 215   | 82          | 70              | 0           | 1                | 11  | normalCdf (70), normalPdf (47), normalQuantile (32), sigmoid (27), logGamma (15)                                             |
| render/diagram       | 189   | 19          | 0               | 19          | 0                | 0   | Diagram (170), MathText (57), link (50), variable (50), factor (22)                                                          |
| methods/data         | 175   | 101         | 49              | 1           | 0                | 51  | moons (50), blobs (23), semiSupervisedSplit (20), circles (12), forrester (10)                                               |
| compute/learning     | 142   | 142         | 104             | 3           | 0                | 35  | dataset (81), rbf (25), crossValidate (25), meanSquaredError (23), kFold (21)                                                |
| methods/learning     | 120   | 98          | 65              | 2           | 0                | 31  | linearRegression (30), kNearestNeighbours (20), polynomialFeatures (19), logisticRegression (17), randomFourierFeatures (11) |
| render/ui            | 100   | 14          | 0               | 0           | 14               | 0   | Button (76), Table (25), TableBody (25), TableCell (25), TableHead (25)                                                      |
| compute/probability  | 81    | 61          | 55              | 0           | 0                | 6   | Normal (15), Binomial (10), standardDeviation (9), MultivariateNormal (7), median (7)                                        |
| compute/nn           | 79    | 113         | 61              | 7           | 0                | 45  | Mlp (15), xavierUniform (10), trainingLoop (10), networkCost (5), linearRecurrence (5)                                       |
| compute/signal       | 55    | 39          | 34              | 0           | 0                | 5   | getWindow (39), hzToMel (12), lfilter (11), melToHz (11), freqz (10)                                                         |
| methods/neural       | 47    | 81          | 12              | 2           | 0                | 67  | consistencyTraining (7), predictProbabilities (7), pinnVariantRun (4), TwoTower (2), embed (2)                               |
| methods/theory       | 45    | 18          | 12              | 2           | 0                | 4   | targetFunction (17), drawTrainingSet (15), unitGrid (10), fitAndPredict (5), tailBounds (4)                                  |
| compute/optim        | 34    | 18          | 15              | 0           | 0                | 3   | adamRule (19), gradientDescent (8), objectiveFn (7), lbfgs (5), projectSimplex (5)                                           |
| compute/graph        | 33    | 33          | 28              | 0           | 0                | 5   | fromEdges (8), pointAffinity (6), propagate (6), treeFromParents (5), laplacian (5)                                          |
| compute/dynamics     | 22    | 11          | 9               | 0           | 0                | 2   | rungeKutta (7), solveIvp (7), dormandPrince (3), symplectic (3), withEvents (3)                                              |
| methods/unsupervised | 22    | 36          | 27              | 0           | 0                | 9   | pca (4), semiSupervisedMixtureSteps (3), knnScore (2), localOutlierFactor (2), localOutlierScore (2)                         |
| methods/optimisation | 19    | 27          | 11              | 3           | 0                | 13  | bayesianOptimisationSteps (8), expectedImprovement (4), ucbBeta (4), multiFidelitySteps (3), successiveHalvingSchedule (3)   |
| compute/logic        | 15    | 60          | 33              | 0           | 0                | 27  | parseProgram (5), termToString (5), prologProgram (3), solveQuery (3), clauseToString (3)                                    |
| methods/generative   | 14    | 36          | 13              | 0           | 0                | 23  | gaussianMixtureData (3), linearSchedule (3), autoencoderRun (2), ddimSampler (2), mixtureLogDensity (2)                      |
| render/design        | 12    | 6           | 0               | 6           | 0                | 0   | seriesColor (10), useTheme (10), chrome (2), MARKER_SHAPES (1), sequential (1)                                               |
| render/misc          | 12    | 1           | 0               | 0           | 1                | 0   | cn (12)                                                                                                                      |
| compute/systems      | 10    | 1           | 1               | 0           | 0                | 0   | transferFunction (10)                                                                                                        |
| compute/inference    | 7     | 30          | 23              | 0           | 0                | 7   | forwardBackward (1), variableElimination (1), viterbi (1), epLogEvidence (1), expectationPropagation (1)                     |
| compute/transport    | 6     | 3           | 2               | 0           | 0                | 1   | monotonePlan (6), sinkhorn (6), wasserstein1d (6)                                                                            |
| methods/inference    | 6     | 19          | 3               | 0           | 0                | 16  | rateStream (2), learnerModelRun (1), clutterEp (1), clutterLogLikelihood (1), clutterPosterior (1)                           |
| methods/gym          | 5     | 17          | 12              | 0           | 0                | 5   | greedyPath (3), valueIteration (3), GRID_ACTION_NAMES (2), train (2), greedyPolicy (2)                                       |
| render/gym           | 5     | 3           | 0               | 0           | 0                | 3   | GridView (5), GymTrainer (1), trainingRun (1)                                                                                |
| methods/dynamics     | 3     | 15          | 11              | 0           | 0                | 4   | bifurcationDiagram (1), cobweb (1), henonMap (1), logisticMap (1), lyapunovCurve (1)                                         |
| methods/evaluation   | 3     | 3           | 3               | 0           | 0                | 0   | boxIou (2), boundaryF1 (1), maskIou (1)                                                                                      |
| methods/information  | 2     | 2           | 2               | 0           | 0                | 0   | blahutArimotoCapacity (1), rateDistortionCurve (1)                                                                           |
| methods/vision       | 2     | 6           | 3               | 0           | 0                | 3   | anchorGrid (2), anchorFreeTargets (1), decodeBoxes (1), encodeBoxes (1), matchAnchors (1)                                    |
| methods/algorithms   | 1     | 5           | 0               | 0           | 0                | 5   | atomText (1), countByLength (1), runStringProgram (1), topPrograms (1), versionSpace (1)                                     |
| methods/retrieval    | 1     | 1           | 0               | 0           | 0                | 1   | recommenderRun (1)                                                                                                           |
| methods/timeseries   | 1     | 1           | 0               | 0           | 0                | 1   | constantVelocityModel (1)                                                                                                    |
<!-- /generated:usage-by-area -->

What stands out:
- **Figures and randomness are used by nearly every note that has a figure.** `aifn-render` is imported by 1274 notes
  (`Figure` by all of them, `Plot`, `useAxis` and `useFigureState` by more than 1100), `aifn-compute/foundation/random`
  by 643 (`stream` alone by 648 counting worker calls), `foundation/tensor` by 608. Past those three, the numerics
  (`numerics/special` 154 notes, `probability/stats` 39, `numerics/linalg` 44, `foundation/fourier` 42) and the learning
  kit (`learning/estimators` 82, `learning/metrics` 60, `learning/validate` 33) follow.
- **The datasets are the third heavy user of the methods package.** `aifn-methods/data/synthetic` is imported by 134
  notes (179 import lines), mostly `moons` (50), `blobs` (23), `semiSupervisedSplit` (20) and the planted-truth
  generators. Half of the 101 values used from `aifn-methods/data` (synthetic, real, objectives, signals, targets) have no omai counterpart.
- **The methods package is a long tail.** 233 of its 466 used values have no counterpart, but they are spread over 84
  import paths with one to a few notes each (`neural/pinn`, `neural/operator`, `generative/*`, `neural/semi-supervised`,
  `inference/rating-models`). Much of that is the AIFN methods layer (functions called `...Run`, `...Steps`,
  `...Trace`) that omai deliberately has no layer for; see `gaps.md`.
- **The notes do their own numerics too.** 103 `.ts` files of 120 lines or more under `content/notes` (largest:
  group-sequential design tables, 4860 lines; ordinal regression, 1325; splines, 821; empirical mode decomposition, 611)
  implement things in a note instead of importing them. `future.md` lists them.

## The 40 most used values

<!-- generated:top-symbols -->
| symbol                                | notes | status    | in omai                          |
| ------------------------------------- | ----- | --------- | -------------------------------- |
| `render/layout#Figure`                | 1274  | design    | `render/react#FigureFrame`       |
| `render/viz#Readout`                  | 1188  | unmatched | -                                |
| `render/viz#Plot`                     | 1157  | design    | `render/react#Chart`             |
| `render/viz#useAxis`                  | 1157  | design    | `render/charts#cartesian`        |
| `render/state#useFigureState`         | 1142  | design    | `render/react#SpaceControls`     |
| `render/viz#formatNumber`             | 898   | exact     | `render/theme#formatNumber`      |
| `render/viz#Curve`                    | 826   | design    | `render/charts#lines`            |
| `render/state#int`                    | 682   | design    | `render/react#SpaceControls`     |
| `render/viz#Points`                   | 675   | design    | `render/charts#points`           |
| `compute/foundation/random#stream`    | 648   | design    | `engine/random#key`              |
| `render/state#choice`                 | 620   | design    | `render/react#SpaceControls`     |
| `compute/foundation/tensor#toFlat`    | 577   | exact     | `engine/tensor#toFlat`           |
| `render/state#slider`                 | 554   | design    | `render/react#SpaceControls`     |
| `render/state#float`                  | 527   | design    | `render/react#SpaceControls`     |
| `compute/foundation/random#normal`    | 316   | exact     | `engine/random#normal`           |
| `compute/foundation/tensor#linspace`  | 286   | exact     | `engine/tensor#linspace`         |
| `render/viz#seriesLayers`             | 248   | design    | `render/charts#layer`            |
| `compute/foundation/random#uniform`   | 244   | exact     | `engine/random#uniform`          |
| `render/viz#Raster`                   | 212   | design    | `render/charts#heatmap`          |
| `render/controls#Player`              | 191   | exact     | `render/react#Player`            |
| `render/viz#Segments`                 | 183   | design    | `render/charts#lines`            |
| `render/diagram#Diagram`              | 170   | design    | `render/structures#diagram`      |
| `render/viz#Plots`                    | 159   | design    | `render/charts#subplots`         |
| `render/viz#Bars`                     | 146   | design    | `render/charts#bars`             |
| `render/state#setting`                | 142   | design    | `render/react#SpaceControls`     |
| `render/state#useComputed`            | 131   | design    | `render/producer#workerProducer` |
| `compute/foundation/random#child`     | 126   | design    | `engine/random#foldIn`           |
| `compute/foundation/tensor#tensor`    | 124   | exact     | `engine/tensor#tensor`           |
| `compute/foundation/tensor#fromData`  | 120   | alias     | `engine/tensor#fromStorage`      |
| `compute/foundation/tensor#toRows`    | 101   | exact     | `engine/base#toRows`             |
| `render/viz#FrameContext`             | 98    | design    | `render/react#useFrameSize`      |
| `compute/foundation/trace#trace`      | 82    | alias     | `engine/trace#runAlgorithm`      |
| `compute/learning/estimators#dataset` | 81    | design    | `engine/data#Dataset`            |
| `compute/foundation/random#normals`   | 78    | alias     | `engine/random#normal`           |
| `render/ui#Button`                    | 76    | internal  | -                                |
| `render/viz#Annotation`               | 71    | design    | `render/charts#annotations`      |
| `compute/numerics/special#normalCdf`  | 70    | alias     | `engine/numerics/special#ndtr`   |
| `render/viz#Vectors`                  | 64    | design    | `render/charts#fieldArrows`      |
| `render/viz#Area`                     | 63    | design    | `render/charts#band`             |
| `render/diagram#MathText`             | 57    | design    | `render/react#Tex`               |
<!-- /generated:top-symbols -->

## The most used import paths

<!-- generated:import-paths -->
| import path                              | notes | symbols |
| ---------------------------------------- | ----- | ------- |
| `aifn-render`                            | 1274  | 118     |
| `aifn-compute/foundation/random`         | 643   | 16      |
| `aifn-compute/foundation/tensor`         | 608   | 78      |
| `aifn-compute/numerics/special`          | 154   | 26      |
| `aifn-methods/data/synthetic`            | 134   | 149     |
| `aifn-compute/foundation/trace`          | 111   | 13      |
| `aifn-compute/learning/estimators`       | 82    | 18      |
| `aifn-compute/learning/metrics`          | 60    | 48      |
| `aifn-methods/learning/preprocessing`    | 47    | 20      |
| `aifn-methods/learning/linear`           | 45    | 20      |
| `aifn-compute/numerics/linalg`           | 44    | 17      |
| `aifn-compute/foundation/fourier`        | 42    | 11      |
| `aifn-compute/probability/stats`         | 39    | 28      |
| `aifn-compute/signal/windows`            | 36    | 4       |
| `aifn-compute/learning/validate`         | 33    | 20      |
| `aifn-compute/probability/distributions` | 32    | 43      |
| `aifn-compute/graph`                     | 26    | 12      |
| `aifn-compute/learning/kernels`          | 26    | 11      |
| `aifn-methods/learning/neighbours`       | 23    | 3       |
| `aifn-methods/learning/semi-supervised`  | 23    | 25      |
| `aifn-methods/theory`                    | 23    | 5       |
| `aifn-compute/nn/layers`                 | 21    | 13      |
| `aifn-methods/data`                      | 21    | 20      |
| `aifn-methods/learning/generalised/glm`  | 21    | 3       |
| `aifn-compute/nn/functional`             | 20    | 18      |
| `aifn-compute/optim/first-order`         | 20    | 9       |
| `aifn-compute/probability/tests`         | 20    | 40      |
| `aifn-compute/dynamics/ode`              | 19    | 18      |
| `aifn-compute/foundation/convolution`    | 19    | 6       |
| `aifn-methods/data/objectives`           | 19    | 9       |
| `aifn-compute/learning/losses`           | 18    | 10      |
| `aifn-compute/foundation/pytree`         | 17    | 4       |
| `aifn-compute/learning/compose`          | 17    | 8       |
| `aifn-compute/nn/graph`                  | 17    | 23      |
| `aifn-methods/data/real`                 | 17    | 7       |
| `aifn-compute/graph/propagation`         | 15    | 12      |
| `aifn-compute/learning/selection`        | 15    | 13      |
| `aifn-compute/foundation/space`          | 14    | 8       |
| `aifn-compute/learning/explain`          | 14    | 48      |
| `aifn-compute/nn/training`               | 13    | 8       |
<!-- /generated:import-paths -->

`aifn-compute/foundation/random` and `aifn-render` are the two entry points that a port touches in nearly every note.
The 175 import paths with at least one note are in `data/usage.json` (`subpaths`).

## Worker calls

Heavy work runs in a worker through `useComputed` and `call('module/export', ...)`, with the address a string. The
scanner reads these as uses of the named export. Six distinct calls appear (the rest of a figure's work is on the main
thread):

<!-- generated:worker-calls -->
| worker call                | notes |
| -------------------------- | ----- |
| `foundation/random/stream` | 13    |
| `neural/pinn/pinnRun`      | 2     |
| `data/real/karateClub`     | 1     |
| `foundation/random/child`  | 1     |
| `neural/pinn/burgers`      | 1     |
| `neural/pinn/heat`         | 1     |
<!-- /generated:worker-calls -->

## Other places that import the engine

- `lab/` (an explorer of the engine, never the site): 262 of its 415 files import aifn, 1113 declarations that no note
  uses (574 of them without an omai counterpart by name). Kept apart; they feed `future.md`.
- `site/src`: 8 files (the shared widgets and providers). `scripts/`: none. `plugins/`: one file.
- Two dynamic imports load data modules lazily (`aifn-methods/data/real/ecg` and `.../fonts`).
