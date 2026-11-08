# Generated tables

Written by `scripts/aifn-usage/match.ts`; do not edit.

## Usage by area

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

## The 40 most used values

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

## The most used import paths

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

## Worker calls (`call('module/export')`)

| worker call                | notes |
| -------------------------- | ----- |
| `foundation/random/stream` | 13    |
| `neural/pinn/pinnRun`      | 2     |
| `data/real/karateClub`     | 1     |
| `foundation/random/child`  | 1     |
| `neural/pinn/burgers`      | 1     |
| `neural/pinn/heat`         | 1     |

## Used symbols with no omai counterpart yet (values)

| symbol                                                                     | kind     | notes | sole | status    | aifn source                                                          |
| -------------------------------------------------------------------------- | -------- | ----- | ---- | --------- | -------------------------------------------------------------------- |
| `aifn-render/viz#Readout`                                                  | function | 1188  | 953  | unmatched | `render/viz/Readout.tsx:5`                                           |
| `aifn-compute/learning/explain#simplifyRule`                               | function | 9     | 0    | unmatched | `compute/learning/explain/rules.ts:81`                               |
| `aifn-methods/data/synthetic#gaussians`                                    | function | 9     | 0    | unmatched | `methods/data/synthetic/points.ts:1007`                              |
| `aifn-methods/learning/semi-supervised#rowArgmax`                          | function | 7     | 0    | unmatched | `methods/learning/semi-supervised/self-training.ts:112`              |
| `aifn-methods/neural/semi-supervised#consistencyTraining`                  | function | 7     | 0    | unmatched | `methods/neural/semi-supervised/consistency.ts:430`                  |
| `aifn-methods/neural/semi-supervised#predictProbabilities`                 | function | 7     | 0    | unmatched | `methods/neural/semi-supervised/consistency.ts:225`                  |
| `aifn-compute/foundation/fourier#isPowerOfTwo`                             | function | 6     | 0    | unmatched | `compute/foundation/fourier/fft.ts:65`                               |
| `aifn-compute/transport#monotonePlan`                                      | function | 6     | 0    | unmatched | `compute/transport/oneD.ts:117`                                      |
| `aifn-methods/data/objectives#svmTuningSurface`                            | function | 6     | 0    | unmatched | `methods/data/objectives/svm.ts:88`                                  |
| `aifn-compute/foundation/fourier#dctMatrix`                                | function | 5     | 0    | unmatched | `compute/foundation/fourier/dct.ts:26`                               |
| `aifn-render/gym#GridView`                                                 | function | 5     | 0    | unmatched | `render/gym/GridView.tsx:73`                                         |
| `aifn-methods/data/synthetic#learningCurves`                               | function | 4     | 0    | unmatched | `methods/data/synthetic/learning-curves.ts:114`                      |
| `aifn-methods/data/synthetic#swissRoll2d`                                  | function | 4     | 0    | unmatched | `methods/data/synthetic/densities.ts:418`                            |
| `aifn-methods/neural/pinn#pinnVariantRun`                                  | function | 4     | 0    | unmatched | `methods/neural/pinn/variants.ts:248`                                |
| `aifn-compute/learning/metrics#relativeL2Error`                            | const    | 3     | 0    | unmatched | `compute/learning/metrics/regression.ts:220`                         |
| `aifn-compute/nn/layers#ActivationLayer`                                   | function | 3     | 0    | unmatched | `compute/nn/layers/layers.ts:834`                                    |
| `aifn-compute/nn/training#methodTraining`                                  | function | 3     | 0    | unmatched | `compute/nn/training/method.ts:89`                                   |
| `aifn-methods/data/signals#syntheticEcg`                                   | function | 3     | 0    | unmatched | `methods/data/signals/heartbeat.ts:102`                              |
| `aifn-methods/data/synthetic#shapeTextureDataset`                          | function | 3     | 0    | unmatched | `methods/data/synthetic/shape-texture.ts:219`                        |
| `aifn-methods/generative/diffusion#gaussianMixtureData`                    | function | 3     | 0    | unmatched | `methods/generative/diffusion/mixture.ts:45`                         |
| `aifn-methods/learning/semi-supervised#laplacianRegularisedLeastSquares`   | function | 3     | 0    | unmatched | `methods/learning/semi-supervised/laplacian-rls.ts:70`               |
| `aifn-methods/theory/concentration#concentrationStudy`                     | function | 3     | 0    | unmatched | `methods/theory/concentration/concentration.ts:214`                  |
| `aifn-methods/unsupervised/clustering#semiSupervisedMixtureSteps`          | function | 3     | 0    | unmatched | `methods/unsupervised/clustering/semi-supervised-mixture.ts:293`     |
| `aifn-render/controls#StatusText`                                          | function | 3     | 0    | unmatched | `render/controls/base/StatusText.tsx:10`                             |
| `aifn-compute/dynamics/fields#partials`                                    | function | 2     | 0    | unmatched | `compute/dynamics/fields/partials.ts:71`                             |
| `aifn-compute/foundation/tensor#get`                                       | function | 2     | 0    | unmatched | `compute/foundation/tensor/structure.ts:489`                         |
| `aifn-compute/logic/induction#allInstances`                                | function | 2     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:234`                        |
| `aifn-compute/nn/attention#positionMask`                                   | function | 2     | 0    | unmatched | `compute/nn/attention/masks.ts:89`                                   |
| `aifn-compute/nn/attention#positionRange`                                  | function | 2     | 0    | unmatched | `compute/nn/attention/masks.ts:26`                                   |
| `aifn-compute/nn/attention#sparseAttentionMask`                            | function | 2     | 0    | unmatched | `compute/nn/attention/sparse.ts:73`                                  |
| `aifn-compute/nn/graph#batchGraphs`                                        | function | 2     | 0    | unmatched | `compute/nn/graph/pooling.ts:77`                                     |
| `aifn-compute/nn/sets#padSets`                                             | function | 2     | 0    | unmatched | `compute/nn/sets/sets.ts:56`                                         |
| `aifn-compute/numerics/interpolate#evaluatePiecewise`                      | function | 2     | 0    | unmatched | `compute/numerics/interpolate/piecewise.ts:107`                      |
| `aifn-methods/data/synthetic#blockModelProbabilities`                      | function | 2     | 0    | unmatched | `methods/data/synthetic/graphs.ts:144`                               |
| `aifn-methods/data/synthetic#pairedShapes`                                 | function | 2     | 0    | unmatched | `methods/data/synthetic/paired.ts:226`                               |
| `aifn-methods/data/synthetic#shuffleDataset`                               | function | 2     | 0    | unmatched | `methods/data/synthetic/points.ts:1300`                              |
| `aifn-methods/generative/autoencoders#autoencoderRun`                      | function | 2     | 0    | unmatched | `methods/generative/autoencoders/run.ts:151`                         |
| `aifn-methods/generative/diffusion#mixtureLogDensity`                      | function | 2     | 0    | unmatched | `methods/generative/diffusion/mixture.ts:193`                        |
| `aifn-methods/generative/diffusion#mixtureNoisePredictor`                  | function | 2     | 0    | unmatched | `methods/generative/diffusion/mixture.ts:231`                        |
| `aifn-methods/gym#GRID_ACTION_NAMES`                                       | const    | 2     | 0    | unmatched | `methods/gym/mdp.ts:62`                                              |
| `aifn-methods/gym#train`                                                   | function | 2     | 0    | unmatched | `methods/gym/train.ts:189`                                           |
| `aifn-methods/inference/rating-models#rateStream`                          | function | 2     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:476`                    |
| `aifn-methods/learning/kernel-methods#dualDecision`                        | function | 2     | 0    | unmatched | `methods/learning/kernel-methods/svm.ts:379`                         |
| `aifn-methods/learning/semi-supervised#symmetricHinge`                     | function | 2     | 0    | unmatched | `methods/learning/semi-supervised/transductive-svm.ts:45`            |
| `aifn-methods/learning/semi-supervised#symmetricSigmoid`                   | function | 2     | 0    | unmatched | `methods/learning/semi-supervised/transductive-svm.ts:59`            |
| `aifn-methods/neural/contrastive#embed`                                    | function | 2     | 0    | unmatched | `methods/neural/contrastive/clip.ts:246`                             |
| `aifn-methods/neural/graph#propagationStack`                               | function | 2     | 0    | unmatched | `methods/neural/graph/depth.ts:85`                                   |
| `aifn-methods/neural/language-models#charCorpus`                           | function | 2     | 0    | unmatched | `methods/neural/language-models/corpus.ts:54`                        |
| `aifn-methods/neural/language-models#decodeChars`                          | function | 2     | 0    | unmatched | `methods/neural/language-models/corpus.ts:92`                        |
| `aifn-methods/neural/language-models#encodeChars`                          | function | 2     | 0    | unmatched | `methods/neural/language-models/corpus.ts:73`                        |
| `aifn-methods/neural/language-models#kneserNey`                            | function | 2     | 0    | unmatched | `methods/neural/language-models/ngram.ts:159`                        |
| `aifn-methods/neural/pinn#heatProblem`                                     | const    | 2     | 0    | unmatched | `methods/neural/pinn/problems.ts:232`                                |
| `aifn-methods/neural/pinn#pinnRun`                                         | function | 2     | 0    | unmatched | `methods/neural/pinn/run.ts:176`                                     |
| `aifn-methods/optimisation/bayesian#maxValueEntropySearch`                 | function | 2     | 0    | unmatched | `methods/optimisation/bayesian/entropy.ts:189`                       |
| `aifn-methods/unsupervised/clustering#CORE`                                | const    | 2     | 0    | unmatched | `methods/unsupervised/clustering/density.ts:26`                      |
| `aifn-methods/vision/detection#anchorGrid`                                 | function | 2     | 0    | unmatched | `methods/vision/detection/anchors.ts:67`                             |
| `aifn-compute/dynamics/control#lqgSimulation`                              | function | 1     | 0    | unmatched | `compute/dynamics/control/lqg.ts:199`                                |
| `aifn-compute/foundation/tensor#blellochScanSteps`                         | function | 1     | 0    | unmatched | `compute/foundation/tensor/scan.ts:322`                              |
| `aifn-compute/foundation/tensor#hillisSteeleScanSteps`                     | function | 1     | 0    | unmatched | `compute/foundation/tensor/scan.ts:250`                              |
| `aifn-compute/foundation/tensor#toComplexFlat`                             | function | 1     | 0    | unmatched | `compute/foundation/tensor/create.ts:432`                            |
| `aifn-compute/foundation/trace#extend`                                     | function | 1     | 0    | unmatched | `compute/foundation/trace/runners.ts:716`                            |
| `aifn-compute/graph/isomorphism#folkloreWeisfeilerLeman2`                  | function | 1     | 0    | unmatched | `compute/graph/isomorphism/weisfeiler-leman.ts:289`                  |
| `aifn-compute/graph/isomorphism#subgraphWeisfeilerLeman`                   | function | 1     | 0    | unmatched | `compute/graph/isomorphism/subgraphs.ts:105`                         |
| `aifn-compute/graph/measures#pairHeuristics`                               | function | 1     | 0    | unmatched | `compute/graph/measures/links.ts:43`                                 |
| `aifn-compute/graph/rewiring#fosrRewire`                                   | function | 1     | 0    | unmatched | `compute/graph/rewiring/rewiring.ts:77`                              |
| `aifn-compute/graph/rewiring#withVirtualNode`                              | function | 1     | 0    | unmatched | `compute/graph/rewiring/rewiring.ts:40`                              |
| `aifn-compute/inference/filtering#laggedObservations`                      | function | 1     | 0    | unmatched | `compute/inference/filtering/changepoint.ts:769`                     |
| `aifn-compute/inference/filtering#normalisedInnovationSquared`             | function | 1     | 0    | unmatched | `compute/inference/filtering/kalman.ts:956`                          |
| `aifn-compute/inference/filtering#regressionNormalGamma`                   | function | 1     | 0    | unmatched | `compute/inference/filtering/changepoint.ts:806`                     |
| `aifn-compute/inference/filtering#runLengthMass`                           | function | 1     | 0    | unmatched | `compute/inference/filtering/changepoint.ts:433`                     |
| `aifn-compute/inference/filtering#runLengthRow`                            | function | 1     | 0    | unmatched | `compute/inference/filtering/changepoint.ts:329`                     |
| `aifn-compute/inference/message-passing#decodeBeliefs`                     | function | 1     | 0    | unmatched | `compute/inference/message-passing/bp.ts:669`                        |
| `aifn-compute/inference/model#discreteFactor`                              | function | 1     | 0    | unmatched | `compute/inference/model/factors.ts:152`                             |
| `aifn-compute/learning/calibration#topLabelConfidence`                     | function | 1     | 0    | unmatched | `compute/learning/calibration/maps.ts:493`                           |
| `aifn-compute/learning/explain#applyRuleList`                              | function | 1     | 0    | unmatched | `compute/learning/explain/rules.ts:662`                              |
| `aifn-compute/learning/explain#denseFunction`                              | function | 1     | 0    | unmatched | `compute/learning/explain/network.ts:215`                            |
| `aifn-compute/learning/explain#denseOutput`                                | function | 1     | 0    | unmatched | `compute/learning/explain/network.ts:245`                            |
| `aifn-compute/learning/explain#featureMaps`                                | function | 1     | 0    | unmatched | `compute/learning/explain/cam.ts:51`                                 |
| `aifn-compute/learning/explain#fromMlpParams`                              | function | 1     | 0    | unmatched | `compute/learning/explain/network.ts:268`                            |
| `aifn-compute/learning/explain#normaliseMap`                               | function | 1     | 0    | unmatched | `compute/learning/explain/cam.ts:169`                                |
| `aifn-compute/learning/explain#ruleList`                                   | function | 1     | 0    | unmatched | `compute/learning/explain/rules.ts:583`                              |
| `aifn-compute/learning/off-policy#switchDoublyRobust`                      | function | 1     | 0    | unmatched | `compute/learning/off-policy/estimators.ts:428`                      |
| `aifn-compute/learning/rules#RULE_MEASURES`                                | const    | 1     | 0    | unmatched | `compute/learning/rules/quality.ts:71`                               |
| `aifn-compute/learning/rules#aggregateBinary`                              | function | 1     | 0    | unmatched | `compute/learning/rules/boolean-rules.ts:56`                         |
| `aifn-compute/learning/rules#booleanRuleColumnGeneration`                  | function | 1     | 0    | unmatched | `compute/learning/rules/boolean-rules.ts:186`                        |
| `aifn-compute/learning/rules#cbaBuild`                                     | function | 1     | 0    | unmatched | `compute/learning/rules/associative.ts:187`                          |
| `aifn-compute/learning/rules#cbaPredict`                                   | function | 1     | 0    | unmatched | `compute/learning/rules/associative.ts:246`                          |
| `aifn-compute/learning/rules#classAssociationRules`                        | function | 1     | 0    | unmatched | `compute/learning/rules/associative.ts:70`                           |
| `aifn-compute/learning/rules#cmarBuild`                                    | function | 1     | 0    | unmatched | `compute/learning/rules/associative.ts:339`                          |
| `aifn-compute/learning/rules#cmarPredict`                                  | function | 1     | 0    | unmatched | `compute/learning/rules/associative.ts:422`                          |
| `aifn-compute/learning/rules#coverMask`                                    | function | 1     | 0    | unmatched | `compute/learning/rules/conditions.ts:89`                            |
| `aifn-compute/learning/rules#ensembleRules`                                | function | 1     | 0    | unmatched | `compute/learning/rules/rulefit.ts:51`                               |
| `aifn-compute/learning/selection#bootstrapPredictionError`                 | function | 1     | 0    | unmatched | `compute/learning/selection/bootstrap.ts:140`                        |
| `aifn-compute/learning/selection#dic`                                      | function | 1     | 0    | unmatched | `compute/learning/selection/predictive.ts:399`                       |
| `aifn-compute/learning/selection#nestedCvInterval`                         | function | 1     | 0    | unmatched | `compute/learning/selection/nested.ts:170`                           |
| `aifn-compute/learning/selection#softThresholdSure`                        | function | 1     | 0    | unmatched | `compute/learning/selection/criteria.ts:210`                         |
| `aifn-compute/learning/subgroups#bitsetAndCount`                           | function | 1     | 0    | unmatched | `compute/learning/subgroups/cover.ts:158`                            |
| `aifn-compute/learning/subgroups#bitsetCount`                              | function | 1     | 0    | unmatched | `compute/learning/subgroups/cover.ts:139`                            |
| `aifn-compute/learning/subgroups#descriptionKey`                           | function | 1     | 0    | unmatched | `compute/learning/subgroups/language.ts:127`                         |
| `aifn-compute/learning/subgroups#formatDescription`                        | function | 1     | 0    | unmatched | `compute/learning/subgroups/language.ts:147`                         |
| `aifn-compute/learning/subgroups#sdMap`                                    | function | 1     | 0    | unmatched | `compute/learning/subgroups/sdmap.ts:135`                            |
| `aifn-compute/learning/subgroups#selectorLanguage`                         | function | 1     | 0    | unmatched | `compute/learning/subgroups/language.ts:263`                         |
| `aifn-compute/learning/subgroups#standardQuality`                          | function | 1     | 0    | unmatched | `compute/learning/subgroups/quality.ts:146`                          |
| `aifn-compute/learning/subgroups#subgroupOf`                               | function | 1     | 0    | unmatched | `compute/learning/subgroups/subgroups.ts:229`                        |
| `aifn-compute/learning/validate#buffered`                                  | function | 1     | 0    | unmatched | `compute/learning/validate/spatial.ts:261`                           |
| `aifn-compute/learning/validate#knndm`                                     | function | 1     | 0    | unmatched | `compute/learning/validate/spatial.ts:536`                           |
| `aifn-compute/logic/induction#aggregateFeatures`                           | function | 1     | 0    | unmatched | `compute/logic/induction/propositionalise.ts:235`                    |
| `aifn-compute/logic/induction#allConjunctions`                             | function | 1     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:252`                        |
| `aifn-compute/logic/induction#attributeFeatures`                           | function | 1     | 0    | unmatched | `compute/logic/induction/propositionalise.ts:130`                    |
| `aifn-compute/logic/induction#bottomUpSearch`                              | function | 1     | 0    | unmatched | `compute/logic/induction/generality-search.ts:153`                   |
| `aifn-compute/logic/induction#conjunctionCoverage`                         | function | 1     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:316`                        |
| `aifn-compute/logic/induction#conjunctionLevel`                            | function | 1     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:275`                        |
| `aifn-compute/logic/induction#conjunctionText`                             | function | 1     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:358`                        |
| `aifn-compute/logic/induction#consistent`                                  | function | 1     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:339`                        |
| `aifn-compute/logic/induction#existentialFeatures`                         | function | 1     | 0    | unmatched | `compute/logic/induction/propositionalise.ts:191`                    |
| `aifn-compute/logic/induction#featureTable`                                | function | 1     | 0    | unmatched | `compute/logic/induction/propositionalise.ts:332`                    |
| `aifn-compute/logic/induction#hasseEdges`                                  | function | 1     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:292`                        |
| `aifn-compute/logic/induction#instanceText`                                | function | 1     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:385`                        |
| `aifn-compute/logic/induction#predictLinks`                                | function | 1     | 0    | unmatched | `compute/logic/induction/rule-mining.ts:238`                         |
| `aifn-compute/logic/induction#sameConjunction`                             | function | 1     | 0    | unmatched | `compute/logic/induction/conjunctions.ts:100`                        |
| `aifn-compute/logic/induction#targetLiterals`                              | function | 1     | 0    | unmatched | `compute/logic/induction/generalise.ts:400`                          |
| `aifn-compute/logic/induction#topDownSearch`                               | function | 1     | 0    | unmatched | `compute/logic/induction/generality-search.ts:79`                    |
| `aifn-compute/logic/induction#wordify`                                     | function | 1     | 0    | unmatched | `compute/logic/induction/propositionalise.ts:280`                    |
| `aifn-compute/logic/propositional#conjoin`                                 | function | 1     | 0    | unmatched | `compute/logic/propositional/formulas.ts:91`                         |
| `aifn-compute/logic/propositional#disj`                                    | function | 1     | 0    | unmatched | `compute/logic/propositional/formulas.ts:103`                        |
| `aifn-compute/logic/propositional#implies`                                 | function | 1     | 0    | unmatched | `compute/logic/propositional/formulas.ts:117`                        |
| `aifn-compute/logic/propositional#negate`                                  | function | 1     | 0    | unmatched | `compute/logic/propositional/formulas.ts:79`                         |
| `aifn-compute/logic/resolution#formatSolution`                             | function | 1     | 0    | unmatched | `compute/logic/resolution/sld.ts:894`                                |
| `aifn-compute/logic/resolution#herbrandBaseSize`                           | function | 1     | 0    | unmatched | `compute/logic/resolution/forward.ts:237`                            |
| `aifn-compute/logic/resolution#sldSteps`                                   | function | 1     | 0    | unmatched | `compute/logic/resolution/sld.ts:423`                                |
| `aifn-compute/logic/terms#PrologSyntaxError`                               | class    | 1     | 0    | unmatched | `compute/logic/terms/syntax.ts:36`                                   |
| `aifn-compute/logic/terms#unificationTrace`                                | function | 1     | 0    | unmatched | `compute/logic/terms/unification.ts:115`                             |
| `aifn-compute/nn/attention#attentionReach`                                 | function | 1     | 0    | unmatched | `compute/nn/attention/sparse.ts:144`                                 |
| `aifn-compute/nn/attention#blockScores`                                    | function | 1     | 0    | unmatched | `compute/nn/attention/block-sparse.ts:48`                            |
| `aifn-compute/nn/attention#blockSparseMask`                                | function | 1     | 0    | unmatched | `compute/nn/attention/block-sparse.ts:136`                           |
| `aifn-compute/nn/attention#capturedWeight`                                 | function | 1     | 0    | unmatched | `compute/nn/attention/block-sparse.ts:161`                           |
| `aifn-compute/nn/attention#flashAttentionSteps`                            | function | 1     | 0    | unmatched | `compute/nn/attention/flash.ts:100`                                  |
| `aifn-compute/nn/attention#maskDensity`                                    | function | 1     | 0    | unmatched | `compute/nn/attention/sparse.ts:124`                                 |
| `aifn-compute/nn/attention#selectBlocks`                                   | function | 1     | 0    | unmatched | `compute/nn/attention/block-sparse.ts:107`                           |
| `aifn-compute/nn/attention#t5RelativeBucket`                               | function | 1     | 0    | unmatched | `compute/nn/attention/positions.ts:385`                              |
| `aifn-compute/nn/decoding#expectedTokensPerCall`                           | function | 1     | 0    | unmatched | `compute/nn/decoding/decoders.ts:504`                                |
| `aifn-compute/nn/decoding#greedyDecoding`                                  | function | 1     | 0    | unmatched | `compute/nn/decoding/decoders.ts:123`                                |
| `aifn-compute/nn/decoding#samplingDecoding`                                | function | 1     | 0    | unmatched | `compute/nn/decoding/decoders.ts:161`                                |
| `aifn-compute/nn/equivariant#p4Act`                                        | function | 1     | 0    | unmatched | `compute/nn/equivariant/p4.ts:96`                                    |
| `aifn-compute/nn/equivariant#p4Lift`                                       | function | 1     | 0    | unmatched | `compute/nn/equivariant/p4.ts:124`                                   |
| `aifn-compute/nn/equivariant#rot90`                                        | function | 1     | 0    | unmatched | `compute/nn/equivariant/p4.ts:77`                                    |
| `aifn-compute/nn/functional#dilatedTapCoverage`                            | function | 1     | 0    | unmatched | `compute/nn/functional/receptive.ts:127`                             |
| `aifn-compute/nn/graph#adjacencySpectrum`                                  | function | 1     | 0    | unmatched | `compute/nn/graph/smoothing.ts:88`                                   |
| `aifn-compute/nn/graph#attentionPool`                                      | function | 1     | 0    | unmatched | `compute/nn/graph/pooling.ts:140`                                    |
| `aifn-compute/nn/graph#diffPool`                                           | function | 1     | 0    | unmatched | `compute/nn/graph/pooling.ts:180`                                    |
| `aifn-compute/nn/graph#graphAttention`                                     | function | 1     | 0    | unmatched | `compute/nn/graph/layers.ts:268`                                     |
| `aifn-compute/nn/graph#graphormerAttention`                                | function | 1     | 0    | unmatched | `compute/nn/graph/transformer.ts:80`                                 |
| `aifn-compute/nn/graph#hypergraphMeanPass`                                 | function | 1     | 0    | unmatched | `compute/nn/graph/hypergraph.ts:52`                                  |
| `aifn-compute/nn/graph#pnaAggregate`                                       | function | 1     | 0    | unmatched | `compute/nn/graph/expressive.ts:214`                                 |
| `aifn-compute/nn/graph#propagationSensitivity`                             | function | 1     | 0    | unmatched | `compute/nn/graph/smoothing.ts:201`                                  |
| `aifn-compute/nn/graph#rgcnTerms`                                          | function | 1     | 0    | unmatched | `compute/nn/graph/relational.ts:53`                                  |
| `aifn-compute/nn/graph#spatialBias`                                        | function | 1     | 0    | unmatched | `compute/nn/graph/transformer.ts:34`                                 |
| `aifn-compute/nn/sequence#eluFeatureMap`                                   | function | 1     | 0    | unmatched | `compute/nn/sequence/linear-attention.ts:49`                         |
| `aifn-compute/nn/sequence#implicitFilter`                                  | function | 1     | 0    | unmatched | `compute/nn/sequence/hyena.ts:91`                                    |
| `aifn-compute/nn/sequence#mlstmParallel`                                   | function | 1     | 0    | unmatched | `compute/nn/sequence/mlstm.ts:162`                                   |
| `aifn-compute/nn/sequence#mlstmRecurrent`                                  | function | 1     | 0    | unmatched | `compute/nn/sequence/mlstm.ts:111`                                   |
| `aifn-compute/nn/sequence#neighbourPairs`                                  | function | 1     | 0    | unmatched | `compute/nn/sequence/scan-orders.ts:127`                             |
| `aifn-compute/nn/sequence#positiveRandomFeatures`                          | function | 1     | 0    | unmatched | `compute/nn/sequence/linear-attention.ts:183`                        |
| `aifn-compute/nn/sequence#routePositions`                                  | function | 1     | 0    | unmatched | `compute/nn/sequence/scan-orders.ts:107`                             |
| `aifn-compute/nn/sequence#scanOrder`                                       | function | 1     | 0    | unmatched | `compute/nn/sequence/scan-orders.ts:50`                              |
| `aifn-compute/nn/sequence#semiseparableMask`                               | function | 1     | 0    | unmatched | `compute/nn/sequence/ssd.ts:90`                                      |
| `aifn-compute/nn/sequence#sequenceDistance`                                | function | 1     | 0    | unmatched | `compute/nn/sequence/scan-orders.ts:150`                             |
| `aifn-compute/nn/sequence#visibleTo`                                       | function | 1     | 0    | unmatched | `compute/nn/sequence/scan-orders.ts:168`                             |
| `aifn-compute/nn/sets#criticalSet`                                         | function | 1     | 0    | unmatched | `compute/nn/sets/sets.ts:118`                                        |
| `aifn-compute/nn/training#activations`                                     | function | 1     | 0    | unmatched | `compute/nn/training/inspect.ts:77`                                  |
| `aifn-compute/numerics/factorisation#distanceDistortion`                   | function | 1     | 0    | unmatched | `compute/numerics/factorisation/projection.ts:169`                   |
| `aifn-compute/numerics/interpolate#linearInterpolant`                      | function | 1     | 0    | unmatched | `compute/numerics/interpolate/piecewise.ts:202`                      |
| `aifn-compute/numerics/neighbours#codePrefixTree`                          | function | 1     | 0    | unmatched | `compute/numerics/neighbours/residual.ts:225`                        |
| `aifn-compute/numerics/neighbours#residualQuantiser`                       | function | 1     | 0    | unmatched | `compute/numerics/neighbours/residual.ts:87`                         |
| `aifn-compute/numerics/neighbours#rqDecode`                                | function | 1     | 0    | unmatched | `compute/numerics/neighbours/residual.ts:178`                        |
| `aifn-compute/numerics/neighbours#rqEncode`                                | function | 1     | 0    | unmatched | `compute/numerics/neighbours/residual.ts:147`                        |
| `aifn-compute/numerics/quadrature#maxGap`                                  | function | 1     | 0    | unmatched | `compute/numerics/quadrature/designs.ts:438`                         |
| `aifn-compute/numerics/quadrature#randomShift`                             | function | 1     | 0    | unmatched | `compute/numerics/quadrature/designs.ts:90`                          |
| `aifn-compute/numerics/quadrature#starDiscrepancy`                         | function | 1     | 0    | unmatched | `compute/numerics/quadrature/designs.ts:338`                         |
| `aifn-compute/numerics/regularisation#convolutionMatrix`                   | function | 1     | 0    | unmatched | `compute/numerics/regularisation/tikhonov.ts:562`                    |
| `aifn-compute/optim/programming#dynamicProgram`                            | function | 1     | 0    | unmatched | `compute/optim/programming/dp.ts:93`                                 |
| `aifn-compute/optim/proximal#fusedLassoSignal`                             | function | 1     | 0    | unmatched | `compute/optim/proximal/totalVariation.ts:181`                       |
| `aifn-compute/optim/proximal#thresholdingPenalty`                          | function | 1     | 0    | unmatched | `compute/optim/proximal/thresholding.ts:121`                         |
| `aifn-compute/probability/information#correlationForMutualInformation`     | function | 1     | 0    | unmatched | `compute/probability/information/bounds.ts:138`                      |
| `aifn-compute/probability/information#donskerVaradhanBound`                | function | 1     | 0    | unmatched | `compute/probability/information/bounds.ts:119`                      |
| `aifn-compute/probability/likelihoods#distributionalFamily`                | function | 1     | 0    | unmatched | `compute/probability/likelihoods/distributional.ts:461`              |
| `aifn-compute/probability/likelihoods#link`                                | function | 1     | 0    | unmatched | `compute/probability/likelihoods/families.ts:129`                    |
| `aifn-compute/probability/tests#bayesianSignedRankTest`                    | function | 1     | 0    | unmatched | `compute/probability/tests/comparison.ts:901`                        |
| `aifn-compute/probability/tests#expectedMaxOfDraws`                        | function | 1     | 0    | unmatched | `compute/probability/tests/comparison.ts:960`                        |
| `aifn-compute/signal/image#upsampleBilinear`                               | function | 1     | 0    | unmatched | `compute/signal/image/resample.ts:94`                                |
| `aifn-compute/signal/sparse#convolutionalDictionaryLearningSteps`          | function | 1     | 0    | unmatched | `compute/signal/sparse/convolutional.ts:387`                         |
| `aifn-compute/signal/sparse#convolutionalSynthesis`                        | function | 1     | 0    | unmatched | `compute/signal/sparse/convolutional.ts:151`                         |
| `aifn-compute/signal/sparse#dictionaryLearningSteps`                       | function | 1     | 0    | unmatched | `compute/signal/sparse/dictionary.ts:229`                            |
| `aifn-compute/signal/sparse#sparseCode`                                    | function | 1     | 0    | unmatched | `compute/signal/sparse/code.ts:86`                                   |
| `aifn-methods/algorithms/program-synthesis#atomText`                       | function | 1     | 0    | unmatched | `methods/algorithms/program-synthesis/strings.ts:162`                |
| `aifn-methods/algorithms/program-synthesis#countByLength`                  | function | 1     | 0    | unmatched | `methods/algorithms/program-synthesis/strings.ts:349`                |
| `aifn-methods/algorithms/program-synthesis#runStringProgram`               | function | 1     | 0    | unmatched | `methods/algorithms/program-synthesis/strings.ts:142`                |
| `aifn-methods/algorithms/program-synthesis#topPrograms`                    | function | 1     | 0    | unmatched | `methods/algorithms/program-synthesis/strings.ts:399`                |
| `aifn-methods/algorithms/program-synthesis#versionSpace`                   | function | 1     | 0    | unmatched | `methods/algorithms/program-synthesis/strings.ts:333`                |
| `aifn-methods/data/objectives#tuningTables`                                | function | 1     | 0    | unmatched | `methods/data/objectives/tuning-tables.ts:88`                        |
| `aifn-methods/data/signals#coupledProcesses`                               | function | 1     | 0    | unmatched | `methods/data/signals/processes.ts:541`                              |
| `aifn-methods/data/signals#syntheticBeat`                                  | function | 1     | 0    | unmatched | `methods/data/signals/heartbeat.ts:37`                               |
| `aifn-methods/data/signals#unevenSinusoids`                                | function | 1     | 0    | unmatched | `methods/data/signals/processes.ts:460`                              |
| `aifn-methods/data/synthetic#ATOM_FEATURES`                                | const    | 1     | 0    | unmatched | `methods/data/synthetic/molecules.ts:226`                            |
| `aifn-methods/data/synthetic#CLOUD_SHAPES`                                 | const    | 1     | 0    | unmatched | `methods/data/synthetic/sets.ts:71`                                  |
| `aifn-methods/data/synthetic#GROWTH_TRUTH`                                 | const    | 1     | 0    | unmatched | `methods/data/synthetic/growth.ts:30`                                |
| `aifn-methods/data/synthetic#PLANTED_MAIN`                                 | const    | 1     | 0    | unmatched | `methods/data/synthetic/subgroups.ts:72`                             |
| `aifn-methods/data/synthetic#banditProblem`                                | function | 1     | 0    | unmatched | `methods/data/synthetic/bandits.ts:135`                              |
| `aifn-methods/data/synthetic#bishopInverse`                                | function | 1     | 0    | unmatched | `methods/data/synthetic/inverse.ts:90`                               |
| `aifn-methods/data/synthetic#clustersProblem`                              | function | 1     | 0    | unmatched | `methods/data/synthetic/thresholds.ts:69`                            |
| `aifn-methods/data/synthetic#cocktailParty`                                | function | 1     | 0    | unmatched | `methods/data/synthetic/factors.ts:53`                               |
| `aifn-methods/data/synthetic#countFingerprints`                            | function | 1     | 0    | unmatched | `methods/data/synthetic/molecules.ts:315`                            |
| `aifn-methods/data/synthetic#curve1d`                                      | function | 1     | 0    | unmatched | `methods/data/synthetic/curves.ts:94`                                |
| `aifn-methods/data/synthetic#drawLine`                                     | function | 1     | 0    | unmatched | `methods/data/synthetic/thresholds.ts:159`                           |
| `aifn-methods/data/synthetic#feasibilityTask`                              | function | 1     | 0    | unmatched | `methods/data/synthetic/explanation.ts:50`                           |
| `aifn-methods/data/synthetic#fitBoth`                                      | function | 1     | 0    | unmatched | `methods/data/synthetic/thresholds.ts:201`                           |
| `aifn-methods/data/synthetic#flippedMask`                                  | function | 1     | 0    | unmatched | `methods/data/synthetic/modifiers.ts:366`                            |
| `aifn-methods/data/synthetic#focalPlayerStream`                            | function | 1     | 0    | unmatched | `methods/data/synthetic/rating-streams.ts:282`                       |
| `aifn-methods/data/synthetic#gaussianPairLogRatios`                        | function | 1     | 0    | unmatched | `methods/data/synthetic/mutual-information.ts:42`                    |
| `aifn-methods/data/synthetic#interactionTask`                              | function | 1     | 0    | unmatched | `methods/data/synthetic/explanation.ts:212`                          |
| `aifn-methods/data/synthetic#latentFactorModel`                            | function | 1     | 0    | unmatched | `methods/data/synthetic/factors.ts:196`                              |
| `aifn-methods/data/synthetic#latentFactors`                                | function | 1     | 0    | unmatched | `methods/data/synthetic/factors.ts:231`                              |
| `aifn-methods/data/synthetic#learnerResponses`                             | function | 1     | 0    | unmatched | `methods/data/synthetic/learners.ts:181`                             |
| `aifn-methods/data/synthetic#learningCurveSurface`                         | function | 1     | 0    | unmatched | `methods/data/synthetic/learning-curve-surface.ts:135`               |
| `aifn-methods/data/synthetic#motifSeries`                                  | function | 1     | 0    | unmatched | `methods/data/synthetic/sequences.ts:408`                            |
| `aifn-methods/data/synthetic#numberSets`                                   | function | 1     | 0    | unmatched | `methods/data/synthetic/sets.ts:44`                                  |
| `aifn-methods/data/synthetic#odeFailureCase`                               | function | 1     | 0    | unmatched | `methods/data/synthetic/walks.ts:72`                                 |
| `aifn-methods/data/synthetic#parallelWaves`                                | function | 1     | 0    | unmatched | `methods/data/synthetic/parallel-waves.ts:56`                        |
| `aifn-methods/data/synthetic#parityStrings`                                | function | 1     | 0    | unmatched | `methods/data/synthetic/recall.ts:65`                                |
| `aifn-methods/data/synthetic#periodicGaussianField`                        | function | 1     | 0    | unmatched | `methods/data/synthetic/operators.ts:86`                             |
| `aifn-methods/data/synthetic#plantedSubgroups`                             | function | 1     | 0    | unmatched | `methods/data/synthetic/subgroups.ts:106`                            |
| `aifn-methods/data/synthetic#quantileModelOutputs`                         | function | 1     | 0    | unmatched | `methods/data/synthetic/predictions.ts:159`                          |
| `aifn-methods/data/synthetic#randomWalk`                                   | function | 1     | 0    | unmatched | `methods/data/synthetic/sequences.ts:339`                            |
| `aifn-methods/data/synthetic#ratingPopulation`                             | function | 1     | 0    | unmatched | `methods/data/synthetic/rating-streams.ts:170`                       |
| `aifn-methods/data/synthetic#renderShapes`                                 | function | 1     | 0    | unmatched | `methods/data/synthetic/shape-texture.ts:138`                        |
| `aifn-methods/data/synthetic#shapeClouds`                                  | function | 1     | 0    | unmatched | `methods/data/synthetic/sets.ts:124`                                 |
| `aifn-methods/data/synthetic#strokeGlyphs`                                 | function | 1     | 0    | unmatched | `methods/data/synthetic/factors.ts:131`                              |
| `aifn-methods/data/synthetic#thresholdLearningCurve`                       | function | 1     | 0    | unmatched | `methods/data/synthetic/thresholds.ts:253`                           |
| `aifn-methods/data/synthetic#twoViewDataset`                               | function | 1     | 0    | unmatched | `methods/data/synthetic/two-view.ts:92`                              |
| `aifn-methods/data/synthetic#uniformProblem`                               | function | 1     | 0    | unmatched | `methods/data/synthetic/thresholds.ts:105`                           |
| `aifn-methods/data/targets#nonCentredFunnel`                               | function | 1     | 0    | unmatched | `methods/data/targets/targets.ts:275`                                |
| `aifn-methods/dynamics/maps#lyapunovCurve`                                 | function | 1     | 0    | unmatched | `methods/dynamics/maps/index.ts:473`                                 |
| `aifn-methods/dynamics/maps#sineMap`                                       | function | 1     | 0    | unmatched | `methods/dynamics/maps/index.ts:155`                                 |
| `aifn-methods/dynamics/pde#evaluateSeries`                                 | function | 1     | 0    | unmatched | `methods/dynamics/pde/spectral.ts:43`                                |
| `aifn-methods/dynamics/pde#heatSolution`                                   | function | 1     | 0    | unmatched | `methods/dynamics/pde/spectral.ts:82`                                |
| `aifn-methods/generative/autoencoders#m2ClassProbabilities`                | function | 1     | 0    | unmatched | `methods/generative/autoencoders/m2.ts:219`                          |
| `aifn-methods/generative/autoencoders#m2Decode`                            | function | 1     | 0    | unmatched | `methods/generative/autoencoders/m2.ts:237`                          |
| `aifn-methods/generative/autoencoders#m2Model`                             | function | 1     | 0    | unmatched | `methods/generative/autoencoders/m2.ts:108`                          |
| `aifn-methods/generative/autoencoders#semiSupervisedVaeTraining`           | function | 1     | 0    | unmatched | `methods/generative/autoencoders/m2.ts:312`                          |
| `aifn-methods/generative/boltzmann#capacityCurve`                          | function | 1     | 0    | unmatched | `methods/generative/boltzmann/hopfield.ts:257`                       |
| `aifn-methods/generative/boltzmann#corruptPattern`                         | function | 1     | 0    | unmatched | `methods/generative/boltzmann/hopfield.ts:216`                       |
| `aifn-methods/generative/boltzmann#dbnRun`                                 | function | 1     | 0    | unmatched | `methods/generative/boltzmann/dbn.ts:223`                            |
| `aifn-methods/generative/boltzmann#overlaps`                               | function | 1     | 0    | unmatched | `methods/generative/boltzmann/hopfield.ts:232`                       |
| `aifn-methods/generative/diffusion#alphaBarAt`                             | function | 1     | 0    | unmatched | `methods/generative/diffusion/schedules.ts:147`                      |
| `aifn-methods/generative/diffusion#forwardNoise`                           | function | 1     | 0    | unmatched | `methods/generative/diffusion/forward.ts:41`                         |
| `aifn-methods/generative/diffusion#mixtureScore`                           | function | 1     | 0    | unmatched | `methods/generative/diffusion/mixture.ts:213`                        |
| `aifn-methods/generative/diffusion#sampleMixture`                          | function | 1     | 0    | unmatched | `methods/generative/diffusion/mixture.ts:82`                         |
| `aifn-methods/generative/energy#jemRun`                                    | function | 1     | 0    | unmatched | `methods/generative/energy/run.ts:194`                               |
| `aifn-methods/generative/gan#classAndFakeProbabilities`                    | function | 1     | 0    | unmatched | `methods/generative/gan/semi-supervised.ts:134`                      |
| `aifn-methods/generative/gan#complementSampler`                            | function | 1     | 0    | unmatched | `methods/generative/gan/semi-supervised.ts:179`                      |
| `aifn-methods/generative/gan#generatedPoints`                              | function | 1     | 0    | unmatched | `methods/generative/gan/semi-supervised.ts:365`                      |
| `aifn-methods/generative/gan#initSemiSupervisedGan`                        | function | 1     | 0    | unmatched | `methods/generative/gan/semi-supervised.ts:342`                      |
| `aifn-methods/generative/gan#semiSupervisedGan`                            | function | 1     | 0    | unmatched | `methods/generative/gan/semi-supervised.ts:93`                       |
| `aifn-methods/generative/gan#semiSupervisedGanTraining`                    | function | 1     | 0    | unmatched | `methods/generative/gan/semi-supervised.ts:271`                      |
| `aifn-methods/gym#greedyActions`                                           | function | 1     | 0    | unmatched | `methods/gym/mdp.ts:263`                                             |
| `aifn-methods/gym#gymSetup`                                                | function | 1     | 0    | unmatched | `methods/gym/index.ts:175`                                           |
| `aifn-methods/gym/environments#MAZES`                                      | const    | 1     | 0    | unmatched | `methods/gym/environments/gridworlds.ts:406`                         |
| `aifn-methods/inference/learner-models#learnerModelRun`                    | function | 1     | 0    | unmatched | `methods/inference/learner-models/run.ts:231`                        |
| `aifn-methods/inference/mixture-models#clutterEp`                          | function | 1     | 0    | unmatched | `methods/inference/mixture-models/clutter.ts:225`                    |
| `aifn-methods/inference/mixture-models#clutterLogLikelihood`               | function | 1     | 0    | unmatched | `methods/inference/mixture-models/clutter.ts:167`                    |
| `aifn-methods/inference/mixture-models#clutterPosterior`                   | function | 1     | 0    | unmatched | `methods/inference/mixture-models/clutter.ts:266`                    |
| `aifn-methods/inference/mixture-models#sampleClutter`                      | function | 1     | 0    | unmatched | `methods/inference/mixture-models/clutter.ts:196`                    |
| `aifn-methods/inference/rating-models#LICHESS_PROVISIONAL_RD`              | const    | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:890`                    |
| `aifn-methods/inference/rating-models#THURSTONE_BETA`                      | const    | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:24`                     |
| `aifn-methods/inference/rating-models#chessComSpec`                        | function | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:906`                    |
| `aifn-methods/inference/rating-models#chessComStarts`                      | function | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:928`                    |
| `aifn-methods/inference/rating-models#fideSpec`                            | function | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:950`                    |
| `aifn-methods/inference/rating-models#lichessSpec`                         | function | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:875`                    |
| `aifn-methods/inference/rating-models#ratingScaleMap`                      | function | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:829`                    |
| `aifn-methods/inference/rating-models#settlingGames`                       | function | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:747`                    |
| `aifn-methods/inference/rating-models#trueSkillEp`                         | function | 1     | 0    | unmatched | `methods/inference/rating-models/examples.ts:303`                    |
| `aifn-methods/inference/rating-models#trueSkillThroughTime`                | function | 1     | 0    | unmatched | `methods/inference/rating-models/dynamics.ts:568`                    |
| `aifn-methods/learning/gaussian-processes#fitMultiFidelityGp`              | function | 1     | 0    | unmatched | `methods/learning/gaussian-processes/multi-fidelity.ts:135`          |
| `aifn-methods/learning/gaussian-processes#samplePrior`                     | function | 1     | 0    | unmatched | `methods/learning/gaussian-processes/regression.ts:212`              |
| `aifn-methods/learning/generalised/gam#gamlssProblem`                      | function | 1     | 0    | unmatched | `methods/learning/generalised/gam/gamlss.ts:158`                     |
| `aifn-methods/learning/generalised/gam#gamlssTrace`                        | function | 1     | 0    | unmatched | `methods/learning/generalised/gam/gamlss.ts:496`                     |
| `aifn-methods/learning/linear#forwardStagewise`                            | function | 1     | 0    | unmatched | `methods/learning/linear/subsets.ts:389`                             |
| `aifn-methods/learning/linear#ridgeHyperDescentSteps`                      | function | 1     | 0    | unmatched | `methods/learning/linear/hypergradient.ts:219`                       |
| `aifn-methods/learning/linear#ridgeHypergradient`                          | function | 1     | 0    | unmatched | `methods/learning/linear/hypergradient.ts:145`                       |
| `aifn-methods/learning/linear#ruleFitPath`                                 | function | 1     | 0    | unmatched | `methods/learning/linear/ruleFit.ts:114`                             |
| `aifn-methods/learning/linear#ruleImportances`                             | function | 1     | 0    | unmatched | `methods/learning/linear/ruleFit.ts:231`                             |
| `aifn-methods/learning/mixture-density#mixtureDensityRun`                  | function | 1     | 0    | unmatched | `methods/learning/mixture-density/training.ts:174`                   |
| `aifn-methods/learning/preprocessing#fitTransform`                         | function | 1     | 0    | unmatched | `methods/learning/preprocessing/transformer.ts:55`                   |
| `aifn-methods/learning/semi-supervised#clusterClasses`                     | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/cluster-then-label.ts:35`          |
| `aifn-methods/learning/semi-supervised#clusterThenLabelProbabilities`      | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/cluster-then-label.ts:66`          |
| `aifn-methods/learning/semi-supervised#entropyMinimisationPath`            | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/entropy-minimisation.ts:264`       |
| `aifn-methods/learning/semi-supervised#entropyMinimisationProbabilities`   | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/entropy-minimisation.ts:295`       |
| `aifn-methods/learning/semi-supervised#featureDropout`                     | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/student-noise.ts:80`               |
| `aifn-methods/learning/semi-supervised#gaussianNoveltyScores`              | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/gaussian-novelty.ts:54`            |
| `aifn-methods/learning/semi-supervised#inputJitter`                        | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/student-noise.ts:39`               |
| `aifn-methods/learning/semi-supervised#probabilities`                      | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/self-training.ts:139`              |
| `aifn-methods/learning/semi-supervised#rhoPathDistances`                   | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/density-sensitive-distance.ts:117` |
| `aifn-methods/learning/semi-supervised#rhoPathExtend`                      | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/density-sensitive-distance.ts:206` |
| `aifn-methods/learning/semi-supervised#rhoPathNodes`                       | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/density-sensitive-distance.ts:260` |
| `aifn-methods/learning/semi-supervised#rhoQueryNeighbours`                 | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/density-sensitive-distance.ts:183` |
| `aifn-methods/learning/semi-supervised#unlabelledDensity`                  | function | 1     | 0    | unmatched | `methods/learning/semi-supervised/threshold-learning.ts:116`         |
| `aifn-methods/learning/survival#coxPartialLikelihood`                      | function | 1     | 0    | unmatched | `methods/learning/survival/cox.ts:302`                               |
| `aifn-methods/learning/weak-supervision#labelModelReport`                  | function | 1     | 0    | unmatched | `methods/learning/weak-supervision/compare.ts:93`                    |
| `aifn-methods/neural/contrastive#contrastiveTrainingRun`                   | function | 1     | 0    | unmatched | `methods/neural/contrastive/clip.ts:578`                             |
| `aifn-methods/neural/convolutional#capsulePredictions`                     | function | 1     | 0    | unmatched | `methods/neural/convolutional/capsules.ts:46`                        |
| `aifn-methods/neural/convolutional#convolutionAlgorithmCosts`              | function | 1     | 0    | unmatched | `methods/neural/convolutional/fast.ts:330`                           |
| `aifn-methods/neural/convolutional#deformableConvolutionAt`                | function | 1     | 0    | unmatched | `methods/neural/convolutional/deformable.ts:68`                      |
| `aifn-methods/neural/convolutional#dynamicRoutingTrace`                    | function | 1     | 0    | unmatched | `methods/neural/convolutional/capsules.ts:82`                        |
| `aifn-methods/neural/convolutional#winogradRelativeError`                  | function | 1     | 0    | unmatched | `methods/neural/convolutional/fast.ts:237`                           |
| `aifn-methods/neural/full-batch#comparisonModel`                           | function | 1     | 0    | unmatched | `methods/neural/full-batch/comparison.ts:74`                         |
| `aifn-methods/neural/full-batch#fullBatchComparison`                       | function | 1     | 0    | unmatched | `methods/neural/full-batch/comparison.ts:240`                        |
| `aifn-methods/neural/graph#adjacencySequence`                              | function | 1     | 0    | unmatched | `methods/neural/graph/generation.ts:35`                              |
| `aifn-methods/neural/graph#linkPredictionRun`                              | function | 1     | 0    | unmatched | `methods/neural/graph/links.ts:178`                                  |
| `aifn-methods/neural/graph#meanPropagation`                                | function | 1     | 0    | unmatched | `methods/neural/graph/depth.ts:129`                                  |
| `aifn-methods/neural/graph#memoryTrace`                                    | function | 1     | 0    | unmatched | `methods/neural/graph/temporal.ts:96`                                |
| `aifn-methods/neural/graph#sequenceBandwidth`                              | function | 1     | 0    | unmatched | `methods/neural/graph/generation.ts:64`                              |
| `aifn-methods/neural/graph#splitEdges`                                     | function | 1     | 0    | unmatched | `methods/neural/graph/links.ts:107`                                  |
| `aifn-methods/neural/graph#staticReach`                                    | function | 1     | 0    | unmatched | `methods/neural/graph/temporal.ts:202`                               |
| `aifn-methods/neural/graph#timeRespectingReach`                            | function | 1     | 0    | unmatched | `methods/neural/graph/temporal.ts:173`                               |
| `aifn-methods/neural/grokking#ModularMlp`                                  | function | 1     | 0    | unmatched | `methods/neural/grokking/grokking.ts:84`                             |
| `aifn-methods/neural/grokking#grokkingRun`                                 | function | 1     | 0    | unmatched | `methods/neural/grokking/grokking.ts:247`                            |
| `aifn-methods/neural/ode#discInRing`                                       | function | 1     | 0    | unmatched | `methods/neural/ode/run.ts:75`                                       |
| `aifn-methods/neural/ode#reflectionData`                                   | function | 1     | 0    | unmatched | `methods/neural/ode/run.ts:43`                                       |
| `aifn-methods/neural/ode-mixtures#endpointTask`                            | function | 1     | 0    | unmatched | `methods/neural/ode-mixtures/tasks.ts:42`                            |
| `aifn-methods/neural/ode-mixtures#svfmRun`                                 | function | 1     | 0    | unmatched | `methods/neural/ode-mixtures/run.ts:256`                             |
| `aifn-methods/neural/operator#deepOnetLoss`                                | function | 1     | 0    | unmatched | `methods/neural/operator/deeponet.ts:132`                            |
| `aifn-methods/neural/operator#fitDenseOperator`                            | function | 1     | 0    | unmatched | `methods/neural/operator/linear.ts:55`                               |
| `aifn-methods/neural/operator#fitPhysicsInformedMultiplier`                | function | 1     | 0    | unmatched | `methods/neural/operator/linear.ts:195`                              |
| `aifn-methods/neural/operator#meshHeatRollout`                             | function | 1     | 0    | unmatched | `methods/neural/operator/mesh.ts:80`                                 |
| `aifn-methods/neural/pinn#convectionProblem`                               | function | 1     | 0    | unmatched | `methods/neural/pinn/problems.ts:306`                                |
| `aifn-methods/neural/pinn#layerProblem`                                    | function | 1     | 0    | unmatched | `methods/neural/pinn/problems.ts:275`                                |
| `aifn-methods/neural/post-training#coverageProblem`                        | function | 1     | 0    | unmatched | `methods/neural/post-training/grpo.ts:419`                           |
| `aifn-methods/neural/post-training#grpoTrace`                              | function | 1     | 0    | unmatched | `methods/neural/post-training/grpo.ts:270`                           |
| `aifn-methods/neural/post-training#passAtKExact`                           | function | 1     | 0    | unmatched | `methods/neural/post-training/advantages.ts:242`                     |
| `aifn-methods/neural/post-training#postTrainingTrace`                      | function | 1     | 0    | unmatched | `methods/neural/post-training/toy.ts:180`                            |
| `aifn-methods/neural/post-training#solveProbabilities`                     | function | 1     | 0    | unmatched | `methods/neural/post-training/grpo.ts:450`                           |
| `aifn-methods/neural/quantisation#LLAMA_3_1_70B`                           | const    | 1     | 0    | unmatched | `methods/neural/quantisation/cost.ts:278`                            |
| `aifn-methods/neural/quantisation#LLAMA_3_1_8B`                            | const    | 1     | 0    | unmatched | `methods/neural/quantisation/cost.ts:261`                            |
| `aifn-methods/neural/quantisation#decodeThroughput`                        | function | 1     | 0    | unmatched | `methods/neural/quantisation/cost.ts:134`                            |
| `aifn-methods/neural/quantisation#fineTuningCompute`                       | function | 1     | 0    | unmatched | `methods/neural/quantisation/cost.ts:453`                            |
| `aifn-methods/neural/reward-models#bestOfNCurve`                           | function | 1     | 0    | unmatched | `methods/neural/reward-models/preferences.ts:247`                    |
| `aifn-methods/neural/reward-models#bestOfNKl`                              | function | 1     | 0    | unmatched | `methods/neural/reward-models/preferences.ts:183`                    |
| `aifn-methods/neural/reward-models#bestOfNWeights`                         | function | 1     | 0    | unmatched | `methods/neural/reward-models/preferences.ts:214`                    |
| `aifn-methods/neural/reward-models#syntheticPreferences`                   | function | 1     | 0    | unmatched | `methods/neural/reward-models/preferences.ts:72`                     |
| `aifn-methods/neural/semi-supervised#sharpen`                              | function | 1     | 0    | unmatched | `methods/neural/semi-supervised/consistency.ts:244`                  |
| `aifn-methods/neural/sequence#angularRotations`                            | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:295`                         |
| `aifn-methods/neural/sequence#associativeRecallByDimension`                | function | 1     | 0    | unmatched | `methods/neural/sequence/recall.ts:129`                              |
| `aifn-methods/neural/sequence#convolutionMultiplies`                       | function | 1     | 0    | unmatched | `methods/neural/sequence/cost.ts:25`                                 |
| `aifn-methods/neural/sequence#linformerMatrix`                             | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:221`                         |
| `aifn-methods/neural/sequence#lowRankAt`                                   | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:174`                         |
| `aifn-methods/neural/sequence#lowRankSeries`                               | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:156`                         |
| `aifn-methods/neural/sequence#lshAttentionMatrix`                          | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:316`                         |
| `aifn-methods/neural/sequence#nystromMatrix`                               | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:244`                         |
| `aifn-methods/neural/sequence#onesUntilZero`                               | function | 1     | 0    | unmatched | `methods/neural/sequence/parity.ts:90`                               |
| `aifn-methods/neural/sequence#parityAccuracyByLength`                      | function | 1     | 0    | unmatched | `methods/neural/sequence/parity.ts:128`                              |
| `aifn-methods/neural/sequence#parityStates`                                | function | 1     | 0    | unmatched | `methods/neural/sequence/parity.ts:50`                               |
| `aifn-methods/neural/sequence#rowTotalVariation`                           | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:348`                         |
| `aifn-methods/neural/sequence#segmentMeans`                                | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:196`                         |
| `aifn-methods/neural/sequence#topicTokens`                                 | function | 1     | 0    | unmatched | `methods/neural/sequence/approximate.ts:117`                         |
| `aifn-methods/optimisation/bayesian#multiPointExpectedImprovement`         | function | 1     | 0    | unmatched | `methods/optimisation/bayesian/batch.ts:105`                         |
| `aifn-methods/optimisation/multi-fidelity#bohbSteps`                       | function | 1     | 0    | unmatched | `methods/optimisation/multi-fidelity/bohb.ts:129`                    |
| `aifn-methods/optimisation/multi-fidelity#fitPowerLaw`                     | function | 1     | 0    | unmatched | `methods/optimisation/multi-fidelity/extrapolate.ts:76`              |
| `aifn-methods/optimisation/multi-objective#chebyshevScalarisation`         | function | 1     | 0    | unmatched | `methods/optimisation/multi-objective/scalarisation.ts:66`           |
| `aifn-methods/optimisation/multi-objective#expectedHypervolumeImprovement` | function | 1     | 0    | unmatched | `methods/optimisation/multi-objective/hypervolume.ts:270`            |
| `aifn-methods/optimisation/multi-objective#simplexLattice`                 | function | 1     | 0    | unmatched | `methods/optimisation/multi-objective/scalarisation.ts:117`          |
| `aifn-methods/optimisation/population#pbtToyProblem`                       | function | 1     | 0    | unmatched | `methods/optimisation/population/pbt.ts:60`                          |
| `aifn-methods/optimisation/population#populationBasedTrainingSteps`        | function | 1     | 0    | unmatched | `methods/optimisation/population/pbt.ts:172`                         |
| `aifn-methods/optimisation/portfolio#averageRank`                          | function | 1     | 0    | unmatched | `methods/optimisation/portfolio/portfolio.ts:291`                    |
| `aifn-methods/optimisation/portfolio#greedyPortfolio`                      | function | 1     | 0    | unmatched | `methods/optimisation/portfolio/portfolio.ts:140`                    |
| `aifn-methods/optimisation/portfolio#leaveOneTaskOut`                      | function | 1     | 0    | unmatched | `methods/optimisation/portfolio/portfolio.ts:247`                    |
| `aifn-methods/optimisation/portfolio#portfolioLoss`                        | function | 1     | 0    | unmatched | `methods/optimisation/portfolio/portfolio.ts:215`                    |
| `aifn-methods/retrieval/recommenders#recommenderRun`                       | function | 1     | 0    | unmatched | `methods/retrieval/recommenders/run.ts:195`                          |
| `aifn-methods/theory/adaptive-holdout#expectedGaussianMaximum`             | function | 1     | 0    | unmatched | `methods/theory/adaptive-holdout/maximum.ts:29`                      |
| `aifn-methods/theory/double-descent#doubleDescent`                         | function | 1     | 0    | unmatched | `methods/theory/double-descent/double-descent.ts:186`                |
| `aifn-methods/theory/shrinkage#jamesSteinExactRisk`                        | function | 1     | 0    | unmatched | `methods/theory/shrinkage/shrinkage.ts:165`                          |
| `aifn-methods/timeseries#constantVelocityModel`                            | function | 1     | 0    | unmatched | `methods/timeseries/tracking.ts:51`                                  |
| `aifn-methods/unsupervised/clustering#BORDER`                              | const    | 1     | 0    | unmatched | `methods/unsupervised/clustering/density.ts:28`                      |
| `aifn-methods/unsupervised/clustering#classPosterior`                      | function | 1     | 0    | unmatched | `methods/unsupervised/clustering/semi-supervised-mixture.ts:185`     |
| `aifn-methods/unsupervised/clustering#constrainedKMeansSteps`              | function | 1     | 0    | unmatched | `methods/unsupervised/clustering/constrained.ts:189`                 |
| `aifn-methods/unsupervised/clustering#constraintClosure`                   | function | 1     | 0    | unmatched | `methods/unsupervised/clustering/constrained.ts:77`                  |
| `aifn-methods/unsupervised/clustering#relevantComponentMetric`             | function | 1     | 0    | unmatched | `methods/unsupervised/clustering/constrained.ts:385`                 |
| `aifn-methods/unsupervised/embedding/linear#latentGaussianSteps`           | function | 1     | 0    | unmatched | `methods/unsupervised/embedding/linear/latent.ts:143`                |
| `aifn-methods/unsupervised/embedding/neighbour#fuzzyGraph`                 | function | 1     | 0    | unmatched | `methods/unsupervised/embedding/neighbour/umap.ts:122`               |
| `aifn-methods/vision/detection#anchorFreeTargets`                          | function | 1     | 0    | unmatched | `methods/vision/detection/anchors.ts:230`                            |
| `aifn-methods/vision/detection#pyramidLevel`                               | function | 1     | 0    | unmatched | `methods/vision/detection/anchors.ts:271`                            |
| `aifn-render/controls#CodeEditor`                                          | function | 1     | 0    | unmatched | `render/controls/code/CodeEditor.tsx:268`                            |
| `aifn-render/gym#GymTrainer`                                               | function | 1     | 0    | unmatched | `render/gym/GymTrainer.tsx:77`                                       |
| `aifn-render/gym#trainingRun`                                              | function | 1     | 0    | unmatched | `render/gym/trainingRun.ts:22`                                       |

## Used types with no omai counterpart

| symbol | kind | notes | sole | status | aifn source |
| ------ | ---- | ----- | ---- | ------ | ----------- |
