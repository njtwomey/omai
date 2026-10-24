import type { DatasetGenerator, DatasetRef } from '../data/index.ts'
import { instrumentalVariables, linearScm, treatmentEffects } from './causal.ts'
import { changepointSeries } from './changepoints.ts'
import { blobs, checkerboard, circles, gaussianQuantiles, hastie10_2, moons, spirals, xor } from './classification.ts'
import { anisotropicBlobs, rings, unequalVarianceBlobs } from './clustering.ts'
import { eloGames, roundRobin } from './comparisons.ts'
import { driftingTopics, naiveBayesCorpus, nearDuplicateCorpus, tinyShakespeare, topicCorpus } from './corpus.ts'
import { densitySamples } from './densities.ts'
import { hawkesProcess, inhomogeneousPoisson, poissonProcess, spatialPoisson } from './events.ts'
import { abTest } from './experiments.ts'
import { contextualSbm, graphFamilies, kinshipTriples } from './graphs.ts'
import { barsAndStripes, digitGlyphs, fashionMnistTest, geometricScenes, shapes } from './images.ts'
import { makeClassification, makeMultilabelClassification } from './make-classification.ts'
import { sCurve, severedSphere, swissRoll, torus } from './manifolds.ts'
import { antiderivativeOperator, burgersOperator, heatOperator } from './operators.ts'
import { blobsWithOutliers, linearOutliers, moonsWithOutliers } from './outliers.ts'
import { miscalibratedPredictions, survivalData } from './predictions.ts'
import { clickLogs, implicitFeedback, lowRankRatings, movielensSmall, zipfianCatalogue } from './ranking.ts'
import { friedman1, friedman2, friedman3, makeRegression, sparseUncorrelated } from './regression.ts'
import {
  additiveModel,
  dopplerCurve,
  heteroscedastic,
  piecewiseLinear,
  quantileCurves,
  sincCurve,
  sineCurve,
  stepCurve,
} from './regression-curves.ts'
import { californiaHousing, growthCurves } from './regression-real.ts'
import {
  additionTask,
  associativeRecall,
  copyTask,
  dyckTask,
  modularArithmetic,
  parityTask,
  reverseTask,
  sortTask,
} from './sequence-tasks.ts'
import { arma, electrocardiogram, seasonalSeries, sinusoids } from './timeseries.ts'
import { loggedBandit, offlineMdp } from './trajectories.ts'
import { crowdLabels, labellingFunctions, labelProportionBags, positiveUnlabelled } from './weak-labels.ts'

/**
 * Every dataset generator of this area by name, the catalogue that `makeFromRecipe` looks generators up in and that
 * documentation and controls are built from.
 *
 * @example
 * ```ts
 * import { makeFromRecipe } from 'whitebox-engine/data'
 * import { datasetGenerators } from 'whitebox-engine/datasets'
 * makeFromRecipe({ generator: 'moons', knobs: { n: 10 } }, { generators: datasetGenerators }).data.kind // => 'classification'
 * ```
 */
export const datasetGenerators: Readonly<Record<string, DatasetGenerator>> = {
  abTest,
  additionTask,
  additiveModel,
  anisotropicBlobs,
  antiderivativeOperator,
  arma,
  associativeRecall,
  barsAndStripes,
  blobs,
  blobsWithOutliers,
  burgersOperator,
  changepointSeries,
  checkerboard,
  circles,
  clickLogs,
  contextualSbm,
  copyTask,
  crowdLabels,
  densitySamples,
  digitGlyphs,
  dopplerCurve,
  driftingTopics,
  dyckTask,
  eloGames,
  friedman1,
  friedman2,
  friedman3,
  gaussianQuantiles,
  geometricScenes,
  graphFamilies,
  growthCurves,
  hastie10_2,
  hawkesProcess,
  heatOperator,
  heteroscedastic,
  implicitFeedback,
  inhomogeneousPoisson,
  instrumentalVariables,
  kinshipTriples,
  labelProportionBags,
  labellingFunctions,
  linearOutliers,
  linearScm,
  loggedBandit,
  lowRankRatings,
  makeClassification,
  makeMultilabelClassification,
  makeRegression,
  miscalibratedPredictions,
  modularArithmetic,
  moons,
  moonsWithOutliers,
  naiveBayesCorpus,
  nearDuplicateCorpus,
  offlineMdp,
  parityTask,
  piecewiseLinear,
  poissonProcess,
  positiveUnlabelled,
  quantileCurves,
  reverseTask,
  rings,
  roundRobin,
  sCurve,
  seasonalSeries,
  severedSphere,
  shapes,
  sincCurve,
  sineCurve,
  sinusoids,
  sortTask,
  sparseUncorrelated,
  spatialPoisson,
  spirals,
  stepCurve,
  survivalData,
  swissRoll,
  topicCorpus,
  torus,
  treatmentEffects,
  unequalVarianceBlobs,
  xor,
  zipfianCatalogue,
} as Readonly<Record<string, DatasetGenerator>>

/**
 * The remote datasets of this area by name (pinned by SHA-256, fetched and parsed by `load`).
 *
 * @example
 * ```ts
 * import { remoteDatasets } from 'whitebox-engine/datasets'
 * Object.keys(remoteDatasets).length // => 5
 * ```
 */
export const remoteDatasets: Readonly<Record<string, DatasetRef>> = {
  californiaHousing,
  electrocardiogram,
  fashionMnistTest,
  movielensSmall,
  tinyShakespeare,
} as Readonly<Record<string, DatasetRef>>
