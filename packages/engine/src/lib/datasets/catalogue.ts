/**
 * The catalogue of every dataset generator and remote dataset in this domain, by name, and the tagged registry built
 * from it (`datasetRegistry`: every generator, vendored real dataset and remote asset with its kind, modality, real
 * flag, size, licence and source beside its title). It is where `makeFromRecipe` looks generators up, where `load`
 * finds pinned remote assets, and what documentation and dataset selectors are built from.
 */

import { MdmlError, registry } from '../base/index.ts'
import type { DatasetGenerator, DatasetRef } from '../data/index.ts'
import { spokenDigits } from './audio.ts'
import { captionedShapes } from './captions.ts'
import { discontinuityDesign, instrumentalVariables, linearScm, panelEffects, treatmentEffects } from './causal.ts'
import { changepointSeries } from './changepoints.ts'
import { blobs, checkerboard, circles, gaussianQuantiles, hastie10_2, moons, spirals, xor } from './classification.ts'
import {
  breastCancer,
  digitImages,
  digits,
  iris,
  penguins,
  penguinsClassification,
  titanic,
  wine,
} from './classification-real.ts'
import { anisotropicBlobs, oldFaithful, rings, unequalVarianceBlobs } from './clustering.ts'
import { eloGames, roundRobin } from './comparisons.ts'
import {
  driftingTopics,
  naiveBayesCorpus,
  nearDuplicateCorpus,
  nurseryRhymes,
  taggedCorpus,
  tinyShakespeare,
  topicCorpus,
  typoPairs,
  workedExampleCorpus,
} from './corpus.ts'
import { densitySamples } from './densities.ts'
import { hawkesProcess, inhomogeneousPoisson, poissonProcess, spatialPoisson } from './events.ts'
import { abTest, carryoverExperiment, networkExperiment } from './experiments.ts'
import { biasedCredit, germanCredit, titanicSurvival } from './fairness.ts'
import { glmCurve } from './glm-curves.ts'
import { contextualSbm, graphFamilies, karateClub, kinshipTriples, lesMiserables, smallMolecules } from './graphs.ts'
import { dishonestCasino, regimeSwitching } from './hidden-markov.ts'
import { barsAndStripes, digitGlyphs, fashionMnistTest, geometricScenes, shapes } from './images.ts'
import { mixedSources, toySuperposition, twoViews } from './latent.ts'
import { makeClassification, makeMultilabelClassification } from './make-classification.ts'
import { sCurve, severedSphere, swissRoll, torus } from './manifolds.ts'
import { antiderivativeOperator, burgersOperator, heatOperator } from './operators.ts'
import { blobsWithOutliers, linearOutliers, moonsWithOutliers } from './outliers.ts'
import { miscalibratedPredictions, survivalData } from './predictions.ts'
import { clickLogs, implicitFeedback, lowRankRatings, movielensSmall, zipfianCatalogue } from './ranking.ts'
import {
  type DatasetEntry,
  type DatasetModality,
  type DatasetRegistry,
  generatorEntry,
  type RemoteEntryTags,
  remoteEntry,
  vendoredEntry,
} from './registry.ts'
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
import {
  anscombe,
  californiaHousing,
  diabetes,
  growthCurves,
  linnerud,
  mtcars,
  mtcarsRegression,
} from './regression-real.ts'
import { plantedRuleList } from './rule-lists.ts'
import {
  additionTask,
  associativeRecall,
  copyTask,
  dyckTask,
  inContextRegression,
  modularArithmetic,
  parityTask,
  reverseTask,
  sortTask,
} from './sequence-tasks.ts'
import {
  airPassengers,
  arma,
  coalMiningCounts,
  coalMiningDisasters,
  coupledSeries,
  electrocardiogram,
  garchReturns,
  plantedSeries,
  seasonalSeries,
  sinusoids,
  sunspots,
} from './timeseries.ts'
import { loggedBandit, offlineMdp } from './trajectories.ts'
import { plantedTransactions } from './transactions.ts'
import {
  clusterAssumption,
  crowdLabels,
  labellingFunctions,
  labelProportionBags,
  multipleInstanceBags,
  positiveUnlabelled,
} from './weak-labels.ts'

/**
 * Every dataset generator of this area by name, the catalogue that `makeFromRecipe` looks generators up in and that
 * documentation and controls are built from.
 *
 * @example
 * ```ts
 * import { makeFromRecipe } from 'mdml-engine/data'
 * import { datasetGenerators } from 'mdml-engine/datasets'
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
  biasedCredit,
  blobs,
  blobsWithOutliers,
  burgersOperator,
  captionedShapes,
  carryoverExperiment,
  changepointSeries,
  checkerboard,
  circles,
  clickLogs,
  clusterAssumption,
  contextualSbm,
  copyTask,
  coupledSeries,
  crowdLabels,
  densitySamples,
  digitGlyphs,
  discontinuityDesign,
  dishonestCasino,
  dopplerCurve,
  driftingTopics,
  dyckTask,
  eloGames,
  friedman1,
  friedman2,
  friedman3,
  garchReturns,
  gaussianQuantiles,
  geometricScenes,
  glmCurve,
  graphFamilies,
  growthCurves,
  hastie10_2,
  hawkesProcess,
  heatOperator,
  heteroscedastic,
  implicitFeedback,
  inContextRegression,
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
  mixedSources,
  modularArithmetic,
  moons,
  moonsWithOutliers,
  multipleInstanceBags,
  naiveBayesCorpus,
  nearDuplicateCorpus,
  networkExperiment,
  offlineMdp,
  panelEffects,
  parityTask,
  piecewiseLinear,
  plantedRuleList,
  plantedSeries,
  plantedTransactions,
  poissonProcess,
  positiveUnlabelled,
  quantileCurves,
  regimeSwitching,
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
  spokenDigits,
  stepCurve,
  survivalData,
  swissRoll,
  taggedCorpus,
  topicCorpus,
  torus,
  toySuperposition,
  treatmentEffects,
  twoViews,
  typoPairs,
  unequalVarianceBlobs,
  xor,
  zipfianCatalogue,
} as Readonly<Record<string, DatasetGenerator>>

/**
 * The remote datasets of this area by name (pinned by SHA-256, fetched and parsed by `load`).
 *
 * @example
 * ```ts
 * import { remoteDatasets } from 'mdml-engine/datasets'
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

// ----------------------------------------------------------------------------------------------- the tagged registry

/** A generator of the catalogue by name, with its title; the modality is the group's. */
const g = (modality: DatasetModality, items: readonly (readonly [name: string, title: string])[]): DatasetEntry[] =>
  items.map(([name, title]) => {
    const generator = datasetGenerators[name]
    if (generator === undefined) throw new MdmlError('value', `datasetRegistry: no generator '${name}'`)
    return generatorEntry(generator, { title, modality })
  })

/** A remote asset of the catalogue by name. */
const r = (name: string, tags: RemoteEntryTags): DatasetEntry => {
  const ref = remoteDatasets[name]
  if (ref === undefined) throw new MdmlError('value', `datasetRegistry: no remote dataset '${name}'`)
  return remoteEntry(ref, tags)
}

const SKLEARN = 'BSD-3-Clause (scikit-learn); the data are from the UCI repository (CC BY 4.0)'
const R_DATA = 'Public domain data; table from R (GPL-2 | GPL-3)'
const TITANIC = 'Public domain (British Board of Trade, 1912); table from R (GPL-2 | GPL-3)'
const COAL = 'Public domain data; dataset from the R package boot (GPL-2)'
const NETWORKX = 'Public domain data (NetworkX, BSD-3-Clause)'

/**
 * The library's dataset registry: every generator of the catalogue, every vendored real dataset and every remote
 * asset, tagged (see `registry.ts`). Extend it with `registerDatasets`; query it with `whereDatasets`.
 *
 * @concept datasets
 * @example
 * ```ts
 * import { lookup } from 'mdml-engine/base'
 * import { datasetRegistry } from 'mdml-engine/datasets'
 * datasetRegistry.entries.length > 100 // => true
 * lookup(datasetRegistry, 'iris').size // => 150
 * ```
 */
export const datasetRegistry: DatasetRegistry = registry<DatasetEntry>([
  ...g('tabular', [
    ['abTest', 'A/B test with known effects'],
    ['additiveModel', 'Additive model (GAM)'],
    ['anisotropicBlobs', 'Anisotropic blobs'],
    ['antiderivativeOperator', 'Antiderivative operator'],
    ['biasedCredit', 'Credit with a protected group'],
    ['blobs', 'Blobs'],
    ['blobsWithOutliers', 'Blobs with outliers'],
    ['burgersOperator', 'Burgers operator'],
    ['checkerboard', 'Checkerboard'],
    ['circles', 'Concentric circles'],
    ['clusterAssumption', 'Clusters and classes (semi-supervised)'],
    ['densitySamples', 'Samples from a known density'],
    ['dopplerCurve', 'Doppler curve'],
    ['friedman1', 'Friedman 1'],
    ['friedman2', 'Friedman 2'],
    ['friedman3', 'Friedman 3'],
    ['gaussianQuantiles', 'Gaussian quantiles'],
    ['glmCurve', 'GLM curve (counts and positive responses)'],
    ['growthCurves', 'Growth curves (CDC charts)'],
    ['hastie10_2', 'Hastie 10.2'],
    ['heatOperator', 'Heat operator'],
    ['heteroscedastic', 'Heteroscedastic noise'],
    ['instrumentalVariables', 'Instrumental variables'],
    ['labelProportionBags', 'Label-proportion bags'],
    ['labellingFunctions', 'Labelling functions'],
    ['linearOutliers', 'Line with outliers'],
    ['linearScm', 'Linear structural causal model'],
    ['makeClassification', 'Synthetic classification (make_classification)'],
    ['makeMultilabelClassification', 'Synthetic multilabel (make_multilabel_classification)'],
    ['makeRegression', 'Synthetic regression (make_regression)'],
    ['miscalibratedPredictions', 'Miscalibrated predictions'],
    ['moons', 'Two moons'],
    ['moonsWithOutliers', 'Two moons with outliers'],
    ['multipleInstanceBags', 'Multiple-instance bags'],
    ['piecewiseLinear', 'Piecewise linear'],
    ['plantedRuleList', 'Planted rule list'],
    ['positiveUnlabelled', 'Positive-unlabelled'],
    ['quantileCurves', 'Curves with known quantiles'],
    ['rings', 'Rings'],
    ['sCurve', 'S curve'],
    ['severedSphere', 'Severed sphere'],
    ['sincCurve', 'Sinc curve'],
    ['sineCurve', 'Sine curve'],
    ['sparseUncorrelated', 'Sparse uncorrelated'],
    ['spirals', 'Spirals'],
    ['stepCurve', 'Step'],
    ['survivalData', 'Survival data with known hazards'],
    ['swissRoll', 'Swiss roll'],
    ['torus', 'Torus'],
    ['treatmentEffects', 'Treatment effects'],
    ['unequalVarianceBlobs', 'Blobs of unequal variance'],
    ['xor', 'XOR'],
    ['panelEffects', 'Panel with staggered adoption'],
    ['discontinuityDesign', 'Regression discontinuity design'],
    ['networkExperiment', 'Network experiment with spillover'],
    ['carryoverExperiment', 'Chained experiments with carryover'],
    ['mixedSources', 'Mixed independent sources'],
    ['twoViews', 'Two views of shared factors'],
    ['toySuperposition', 'Toy model of superposition'],
  ]),
  ...g('image', [
    ['barsAndStripes', 'Bars and stripes'],
    ['digitGlyphs', 'Digit glyphs'],
    ['geometricScenes', 'Geometric scenes'],
    ['shapes', 'Shapes'],
    ['captionedShapes', 'Captioned shapes'],
  ]),
  ...g('timeSeries', [
    ['arma', 'ARMA process'],
    ['changepointSeries', 'Changepoints'],
    ['regimeSwitching', 'Two Gaussian regimes'],
    ['seasonalSeries', 'Seasonal series'],
    ['sinusoids', 'Sinusoids'],
    ['plantedSeries', 'Planted motif, discord and regime'],
    ['garchReturns', 'GARCH returns'],
    ['coupledSeries', 'Coupled series'],
  ]),
  ...g('text', [
    ['driftingTopics', 'Drifting topics'],
    ['naiveBayesCorpus', 'Labelled corpus (naive Bayes)'],
    ['nearDuplicateCorpus', 'Near-duplicate documents'],
    ['topicCorpus', 'Topic corpus (LDA)'],
    ['taggedCorpus', 'Tagged corpus (tag HMM)'],
    ['typoPairs', 'Typos and intended words'],
    ['plantedTransactions', 'Baskets with planted itemsets'],
  ]),
  ...g('sequence', [
    ['additionTask', 'Addition'],
    ['associativeRecall', 'Associative recall'],
    ['copyTask', 'Copy'],
    ['dishonestCasino', 'The dishonest casino'],
    ['dyckTask', 'Dyck brackets'],
    ['modularArithmetic', 'Modular arithmetic'],
    ['parityTask', 'Parity'],
    ['reverseTask', 'Reverse'],
    ['sortTask', 'Sort'],
    ['inContextRegression', 'In-context linear regression'],
  ]),
  ...g('events', [
    ['hawkesProcess', 'Hawkes process'],
    ['inhomogeneousPoisson', 'Inhomogeneous Poisson process'],
    ['poissonProcess', 'Poisson process'],
    ['spatialPoisson', 'Spatial Poisson process'],
  ]),
  ...g('graph', [
    ['contextualSbm', 'Contextual stochastic block model'],
    ['graphFamilies', 'Random graph families'],
    ['kinshipTriples', 'Kinship triples'],
  ]),
  ...g('interactions', [
    ['clickLogs', 'Click logs'],
    ['crowdLabels', 'Crowd labels'],
    ['eloGames', 'Games between rated players'],
    ['implicitFeedback', 'Implicit feedback'],
    ['lowRankRatings', 'Low-rank ratings'],
    ['roundRobin', 'Round-robin tournament'],
    ['zipfianCatalogue', 'Zipfian catalogue'],
  ]),
  ...g('trajectories', [
    ['loggedBandit', 'Logged bandit'],
    ['offlineMdp', 'Offline MDP episodes'],
  ]),
  ...g('audio', [['spokenDigits', 'Spoken digits (synthetic)']]),
  // Vendored real datasets, with the provenance their metadata records.
  vendoredEntry('iris', iris, {
    title: 'Iris',
    description: 'Sepal and petal measurements of 150 irises of three species (Fisher, 1936).',
    kind: 'classification',
    modality: 'tabular',
    size: 150,
    licence: SKLEARN,
    source: 'scikit-learn load_iris',
  }),
  vendoredEntry('wine', wine, {
    title: 'Wine',
    description: 'Chemical analysis of 178 wines from three cultivars in the same region of Italy.',
    kind: 'classification',
    modality: 'tabular',
    size: 178,
    licence: SKLEARN,
    source: 'scikit-learn load_wine',
  }),
  vendoredEntry('breastCancer', breastCancer, {
    title: 'Breast cancer (Wisconsin)',
    description: 'Features of cell nuclei in 569 breast masses, malignant or benign (Street et al., 1993).',
    kind: 'classification',
    modality: 'tabular',
    size: 569,
    licence: SKLEARN,
    source: 'scikit-learn load_breast_cancer',
  }),
  vendoredEntry('digits', digits, {
    title: 'Digits (8 by 8)',
    description: '1797 handwritten digits as 64 pixel features with grey levels 0 to 16.',
    kind: 'classification',
    modality: 'tabular',
    size: 1797,
    licence: SKLEARN,
    source: 'scikit-learn load_digits',
  }),
  vendoredEntry('digitImages', digitImages, {
    title: 'Digit images (8 by 8)',
    description: '1797 handwritten digits as 8 by 8 images scaled to [0, 1].',
    kind: 'image',
    modality: 'image',
    size: 1797,
    licence: SKLEARN,
    source: 'scikit-learn load_digits',
  }),
  vendoredEntry('penguins', penguins, {
    title: 'Palmer penguins (table)',
    description: 'Species, island, measurements, sex and year of 344 penguins, with missing values.',
    kind: 'table',
    modality: 'tabular',
    size: 344,
    licence: 'CC0-1.0',
    source: 'palmerpenguins penguins.csv',
  }),
  vendoredEntry('penguinsClassification', penguinsClassification, {
    title: 'Palmer penguins',
    description: 'The four measurements of the 342 penguins with complete measurements, labelled by species.',
    kind: 'classification',
    modality: 'tabular',
    size: 342,
    licence: 'CC0-1.0',
    source: 'palmerpenguins penguins.csv',
  }),
  vendoredEntry('titanic', titanic, {
    title: 'Titanic (table)',
    description: 'Class, sex, age and survival of the 2201 people aboard the Titanic.',
    kind: 'table',
    modality: 'tabular',
    size: 2201,
    licence: TITANIC,
    source: 'R datasets::Titanic via Rdatasets',
  }),
  vendoredEntry('titanicSurvival', titanicSurvival, {
    title: 'Titanic survival',
    description:
      'Survival of the 2201 people aboard the Titanic from class, age and sex, sex as a protected attribute.',
    kind: 'classification',
    modality: 'tabular',
    size: 2201,
    licence: TITANIC,
    source: 'R datasets::Titanic via Rdatasets',
  }),
  vendoredEntry('germanCredit', germanCredit, {
    title: 'German credit (Statlog)',
    description: '1000 loan applicants with twelve attributes, each a good or bad credit risk (Hofmann, 1994).',
    kind: 'classification',
    modality: 'tabular',
    size: 1000,
    licence: 'CC BY 4.0 (UCI Machine Learning Repository)',
    source: 'UCI Statlog (German credit), german.data',
  }),
  vendoredEntry('diabetes', () => diabetes(), {
    title: 'Diabetes',
    description: 'Disease progression one year after baseline for 442 diabetes patients (Efron et al., 2004).',
    kind: 'regression',
    modality: 'tabular',
    size: 442,
    licence: 'BSD-3-Clause (scikit-learn)',
    source: 'scikit-learn load_diabetes',
  }),
  ...([1, 2, 3, 4] as const).map((set) =>
    vendoredEntry(`anscombe${set}`, () => anscombe(set), {
      title: `Anscombe's quartet, set ${set}`,
      description: `Eleven points of Anscombe's quartet, set ${set}: the same summary statistics, a different shape.`,
      kind: 'regression',
      modality: 'tabular',
      size: 11,
      licence: R_DATA,
      source: 'R datasets::anscombe via Rdatasets',
    }),
  ),
  vendoredEntry('linnerud', linnerud, {
    title: 'Linnerud',
    description: 'Exercise and physiological measurements of 20 men (Tenenhaus, 1998), three targets.',
    kind: 'regression',
    modality: 'tabular',
    size: 20,
    licence: 'BSD-3-Clause (scikit-learn)',
    source: 'scikit-learn load_linnerud',
  }),
  vendoredEntry('mtcars', mtcars, {
    title: 'Motor Trend cars (table)',
    description: 'Fuel consumption and ten aspects of design and performance of 32 cars (Motor Trend, 1974).',
    kind: 'table',
    modality: 'tabular',
    size: 32,
    licence: R_DATA,
    source: 'R datasets::mtcars via Rdatasets',
  }),
  vendoredEntry('mtcarsRegression', mtcarsRegression, {
    title: 'Motor Trend cars',
    description: 'Miles per gallon of 32 cars from the other ten measurements (Motor Trend, 1974).',
    kind: 'regression',
    modality: 'tabular',
    size: 32,
    licence: R_DATA,
    source: 'R datasets::mtcars via Rdatasets',
  }),
  vendoredEntry('oldFaithful', oldFaithful, {
    title: 'Old Faithful',
    description: 'Eruption durations and waiting times of 272 Old Faithful eruptions, a two-cluster classic.',
    kind: 'classification',
    modality: 'tabular',
    size: 272,
    licence: R_DATA,
    source: 'R datasets::faithful via Rdatasets',
  }),
  vendoredEntry('airPassengers', airPassengers, {
    title: 'Airline passengers',
    description: 'Monthly international airline passengers in thousands, 1949 to 1960 (Box and Jenkins).',
    kind: 'timeseries',
    modality: 'timeSeries',
    size: 144,
    licence: 'Public domain data; series from R (GPL-2 | GPL-3)',
    source: 'R datasets::AirPassengers via Rdatasets',
  }),
  vendoredEntry('sunspots', sunspots, {
    title: 'Sunspots',
    description: 'Yearly mean sunspot numbers, 1700 to 2008.',
    kind: 'timeseries',
    modality: 'timeSeries',
    size: 309,
    licence: 'Public domain (NOAA)',
    source: 'statsmodels datasets.sunspots',
  }),
  vendoredEntry('coalMiningDisasters', coalMiningDisasters, {
    title: 'Coal-mining disasters (events)',
    description: 'Dates of the 191 explosions that killed ten or more British coal miners, 1851 to 1962.',
    kind: 'events',
    modality: 'events',
    size: 191,
    licence: COAL,
    source: 'boot::coal via Rdatasets',
  }),
  vendoredEntry('coalMiningCounts', coalMiningCounts, {
    title: 'Coal-mining disasters (yearly counts)',
    description: 'British coal-mining disasters as yearly counts, 1851 to 1962, the classic changepoint series.',
    kind: 'timeseries',
    modality: 'timeSeries',
    size: 112,
    licence: COAL,
    source: 'boot::coal via Rdatasets',
  }),
  vendoredEntry('karateClub', karateClub, {
    title: "Zachary's karate club",
    description: 'The friendships of 34 members of a karate club that split in two (Zachary, 1977).',
    kind: 'graph',
    modality: 'graph',
    size: 34,
    licence: NETWORKX,
    source: 'NetworkX karate_club_graph',
  }),
  vendoredEntry('lesMiserables', lesMiserables, {
    title: 'Les Misérables',
    description: 'Co-appearances of the 77 characters of Les Misérables (Knuth, 1993).',
    kind: 'graph',
    modality: 'graph',
    size: 77,
    licence: NETWORKX,
    source: 'NetworkX les_miserables_graph',
  }),
  vendoredEntry('smallMolecules', smallMolecules, {
    title: 'Small molecules',
    description: '36 common small molecules as heavy-atom graphs from SMILES, labelled aromatic or not.',
    kind: 'graph',
    modality: 'graph',
    size: 36,
    licence: 'Public domain (chemical facts)',
    source: 'mdml-js (SMILES of common molecules)',
  }),
  vendoredEntry('nurseryRhymes', nurseryRhymes, {
    title: 'Nursery rhymes',
    description: 'Sixteen traditional English nursery rhymes, one document each, titles as classes.',
    kind: 'corpus',
    modality: 'text',
    size: 16,
    licence: 'Public domain',
  }),
  vendoredEntry('deerwester', () => workedExampleCorpus('deerwester'), {
    title: 'Deerwester memo titles',
    description: 'The nine memo titles of the latent semantic analysis worked example (Deerwester et al., 1990).',
    kind: 'corpus',
    modality: 'text',
    size: 9,
    source: 'Deerwester et al. (1990), Indexing by latent semantic analysis',
  }),
  vendoredEntry('caesar', () => workedExampleCorpus('caesar'), {
    title: 'Julius Caesar passages',
    description: 'Two Julius Caesar passages, the inverted-index example of Introduction to Information Retrieval.',
    kind: 'corpus',
    modality: 'text',
    size: 2,
    licence: 'Public domain (Shakespeare)',
    source: 'Manning, Raghavan and Schütze (2008), chapter 1',
  }),
  // Remote assets, pinned by hash and fetched on demand.
  r('californiaHousing', {
    title: 'California housing',
    description: 'Median house values of 20640 California census block groups from eight attributes (1990 census).',
    kind: 'regression',
    modality: 'tabular',
    size: 20640,
    source: 'StatLib cal_housing.tgz (as fetched by scikit-learn)',
  }),
  r('electrocardiogram', {
    title: 'Electrocardiogram',
    description: 'MIT-BIH record 208 as shipped with SciPy: 108000 samples at 360 Hz, five minutes of ECG.',
    kind: 'timeseries',
    modality: 'timeSeries',
    size: 108000,
    source: 'scipy.datasets.electrocardiogram',
  }),
  r('fashionMnistTest', {
    title: 'Fashion-MNIST (test set)',
    description: 'The 10000 test images of Fashion-MNIST: 28 by 28 greyscale articles of clothing in ten classes.',
    kind: 'image',
    modality: 'image',
    size: 10000,
    source: 'zalandoresearch/fashion-mnist t10k files',
  }),
  r('movielensSmall', {
    title: 'MovieLens (small)',
    description: '100836 ratings of 9742 movies by 610 users (ml-latest-small), with timestamps.',
    kind: 'interactions',
    modality: 'interactions',
    size: 100836,
    source: 'GroupLens ml-latest-small.zip',
  }),
  r('tinyShakespeare', {
    title: 'Tiny Shakespeare',
    description: '1.1 MB of Shakespeare plays as passages separated by blank lines (Karpathy’s char-rnn data).',
    kind: 'corpus',
    modality: 'text',
    source: 'karpathy/char-rnn data/tinyshakespeare/input.txt',
  }),
])
