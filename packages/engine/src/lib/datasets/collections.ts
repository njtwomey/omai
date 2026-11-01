/**
 * Themed collections of datasets for pickers and hub entries (two-dimensional classification, one-dimensional
 * regression and so on): each entry is a generator with demo knobs or a vendored real dataset, and `loadEntry` makes or
 * loads one.
 */

import { MdmlError } from '../base/index.ts'
import type { Dataset } from '../data/index.ts'
import { type Generated, generate, type Knobs } from '../data/index.ts'
import type { Key } from '../random/index.ts'
import { datasetGenerators } from './catalogue.ts'
import { breastCancer, digits, iris, penguinsClassification, wine } from './classification-real.ts'
import {
  balanceScale,
  banknoteAuthentication,
  ecoli,
  glassIdentification,
  haberman,
  ionosphere,
  lenses,
  seeds,
  sonar,
  spambase,
  vehicleSilhouettes,
  wholesaleCustomers,
  yeast,
} from './classification-uci.ts'
import { oldFaithful } from './clustering.ts'
import { germanCredit, titanicSurvival } from './fairness.ts'
import { diabetes, linnerud, mtcarsRegression } from './regression-real.ts'
import {
  airfoilSelfNoise,
  concreteStrength,
  energyEfficiency,
  wineQualityRed,
  wineQualityWhite,
  yachtHydrodynamics,
} from './regression-uci.ts'
import {
  abalone,
  adultIncome,
  autoMpg,
  bikeSharingDaily,
  carEvaluation,
  congressionalVoting,
  dermatology,
  forestFires,
  heartDisease,
  hepatitis,
  mushroom,
  servo,
  studentMath,
  ticTacToe,
  zoo,
} from './tables-uci.ts'

/** One dataset in a collection: a generator with demo knobs, or a vendored real dataset. */
export type CollectionEntry =
  | {
      readonly name: string
      readonly title: string
      /** A generator name in {@link datasetGenerators}. */
      readonly generator: string
      /** Knob values for demos (missing knobs take the generator's defaults). */
      readonly knobs?: Knobs
    }
  | { readonly name: string; readonly title: string; readonly load: () => Dataset }

/** A themed list of datasets for pickers (classification in two dimensions, one-dimensional regression, ...). */
export interface DatasetCollection {
  readonly id: string
  readonly title: string
  readonly entries: readonly CollectionEntry[]
}

function g(name: string, title: string, knobs?: Knobs): CollectionEntry
function g(name: string, title: string, generator: string, knobs: Knobs): CollectionEntry
function g(name: string, title: string, a?: Knobs | string, b?: Knobs): CollectionEntry {
  const generator = typeof a === 'string' ? a : name
  const knobs = typeof a === 'string' ? b : a
  return knobs === undefined ? { name, title, generator } : { name, title, generator, knobs }
}

/**
 * The dataset collections, by id: what hub entries name and dataset pickers offer.
 *
 * @example
 * ```ts
 * import { datasetCollections } from 'mdml-engine/datasets'
 * datasetCollections['classification2d']?.entries.map((e) => e.name).includes('moons') // => true
 * ```
 */
export const datasetCollections: Readonly<Record<string, DatasetCollection>> = {
  classification2d: {
    id: 'classification2d',
    title: 'Classification in two dimensions',
    entries: [
      g('moons', 'Two moons', { n: 200, noise: 0.2 }),
      g('circles', 'Concentric circles', { n: 200, noise: 0.08 }),
      g('blobs', 'Blobs', { n: 200 }),
      g('xor', 'XOR', { n: 200 }),
      g('spirals', 'Spirals', { n: 200 }),
      g('checkerboard', 'Checkerboard', { n: 300 }),
      g('anisotropicBlobs', 'Anisotropic blobs', { n: 200 }),
      g('unequalVarianceBlobs', 'Blobs of unequal variance', { n: 200 }),
      g('gaussianQuantiles', 'Gaussian quantiles', { n: 200 }),
    ],
  },
  classificationTabular: {
    id: 'classificationTabular',
    title: 'Tabular classification',
    entries: [
      { name: 'iris', title: 'Iris', load: iris },
      { name: 'wine', title: 'Wine', load: wine },
      { name: 'breastCancer', title: 'Breast cancer (Wisconsin)', load: breastCancer },
      { name: 'penguins', title: 'Palmer penguins', load: penguinsClassification },
      { name: 'digits', title: 'Digits (8 by 8)', load: digits },
      { name: 'banknoteAuthentication', title: 'Banknote authentication', load: banknoteAuthentication },
      { name: 'ionosphere', title: 'Ionosphere', load: ionosphere },
      { name: 'sonar', title: 'Sonar (mines versus rocks)', load: sonar },
      { name: 'glassIdentification', title: 'Glass identification', load: glassIdentification },
      { name: 'seeds', title: 'Seeds', load: seeds },
      { name: 'ecoli', title: 'E. coli protein localisation', load: ecoli },
      { name: 'yeast', title: 'Yeast protein localisation', load: yeast },
      { name: 'haberman', title: "Haberman's survival", load: haberman },
      { name: 'balanceScale', title: 'Balance scale', load: balanceScale },
      { name: 'vehicleSilhouettes', title: 'Vehicle silhouettes', load: vehicleSilhouettes },
      { name: 'wholesaleCustomers', title: 'Wholesale customers', load: wholesaleCustomers },
      { name: 'lenses', title: 'Contact lenses', load: lenses },
      { name: 'spambase', title: 'Spambase (subsample)', load: spambase },
      g('makeClassification', 'Synthetic (make_classification)'),
    ],
  },
  fairnessTabular: {
    id: 'fairnessTabular',
    title: 'Classification with a protected attribute',
    entries: [
      g('biasedLabels', 'Credit: biased labels', 'biasedCredit', { labelBias: 0.3, baseRateGap: 0.3 }),
      g('underRepresented', 'Credit: an under-represented, noisier group', 'biasedCredit', {
        minorityShare: 0.12,
        minorityNoise: 2.5,
        labelBias: 0,
        baseRateGap: 0,
      }),
      g('baseRateGap', 'Credit: a real base-rate gap, no bias', 'biasedCredit', {
        labelBias: 0,
        baseRateGap: 1,
        minorityNoise: 1,
      }),
      g('strongProxy', 'Credit: biased labels and a strong proxy', 'biasedCredit', {
        proxyStrength: 3,
        labelBias: 0.3,
      }),
      g('fairCredit', 'Credit: no bias at all', 'biasedCredit', {
        labelBias: 0,
        baseRateGap: 0,
        minorityNoise: 1,
        proxyStrength: 0,
      }),
      { name: 'germanCredit', title: 'German credit (Statlog)', load: germanCredit },
      { name: 'titanicSurvival', title: 'Titanic survival', load: titanicSurvival },
    ],
  },
  regression1d: {
    id: 'regression1d',
    title: 'Regression in one dimension',
    entries: [
      g('sineCurve', 'Sine curve'),
      g('linearOutliers', 'Line with outliers'),
      g('sincCurve', 'Sinc curve'),
      g('stepCurve', 'Step'),
      g('piecewiseLinear', 'Piecewise linear'),
      g('dopplerCurve', 'Doppler'),
      g('heteroscedastic', 'Heteroscedastic noise'),
      g('quantileCurves', 'Skewed noise (quantiles)'),
    ],
  },
  glm1d: {
    id: 'glm1d',
    title: 'Counts and positive responses in one dimension',
    entries: [
      g('poissonCounts', 'Poisson counts', 'glmCurve', { family: 'poisson' }),
      g('overdispersedCounts', 'Overdispersed counts (negative binomial)', 'glmCurve', { family: 'negativeBinomial' }),
      g('gammaResponses', 'Positive skewed (gamma)', 'glmCurve', { family: 'gamma' }),
      g('inverseGaussianResponses', 'Positive, heavy right tail (inverse Gaussian)', 'glmCurve', {
        family: 'inverseGaussian',
        dispersion: 0.1,
      }),
      g('tweedieResponses', 'Zeros and positive amounts (Tweedie)', 'glmCurve', { family: 'tweedie' }),
    ],
  },
  regressionTabular: {
    id: 'regressionTabular',
    title: 'Tabular regression',
    entries: [
      { name: 'diabetes', title: 'Diabetes', load: () => diabetes() },
      { name: 'mtcars', title: 'Motor Trend cars', load: mtcarsRegression },
      { name: 'linnerud', title: 'Linnerud', load: linnerud },
      { name: 'concreteStrength', title: 'Concrete compressive strength', load: concreteStrength },
      { name: 'energyEfficiency', title: 'Energy efficiency', load: energyEfficiency },
      { name: 'yachtHydrodynamics', title: 'Yacht hydrodynamics', load: yachtHydrodynamics },
      { name: 'airfoilSelfNoise', title: 'Airfoil self-noise', load: airfoilSelfNoise },
      { name: 'wineQualityRed', title: 'Wine quality (red)', load: wineQualityRed },
      { name: 'wineQualityWhite', title: 'Wine quality (white)', load: wineQualityWhite },
      g('friedman1', 'Friedman 1'),
      g('makeRegression', 'Synthetic (make_regression)'),
      g('sparseUncorrelated', 'Sparse uncorrelated'),
    ],
  },
  mixedTables: {
    id: 'mixedTables',
    title: 'Tables with categorical features and missing values',
    entries: [
      { name: 'autoMpg', title: 'Auto MPG', load: autoMpg },
      { name: 'abalone', title: 'Abalone', load: abalone },
      { name: 'heartDisease', title: 'Heart disease (Cleveland)', load: heartDisease },
      { name: 'hepatitis', title: 'Hepatitis', load: hepatitis },
      { name: 'zoo', title: 'Zoo', load: zoo },
      { name: 'ticTacToe', title: 'Tic-tac-toe endgame', load: ticTacToe },
      { name: 'carEvaluation', title: 'Car evaluation', load: carEvaluation },
      { name: 'congressionalVoting', title: 'Congressional voting records', load: congressionalVoting },
      { name: 'mushroom', title: 'Mushroom (subsample)', load: mushroom },
      { name: 'adultIncome', title: 'Adult census income (subsample)', load: adultIncome },
      { name: 'dermatology', title: 'Dermatology', load: dermatology },
      { name: 'forestFires', title: 'Forest fires', load: forestFires },
      { name: 'studentMath', title: 'Student performance (mathematics)', load: studentMath },
      { name: 'servo', title: 'Servo', load: servo },
      { name: 'bikeSharingDaily', title: 'Bike sharing (daily)', load: bikeSharingDaily },
    ],
  },
  clusteringTabular: {
    id: 'clusteringTabular',
    title: 'Clustering real tabular data',
    entries: [
      { name: 'oldFaithful', title: 'Old Faithful', load: () => oldFaithful().data },
      { name: 'seeds', title: 'Seeds', load: seeds },
      { name: 'wholesaleCustomers', title: 'Wholesale customers', load: wholesaleCustomers },
      { name: 'glassIdentification', title: 'Glass identification', load: glassIdentification },
      { name: 'ecoli', title: 'E. coli protein localisation', load: ecoli },
      { name: 'yeast', title: 'Yeast protein localisation', load: yeast },
    ],
  },
  clustering2d: {
    id: 'clustering2d',
    title: 'Clustering in two dimensions',
    entries: [
      g('blobs', 'Blobs', { n: 300 }),
      g('anisotropicBlobs', 'Anisotropic blobs', { n: 300 }),
      g('unequalVarianceBlobs', 'Blobs of unequal variance', { n: 300 }),
      g('moons', 'Two moons', { n: 300, noise: 0.06 }),
      g('circles', 'Concentric circles', { n: 300, noise: 0.05, factor: 0.5 }),
      g('rings', 'Rings', { n: 300 }),
    ],
  },
  density2d: {
    id: 'density2d',
    title: 'Densities in two dimensions',
    entries: [g('densitySamples', 'Samples from a known density'), g('moons', 'Two moons', { n: 400, noise: 0.1 })],
  },
  anomaly2d: {
    id: 'anomaly2d',
    title: 'Anomalies in two dimensions',
    entries: [
      g('blobsWithOutliers', 'Blobs with outliers'),
      g('moonsWithOutliers', 'Two moons with outliers'),
      g('blobs', 'Blobs (no outliers)', { n: 300 }),
      g('moons', 'Two moons (no outliers)', { n: 300, noise: 0.08 }),
    ],
  },
  manifold3d: {
    id: 'manifold3d',
    title: 'Manifolds in three dimensions',
    entries: [
      g('swissRoll', 'Swiss roll', { n: 300 }),
      g('sCurve', 'S curve', { n: 300 }),
      g('severedSphere', 'Severed sphere', { n: 300 }),
      g('torus', 'Torus', { n: 300 }),
    ],
  },
  timeSeries: {
    id: 'timeSeries',
    title: 'Time series',
    entries: [
      g('sinusoids', 'Sinusoids'),
      g('seasonalSeries', 'Seasonal series'),
      g('arma', 'ARMA process'),
      g('changepointSeries', 'Changepoints'),
    ],
  },
  sequences: {
    id: 'sequences',
    title: 'Sequences with hidden states',
    entries: [
      g('dishonestCasino', 'The dishonest casino', { length: 1000 }),
      g('regimeSwitching', 'Two Gaussian regimes', { length: 300 }),
    ],
  },
  comparisons: {
    id: 'comparisons',
    title: 'Games and tournaments',
    entries: [
      g('eloGames', 'Games between players of fixed skill', { players: 12, games: 400 }),
      {
        name: 'driftingSkills',
        title: 'Games between players whose skill drifts',
        generator: 'eloGames',
        knobs: { players: 12, games: 600, drift: 15 },
      },
      g('roundRobin', 'Round-robin tournament', { players: 10, rounds: 4 }),
    ],
  },
  textCorpora: {
    id: 'textCorpora',
    title: 'Text corpora',
    entries: [
      g('topicCorpus', 'Topic corpus'),
      g('naiveBayesCorpus', 'Labelled corpus'),
      g('driftingTopics', 'Drifting topics'),
    ],
  },
}

/**
 * Generate or load one dataset of a collection entry.
 *
 * @param entry - The entry.
 * @param key - The key for generated datasets.
 * @param knobs - Knob values overriding the entry's demo knobs.
 * @returns The dataset, with its generating truth for generated entries.
 * @throws MdmlError `value` when the entry names an unknown generator.
 * @example
 * ```ts
 * import { datasetCollections, loadEntry } from 'mdml-engine/datasets'
 * import { key } from 'mdml-engine/random'
 * const moons = datasetCollections['classification2d']?.entries[0]
 * moons === undefined ? 0 : loadEntry(moons, key(0)).data.kind // => 'classification'
 * ```
 */
export function loadEntry(
  entry: CollectionEntry,
  key: Key,
  knobs: Knobs = {},
): { readonly data: Dataset } & Partial<Generated> {
  if ('load' in entry) return { data: entry.load() }
  const generator = datasetGenerators[entry.generator]
  if (generator === undefined) throw new MdmlError('value', `loadEntry: unknown generator '${entry.generator}'`)
  return generate(generator, { key, knobs: { ...entry.knobs, ...knobs } })
}
