/**
 * Random-matrix tools for covariance and correlation matrices: the eigenvalue spectrum, the Marchenko-Pastur bulk edge
 * and an estimate of the noise variance from the median eigenvalue, the spiked-covariance (Baik-Ben Arous-Peche)
 * threshold with the limiting position and eigenvector overlap of a spiked eigenvalue, and the cleaning of a
 * correlation matrix by its spectrum: denoising (eigenvalues inside the bulk replaced by their mean, a target or zero)
 * and detoning (the market mode removed). `spectralCleaningEstimator` fits them to data on the covariance estimator
 * pattern; the exposition shows the sample eigenvalues against the Marchenko-Pastur quantiles and the correlation
 * matrix before and after.
 */

import { OmaiError } from '../base/index.ts'
import { marchenkoPastur, marchenkoPasturEdges } from '../dists/index.ts'
import { curveSet, type Exposition, type ValueGrids, valueGrids } from '../exposition/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  type IterativeEstimator,
  iterative,
  type LearnData,
} from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { bool, choice, dict, int, real } from '../space/index.ts'
import { item, type TensorLike, tensor, toArray, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  conditionOf,
  fromCorrelation,
  fromEigen,
  moments,
  type Rows,
  squareRows,
  symmetricEigen,
  toCorrelation,
} from './_matrix.ts'
import { type CovarianceParams, type CovarianceVerbs, covarianceVerbs, mahalanobisRecord } from './shrinkage.ts'

/** The eigenvalues of a symmetric matrix, descending, with their eigenvectors as rows (`eigenvectors[k]` for `eigenvalues[k]`). */
export interface EigenSpectrum {
  readonly eigenvalues: number[]
  readonly eigenvectors: number[][]
}

/**
 * The eigenvalues (descending) and eigenvectors of a symmetric matrix, the starting point of every spectral method.
 *
 * @param matrix - A symmetric matrix `[p, p]` (its symmetric part is used).
 * @returns The eigenvalues, largest first, and the unit eigenvectors as rows in the same order.
 * @index function Eigenvalues and eigenvectors of a symmetric matrix, largest first.
 * @example
 * ```ts
 * import { eigenSpectrum } from 'omai-engine/covariance'
 * eigenSpectrum([[2, 0], [0, 1]]).eigenvalues // => [2, 1]
 * ```
 */
export function eigenSpectrum(matrix: TensorLike): EigenSpectrum {
  const a = squareRows(matrix, 'eigenSpectrum')
  const p = a.length
  const { values, vectors } = symmetricEigen(a)
  return {
    eigenvalues: values,
    eigenvectors: values.map((_, k) => Array.from({ length: p }, (__, i) => vectors[i * p + k] ?? 0)),
  }
}

function checkRatio(who: string, ratio: number, variance: number): void {
  if (!(ratio > 0) || !Number.isFinite(ratio))
    throw new OmaiError('value', `${who}: ratio must be positive (got ${ratio})`)
  if (!(variance > 0) || !Number.isFinite(variance))
    throw new OmaiError('value', `${who}: variance must be positive (got ${variance})`)
}

function medianOf(values: readonly number[]): number {
  const s = [...values].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 === 1 ? (s[m] ?? 0) : 0.5 * ((s[m - 1] ?? 0) + (s[m] ?? 0))
}

/**
 * The noise variance behind a sample correlation or covariance spectrum, from its median eigenvalue (the spikes sit far
 * above the median and do not move it): the median of the eigenvalues divided by the median of the unit-variance
 * Marchenko-Pastur law of the same aspect ratio. For a ratio above 1 only the `p / ratio` non-zero eigenvalues are used,
 * as the eigenvalues of the dual `n x n` matrix.
 *
 * @param eigenvalues - The eigenvalues of the sample matrix (any order).
 * @param ratio - The aspect ratio `p / n`.
 * @returns The variance of the noise bulk.
 * @example
 * ```ts
 * import { noiseVariance } from 'omai-engine/covariance'
 * Math.round(noiseVariance([1, 1, 1, 1], 1e-9) * 1e6) / 1e6 // => 1
 * ```
 */
export function noiseVariance(eigenvalues: readonly number[], ratio: number): number {
  checkRatio('noiseVariance', ratio, 1)
  const sorted = [...eigenvalues].sort((a, b) => b - a)
  if (ratio <= 1) return medianOf(sorted) / item(marchenkoPastur(ratio, 1).ppf(0.5))
  const nonZero = Math.max(1, Math.min(sorted.length, Math.round(sorted.length / ratio)))
  return medianOf(sorted.slice(0, nonZero)) / ratio / item(marchenkoPastur(1 / ratio, 1).ppf(0.5))
}

/**
 * The Baik-Ben Arous-Peche threshold of the spiked covariance model: a population eigenvalue `spike` above noise of the
 * given variance separates from the Marchenko-Pastur bulk in the sample covariance only when it exceeds
 * `variance (1 + sqrt(ratio))`; below it the spike is invisible, the sample eigenvalue sticking to the bulk edge.
 *
 * @param ratio - The aspect ratio `p / n`.
 * @param variance - The noise variance (default 1).
 * @returns The smallest detectable population eigenvalue.
 * @index function Baik-Ben Arous-Peche threshold of a spiked covariance eigenvalue.
 * @example
 * ```ts
 * import { bbpThreshold } from 'omai-engine/covariance'
 * bbpThreshold(0.25) // => 1.5
 * ```
 */
export function bbpThreshold(ratio: number, variance = 1): number {
  checkRatio('bbpThreshold', ratio, variance)
  return variance * (1 + Math.sqrt(ratio))
}

/**
 * The almost-sure limit of the sample eigenvalue of a population spike `spike` in noise of the given variance:
 * `spike (1 + variance ratio / (spike - variance))` above the Baik-Ben Arous-Peche threshold, and the bulk edge
 * `variance (1 + sqrt(ratio))^2` at or below it. The sample eigenvalue overshoots the population one by the factor
 * `1 + variance ratio / (spike - variance)`.
 *
 * @param spike - The population eigenvalue of the spike.
 * @param ratio - The aspect ratio `p / n`.
 * @param variance - The noise variance (default 1).
 * @returns The limiting sample eigenvalue.
 * @index function Limiting sample eigenvalue of a spiked population eigenvalue.
 * @example
 * ```ts
 * import { spikedEigenvalueLimit } from 'omai-engine/covariance'
 * spikedEigenvalueLimit(1.5, 0.25) // => 2.25
 * spikedEigenvalueLimit(1.2, 0.25) // => 2.25
 * ```
 */
export function spikedEigenvalueLimit(spike: number, ratio: number, variance = 1): number {
  checkRatio('spikedEigenvalueLimit', ratio, variance)
  if (!(spike > 0)) throw new OmaiError('value', `spikedEigenvalueLimit: spike must be positive (got ${spike})`)
  if (spike <= bbpThreshold(ratio, variance)) return marchenkoPasturEdges(ratio, variance).upper
  return spike * (1 + (variance * ratio) / (spike - variance))
}

/**
 * The limit of the squared cosine between the sample eigenvector of a spiked eigenvalue and the population spike
 * direction: `(1 - ratio variance^2 / (spike - variance)^2) / (1 + ratio variance / (spike - variance))` above the
 * threshold and 0 at or below it, where the sample eigenvector carries no information about the spike.
 *
 * @param spike - The population eigenvalue of the spike.
 * @param ratio - The aspect ratio `p / n`.
 * @param variance - The noise variance (default 1).
 * @returns The limiting squared overlap, in `[0, 1)`.
 * @index function Limiting overlap of a sample eigenvector with a spike direction.
 * @example
 * ```ts
 * import { spikedEigenvectorOverlap } from 'omai-engine/covariance'
 * spikedEigenvectorOverlap(1.5, 0.25) // => 0
 * ```
 */
export function spikedEigenvectorOverlap(spike: number, ratio: number, variance = 1): number {
  checkRatio('spikedEigenvectorOverlap', ratio, variance)
  if (!(spike > 0)) throw new OmaiError('value', `spikedEigenvectorOverlap: spike must be positive (got ${spike})`)
  if (spike <= bbpThreshold(ratio, variance)) return 0
  const gap = spike - variance
  return (1 - (ratio * variance * variance) / (gap * gap)) / (1 + (ratio * variance) / gap)
}

/** How the eigenvalues inside the Marchenko-Pastur bulk are replaced by {@link denoiseCorrelation}. */
export type DenoiseMethod = 'constant' | 'target' | 'zero'

/** Options of {@link denoiseCorrelation}. */
export interface DenoiseOptions {
  /** The aspect ratio `p / n` of dimension to sample size; give this or `sampleSize`. */
  readonly ratio?: number
  /** The number of observations behind the matrix (the ratio is `p / sampleSize`). */
  readonly sampleSize?: number
  /** The noise variance (default: estimated from the median eigenvalue, see {@link noiseVariance}). */
  readonly variance?: number
  /** The bulk eigenvalues become their mean (`'constant'`, the default, which keeps the trace), a `target` or zero. */
  readonly method?: DenoiseMethod
  /** The value of the bulk eigenvalues for `'target'` (default the noise variance). */
  readonly target?: number
  /** A relative margin above the Marchenko-Pastur edge before an eigenvalue counts as signal (default 0). */
  readonly margin?: number
}

/** A denoised correlation matrix with what was decided. */
export interface DenoisedCorrelation {
  /** The cleaned matrix, scaled back to a unit diagonal. */
  readonly correlation: Rows
  /** The sample eigenvalues, descending. */
  readonly eigenvalues: number[]
  /** The eigenvalues after the bulk was replaced, before the rescaling to a unit diagonal. */
  readonly cleanedEigenvalues: number[]
  /** How many eigenvalues lie above the bulk edge and were kept. */
  readonly nSignal: number
  /** The noise variance used. */
  readonly variance: number
  /** The upper edge of the Marchenko-Pastur bulk, `variance (1 + sqrt(ratio))^2`. */
  readonly edge: number
}

function aspectRatio(
  who: string,
  p: number,
  options: { ratio?: number | undefined; sampleSize?: number | undefined },
): number {
  if (options.ratio !== undefined) return options.ratio
  if (options.sampleSize !== undefined && options.sampleSize > 0) return p / options.sampleSize
  throw new OmaiError('value', `${who}: give the aspect ratio \`ratio\` or the \`sampleSize\` behind the matrix`)
}

/**
 * Denoise a correlation matrix by its spectrum (Laloux, Cizeau, Bouchaud and Potters 1999; Lopez de Prado 2020): the
 * eigenvalues of the sample matrix that fall inside the Marchenko-Pastur bulk `variance (1 -+ sqrt(ratio))^2` are
 * indistinguishable from the noise of `n` observations of `p` independent variables, and are replaced by their mean
 * (keeping the trace), by a target value or by zero; the eigenvalues above the bulk edge are kept. The matrix is
 * rebuilt from the original eigenvectors and rescaled to a unit diagonal.
 *
 * @param correlation - A sample correlation (or covariance) matrix `[p, p]`.
 * @param options - The aspect ratio or sample size, the noise variance, the replacement and the margin.
 * @returns The cleaned correlation matrix with the eigenvalues, the number of signal eigenvalues, the noise variance and the edge.
 * @index function Denoise a correlation matrix: bulk eigenvalues below the Marchenko-Pastur edge replaced.
 * @example
 * ```ts
 * import { denoiseCorrelation } from 'omai-engine/covariance'
 * const c = [[1, 0.9, 0.1], [0.9, 1, 0.1], [0.1, 0.1, 1]]
 * denoiseCorrelation(c, { sampleSize: 30, variance: 0.8 }).nSignal // => 1
 * ```
 */
export function denoiseCorrelation(correlation: TensorLike, options: DenoiseOptions = {}): DenoisedCorrelation {
  const c = squareRows(correlation, 'denoiseCorrelation')
  const p = c.length
  const ratio = aspectRatio('denoiseCorrelation', p, options)
  const { values, vectors } = symmetricEigen(c)
  const variance = options.variance ?? noiseVariance(values, ratio)
  checkRatio('denoiseCorrelation', ratio, variance)
  const edge = marchenkoPasturEdges(ratio, variance).upper
  const cut = edge * (1 + (options.margin ?? 0))
  const nSignal = values.filter((v) => v > cut).length
  const bulk = values.slice(nSignal)
  const method = options.method ?? 'constant'
  const replacement =
    method === 'zero'
      ? 0
      : method === 'target'
        ? (options.target ?? variance)
        : bulk.reduce((a, v) => a + v, 0) / Math.max(1, bulk.length)
  const cleanedEigenvalues = values.map((v, k) => (k < nSignal ? v : replacement))
  const rebuilt = fromEigen(cleanedEigenvalues, vectors)
  return {
    correlation: toCorrelation(rebuilt).correlation,
    eigenvalues: values,
    cleanedEigenvalues,
    nSignal,
    variance,
    edge,
  }
}

/** The result of {@link detoneCorrelation}. */
export interface DetonedCorrelation {
  /** The correlation matrix without the removed modes, scaled back to a unit diagonal. */
  readonly correlation: Rows
  /** The sample eigenvalues, descending. */
  readonly eigenvalues: number[]
  /** The eigenvalues of the removed modes. */
  readonly removed: number[]
}

/**
 * Detone a correlation matrix (Lopez de Prado 2020): remove its largest eigencomponents, the market mode that every
 * variable shares, so that what is left is the structure between the variables. The matrix `C - sum_k lambda_k v_k v_k^T`
 * over the removed modes is rescaled to a unit diagonal; it is singular by one rank per mode removed.
 *
 * @param correlation - A correlation matrix `[p, p]`.
 * @param options - The number of leading eigencomponents to remove (default 1).
 * @returns The detoned correlation matrix, the eigenvalues and the removed ones.
 * @index function Detone a correlation matrix: remove the market mode (leading eigencomponents).
 * @example
 * ```ts
 * import { detoneCorrelation } from 'omai-engine/covariance'
 * const c = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
 * detoneCorrelation(c).correlation[0]?.[1] // => -0.5
 * ```
 */
export function detoneCorrelation(
  correlation: TensorLike,
  options: { readonly components?: number } = {},
): DetonedCorrelation {
  const c = squareRows(correlation, 'detoneCorrelation')
  const p = c.length
  const m = options.components ?? 1
  if (!Number.isInteger(m) || m < 0 || m >= p)
    throw new OmaiError('value', `detoneCorrelation: components must be an integer in [0, ${p - 1}] (got ${m})`)
  const { values, vectors } = symmetricEigen(c)
  const kept = values.map((v, k) => (k < m ? 0 : v))
  return {
    correlation: toCorrelation(fromEigen(kept, vectors)).correlation,
    eigenvalues: values,
    removed: values.slice(0, m),
  }
}

/** Which cleaning {@link spectralCleaningEstimator} applies. */
export type SpectralCleaningMethod = 'denoise' | 'detone' | 'denoise-detone'

/** Hyperparameters of {@link spectralCleaningEstimator}. */
export interface SpectralCleaningHyper {
  /** Denoise, detone, or denoise then detone (default `'denoise'`). */
  readonly method?: SpectralCleaningMethod
  /** How the bulk eigenvalues are replaced when denoising (default `'constant'`). */
  readonly noise?: DenoiseMethod
  /** The value of the bulk eigenvalues for `noise: 'target'` (default the noise variance). */
  readonly target?: number
  /** The noise variance; 0 estimates it from the median eigenvalue (default 0). */
  readonly variance?: number
  /** The relative margin above the Marchenko-Pastur edge before an eigenvalue counts as signal (default 0). */
  readonly margin?: number
  /** The number of leading eigencomponents detoning removes (default 1). */
  readonly components?: number
  /** Whether the data are already centred (default false). */
  readonly assumeCentered?: boolean
}

/** The state of a spectral cleaning fit. */
export interface SpectralCleaningState {
  readonly location: number[]
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly shrinkage: number
  readonly logdet: number
  /** The sample correlation matrix. */
  readonly sampleCorrelation: number[][]
  /** The cleaned correlation matrix. */
  readonly cleanedCorrelation: number[][]
  /** The sample eigenvalues, descending. */
  readonly sampleEigenvalues: number[]
  /** The eigenvalues of the cleaned correlation matrix, descending. */
  readonly cleanedEigenvalues: number[]
  readonly nSignal: number
  readonly variance: number
  readonly edge: number
  readonly ratio: number
  readonly done: boolean
}

/** The fitted parameters of the spectral cleaning estimator. */
export type SpectralCleaningParams = CovarianceParams & {
  readonly nSignal: number
  readonly variance: number
  readonly edge: number
  readonly ratio: number
}

/** The verbs of a fitted spectral cleaning (those of any fitted covariance). */
export type SpectralCleaningVerbs = CovarianceVerbs

type Resolved = Required<Omit<SpectralCleaningHyper, 'target'>> & { readonly target: number | undefined }

function resolve(h: SpectralCleaningHyper): Resolved {
  return {
    method: h.method ?? 'denoise',
    noise: h.noise ?? 'constant',
    target: h.target,
    variance: h.variance ?? 0,
    margin: h.margin ?? 0,
    components: h.components ?? 1,
    assumeCentered: h.assumeCentered ?? false,
  }
}

function cleaningState(x: TensorLike, h: Resolved, done: boolean): SpectralCleaningState {
  const m = moments(x, 'spectralCleaning', { assumeCentered: h.assumeCentered })
  const ratio = m.d / m.n
  const { correlation, scale } = toCorrelation(m.covariance)
  const sample = symmetricEigen(correlation)
  let cleaned = correlation
  let nSignal = 0
  let variance = h.variance > 0 ? h.variance : noiseVariance(sample.values, ratio)
  let edge = marchenkoPasturEdges(ratio, variance).upper
  if (h.method !== 'detone') {
    const r = denoiseCorrelation(correlation, {
      ratio,
      ...(h.variance > 0 ? { variance: h.variance } : {}),
      method: h.noise,
      ...(h.target === undefined ? {} : { target: h.target }),
      margin: h.margin,
    })
    cleaned = r.correlation
    nSignal = r.nSignal
    variance = r.variance
    edge = r.edge
  }
  if (h.method !== 'denoise')
    cleaned = detoneCorrelation(cleaned, { components: Math.min(h.components, m.d - 1) }).correlation
  const covariance = fromCorrelation(cleaned, scale)
  const precision = toArray(linalg.pinv(covariance)) as number[][]
  const cleanedEigenvalues = symmetricEigen(cleaned).values
  const positive = symmetricEigen(covariance).values.filter((v) => v > 1e-12)
  return {
    location: m.location,
    covariance,
    precision,
    shrinkage: 0,
    logdet: positive.reduce((a, v) => a + Math.log(v), 0),
    sampleCorrelation: correlation,
    cleanedCorrelation: cleaned,
    sampleEigenvalues: sample.values,
    cleanedEigenvalues,
    nSignal,
    variance,
    edge,
    ratio,
    done,
  }
}

/**
 * Spectral cleaning of a covariance matrix as a stepped {@link Algorithm}: the sample correlation matrix is denoised by
 * the Marchenko-Pastur edge, detoned, or both, then scaled back by the sample standard deviations. One step; it records
 * the number of signal eigenvalues, the bulk edge, the noise variance and the condition number before and after.
 *
 * @param x - Observations `[n, d]`.
 * @param options - The cleaning and its knobs.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { spectralCleaningAlgorithm } from 'omai-engine/covariance'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 1, 0], [-1, -1, 0.5], [1, 0.8, -1], [-1, -1, 1], [0.5, 1, 0], [-0.5, -1, -0.5]]
 * runAlgorithm(spectralCleaningAlgorithm(x), { maxSteps: 2 }).state.cleanedCorrelation[0]?.[0] // => 1
 * ```
 */
export function spectralCleaningAlgorithm(
  x: TensorLike,
  options: SpectralCleaningHyper = {},
): Algorithm<SpectralCleaningState> {
  const h = resolve(options)
  const d = moments(x, 'spectralCleaningAlgorithm').d
  return {
    name: `spectral-cleaning-${h.method}`,
    config: { method: h.method, noise: h.noise, variance: h.variance, margin: h.margin, components: h.components },
    schema: channels({
      metrics: {
        nSignal: { description: 'the number of sample eigenvalues above the Marchenko-Pastur edge' },
        edge: { description: 'the upper edge of the Marchenko-Pastur bulk' },
        variance: { description: 'the noise variance of the bulk' },
        conditionBefore: { description: 'condition number of the sample correlation matrix' },
        conditionAfter: { description: 'condition number of the cleaned correlation matrix', direction: 'lower' },
      },
      tensors: {
        correlation: { role: 'param', description: 'the cleaned correlation matrix', shape: ['d', 'd'] },
        eigenvalues: { role: 'state', description: 'the sample eigenvalues, descending', shape: ['d'] },
      },
    }),
    init: () => cleaningState(x, h, false),
    step: (s) => ({ ...s, done: true }),
    done: (s) => (s.done ? 'completed' : undefined),
    observe: (s) => ({
      metrics: {
        nSignal: s.nSignal,
        edge: s.edge,
        variance: s.variance,
        ...(Number.isFinite(conditionOf(s.sampleEigenvalues))
          ? { conditionBefore: conditionOf(s.sampleEigenvalues) }
          : {}),
        ...(Number.isFinite(conditionOf(s.cleanedEigenvalues))
          ? { conditionAfter: conditionOf(s.cleanedEigenvalues) }
          : {}),
      },
      tensors: {
        correlation: { role: 'param', value: s.cleanedCorrelation.flat(), shape: [d, d] },
        eigenvalues: { role: 'state', value: s.sampleEigenvalues, shape: [d] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SpectralCleaningState,
  }
}

/** The spectrum record of a cleaning: the sample eigenvalues against the Marchenko-Pastur quantiles and the cleaned ones. */
function spectrumRecords(s: SpectralCleaningState): Exposition[] {
  const p = s.sampleEigenvalues.length
  const ranks = Array.from({ length: p }, (_, i) => i + 1)
  const probs = ranks.map((i) => 1 - (i - 0.5) / p)
  const expected = toFlat(marchenkoPastur(s.ratio, s.variance).ppf(tensor(probs)))
  const records: Exposition[] = [
    curveSet({
      x: ranks,
      xLabel: 'eigenvalue rank',
      yLabel: 'eigenvalue',
      yLog: s.sampleEigenvalues.every((v) => v > 0) && s.cleanedEigenvalues.every((v) => v > 1e-12),
      series: [
        { name: 'sample', y: s.sampleEigenvalues },
        { name: 'cleaned', y: s.cleanedEigenvalues },
        { name: 'Marchenko-Pastur quantiles', y: expected },
      ],
      marks: [{ x: s.nSignal + 0.5, label: `${s.nSignal} signal` }],
    }),
  ]
  const grids: ValueGrids = valueGrids({
    grids: [
      { name: 'sample correlation', values: s.sampleCorrelation },
      { name: 'cleaned correlation', values: s.cleanedCorrelation },
    ],
    scale: 'shared',
    diverging: true,
    axes: ['variable', 'variable'],
    valueName: 'correlation',
  })
  return [...records, grids]
}

/**
 * The spectral cleaning estimator (denoising and detoning of the sample correlation matrix by the Marchenko-Pastur
 * edge) on the iterative base, as a covariance estimator: location, covariance, precision (pseudo-inverse),
 * `mahalanobis` and `score`. Its exposition is the eigenvalue spectrum against the Marchenko-Pastur quantiles with the
 * signal count marked, the correlation matrix before and after, and the Mahalanobis anomaly scores.
 *
 * @param hyper - The cleaning (`'denoise'`, `'detone'` or `'denoise-detone'`) and its knobs.
 * @returns The iterative estimator.
 * @index estimator Covariance by denoising and detoning the correlation spectrum.
 * @example
 * ```ts
 * import { spectralCleaningEstimator } from 'omai-engine/covariance'
 * const x = [[1, 1, 0], [-1, -1, 0.5], [1, 0.8, -1], [-1, -1, 1], [0.5, 1, 0], [-0.5, -1, -0.5]]
 * spectralCleaningEstimator({ method: 'detone' }).fit({ x }).params.nSignal // => 0
 * ```
 */
export function spectralCleaningEstimator(
  hyper: SpectralCleaningHyper = {},
): IterativeEstimator<SpectralCleaningHyper, SpectralCleaningState, SpectralCleaningParams, SpectralCleaningVerbs> {
  return iterative<SpectralCleaningHyper, SpectralCleaningState, SpectralCleaningParams, SpectralCleaningVerbs>({
    name: 'spectral-cleaning',
    kind: 'density',
    defaults: {
      ...resolve(hyper),
      ...(hyper.target === undefined ? {} : { target: hyper.target }),
    } as SpectralCleaningHyper,
    hyperSpace: (h) =>
      dict({
        method: choice(['denoise', 'detone', 'denoise-detone'] as const, {
          label: 'cleaning',
          description: 'Denoise the bulk, remove the market mode, or both.',
          default: h.method ?? 'denoise',
        }),
        noise: choice(['constant', 'target', 'zero'] as const, {
          label: 'bulk eigenvalues',
          description: 'What the eigenvalues inside the Marchenko-Pastur bulk become.',
          default: h.noise ?? 'constant',
        }),
        margin: real(0, 1, {
          label: 'margin',
          description: 'Relative margin above the bulk edge before an eigenvalue counts as signal.',
          default: h.margin ?? 0,
        }),
        components: int(1, 10, {
          label: 'modes removed',
          description: 'Leading eigencomponents removed by detoning.',
          default: h.components ?? 1,
        }),
        assumeCentered: bool({
          label: 'assume centred',
          description: 'Treat the data as already centred (do not subtract the mean).',
          default: h.assumeCentered ?? false,
        }),
      }),
    algorithm: (data, h) => spectralCleaningAlgorithm(data.x, h),
    maxSteps: () => 1,
    params: (s) => ({
      location: s.location,
      covariance: s.covariance,
      precision: s.precision,
      shrinkage: s.shrinkage,
      nSignal: s.nSignal,
      variance: s.variance,
      edge: s.edge,
      ratio: s.ratio,
    }),
    verbs: covarianceVerbs,
    records: (s, data: LearnData) => [
      ...spectrumRecords(s),
      mahalanobisRecord('spectralCleaning', data.x, {
        location: s.location,
        covariance: s.covariance,
        precision: s.precision,
        shrinkage: 0,
      }),
    ],
    logs: { 'solver/nSignal': Log(), 'solver/conditionAfter': Log() },
  })
}

/**
 * The spectral cleaning estimator as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { SpectralCleaning } from 'omai-engine/covariance'
 * const x = [[1, 1, 0], [-1, -1, 0.5], [1, 0.8, -1], [-1, -1, 1], [0.5, 1, 0], [-0.5, -1, -0.5]]
 * new SpectralCleaning({ method: 'detone' }).fit(x).model.params.location.length // => 3
 * ```
 */
export class SpectralCleaning extends ExplainedEstimatorClass<
  SpectralCleaningHyper,
  Fitted<SpectralCleaningParams> & SpectralCleaningVerbs
> {
  /**
   * Make an unfitted spectral cleaning model.
   *
   * @param hyper - The hyperparameters of {@link spectralCleaningEstimator}.
   */
  constructor(hyper: SpectralCleaningHyper = {}) {
    super(spectralCleaningEstimator(hyper))
  }
}
