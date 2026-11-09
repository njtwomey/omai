/**
 * Shrinkage of the sample covariance towards a target other than a scaled identity, with the intensity that minimises
 * the expected squared error in closed form where it exists: the diagonal of the sample covariance, a constant
 * correlation (Ledoit and Wolf 2003, "Honey, I shrunk the sample covariance matrix"), a single market factor (Ledoit and
 * Wolf 2003, "Improved estimation of the covariance matrix of stock returns") and the scaled identity with the
 * Ledoit-Wolf, oracle approximating and Rao-Blackwell Ledoit-Wolf (Chen et al. 2010) intensities. `shrinkagePath` follows
 * the intensity from 0 to 1 (the spectrum bending towards the target, the condition number and the held-out likelihood),
 * as a stepped algorithm; `targetShrinkageEstimator` is the covariance estimator on the iterative base.
 */

import { OmaiError } from '../base/index.ts'
import { curveSet, type Exposition } from '../exposition/index.ts'
import {
  ExplainedEstimatorClass,
  type Fitted,
  type IterativeEstimator,
  iterative,
  type LearnData,
} from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { bool, choice, dict, real } from '../space/index.ts'
import { ledoitWolf, oas } from '../stats/index.ts'
import { type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, Log, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { conditionOf, gaussianLogLikelihood, moments, type Rows, symmetricEigen } from './_matrix.ts'
import { type CovarianceParams, type CovarianceVerbs, covarianceVerbs, mahalanobisRecord } from './shrinkage.ts'

/** What the sample covariance is shrunk towards. */
export type ShrinkageTarget = 'identity' | 'diagonal' | 'constant-correlation' | 'single-factor'

/** How the shrinkage intensity is chosen: a number in `[0, 1]`, or an analytic rule. */
export type ShrinkageIntensity = number | 'analytic' | 'oas' | 'rblw'

/** The moments the targets and intensities are computed from. */
type Moments = ReturnType<typeof moments>

function buildTarget(m: Moments, target: ShrinkageTarget): Rows {
  const { d, covariance: s } = m
  if (target === 'identity') {
    const mu = s.reduce((a, r, i) => a + (r[i] ?? 0), 0) / d
    return s.map((r, i) => r.map((_, j) => (i === j ? mu : 0)))
  }
  if (target === 'diagonal') return s.map((r, i) => r.map((v, j) => (i === j ? v : 0)))
  if (target === 'constant-correlation') {
    const sd = s.map((r, i) => Math.sqrt(r[i] ?? 0))
    let total = 0
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) if (i !== j) total += (s[i]?.[j] ?? 0) / ((sd[i] ?? 1) * (sd[j] ?? 1))
    const rbar = d > 1 ? total / (d * (d - 1)) : 0
    return s.map((r, i) => r.map((v, j) => (i === j ? v : rbar * (sd[i] ?? 0) * (sd[j] ?? 0))))
  }
  const { covMarket, varMarket } = marketMoments(m)
  return s.map((r, i) => r.map((v, j) => (i === j ? v : ((covMarket[i] ?? 0) * (covMarket[j] ?? 0)) / varMarket)))
}

/** The cross-sectional average `m_t` of the (centred) variables, its covariance with each variable and its variance. */
function marketMoments(m: Moments): { market: Float64Array; covMarket: number[]; varMarket: number } {
  const { n, d, centred } = m
  const market = new Float64Array(n)
  for (let t = 0; t < n; t++) {
    let a = 0
    for (let i = 0; i < d; i++) a += centred[t * d + i] ?? 0
    market[t] = a / d
  }
  const covMarket = new Array<number>(d).fill(0)
  let varMarket = 0
  for (let t = 0; t < n; t++) {
    const mt = market[t] ?? 0
    varMarket += (mt * mt) / n
    for (let i = 0; i < d; i++) covMarket[i] = (covMarket[i] ?? 0) + ((centred[t * d + i] ?? 0) * mt) / n
  }
  if (!(varMarket > 0))
    throw new OmaiError('value', 'targetShrinkage: the single-factor target needs a market with variance')
  return { market, covMarket, varMarket }
}

/** The asymptotic variances `pi_ij = (1 / T) sum_t (x_ti x_tj - s_ij)^2` of the sample covariances. */
function piMatrix(m: Moments): Rows {
  const { n, d, centred, covariance: s } = m
  const pi = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  for (let t = 0; t < n; t++)
    for (let i = 0; i < d; i++) {
      const xi = centred[t * d + i] ?? 0
      for (let j = i; j < d; j++) {
        const e = xi * (centred[t * d + j] ?? 0) - (s[i]?.[j] ?? 0)
        const row = pi[i] as number[]
        row[j] = (row[j] ?? 0) + (e * e) / n
      }
    }
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) (pi[i] as number[])[j] = pi[j]?.[i] ?? 0
  return pi
}

function clamp01(x: number): number {
  return Number.isFinite(x) ? Math.max(0, Math.min(1, x)) : 1
}

/** The squared Frobenius distance of the sample covariance to the target. */
function gammaOf(s: Rows, f: Rows): number {
  let g = 0
  for (let i = 0; i < s.length; i++) for (let j = 0; j < s.length; j++) g += ((f[i]?.[j] ?? 0) - (s[i]?.[j] ?? 0)) ** 2
  return g
}

/** The Rao-Blackwell Ledoit-Wolf intensity for the scaled-identity target (Chen, Wiesel, Eldar and Hero 2010). */
function rblwIntensity(m: Moments): number {
  const { n, d, covariance: s } = m
  let trS = 0
  let trS2 = 0
  for (let i = 0; i < d; i++) {
    trS += s[i]?.[i] ?? 0
    for (let j = 0; j < d; j++) trS2 += (s[i]?.[j] ?? 0) * (s[j]?.[i] ?? 0)
  }
  const denom = (n + 2) * (trS2 - (trS * trS) / d)
  if (!(denom > 0)) return 1
  return Math.min(1, (((n - 2) / n) * trS2 + trS * trS) / denom)
}

/** The analytic intensity of the Ledoit-Wolf framework for a target, from the moments. */
function analyticIntensity(m: Moments, target: ShrinkageTarget): number {
  const { n, d, centred, covariance: s } = m
  if (target === 'identity') return clamp01(ledoitWolf(centredToTensor(m), { assumeCentered: true }).shrinkage)
  const f = buildTarget(m, target)
  const gamma = gammaOf(s, f)
  if (!(gamma > 0)) return 1
  const pi = piMatrix(m)
  let piSum = 0
  let piDiag = 0
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) {
      piSum += pi[i]?.[j] ?? 0
      if (i === j) piDiag += pi[i]?.[j] ?? 0
    }
  if (target === 'diagonal') return clamp01((piSum - piDiag) / (gamma * n))
  if (target === 'constant-correlation') {
    // rho = sum_i pi_ii + rbar sum_{i != j} sqrt(s_jj / s_ii) theta_{ii,ij}, theta_{ii,ij} = (1/T) sum_t (x_i^2 - s_ii)(x_i x_j - s_ij).
    const sd = s.map((r, i) => Math.sqrt(r[i] ?? 0))
    let rbar = 0
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) if (i !== j) rbar += (s[i]?.[j] ?? 0) / ((sd[i] ?? 1) * (sd[j] ?? 1))
    rbar /= d * (d - 1)
    const theta = Array.from({ length: d }, () => new Array<number>(d).fill(0))
    for (let t = 0; t < n; t++)
      for (let i = 0; i < d; i++) {
        const xi = centred[t * d + i] ?? 0
        const a = xi * xi - (s[i]?.[i] ?? 0)
        const row = theta[i] as number[]
        for (let j = 0; j < d; j++)
          row[j] = (row[j] ?? 0) + (a * (xi * (centred[t * d + j] ?? 0) - (s[i]?.[j] ?? 0))) / n
      }
    let cross = 0
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) if (i !== j) cross += ((sd[j] ?? 0) / (sd[i] ?? 1)) * (theta[i]?.[j] ?? 0)
    return clamp01((piSum - (piDiag + rbar * cross)) / (gamma * n))
  }
  // Single factor (Ledoit and Wolf 2003): the terms of rho from the market moments.
  const { market, covMarket, varMarket } = marketMoments(m)
  const v1 = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  const v3 = Array.from({ length: d }, () => new Array<number>(d).fill(0))
  for (let t = 0; t < n; t++) {
    const mt = market[t] ?? 0
    for (let i = 0; i < d; i++) {
      const xi = centred[t * d + i] ?? 0
      const row1 = v1[i] as number[]
      const row3 = v3[i] as number[]
      for (let j = 0; j < d; j++) {
        const xj = centred[t * d + j] ?? 0
        row1[j] = (row1[j] ?? 0) + (xi * xi * xj * mt) / n
        row3[j] = (row3[j] ?? 0) + (xi * mt * xj * mt) / n
      }
    }
  }
  let roff1 = 0
  let roff3 = 0
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) {
      const a = (v1[i]?.[j] ?? 0) - (covMarket[i] ?? 0) * (s[i]?.[j] ?? 0)
      const b = (v3[i]?.[j] ?? 0) - varMarket * (s[i]?.[j] ?? 0)
      roff1 += a * ((covMarket[j] ?? 0) / varMarket)
      roff3 += (b * (covMarket[i] ?? 0) * (covMarket[j] ?? 0)) / (varMarket * varMarket)
      if (i === j) {
        roff1 -= a * ((covMarket[i] ?? 0) / varMarket)
        roff3 -= (b * (covMarket[i] ?? 0) ** 2) / (varMarket * varMarket)
      }
    }
  const rho = piDiag + 2 * roff1 - roff3
  return clamp01((piSum - rho) / (gamma * n))
}

function centredToTensor(m: Moments): TensorLike {
  return Array.from({ length: m.n }, (_, t) => Array.from(m.centred.subarray(t * m.d, (t + 1) * m.d)))
}

/** The convex combination `(1 - delta) S + delta F`. */
function mix(s: Rows, f: Rows, delta: number): Rows {
  return s.map((r, i) => r.map((v, j) => (1 - delta) * v + delta * (f[i]?.[j] ?? 0)))
}

function resolveIntensity(m: Moments, target: ShrinkageTarget, intensity: ShrinkageIntensity): number {
  if (typeof intensity === 'number') {
    if (!(intensity >= 0 && intensity <= 1))
      throw new OmaiError('value', `targetShrinkage: intensity must be in [0, 1] (got ${intensity})`)
    return intensity
  }
  if (intensity === 'analytic') return analyticIntensity(m, target)
  if (target !== 'identity')
    throw new OmaiError(
      'value',
      `targetShrinkage: the '${intensity}' intensity is defined for the identity target only`,
    )
  if (intensity === 'oas') return oas(centredToTensor(m), { assumeCentered: true }).shrinkage
  return rblwIntensity(m)
}

/**
 * The target matrix a sample covariance is shrunk towards: a scaled identity with the mean variance, the diagonal of the
 * sample covariance, the constant-correlation matrix (the sample variances with every correlation the average one) or the
 * single-factor matrix (the covariances implied by regressing every variable on the cross-sectional average).
 *
 * @param x - Observations `[n, d]`.
 * @param target - Which target.
 * @param options - Whether the data are already centred (default false).
 * @returns The target covariance `[d, d]`.
 * @index function Shrinkage target matrix: identity, diagonal, constant correlation or single factor.
 * @example
 * ```ts
 * import { shrinkageTargetMatrix } from 'omai-engine/covariance'
 * shrinkageTargetMatrix([[1, 2], [3, 6], [5, 4]], 'diagonal') // => [[2.6666666666666665, 0], [0, 2.6666666666666665]]
 * ```
 */
export function shrinkageTargetMatrix(
  x: TensorLike,
  target: ShrinkageTarget,
  options: { readonly assumeCentered?: boolean } = {},
): number[][] {
  return buildTarget(moments(x, 'shrinkageTargetMatrix', options), target)
}

/** Options of {@link targetShrinkage}. */
export interface TargetShrinkageOptions {
  /** The target (default `'constant-correlation'`). */
  readonly target?: ShrinkageTarget
  /** A number in `[0, 1]`, or `'analytic'` (default; the minimiser of the expected squared error to first order), or for the identity target `'oas'` or `'rblw'`. */
  readonly intensity?: ShrinkageIntensity
  /** Whether the data are already centred (default false). */
  readonly assumeCentered?: boolean
}

/** A covariance shrunk towards a target. */
export interface TargetShrinkage {
  /** The shrunk covariance `(1 - shrinkage) S + shrinkage F`. */
  readonly covariance: number[][]
  /** The sample (maximum likelihood) covariance `S`. */
  readonly sample: number[][]
  /** The target `F`. */
  readonly target: number[][]
  /** The shrinkage intensity used. */
  readonly shrinkage: number
  /** The column means (zeros when `assumeCentered`). */
  readonly location: number[]
}

/**
 * The sample covariance shrunk towards a target with the analytic Ledoit-Wolf intensity (or a given one): `(1 - delta) S
 * + delta F`. The analytic intensity `delta* = (pi - rho) / (gamma T)` estimates the minimiser of the expected squared
 * Frobenius error from the sample's own fourth moments: `pi` is the summed asymptotic variance of the sample
 * covariances, `rho` its covariance with the target's entries and `gamma` the squared distance of target and sample. A
 * target that is right drives it to 1; one that is far from the truth, to 0 as the sample grows.
 *
 * @param x - Observations `[n, d]`.
 * @param options - The target, the intensity and the centring.
 * @returns The shrunk covariance, the sample covariance, the target, the intensity and the location.
 * @throws OmaiError with code `value` for an intensity outside `[0, 1]`, or `'oas'`/`'rblw'` for a target other than the identity.
 * @index function Covariance shrunk towards a diagonal, constant-correlation, single-factor or identity target.
 * @example
 * ```ts
 * import { targetShrinkage } from 'omai-engine/covariance'
 * targetShrinkage([[1, 2], [3, 6], [5, 4]], { target: 'diagonal', intensity: 0.5 }).covariance[0]?.[1] // => 0.6666666666666666
 * ```
 */
export function targetShrinkage(x: TensorLike, options: TargetShrinkageOptions = {}): TargetShrinkage {
  const m = moments(x, 'targetShrinkage', { assumeCentered: options.assumeCentered ?? false })
  const target = options.target ?? 'constant-correlation'
  const f = buildTarget(m, target)
  const shrinkage = resolveIntensity(m, target, options.intensity ?? 'analytic')
  return {
    covariance: mix(m.covariance, f, shrinkage),
    sample: m.covariance,
    target: f,
    shrinkage,
    location: m.location,
  }
}

/** One point of a {@link shrinkagePath}. */
export interface ShrinkagePathPoint {
  /** The intensity. */
  readonly intensity: number
  /** The eigenvalues of the shrunk covariance, descending. */
  readonly eigenvalues: number[]
  /** The ratio of the largest to the smallest eigenvalue (`Infinity` when singular). */
  readonly conditionNumber: number
  /** The largest minus the smallest eigenvalue. */
  readonly spread: number
  /** The mean Gaussian log likelihood of held-out rows (`-Infinity` when the covariance is singular; `NaN` when there is no hold-out). */
  readonly heldOut: number
}

/** Options of {@link shrinkagePathAlgorithm}. */
export interface ShrinkagePathOptions {
  /** The target (default `'constant-correlation'`). */
  readonly target?: ShrinkageTarget
  /** The intensities to visit (default 21 evenly spaced from 0 to 1). */
  readonly intensities?: readonly number[]
  /** Rows held out for the likelihood; when absent the likelihood is cross-validated over `folds` folds of the data. */
  readonly heldOut?: TensorLike
  /** The number of cross-validation folds when no rows are held out (default 5; rows are dealt to folds in turn). */
  readonly folds?: number
  /** Whether the data are already centred (default false). */
  readonly assumeCentered?: boolean
}

/** The state of {@link shrinkagePathAlgorithm}: how many intensities were visited and the points so far. */
export interface ShrinkagePathState {
  readonly index: number
  readonly points: ShrinkagePathPoint[]
}

interface PathContext {
  readonly full: Moments
  readonly target: Rows
  readonly grid: number[]
  /** Per hold-out split: the training moments and target, and the held-out rows with their weight. */
  readonly splits: readonly { train: Moments; target: Rows; test: TensorLike; weight: number }[]
}

function pathContext(x: TensorLike, o: ShrinkagePathOptions): PathContext {
  const full = moments(x, 'shrinkagePath', { assumeCentered: o.assumeCentered ?? false })
  const target = o.target ?? 'constant-correlation'
  const grid = [...(o.intensities ?? Array.from({ length: 21 }, (_, i) => i / 20))]
  if (grid.length === 0 || grid.some((v) => !(v >= 0 && v <= 1)))
    throw new OmaiError('value', 'shrinkagePath: intensities must be a non-empty list within [0, 1]')
  const rows = Array.from({ length: full.n }, (_, t) => Array.from(full.centred.subarray(t * full.d, (t + 1) * full.d)))
  const raw = rows.map((r) => r.map((v, j) => v + (full.location[j] ?? 0)))
  const splits: PathContext['splits'][number][] = []
  if (o.heldOut !== undefined) {
    splits.push({ train: full, target: buildTarget(full, target), test: o.heldOut, weight: 1 })
  } else {
    const folds = Math.min(o.folds ?? 5, full.n)
    if (folds >= 2) {
      for (let k = 0; k < folds; k++) {
        const train = raw.filter((_, t) => t % folds !== k)
        const test = raw.filter((_, t) => t % folds === k)
        if (train.length < 2 || test.length === 0) continue
        const mt = moments(train, 'shrinkagePath', { assumeCentered: o.assumeCentered ?? false })
        splits.push({ train: mt, target: buildTarget(mt, target), test, weight: test.length })
      }
    }
  }
  return { full, target: buildTarget(full, target), grid, splits }
}

function pathPoint(ctx: PathContext, intensity: number): ShrinkagePathPoint {
  const cov = mix(ctx.full.covariance, ctx.target, intensity)
  const eigenvalues = symmetricEigen(cov).values
  let heldOut = Number.NaN
  if (ctx.splits.length > 0) {
    let total = 0
    let weight = 0
    for (const sp of ctx.splits) {
      total +=
        sp.weight * gaussianLogLikelihood(sp.test, sp.train.location, mix(sp.train.covariance, sp.target, intensity))
      weight += sp.weight
    }
    heldOut = total / weight
  }
  return {
    intensity,
    eigenvalues,
    conditionNumber: conditionOf(eigenvalues),
    spread: (eigenvalues[0] ?? 0) - (eigenvalues[eigenvalues.length - 1] ?? 0),
    heldOut,
  }
}

/**
 * The shrinkage path as a stepped {@link Algorithm}: each step moves the intensity along the grid from the sample
 * covariance (0) to the target (1) and records the condition number, the eigenvalue spread, the smallest eigenvalue and the
 * held-out Gaussian log likelihood (of the given rows, or cross-validated), with the spectrum on the tape. It is the
 * figure of shrinkage as a slider: the eigenvalues bending towards the target, the condition number falling and the
 * likelihood rising and then falling again.
 *
 * @param x - Observations `[n, d]`.
 * @param options - The target, the intensities, the hold-out and the centring.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { shrinkagePathAlgorithm } from 'omai-engine/covariance'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1]]
 * runAlgorithm(shrinkagePathAlgorithm(x, { intensities: [0, 0.5, 1], folds: 2 }), { maxSteps: 10 }).state.points.length // => 3
 * ```
 */
export function shrinkagePathAlgorithm(
  x: TensorLike,
  options: ShrinkagePathOptions = {},
): Algorithm<ShrinkagePathState> {
  const ctx = pathContext(x, options)
  const d = ctx.full.d
  return {
    name: 'shrinkage-path',
    config: { target: options.target ?? 'constant-correlation', points: ctx.grid.length },
    schema: channels({
      metrics: {
        intensity: { description: 'the shrinkage intensity towards the target' },
        conditionNumber: { description: 'condition number of the shrunk covariance', direction: 'lower' },
        spread: { description: 'largest minus smallest eigenvalue' },
        minEigenvalue: { description: 'the smallest eigenvalue' },
        heldOut: { description: 'mean held-out Gaussian log likelihood', direction: 'higher' },
      },
      tensors: {
        eigenvalues: {
          role: 'state',
          description: 'the eigenvalues of the shrunk covariance, descending',
          shape: ['d'],
        },
      },
    }),
    init: () => ({ index: 0, points: [] }),
    step: (s) => ({
      index: s.index + 1,
      points: [...s.points, pathPoint(ctx, ctx.grid[s.index] ?? 1)],
    }),
    done: (s) => (s.index >= ctx.grid.length ? 'completed' : undefined),
    observe: (s) => {
      const p = s.points[s.points.length - 1]
      if (p === undefined) return { metrics: {}, tensors: {} }
      return {
        metrics: {
          intensity: p.intensity,
          spread: p.spread,
          minEigenvalue: p.eigenvalues[d - 1] ?? 0,
          ...(Number.isFinite(p.conditionNumber) ? { conditionNumber: p.conditionNumber } : {}),
          ...(Number.isFinite(p.heldOut) ? { heldOut: p.heldOut } : {}),
        },
        tensors: { eigenvalues: { role: 'state', value: p.eigenvalues, shape: [d] } },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ShrinkagePathState,
  }
}

/**
 * The shrinkage path of a sample: for each intensity from 0 to 1, the eigenvalues of the shrunk covariance, its condition
 * number, the eigenvalue spread and the held-out Gaussian log likelihood (of the given rows, or cross-validated over
 * folds). {@link shrinkagePathAlgorithm} run to its end.
 *
 * @param x - Observations `[n, d]`.
 * @param options - The target, the intensities, the hold-out and the centring.
 * @returns One point per intensity.
 * @index function Shrinkage path: condition number, eigenvalue spread and held-out likelihood against intensity.
 * @example
 * ```ts
 * import { shrinkagePath } from 'omai-engine/covariance'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2], [4, 5], [0, 1]]
 * shrinkagePath(x, { intensities: [0, 1], folds: 2 }).map((p) => p.intensity) // => [0, 1]
 * ```
 */
export function shrinkagePath(x: TensorLike, options: ShrinkagePathOptions = {}): ShrinkagePathPoint[] {
  const out = runAlgorithm(shrinkagePathAlgorithm(x, options), { maxSteps: (options.intensities?.length ?? 21) + 1 })
  return out.state.points
}

/** Hyperparameters of {@link targetShrinkageEstimator}. */
export interface TargetShrinkageHyper {
  /** The target (default `'constant-correlation'`). */
  readonly target?: ShrinkageTarget
  /** A fixed intensity in `[0, 1]`, used when `rule` is `'fixed'` (default 0.5). */
  readonly intensity?: number
  /** The intensity rule: `'analytic'` (Ledoit-Wolf), `'oas'` or `'rblw'` (identity target only), or `'fixed'` (default `'analytic'`). */
  readonly rule?: 'analytic' | 'oas' | 'rblw' | 'fixed'
  /** Whether the data are already centred (default false). */
  readonly assumeCentered?: boolean
}

/** The state of a target shrinkage fit. */
export interface TargetShrinkageState {
  readonly location: number[]
  readonly covariance: number[][]
  readonly precision: number[][]
  readonly shrinkage: number
  readonly logdet: number
  readonly done: boolean
}

/** The fitted parameters of the target shrinkage estimator (those of any covariance estimator). */
export type TargetShrinkageParams = CovarianceParams

function shrinkageState(x: TensorLike, h: Required<TargetShrinkageHyper>, done: boolean): TargetShrinkageState {
  const r = targetShrinkage(x, {
    target: h.target,
    intensity: h.rule === 'fixed' ? h.intensity : h.rule,
    assumeCentered: h.assumeCentered,
  })
  const precision = toArray(linalg.pinv(r.covariance)) as number[][]
  const positive = symmetricEigen(r.covariance).values.filter((v) => v > 1e-12)
  return {
    location: r.location,
    covariance: r.covariance,
    precision,
    shrinkage: r.shrinkage,
    logdet: positive.reduce((a, v) => a + Math.log(v), 0),
    done,
  }
}

function resolveHyper(h: TargetShrinkageHyper): Required<TargetShrinkageHyper> {
  return {
    target: h.target ?? 'constant-correlation',
    intensity: h.intensity ?? 0.5,
    rule: h.rule ?? 'analytic',
    assumeCentered: h.assumeCentered ?? false,
  }
}

/**
 * Shrinkage towards a target as a one-step {@link Algorithm} (the intensity from its rule, the covariance, the precision
 * and the log-determinant on the tape); see {@link shrinkagePathAlgorithm} for the stepped path over intensities.
 *
 * @param x - Observations `[n, d]`.
 * @param options - The target, the intensity rule and the centring.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { targetShrinkageAlgorithm } from 'omai-engine/covariance'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2]]
 * runAlgorithm(targetShrinkageAlgorithm(x, { target: 'diagonal', rule: 'fixed', intensity: 1 }), { maxSteps: 2 }).state.covariance[0]?.[1] // => 0
 * ```
 */
export function targetShrinkageAlgorithm(
  x: TensorLike,
  options: TargetShrinkageHyper = {},
): Algorithm<TargetShrinkageState> {
  const h = resolveHyper(options)
  const d = moments(x, 'targetShrinkageAlgorithm').d
  return {
    name: `shrinkage-${h.target}`,
    config: { target: h.target, rule: h.rule, intensity: h.intensity },
    schema: channels({
      metrics: {
        shrinkage: { description: 'the shrinkage intensity towards the target' },
        logdet: { description: 'log-determinant of the estimated covariance matrix' },
      },
      tensors: {
        covariance: { role: 'param', description: 'the shrunk covariance matrix', shape: ['d', 'd'] },
        precision: { role: 'param', description: 'the pseudo-inverse precision matrix', shape: ['d', 'd'] },
      },
    }),
    init: () => shrinkageState(x, h, false),
    step: (s) => ({ ...s, done: true }),
    done: (s) => (s.done ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { shrinkage: s.shrinkage, ...(Number.isFinite(s.logdet) ? { logdet: s.logdet } : {}) },
      tensors: {
        covariance: { role: 'param', value: s.covariance.flat(), shape: [d, d] },
        precision: { role: 'param', value: s.precision.flat(), shape: [d, d] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TargetShrinkageState,
  }
}

function pathRecord(data: LearnData, h: Required<TargetShrinkageHyper>, chosen: number): Exposition {
  const path = shrinkagePath(data.x, { target: h.target, assumeCentered: h.assumeCentered })
  const finite = path.filter((p) => Number.isFinite(p.heldOut))
  const wellConditioned = path.filter((p) => Number.isFinite(p.conditionNumber))
  return curveSet({
    xLabel: 'shrinkage intensity',
    yLabel: 'held-out log likelihood',
    y2Label: 'log10 condition number',
    series: [
      { name: 'held-out log likelihood', x: finite.map((p) => p.intensity), y: finite.map((p) => p.heldOut) },
      {
        name: 'log10 condition number',
        x: wellConditioned.map((p) => p.intensity),
        y: wellConditioned.map((p) => Math.log10(p.conditionNumber)),
        axis: 'right',
      },
    ],
    marks: [{ x: chosen, label: h.rule === 'fixed' ? 'chosen' : h.rule }],
  })
}

/**
 * Covariance shrunk towards a diagonal, constant-correlation, single-factor or identity target with the analytic
 * Ledoit-Wolf intensity (or the OAS, Rao-Blackwell or a fixed one), as a covariance estimator on the iterative base:
 * location, covariance, precision (pseudo-inverse), the intensity, `mahalanobis` and `score`. The exposition is the
 * cross-validated held-out log likelihood and the condition number against the intensity with the chosen one marked,
 * and the Mahalanobis anomaly scores.
 *
 * @param hyper - The target, the intensity rule and the centring.
 * @returns The iterative estimator.
 * @index estimator Covariance shrunk to a diagonal, constant-correlation, factor or identity target.
 * @example
 * ```ts
 * import { targetShrinkageEstimator } from 'omai-engine/covariance'
 * const x = [[1, 2], [3, 6], [5, 4], [2, 2]]
 * targetShrinkageEstimator({ target: 'diagonal', rule: 'fixed', intensity: 1 }).fit({ x }).params.covariance[0]?.[1] // => 0
 * ```
 */
export function targetShrinkageEstimator(
  hyper: TargetShrinkageHyper = {},
): IterativeEstimator<Required<TargetShrinkageHyper>, TargetShrinkageState, TargetShrinkageParams, CovarianceVerbs> {
  return iterative<Required<TargetShrinkageHyper>, TargetShrinkageState, TargetShrinkageParams, CovarianceVerbs>({
    name: 'target-shrinkage',
    kind: 'density',
    defaults: resolveHyper(hyper),
    hyperSpace: (h) =>
      dict({
        target: choice(['identity', 'diagonal', 'constant-correlation', 'single-factor'] as const, {
          label: 'target',
          description: 'What the sample covariance is shrunk towards.',
          default: h.target,
        }),
        rule: choice(['analytic', 'oas', 'rblw', 'fixed'] as const, {
          label: 'intensity rule',
          description: 'The analytic Ledoit-Wolf intensity, OAS or Rao-Blackwell (identity target), or a fixed value.',
          default: h.rule,
        }),
        intensity: real(0, 1, {
          label: 'intensity',
          description: 'The fixed intensity, used when the rule is fixed.',
          default: h.intensity,
        }),
        assumeCentered: bool({
          label: 'assume centred',
          description: 'Treat the data as already centred (do not subtract the mean).',
          default: h.assumeCentered,
        }),
      }),
    algorithm: (data, h) => targetShrinkageAlgorithm(data.x, h),
    maxSteps: () => 1,
    params: (s) => ({ location: s.location, covariance: s.covariance, precision: s.precision, shrinkage: s.shrinkage }),
    verbs: covarianceVerbs,
    records: (s, data, h) => [
      pathRecord(data, h, s.shrinkage),
      mahalanobisRecord('targetShrinkage', data.x, {
        location: s.location,
        covariance: s.covariance,
        precision: s.precision,
        shrinkage: s.shrinkage,
      }),
    ],
    logs: { 'solver/shrinkage': Log() },
  })
}

/**
 * The target shrinkage estimator as a scikit-learn-style class wrapper.
 *
 * @example
 * ```ts
 * import { TargetShrinkageEstimator } from 'omai-engine/covariance'
 * new TargetShrinkageEstimator({ target: 'diagonal', rule: 'fixed', intensity: 1 }).fit([[1, 2], [3, 6], [5, 4], [2, 2]]).model.params.shrinkage // => 1
 * ```
 */
export class TargetShrinkageEstimator extends ExplainedEstimatorClass<
  Required<TargetShrinkageHyper>,
  Fitted<TargetShrinkageParams> & CovarianceVerbs
> {
  /**
   * Make an unfitted target shrinkage model.
   *
   * @param hyper - The hyperparameters of {@link targetShrinkageEstimator}.
   */
  constructor(hyper: TargetShrinkageHyper = {}) {
    super(targetShrinkageEstimator(hyper))
  }
}
