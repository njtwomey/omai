import { WhiteboxError } from '../../base/index.ts'
import { categorical, type Key, uniform } from '../../random/index.ts'
import { choice, dict, real } from '../../space/index.ts'
import { add, exp, fromStorage, matmul, mul, neg, sub, type Tensor, tensor, type Value } from '../../tensor/index.ts'
import type { Generated } from '../generator.ts'
import type { Modifier } from '../recipes.ts'
import { stratifiedIndices, take } from '../rows.ts'
import { type Dataset, rowCount } from '../schema/common.ts'
import type { TabularData } from '../schema/tabular.ts'
import { columnStats, logistic, requireTabular } from './_shared.ts'

/** A standardised projection `z(x) = (x - mean) . direction / sd` of the rows of a feature matrix. */
function projection(
  x: Tensor,
  direction: readonly number[] | undefined,
): { z: Float64Array; score: (v: Value) => Value } {
  const [n = 0, d = 1] = x.shape
  const dir = direction ?? Array.from({ length: d }, (_, j) => (j === 0 ? 1 : 0))
  if (dir.length !== d) throw new WhiteboxError('value', `shift: the direction needs ${d} entries`)
  const { mean } = columnStats(x)
  const raw = Float64Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let j = 0; j < d; j++) s += ((x.data[i * d + j] ?? 0) - (mean[j] ?? 0)) * (dir[j] ?? 0)
    return s
  })
  let ss = 0
  for (const v of raw) ss += v * v
  const sd = Math.sqrt(ss / Math.max(1, n)) || 1
  const w = tensor(dir.map((v) => v / sd))
  const offset = dir.reduce((s, v, j) => s + v * (mean[j] ?? 0), 0) / sd
  return { z: raw.map((v) => v / sd), score: (v) => sub(matmul(v, w), offset) }
}

/** Options of {@link withCovariateShift}. */
export interface CovariateShiftOptions {
  /** The direction of the shift in feature space (default the first feature). */
  readonly direction?: readonly number[]
  /** The log-odds of selection per standard deviation along the direction (default 1). */
  readonly strength?: number
  /** Rows of the shifted sample (default the input size). */
  readonly n?: number
  readonly key: Key
}

/**
 * Covariate shift by biased selection: rows are resampled with probability proportional to `exp(strength z(x))`,
 * where `z` is the standardised projection on `direction`, so `p(x)` changes while `p(y | x)` does not. The truth
 * keeps the regression function and posterior, tilts the log density, and gives the density ratio
 * `w(x) = p_target(x) / p_source(x)` for importance weighting.
 *
 * @param input - Data with a feature matrix and its truth.
 * @param options - Direction, strength, size and key.
 * @returns The shifted sample; `truth.extra.densityRatio` maps features to `w(x)`.
 * @example
 * ```ts
 * import { asGenerated, withCovariateShift, xy } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * const d = asGenerated(xy(Array.from({ length: 100 }, (_, i) => [i]), Array.from({ length: 100 }, (_, i) => i)))
 * const shifted = withCovariateShift(d, { strength: 3, key: key(0) })
 * Array.from(shifted.data.x.data).reduce((a, b) => a + b) / 100 > 60 // => true
 * ```
 */
export function withCovariateShift<D extends TabularData>(
  input: Generated<D>,
  options: CovariateShiftOptions,
): Generated<D> {
  const { data, truth } = requireTabular(input, 'withCovariateShift') as Generated<D>
  const strength = options.strength ?? 1
  const { z, score } = projection(data.x, options.direction)
  const weights = Array.from(z, (v) => Math.exp(strength * v))
  const normaliser = weights.reduce((a, b) => a + b, 0) / Math.max(1, weights.length)
  const n = options.n ?? rowCount(data)
  const rows = Array.from(categorical(options.key, weights, [n]).data)
  const out = take(data, rows)
  const logNorm = Math.log(normaliser)
  const tilt = (v: Value): Value => sub(mul(score(v), strength), logNorm)
  const logProb = truth.logProb
  return {
    data: out,
    truth: {
      ...truth,
      ...(logProb === undefined ? {} : { logProb: (v: Value, y?: Value) => add(logProb(v, y), tilt(v)) }),
      latents: { ...(truth.latents ?? {}), sourceRows: rows },
      extra: {
        ...(truth.extra ?? {}),
        covariateShift: { direction: options.direction, strength },
        densityRatio: (v: Value): Value => exp(tilt(v)),
      },
    },
  }
}

/** Options of {@link withConceptDrift}. */
export interface ConceptDriftOptions {
  /** Drift points as row indices or fractions of the rows (default `[0.5]`). Rows are in time order. */
  readonly at?: readonly number[]
  /** `sudden` (default): switch at each point; `gradual`: the new concept's probability rises linearly over `width`
   * rows; `recurring`: concepts alternate at each point. */
  readonly kind?: 'sudden' | 'gradual' | 'recurring'
  /** Rows over which a gradual drift happens (default a tenth of the rows). */
  readonly width?: number
  readonly key: Key
}

/**
 * Concept drift: the relation between features and target changes at known points of a time-ordered dataset. Under
 * the drifted concept, class labels are cycled (`y -> (y + 1) mod k`) and regression targets are negated, so
 * `p(y | x)` changes while `p(x)` does not. The truth records the drift points as changepoints and the concept of
 * every row; the regression function and posterior describe the original concept, and
 * `truth.extra.driftedRegressionFunction` or `driftedPosterior` the drifted one.
 *
 * @param input - Data with a feature matrix and its truth.
 * @param options - Drift points, kind, width and key.
 * @returns The drifting data; `truth.changepoints` and `truth.latents.concepts`.
 * @example
 * ```ts
 * import { asGenerated, withConceptDrift, xyClasses } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * const d = withConceptDrift(asGenerated(xyClasses([[0], [1], [2], [3]], [0, 0, 0, 0], { classes: ['a', 'b'] })), { key: key(0) })
 * Array.from(d.data.y.data) // => [0, 0, 1, 1]
 * ```
 */
export function withConceptDrift<D extends TabularData>(
  input: Generated<D>,
  options: ConceptDriftOptions,
): Generated<D> {
  const { data, truth } = requireTabular(input, 'withConceptDrift') as Generated<D>
  const n = rowCount(data)
  const points = (options.at ?? [0.5]).map((p) => (p < 1 ? Math.round(p * n) : Math.round(p))).sort((a, b) => a - b)
  const kind = options.kind ?? 'sudden'
  const width = options.width ?? Math.max(1, Math.round(n / 10))
  const u = uniform(options.key, [n]).data
  const concepts = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    let passed = 0
    for (const p of points) if (i >= p) passed++
    if (kind === 'gradual') {
      const p = points[0] ?? n
      const prob = Math.min(1, Math.max(0, (i - (p - width / 2)) / width))
      concepts[i] = (u[i] ?? 1) < prob ? 1 : 0
    } else if (kind === 'recurring') concepts[i] = passed % 2
    else concepts[i] = passed > 0 ? 1 : 0
  }
  let y: Tensor
  const extra: Record<string, unknown> = {}
  if (data.kind === 'classification') {
    const k = data.classes.length
    y = fromStorage(
      Int32Array.from(data.y.data, (c, i) => (concepts[i] === 1 ? (c + 1) % k : c)),
      [n],
      'int32',
    )
    const post = truth.posterior
    if (post !== undefined) {
      const cycle = Array.from({ length: k }, (_, i) =>
        Array.from({ length: k }, (_, j) => (j === (i + 1) % k ? 1 : 0)),
      )
      const c = tensor(cycle)
      extra['driftedPosterior'] = (v: Value) => matmul(post(v), c)
    }
  } else if (data.kind === 'regression') {
    const k = (data.y.data.length / Math.max(1, n)) | 0
    y = fromStorage(
      Float64Array.from(data.y.data as ArrayLike<number>, (v, i) => (concepts[Math.floor(i / k)] === 1 ? -v : v)),
      data.y.shape,
      'float64',
    )
    const f = truth.regressionFunction
    if (f !== undefined) extra['driftedRegressionFunction'] = (v: Value) => neg(f(v))
  } else {
    throw new WhiteboxError('value', `withConceptDrift: needs classification or regression data, got ${data.kind}`)
  }
  return {
    data: { ...data, y },
    truth: {
      ...truth,
      changepoints: points,
      latents: { ...(truth.latents ?? {}), concepts: fromStorage(concepts, [n], 'int32') },
      extra: { ...(truth.extra ?? {}), ...extra, drift: { kind, points, width } },
    },
  }
}

/** Options of {@link shiftedSplit}. */
export interface ShiftedSplitOptions {
  /** The direction to split along (default the first feature). */
  readonly direction?: readonly number[]
  /** The fraction of rows in the training part (default 0.5). */
  readonly trainFraction?: number
  /** `threshold` (default): train below the quantile, test above; `soft`: rows join the test part with probability
   * `logistic(strength z)`, so the parts overlap. */
  readonly mode?: 'threshold' | 'soft'
  readonly strength?: number
  /** Needed for the soft mode. */
  readonly key?: Key
}

/**
 * A domain-shift split: the training and test parts differ in their feature distribution along a direction (the
 * standard "split by a covariate" benchmark for extrapolation and domain adaptation).
 *
 * @param data - Data with a feature matrix.
 * @param options - Direction, training fraction, mode, strength and key.
 * @returns `{ train, test, trainIndices, testIndices }`.
 * @throws WhiteboxError with code `value` for the soft mode without a key.
 * @example
 * ```ts
 * import { shiftedSplit, xy } from 'whitebox-engine/data'
 * const parts = shiftedSplit(xy([[3], [1], [2], [0]], [3, 1, 2, 0]))
 * Array.from(parts.test.x.data).sort() // => [2, 3]
 * ```
 */
export function shiftedSplit<D extends TabularData>(
  data: D,
  options: ShiftedSplitOptions = {},
): { readonly train: D; readonly test: D; readonly trainIndices: number[]; readonly testIndices: number[] } {
  const { z } = projection(data.x, options.direction)
  const n = z.length
  const frac = options.trainFraction ?? 0.5
  let train: number[] = []
  let test: number[] = []
  if (options.mode === 'soft') {
    if (options.key === undefined) throw new WhiteboxError('value', 'shiftedSplit: the soft mode needs a key')
    const u = uniform(options.key, [n]).data
    const strength = options.strength ?? 2
    const order = Array.from(z).sort((a, b) => a - b)
    const cut = order[Math.min(n - 1, Math.max(0, Math.round(frac * n) - 1))] ?? 0
    for (let i = 0; i < n; i++) ((u[i] ?? 0) < logistic(strength * ((z[i] ?? 0) - cut)) ? test : train).push(i)
  } else {
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (z[a] ?? 0) - (z[b] ?? 0) || a - b)
    const m = Math.round(frac * n)
    train = order.slice(0, m).sort((a, b) => a - b)
    test = order.slice(m).sort((a, b) => a - b)
  }
  return { train: take(data, train), test: take(data, test), trainIndices: train, testIndices: test }
}

/** Options of {@link semiSupervisedSplit}. */
export interface SemiSupervisedOptions {
  /** Labelled rows: a fraction or a count (default 0.1). */
  readonly labelled?: number
  /** Stratify the labelled rows by class (default true for classification). */
  readonly stratify?: boolean
  readonly key: Key
}

/**
 * Hide most labels for semi-supervised learning: a labelled part, and an unlabelled part whose targets are replaced
 * by `-1` (classes, sklearn's convention) or NaN (regression), with the hidden targets returned alongside.
 *
 * @param data - Classification or regression data.
 * @param options - Labelled size, stratification and key.
 * @returns `{ labelled, unlabelled, hidden, labelledIndices, unlabelledIndices }`.
 * @throws WhiteboxError with code `value` for other kinds of data.
 * @example
 * ```ts
 * import { semiSupervisedSplit, xyClasses } from 'whitebox-engine/data'
 * import { key } from 'whitebox-engine/random'
 * const s = semiSupervisedSplit(xyClasses([[0], [1], [2], [3]], [0, 1, 0, 1]), { labelled: 2, key: key(0) })
 * Array.from(s.unlabelled.y.data) // => [-1, -1]
 * ```
 */
export function semiSupervisedSplit<D extends TabularData & Dataset>(
  data: D,
  options: SemiSupervisedOptions,
): {
  readonly labelled: D
  readonly unlabelled: D
  readonly hidden: Tensor
  readonly labelledIndices: number[]
  readonly unlabelledIndices: number[]
} {
  if (data.kind !== 'classification' && data.kind !== 'regression')
    throw new WhiteboxError('value', `semiSupervisedSplit: needs classification or regression data, got ${data.kind}`)
  const n = rowCount(data)
  const size = options.labelled ?? 0.1
  const m = size < 1 ? Math.ceil(size * n) : Math.round(size)
  let lab: number[]
  let unl: number[]
  if (data.kind === 'classification' && options.stratify !== false) {
    const s = stratifiedIndices(Array.from(data.y.data), m, options.key)
    lab = s.test
    unl = s.train
  } else {
    const order = Array.from(uniform(options.key, [n]).data, (v, i) => [v, i] as const).sort((a, b) => a[0] - b[0])
    lab = order.slice(0, m).map((p) => p[1])
    unl = order.slice(m).map((p) => p[1])
  }
  const labelled = take(data, lab)
  const rest = take(data, unl)
  const hidden = rest.y
  const blank =
    data.kind === 'classification'
      ? fromStorage(new Int32Array(unl.length).fill(-1), [unl.length], 'int32')
      : fromStorage(new Float64Array(hidden.data.length).fill(Number.NaN), hidden.shape, 'float64')
  return { labelled, unlabelled: { ...rest, y: blank }, hidden, labelledIndices: lab, unlabelledIndices: unl }
}

/**
 * {@link withCovariateShift} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { covariateShiftModifier } from 'whitebox-engine/data'
 * covariateShiftModifier.name // => 'covariateShift'
 * ```
 */
export const covariateShiftModifier: Modifier = {
  name: 'covariateShift',
  description: 'Resample rows with a selection bias along the first feature.',
  knobs: dict({ strength: real(-10, 10, { default: 1, label: 'selection strength' }) }),
  apply: (key, input, knobs) =>
    withCovariateShift(input as Generated<TabularData>, { strength: knobs['strength'] as number, key }),
}

/**
 * {@link withConceptDrift} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { conceptDriftModifier } from 'whitebox-engine/data'
 * conceptDriftModifier.name // => 'conceptDrift'
 * ```
 */
export const conceptDriftModifier: Modifier = {
  name: 'conceptDrift',
  description: 'Change the target relation at the middle of the rows.',
  knobs: dict({
    at: real(0.01, 0.99, { default: 0.5, label: 'drift point (fraction)' }),
    kind: choice(['sudden', 'gradual', 'recurring'], { default: 'sudden' }),
  }),
  apply: (key, input, knobs) =>
    withConceptDrift(input as Generated<TabularData>, {
      at: [knobs['at'] as number],
      kind: knobs['kind'] as 'sudden' | 'gradual' | 'recurring',
      key,
    }),
}
