/**
 * Distribution shift and partial labelling: covariate shift by biased selection along a direction, concept drift at
 * known points of a time-ordered dataset, a domain-shift train-test split for extrapolation and adaptation studies, and
 * a semi-supervised split that hides most labels (as `-1` for classes, sklearn's convention, or NaN for regression).
 *
 * @index function Covariate shift, concept drift and partial labels.
 */

import { MdmlError, sigmoidOf } from '../../base/index.ts'
import { categorical, type Key, normal, uniform } from '../../random/index.ts'
import { choice, dict, real } from '../../space/index.ts'
import {
  add,
  exp,
  fromStorage,
  greater,
  matmul,
  mul,
  neg,
  sliceAxis,
  squeeze,
  sub,
  type Tensor,
  tensor,
  type Value,
  where,
} from '../../tensor/index.ts'
import type { Generated } from '../generator.ts'
import type { Modifier } from '../recipes.ts'
import { splitSize, stratifiedIndices, take } from '../rows.ts'
import { type Dataset, rowCount } from '../schema/common.ts'
import type { ClassificationData, TabularData } from '../schema/tabular.ts'
import {
  columnStats,
  composedFunctions,
  requireClassification,
  requireTabular,
  withoutProgram,
} from './_shared.ts'

/** A standardised projection `z(x) = (x - mean) . direction / sd` of the rows of a feature matrix. */
function projection(
  x: Tensor,
  direction: readonly number[] | undefined,
): { z: Float64Array; score: (v: Value) => Value } {
  const [n = 0, d = 1] = x.shape
  const dir = direction ?? Array.from({ length: d }, (_, j) => (j === 0 ? 1 : 0))
  if (dir.length !== d) throw new MdmlError('value', `shift: the direction needs ${d} entries`)
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
 * import { asGenerated, withCovariateShift, xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
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
      ...withoutProgram(truth),
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
 * import { asGenerated, withConceptDrift, xyClasses } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
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
    throw new MdmlError('value', `withConceptDrift: needs classification or regression data, got ${data.kind}`)
  }
  return {
    data: { ...data, y },
    truth: {
      ...withoutProgram(truth),
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
 * @throws MdmlError with code `value` for the soft mode without a key.
 * @example
 * ```ts
 * import { shiftedSplit, xy } from 'mdml-engine/data'
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
    if (options.key === undefined) throw new MdmlError('value', 'shiftedSplit: the soft mode needs a key')
    const u = uniform(options.key, [n]).data
    const strength = options.strength ?? 2
    const order = Array.from(z).sort((a, b) => a - b)
    const cut = order[Math.min(n - 1, Math.max(0, Math.round(frac * n) - 1))] ?? 0
    for (let i = 0; i < n; i++) ((u[i] ?? 0) < sigmoidOf(strength * ((z[i] ?? 0) - cut)) ? test : train).push(i)
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
 * @throws MdmlError with code `value` for other kinds of data.
 * @example
 * ```ts
 * import { semiSupervisedSplit, xyClasses } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
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
    throw new MdmlError('value', `semiSupervisedSplit: needs classification or regression data, got ${data.kind}`)
  const n = rowCount(data)
  const size = options.labelled ?? 0.1
  const m = splitSize(size, n, 'semiSupervisedSplit')
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

/** Options of {@link withSubpopulationShift}. */
export interface SubpopulationShiftOptions {
  /** The feature that defines the groups (default 0): the group is the rows above `threshold`. */
  readonly column?: number
  /** The cut (default the column's median). */
  readonly threshold?: number
  /** The chance that a row of the group is kept in the source (default 0.1; other rows are all kept). */
  readonly keep?: number
  readonly key: Key
}

/**
 * Subpopulation shift: a group (the rows above a cut of one feature) is under-represented in the source, so a model
 * trained on it does worse on that group in the target, which is the unshifted data. `p(y | x)` is unchanged; `p(x)`
 * changes by the group's weight alone.
 *
 * @param input - Data with a feature matrix and its truth.
 * @param options - Group definition, retention of the group and key.
 * @returns The source sample; `truth.latents.group` `[n]` marks the group's rows, and `truth.extra.densityRatio`
 *   maps features to `p_target(x) / p_source(x)` (constant on each side of the cut), with `truth.extra.subpopulation`
 *   recording the column, cut and retention.
 * @example
 * ```ts
 * import { asGenerated, withSubpopulationShift, xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = asGenerated(xy([[0], [1], [2], [3], [4], [5]], [0, 1, 2, 3, 4, 5]))
 * withSubpopulationShift(d, { keep: 0, key: key(0) }).data.x.shape // => [3, 1]
 * ```
 */
export function withSubpopulationShift<D extends TabularData>(
  input: Generated<D>,
  options: SubpopulationShiftOptions,
): Generated<D> {
  const { data, truth } = requireTabular(input, 'withSubpopulationShift') as Generated<D>
  const [n = 0, d = 1] = data.x.shape
  const column = options.column ?? 0
  if (!(Number.isInteger(column) && column >= 0 && column < d))
    throw new MdmlError('value', `withSubpopulationShift: no column ${column} in ${d}`)
  const values = Array.from({ length: n }, (_, i) => data.x.data[i * d + column] ?? 0)
  const sorted = [...values].sort((a, b) => a - b)
  const threshold = options.threshold ?? ((sorted[Math.floor((n - 1) / 2)] ?? 0) + (sorted[Math.floor(n / 2)] ?? 0)) / 2
  const keep = options.keep ?? 0.1
  const u = uniform(options.key, [n]).data
  const rows: number[] = []
  const inGroup: number[] = []
  let groupRows = 0
  values.forEach((v, i) => {
    const g = v > threshold
    if (g) groupRows++
    if (!g || (u[i] ?? 1) < keep) {
      rows.push(i)
      inGroup.push(g ? 1 : 0)
    }
  })
  const share = groupRows / Math.max(1, n)
  const z = 1 - share + keep * share
  const ratioGroup = keep > 0 ? z / keep : Number.POSITIVE_INFINITY
  const out = take(data, rows)
  const latents = { ...(truth.latents ?? {}), group: fromStorage(Uint8Array.from(inGroup), [rows.length], 'bool') }
  return {
    data: out,
    truth: {
      ...withoutProgram(truth),
      latents,
      extra: {
        ...(truth.extra ?? {}),
        subpopulation: { column, threshold, keep, targetShare: share, sourceShare: (keep * share) / z },
        densityRatio: (x: Value): Value =>
          where(greater(squeeze(sliceAxis(x, 1, column, column + 1), 1), threshold), ratioGroup, z),
      },
    },
  }
}

/** Options of {@link withShortcut}. */
export interface ShortcutOptions {
  /** `source` (default): the shortcut agrees with the label with probability `agreement`; `target`: it is independent of the label. */
  readonly mode?: 'source' | 'target'
  /** The chance that the shortcut equals the label in the source (default 0.95). */
  readonly agreement?: number
  /** Standard deviation of Gaussian noise added to the shortcut value (default 0.1). */
  readonly noise?: number
  readonly key: Key
}

/**
 * Plant a shortcut: append a feature that tracks the class label (its value is the class index, with noise) and
 * agrees with it with probability `agreement` in the source, but is independent of the label in the target. A model
 * that relies on it does well at the source and falls to chance-level reliance at the target. The truth's posterior is
 * that of the honest problem, which ignores the shortcut column.
 *
 * @param input - Classification data with its truth.
 * @param options - Mode, agreement, noise and key.
 * @returns The data with a `shortcut` column; `truth.latents.shortcutAgrees` `[n]` marks the rows where it equals the
 *   label, and `truth.extra.shortcut` records the column, mode and agreement.
 * @example
 * ```ts
 * import { asGenerated, withShortcut, xyClasses } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = asGenerated(xyClasses([[0], [1], [2], [3]], [0, 1, 0, 1]))
 * withShortcut(d, { agreement: 1, noise: 0, key: key(0) }).data.featureNames // => ['x0', 'shortcut']
 * ```
 */
export function withShortcut<D extends ClassificationData>(
  input: Generated<D>,
  options: ShortcutOptions,
): Generated<D> {
  const { data, truth } = requireClassification(input, 'withShortcut') as Generated<D>
  const [n = 0, d = 1] = data.x.shape
  const k = data.classes.length
  const mode = options.mode ?? 'source'
  const agreement = mode === 'source' ? (options.agreement ?? 0.95) : 1 / k
  const u = uniform(options.key, [n, 2]).data
  const z = normal(options.key, [n]).data
  const noise = options.noise ?? 0.1
  const agrees = new Uint8Array(n)
  const wide = new Float64Array(n * (d + 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) wide[i * (d + 1) + j] = data.x.data[i * d + j] ?? 0
    const label = data.y.data[i] ?? 0
    let value = label
    if (mode === 'target') value = Math.min(k - 1, Math.floor((u[i * 2] ?? 0) * k))
    else if ((u[i * 2] ?? 1) >= agreement && k > 1) value = (label + 1 + Math.floor((u[i * 2 + 1] ?? 0) * (k - 1))) % k
    agrees[i] = value === label ? 1 : 0
    wide[i * (d + 1) + d] = value + noise * (z[i] ?? 0)
  }
  const original = (v: Value): Value => sliceAxis(v, 1, 0, d)
  const { logProb: _logProb, ...rest } = withoutProgram(truth)
  return {
    data: { ...data, x: fromStorage(wide, [n, d + 1], 'float64'), featureNames: [...data.featureNames, 'shortcut'] },
    truth: {
      ...rest,
      ...composedFunctions(truth, original),
      relevantFeatures: truth.relevantFeatures ?? Array.from({ length: d }, (_, j) => j),
      latents: { ...(truth.latents ?? {}), shortcutAgrees: fromStorage(agrees, [n], 'bool') },
      extra: { ...(truth.extra ?? {}), shortcut: { column: d, mode, agreement } },
    },
  }
}

/**
 * {@link withCovariateShift} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { covariateShiftModifier } from 'mdml-engine/data'
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
 * import { conceptDriftModifier } from 'mdml-engine/data'
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

/**
 * {@link withSubpopulationShift} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { subpopulationShiftModifier } from 'mdml-engine/data'
 * subpopulationShiftModifier.name // => 'subpopulationShift'
 * ```
 */
export const subpopulationShiftModifier: Modifier = {
  name: 'subpopulationShift',
  description: 'Under-represent the rows above the median of the first feature.',
  knobs: dict({ keep: real(0, 1, { default: 0.1, label: 'retention of the group' }) }),
  apply: (key, input, knobs) =>
    withSubpopulationShift(input as Generated<TabularData>, { keep: knobs['keep'] as number, key }),
}

/**
 * {@link withShortcut} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { shortcutModifier } from 'mdml-engine/data'
 * shortcutModifier.name // => 'shortcut'
 * ```
 */
export const shortcutModifier: Modifier = {
  name: 'shortcut',
  description: 'Append a feature that tracks the label in the source and not in the target.',
  knobs: dict({
    mode: choice(['source', 'target'], { default: 'source' }),
    agreement: real(0, 1, { default: 0.95, label: 'agreement with the label' }),
  }),
  apply: (key, input, knobs) =>
    withShortcut(input as Generated<ClassificationData>, {
      mode: knobs['mode'] as 'source' | 'target',
      agreement: knobs['agreement'] as number,
      key,
    }),
}
