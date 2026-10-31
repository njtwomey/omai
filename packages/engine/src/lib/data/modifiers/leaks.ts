/**
 * Planted leaks: a feature derived from the target (a noisy copy, or a target encoding fitted on all rows) and
 * duplicated rows (exact or jittered copies, spread through the data so they fall on both sides of a split). The truth
 * keeps what was planted, so a leak detector, a validation scheme or a pipeline can be scored against it.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key, normal, permutation } from '../../random/index.ts'
import { choice, dict, real } from '../../space/index.ts'
import { fromStorage, isTensor, sliceAxis, type Value } from '../../tensor/index.ts'
import { takeColumn } from '../_columns.ts'
import type { Generated } from '../generator.ts'
import type { Modifier } from '../recipes.ts'
import { take } from '../rows.ts'
import { type Dataset, rowCount } from '../schema/common.ts'
import type { TabularData } from '../schema/tabular.ts'
import { composedFunctions, requireTabular, withoutProgram } from './_shared.ts'

/** The kinds of leak {@link withLeak} plants. */
export type LeakKind = 'target' | 'targetEncoding'

/** Options of {@link withLeak}. */
export interface LeakOptions {
  /** `target`: the target plus Gaussian noise; `targetEncoding`: the mean target within a bin of the first feature, fitted on all rows. */
  readonly kind?: LeakKind
  /** Standard deviation of the noise on a `target` leak, in units of the target's standard deviation (default 0.1). */
  readonly noise?: number
  /** Bins of the first feature for a `targetEncoding` leak (default 20). */
  readonly bins?: number
  readonly key: Key
}

/**
 * Append a feature that leaks the target: either the target plus a little noise, or the mean of the target within a
 * quantile bin of the first feature computed over all rows, the row itself included (an encoder fitted before the split,
 * the textbook target-leakage mistake). The truth's functions (posterior, regression function, Bayes error) still
 * describe the honest problem without the leak, so they give the error a model should have; `truth.extra.leak` names
 * the column and kind, and `truth.relevantFeatures` stays the original columns.
 *
 * @param input - Classification or regression data with its truth.
 * @param options - The kind, noise or bins, and the key.
 * @returns The data with a `leak` column appended.
 * @example
 * ```ts
 * import { asGenerated, withLeak, xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const out = withLeak(asGenerated(xy([[1], [2], [3]], [1, 2, 3])), { noise: 0, key: key(0) })
 * Array.from(out.data.x.data) // => [1, 1, 2, 2, 3, 3]
 * ```
 */
export function withLeak<D extends TabularData>(input: Generated<D>, options: LeakOptions): Generated<D> {
  const { data, truth } = requireTabular(input, 'withLeak') as Generated<D>
  if (data.kind !== 'classification' && data.kind !== 'regression')
    throw new MdmlError('value', `withLeak: needs classification or regression data, got ${data.kind}`)
  const [n = 0, d = 1] = data.x.shape
  const kind = options.kind ?? 'target'
  const y = Float64Array.from(data.y.data as ArrayLike<number>)
  if (y.length !== n) throw new MdmlError('shape', 'withLeak: needs a single target per row')
  const leak = new Float64Array(n)
  if (kind === 'target') {
    const mean = y.reduce((a, b) => a + b, 0) / Math.max(1, n)
    const sd = Math.sqrt(y.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, n)) || 1
    const z = normal(options.key, [n]).data
    for (let i = 0; i < n; i++) leak[i] = (y[i] ?? 0) + (options.noise ?? 0.1) * sd * (z[i] ?? 0)
  } else {
    const bins = Math.max(1, Math.min(options.bins ?? 20, n))
    const order = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => (data.x.data[a * d] ?? 0) - (data.x.data[b * d] ?? 0) || a - b,
    )
    const sums = new Float64Array(bins)
    const counts = new Float64Array(bins)
    const binOf = new Int32Array(n)
    order.forEach((row, rank) => {
      const b = Math.min(bins - 1, Math.floor((rank * bins) / n))
      binOf[row] = b
      sums[b] = (sums[b] ?? 0) + (y[row] ?? 0)
      counts[b] = (counts[b] ?? 0) + 1
    })
    for (let i = 0; i < n; i++) leak[i] = (sums[binOf[i] ?? 0] ?? 0) / Math.max(1, counts[binOf[i] ?? 0] ?? 1)
  }
  const wide = new Float64Array(n * (d + 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) wide[i * (d + 1) + j] = data.x.data[i * d + j] ?? 0
    wide[i * (d + 1) + d] = leak[i] ?? 0
  }
  const original = (v: Value): Value => sliceAxis(v, 1, 0, d)
  const { logProb: _logProb, ...rest } = withoutProgram(truth)
  return {
    data: { ...data, x: fromStorage(wide, [n, d + 1], 'float64'), featureNames: [...data.featureNames, 'leak'] },
    truth: {
      ...rest,
      ...composedFunctions(truth, original),
      relevantFeatures: truth.relevantFeatures ?? Array.from({ length: d }, (_, j) => j),
      extra: {
        ...(truth.extra ?? {}),
        leak: { column: d, kind, noise: kind === 'target' ? (options.noise ?? 0.1) : 0 },
      },
    },
  }
}

/** Options of {@link withDuplicates}. */
export interface DuplicateOptions {
  /** The share of rows that get a copy, of the original row count (default 0.1). */
  readonly fraction?: number
  /** Standard deviation of Gaussian jitter on the copies' features, in units of each feature's standard deviation (default 0: exact copies; tabular data only). */
  readonly jitter?: number
  /** Shuffle the rows afterwards so that copies sit away from their originals (default true). */
  readonly shuffle?: boolean
  readonly key: Key
}

/**
 * Add copies of a random share of the rows, exact or jittered, and shuffle (by default) so that a random split puts
 * an original and its copy on different sides: the duplicates that inflate a validation score. The truth records for
 * every row the original it comes from and whether it is a copy.
 *
 * @param input - Any dataset with rows, and its truth.
 * @param options - Share, jitter, shuffle and key.
 * @returns The enlarged data; `truth.latents.sourceRow` `[n']` is the original row (in the input's order) of each
 *   row, `isDuplicate` `[n']` marks the copies, and per-row latents the input had are carried along.
 * @example
 * ```ts
 * import { asGenerated, withDuplicates, xy } from 'mdml-engine/data'
 * import { rowCount } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const out = withDuplicates(asGenerated(xy([[1], [2], [3], [4]], [1, 2, 3, 4])), { fraction: 0.5, key: key(0) })
 * rowCount(out.data) // => 6
 * ```
 */
export function withDuplicates<D extends Dataset>(input: Generated<D>, options: DuplicateOptions): Generated<D> {
  const { data, truth } = input
  const n = rowCount(data)
  const m = Math.min(n, Math.round((options.fraction ?? 0.1) * n))
  const chosen = Array.from(permutation(foldIn(options.key, 0), n).data.subarray(0, m))
  const rows = [...Array.from({ length: n }, (_, i) => i), ...chosen]
  const isCopy = rows.map((_, i) => i >= n)
  const order =
    options.shuffle === false
      ? rows.map((_, i) => i)
      : Array.from(permutation(foldIn(options.key, 1), rows.length).data)
  const picked = order.map((p) => rows[p] ?? 0)
  let enlarged: D = take(data, picked)
  const jitter = options.jitter ?? 0
  if (jitter > 0) {
    if (!('x' in enlarged) || !isTensor((enlarged as unknown as TabularData).x))
      throw new MdmlError('value', 'withDuplicates: jitter needs data with a feature matrix')
    const tab = enlarged as unknown as TabularData
    const [rowsN = 0, d = 1] = tab.x.shape
    const sd = Array.from({ length: d }, (_, j) => {
      let mean = 0
      for (let i = 0; i < n; i++) mean += (data as unknown as TabularData).x.data[i * d + j] ?? 0
      mean /= Math.max(1, n)
      let v = 0
      for (let i = 0; i < n; i++) v += (((data as unknown as TabularData).x.data[i * d + j] ?? 0) - mean) ** 2
      return Math.sqrt(v / Math.max(1, n)) || 1
    })
    const z = normal(foldIn(options.key, 2), [rowsN, d]).data
    const x = Float64Array.from(tab.x.data as ArrayLike<number>)
    order.forEach((p, i) => {
      if (isCopy[p])
        for (let j = 0; j < d; j++) x[i * d + j] = (x[i * d + j] ?? 0) + jitter * (sd[j] ?? 1) * (z[i * d + j] ?? 0)
    })
    enlarged = { ...tab, x: fromStorage(x, [rowsN, d], 'float64') } as unknown as D
  }
  const latents: Record<string, unknown> = {}
  for (const [name, value] of Object.entries(truth.latents ?? {}))
    latents[name] =
      isTensor(value as Value) && ((value as { shape: number[] }).shape[0] ?? -1) === n
        ? takeColumn(value, Int32Array.from(picked))
        : value
  latents['sourceRow'] = fromStorage(Int32Array.from(picked), [picked.length], 'int32')
  latents['isDuplicate'] = fromStorage(
    Uint8Array.from(order, (p) => (isCopy[p] ? 1 : 0)),
    [picked.length],
    'bool',
  )
  return {
    data: enlarged,
    truth: { ...truth, latents, extra: { ...(truth.extra ?? {}), duplicates: { copies: m, jitter } } },
  }
}

/**
 * {@link withLeak} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { leakModifier } from 'mdml-engine/data'
 * leakModifier.name // => 'leak'
 * ```
 */
export const leakModifier: Modifier = {
  name: 'leak',
  description: 'Append a feature derived from the target.',
  knobs: dict({
    kind: choice(['target', 'targetEncoding'], { default: 'target' }),
    noise: real(0, 5, { default: 0.1, label: 'noise on the leak' }),
  }),
  apply: (key, input, knobs) =>
    withLeak(input as Generated<TabularData>, {
      kind: knobs['kind'] as LeakKind,
      noise: knobs['noise'] as number,
      key,
    }),
}

/**
 * {@link withDuplicates} as a recipe modifier.
 *
 * @example
 * ```ts
 * import { duplicatesModifier } from 'mdml-engine/data'
 * duplicatesModifier.name // => 'duplicates'
 * ```
 */
export const duplicatesModifier: Modifier = {
  name: 'duplicates',
  description: 'Add exact or jittered copies of a share of the rows, shuffled in.',
  knobs: dict({
    fraction: real(0, 1, { default: 0.1, label: 'share of rows copied' }),
    jitter: real(0, 1, { default: 0, label: 'jitter of the copies' }),
  }),
  apply: (key, input, knobs) =>
    withDuplicates(input, { fraction: knobs['fraction'] as number, jitter: knobs['jitter'] as number, key }),
}
