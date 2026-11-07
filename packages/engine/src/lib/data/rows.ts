/**
 * Row operations on any dataset kind: taking and shuffling rows (every per-row field together), train-test and
 * three-way splits as in scikit-learn's `train_test_split`, stratified splits that keep class proportions, bootstrap
 * resamples with their out-of-bag rows, and stacking datasets of the same kind.
 */

import { MdmlError } from '../base/index.ts'
import { bootstrapIndices, foldIn, type Key, permutation, randint } from '../random/index.ts'
import { isTensor, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { concatColumns, takeColumn } from './_columns.ts'
import { type Dataset, perRowFields, rowCount } from './schema/common.ts'

function indexArray(indices: readonly number[] | Int32Array | Value, name: string): Int32Array {
  if (indices instanceof Int32Array) return indices
  const raw: ArrayLike<number> = Array.isArray(indices)
    ? (indices as readonly number[])
    : isTensor(indices as Value)
      ? ((indices as Tensor).data as ArrayLike<number>)
      : toFlat(indices as Value)
  // Int32Array.from truncates a fraction and turns NaN into row 0: an index that is not a whole number is refused.
  for (let j = 0; j < raw.length; j++) {
    if (!Number.isInteger(raw[j])) throw new MdmlError('value', `${name}: index ${raw[j]} is not an integer`)
  }
  return Int32Array.from(raw)
}

/**
 * The rows of a dataset at the given indices: every per-row field together, other fields (names, vocabularies,
 * metadata) unchanged.
 *
 * @param data - The dataset.
 * @param indices - Row indices (repeats allowed).
 * @returns A dataset of the selected rows, with the same metadata.
 * @throws MdmlError with code `value` for an index outside the rows, or one that is not an integer (NaN, a fraction).
 * @example
 * ```ts
 * import { take, xy } from 'mdml-engine/data'
 * take(xy([[1], [2], [3]], [10, 20, 30]), [2, 0]).y.data // => Float64Array [30, 10]
 * ```
 */
export function take<D extends Dataset>(data: D, indices: readonly number[] | Int32Array | Value): D {
  const idx = indexArray(indices, 'take')
  const n = rowCount(data)
  for (const i of idx) {
    if (i < 0 || i >= n) throw new MdmlError('value', `take: row ${i} outside [0, ${n})`)
  }
  const out: Record<string, unknown> = { ...data }
  const record = data as unknown as Readonly<Record<string, unknown>>
  for (const f of perRowFields(data)) out[f] = takeColumn(record[f], idx)
  return out as unknown as D
}

/**
 * Shuffle the rows of a dataset.
 *
 * @param data - The dataset.
 * @param key - The random key.
 * @returns The shuffled dataset.
 * @example
 * ```ts
 * import { shuffleRows, xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * shuffleRows(xy([[1], [2], [3]], [1, 2, 3]), key(0)).y.shape // => [3]
 * ```
 */
export function shuffleRows<D extends Dataset>(data: D, key: Key): D {
  return take(data, permutation(key, rowCount(data)))
}

/**
 * The class labels of a dataset, used for stratification: `y` of a classification dataset, the labels of a corpus or
 * an image set, or the graph labels of a graph-classification set.
 *
 * @param data - The dataset.
 * @returns The labels `[n]`, or `undefined` when the dataset has none.
 * @example
 * ```ts
 * import { labelsOf, xyClasses } from 'mdml-engine/data'
 * labelsOf(xyClasses([[0], [1]], [1, 0])) // => [1, 0]
 * ```
 */
export function labelsOf(data: Dataset): number[] | undefined {
  switch (data.kind) {
    case 'classification':
      return Array.from(data.y.data)
    case 'corpus':
      return data.labels === undefined ? undefined : Array.from(data.labels.data)
    case 'image':
      return data.labels === undefined ? undefined : Array.from(data.labels.data)
    case 'graph':
      return data.graphLabels === undefined ? undefined : Array.from(data.graphLabels.data as ArrayLike<number>)
    default:
      return undefined
  }
}

function sizeOf(size: number, n: number, name: string): number {
  // A fraction below 1 is a share of the rows (rounded up); from 1 up it is a count of rows and must be whole.
  if (!Number.isFinite(size) || size < 0 || (size >= 1 && !Number.isInteger(size)))
    throw new MdmlError('value', `${name}: a size is a fraction in [0, 1) or a whole number of rows, got ${size}`)
  const m = size < 1 ? Math.ceil(size * n) : size
  if (m > n) throw new MdmlError('value', `${name}: size ${size} does not fit ${n} rows`)
  return m
}

/**
 * Indices of a stratified split: each class contributes to the test part in proportion to its frequency, with
 * leftover rows allotted by largest remainder (as sklearn's `StratifiedShuffleSplit`), then shuffled.
 *
 * @param labels - The class of each row.
 * @param nTest - The number of test rows.
 * @param key - The random key (order within each class).
 * @returns Train and test row indices.
 * @throws MdmlError with code `value` for a number of test rows that is not a whole number in [0, n], or labels that are
 *   not finite.
 * @example
 * ```ts
 * import { stratifiedIndices } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * stratifiedIndices([0, 0, 0, 0, 1, 1], 3, key(0)).test.length // => 3
 * ```
 */
export function stratifiedIndices(
  labels: readonly number[],
  nTest: number,
  key: Key,
): { readonly train: number[]; readonly test: number[] } {
  const n = labels.length
  if (!Number.isInteger(nTest) || nTest < 0 || nTest > n)
    throw new MdmlError('value', `stratifiedIndices: ${nTest} test rows does not fit ${n} rows`)
  if (!labels.every(Number.isFinite)) throw new MdmlError('value', 'stratifiedIndices: labels must be finite')
  const byClass = new Map<number, number[]>()
  for (const [i, c] of labels.entries()) {
    const rows = byClass.get(c)
    if (rows === undefined) byClass.set(c, [i])
    else rows.push(i)
  }
  const classes = [...byClass.keys()].sort((a, b) => a - b)
  const exact = classes.map((c) => ((byClass.get(c)?.length ?? 0) * nTest) / n)
  const alloc = exact.map(Math.floor)
  let left = nTest - alloc.reduce((s, v) => s + v, 0)
  const order = exact.map((v, j) => [v - Math.floor(v), j] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1])
  for (const [, j] of order) {
    if (left <= 0) break
    alloc[j] = (alloc[j] ?? 0) + 1
    left--
  }
  const train: number[] = []
  const test: number[] = []
  for (const [j, c] of classes.entries()) {
    const rows = byClass.get(c) ?? []
    const perm = permutation(foldIn(foldIn(key, 0), c), rows.length).data
    const t = alloc[j] ?? 0
    for (let i = 0; i < rows.length; i++) (i < t ? test : train).push(rows[perm[i] ?? 0] ?? 0)
  }
  const shuffle = (xs: number[], salt: number): number[] => {
    const p = permutation(foldIn(foldIn(key, 1), salt), xs.length).data
    return Array.from(p, (i) => xs[i] ?? 0)
  }
  return { train: shuffle(train, 0), test: shuffle(test, 1) }
}

/** Options of {@link trainTestSplit}. */
export interface SplitOptions {
  /** The fraction of rows held out (default 0.25, as sklearn) or a number of rows. */
  readonly testSize?: number
  /** Shuffle before splitting with this key (default: split in order, without shuffling). */
  readonly key?: Key
  /** Preserve class proportions in both parts (needs a key and class labels). */
  readonly stratify?: boolean
}

/**
 * Split a dataset into training and test parts (sklearn.model_selection.train_test_split, test size rounded up),
 * optionally shuffled and stratified by class.
 *
 * @param data - The dataset.
 * @param options - Test size, shuffling key and stratification. A `testSize` below 1 is a fraction of the rows, from 1 up
 *   a whole number of rows (`1` is one row, not all of them).
 * @returns `{ train, test }`.
 * @throws MdmlError with code `value` when a part would be empty, stratification lacks labels or a key, or `testSize` is
 *   NaN, negative, or a count of rows above 1 that is not whole.
 * @index function Split any dataset kind into train and test rows.
 * @example
 * ```ts
 * import { trainTestSplit, xy } from 'mdml-engine/data'
 * trainTestSplit(xy([[1], [2], [3], [4]], [1, 2, 3, 4])).test.y.shape // => [1]
 * ```
 */
export function trainTestSplit<D extends Dataset>(
  data: D,
  options: SplitOptions = {},
): { readonly train: D; readonly test: D } {
  const n = rowCount(data)
  const nTest = sizeOf(options.testSize ?? 0.25, n, 'trainTestSplit')
  if (nTest <= 0 || nTest >= n)
    throw new MdmlError('value', `trainTestSplit: test size ${nTest} leaves no rows to train on`)
  if (options.stratify === true) {
    const labels = labelsOf(data)
    if (labels === undefined || options.key === undefined)
      throw new MdmlError('value', 'trainTestSplit: stratification needs class labels and a key')
    const { train, test } = stratifiedIndices(labels, nTest, options.key)
    return { train: take(data, train), test: take(data, test) }
  }
  const order =
    options.key === undefined ? Array.from({ length: n }, (_, i) => i) : Array.from(permutation(options.key, n).data)
  return { train: take(data, order.slice(0, n - nTest)), test: take(data, order.slice(n - nTest)) }
}

/** Options of {@link trainValTestSplit}. */
export interface ThreeWaySplitOptions {
  /** Validation fraction or count (default 0.15). */
  readonly valSize?: number
  /** Test fraction or count (default 0.15). */
  readonly testSize?: number
  readonly key?: Key
  readonly stratify?: boolean
}

/**
 * Split a dataset into training, validation and test parts (fractions of the whole, rounded up), optionally shuffled
 * and stratified.
 *
 * @param data - The dataset.
 * @param options - Validation and test sizes, key and stratification.
 * @returns `{ train, val, test }`.
 * @throws MdmlError with code `value` when a part would be empty.
 * @example
 * ```ts
 * import { trainValTestSplit, xy } from 'mdml-engine/data'
 * const parts = trainValTestSplit(xy(Array.from({ length: 10 }, (_, i) => [i]), Array.from({ length: 10 }, (_, i) => i)), { valSize: 2, testSize: 3 })
 * [parts.train.y.shape[0], parts.val.y.shape[0], parts.test.y.shape[0]] // => [5, 2, 3]
 * ```
 */
export function trainValTestSplit<D extends Dataset>(
  data: D,
  options: ThreeWaySplitOptions = {},
): { readonly train: D; readonly val: D; readonly test: D } {
  const n = rowCount(data)
  const nTest = sizeOf(options.testSize ?? 0.15, n, 'trainValTestSplit')
  const nVal = sizeOf(options.valSize ?? 0.15, n, 'trainValTestSplit')
  if (nTest <= 0 || nVal <= 0 || nTest + nVal >= n)
    throw new MdmlError('value', `trainValTestSplit: sizes ${nVal} and ${nTest} leave no rows to train on`)
  const first = trainTestSplit(data, { ...options, testSize: nTest })
  const second = trainTestSplit(first.train, {
    ...options,
    testSize: nVal,
    ...(options.key === undefined ? {} : { key: foldIn(options.key, 1) }),
  })
  return { train: second.train, val: second.test, test: first.test }
}

/**
 * A stratified train-test split (sklearn's `StratifiedShuffleSplit` with one split): each class keeps its proportion
 * in both parts. Labels come from the dataset or are given.
 *
 * @param data - The dataset.
 * @param options - Test size, key, and labels for datasets without class labels.
 * @returns `{ train, test }`.
 * @throws MdmlError with code `value` when there are no labels.
 * @example
 * ```ts
 * import { stratifiedSplit, xyClasses } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const d = xyClasses([[0], [1], [2], [3], [4], [5], [6], [7]], [0, 0, 0, 0, 0, 0, 1, 1])
 * Array.from(stratifiedSplit(d, { testSize: 0.5, key: key(1) }).test.y.data).filter((c) => c === 1).length // => 1
 * ```
 */
export function stratifiedSplit<D extends Dataset>(
  data: D,
  options: { readonly testSize?: number; readonly key: Key; readonly labels?: readonly number[] },
): { readonly train: D; readonly test: D } {
  const labels = options.labels ?? labelsOf(data)
  if (labels === undefined) throw new MdmlError('value', 'stratifiedSplit: the dataset has no class labels')
  const n = rowCount(data)
  const nTest = sizeOf(options.testSize ?? 0.25, n, 'stratifiedSplit')
  const { train, test } = stratifiedIndices(labels, nTest, options.key)
  return { train: take(data, train), test: take(data, test) }
}

/**
 * A bootstrap resample: `n` rows drawn with replacement, and the out-of-bag rows that were never drawn.
 *
 * @param data - The dataset.
 * @param key - The random key.
 * @param size - The resample size (default the number of rows).
 * @returns The resample, the out-of-bag rows and the drawn indices.
 * @throws MdmlError with code `value` for a dataset without rows, or a size that is not a positive whole number.
 * @example
 * ```ts
 * import { bootstrap, xy } from 'mdml-engine/data'
 * import { key } from 'mdml-engine/random'
 * const b = bootstrap(xy([[1], [2], [3], [4]], [1, 2, 3, 4]), key(0))
 * b.sample.y.shape[0] // => 4
 * ```
 */
export function bootstrap<D extends Dataset>(
  data: D,
  key: Key,
  size?: number,
): { readonly sample: D; readonly outOfBag: D; readonly indices: readonly number[] } {
  const n = rowCount(data)
  const m = size ?? n
  if (n === 0) throw new MdmlError('value', 'bootstrap: the dataset has no rows to draw from')
  if (!Number.isInteger(m) || m < 1)
    throw new MdmlError('value', `bootstrap: the size must be a positive whole number, got ${m}`)
  const drawn = Array.from(m === n ? bootstrapIndices(key, n).data : randint(key, [m], 0, n).data)
  const seen = new Uint8Array(n)
  for (const i of drawn) seen[i] = 1
  const oob: number[] = []
  for (let i = 0; i < n; i++) if (seen[i] === 0) oob.push(i)
  return { sample: take(data, drawn), outOfBag: take(data, oob), indices: drawn }
}

/** The fields that name what the values of the rows stand for: stacking parts that differ in them would mislabel rows. */
const VOCABULARIES: readonly string[] = [
  'classes',
  'featureNames',
  'targetNames',
  'labels',
  'outputs',
  'vocabulary',
  'padId',
  'tagNames',
  'playerNames',
  'entityNames',
  'relationNames',
  'channelNames',
  'markNames',
  'maskClasses',
]

/**
 * Stack datasets of the same kind along their rows; non-row fields (names, vocabularies, metadata) come from the
 * first, and the lists that row values index into (`classes`, `featureNames`, `outputs`, `vocabulary`, the sensitive
 * attributes and the like) must agree across the parts. Counts such as `nUsers` are the first part's.
 *
 * @param parts - The datasets.
 * @returns One dataset with all rows in order.
 * @throws MdmlError with code `value` for no parts, mixed kinds, or parts whose classes, names or vocabularies differ.
 * @example
 * ```ts
 * import { concatRows, xy } from 'mdml-engine/data'
 * concatRows([xy([[1]], [1]), xy([[2], [3]], [2, 3])]).y.data // => Float64Array [1, 2, 3]
 * ```
 */
export function concatRows<D extends Dataset>(parts: readonly D[]): D {
  const first = parts[0]
  if (first === undefined) throw new MdmlError('value', 'concatRows: no datasets')
  for (const p of parts) {
    if (p.kind !== first.kind) throw new MdmlError('value', `concatRows: cannot stack ${p.kind} with ${first.kind}`)
  }
  const rowFields = new Set<string>(perRowFields(first))
  const same = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b)
  for (const p of parts) {
    const fields = p as unknown as Readonly<Record<string, unknown>>
    const head = first as unknown as Readonly<Record<string, unknown>>
    for (const f of VOCABULARIES) {
      if (!rowFields.has(f) && !same(fields[f], head[f]))
        throw new MdmlError(
          'value',
          `concatRows: the parts disagree on ${f}, so their rows would mean different things`,
        )
    }
    if (!same(p.meta.sensitive, first.meta.sensitive))
      throw new MdmlError('value', 'concatRows: the parts disagree on the sensitive attributes')
  }
  const out: Record<string, unknown> = { ...first }
  for (const f of perRowFields(first)) {
    out[f] = concatColumns(
      parts.map((p) => {
        const c = (p as unknown as Record<string, unknown>)[f]
        if (c === undefined) throw new MdmlError('value', `concatRows: field ${f} is missing from a part`)
        return c
      }),
    )
  }
  return out as unknown as D
}
