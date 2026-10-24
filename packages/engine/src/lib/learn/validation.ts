import { WhiteboxError } from '../base/index.ts'
import { asTensor, fromStorage, isTensor, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import type { LearnData } from './estimator.ts'

/** A dense row-major matrix read once for fast loops: `data[i * d + j]` is row i, column j. */
export interface MatrixView {
  readonly n: number
  readonly d: number
  readonly data: Float64Array
}

/** Options of {@link asFloatMatrix} and {@link matrixView}. */
export interface MatrixCheckOptions {
  /** Accept NaN entries (missing values), as imputers and NaN-aware scalers do (default false). */
  readonly allowNaN?: boolean
  /** Accept infinite entries (default false). */
  readonly allowInfinite?: boolean
  /** The number of columns required (the features seen in `fit`). */
  readonly nFeatures?: number
  /** The fewest rows accepted (default 1). */
  readonly minSamples?: number
}

/**
 * Read a feature matrix `[n, d]` as float64, with clear errors: a 1-D array is refused with a hint (sklearn's rule),
 * the column count is checked against the features seen in `fit`, and NaN or infinite entries are refused unless
 * allowed.
 *
 * @param x - The features.
 * @param who - The caller (for errors).
 * @param options - Whether NaN and infinities are allowed, and the expected number of columns and rows.
 * @returns The matrix view.
 * @throws WhiteboxError with code `shape` for a non-2-D input or a wrong column count, `value` for non-finite entries.
 * @example
 * ```ts
 * import { matrixView } from 'whitebox-engine/learn'
 * matrixView([[1, 2], [3, 4]], 'demo').d // => 2
 * matrixView([1, 2], 'demo') // => throws shape
 * matrixView([[1, NaN]], 'demo') // => throws value
 * ```
 */
export function matrixView(x: TensorLike, who: string, options: MatrixCheckOptions = {}): MatrixView {
  const v = asTensor(x)
  if (!isTensor(v)) throw new WhiteboxError('value', `${who}: expected concrete data, not a traced value`)
  const t = v
  if (t.shape.length !== 2) {
    throw new WhiteboxError(
      'shape',
      `${who}: expected a 2-D feature matrix [n, d], got shape [${t.shape.join(', ')}]` +
        (t.shape.length === 1 ? ' (reshape a single feature to [n, 1], a single sample to [1, d])' : ''),
      { shape: t.shape },
    )
  }
  const n = t.shape[0] ?? 0
  const d = t.shape[1] ?? 0
  if (n < (options.minSamples ?? 1)) {
    throw new WhiteboxError('shape', `${who}: needs at least ${options.minSamples ?? 1} samples, got ${n}`)
  }
  if (options.nFeatures !== undefined && d !== options.nFeatures) {
    throw new WhiteboxError('shape', `${who}: expected ${options.nFeatures} features (as in fit), got ${d}`)
  }
  const data = t.dtype === 'float64' ? (t.data as Float64Array) : Float64Array.from(toFlat(t))
  for (let i = 0; i < data.length; i++) {
    const v = data[i] ?? 0
    if (Number.isNaN(v) && options.allowNaN !== true) {
      throw new WhiteboxError('value', `${who}: NaN at row ${Math.floor(i / d)}, column ${i % d} (impute first)`)
    }
    if (!Number.isNaN(v) && !Number.isFinite(v) && options.allowInfinite !== true) {
      throw new WhiteboxError('value', `${who}: infinite value at row ${Math.floor(i / d)}, column ${i % d}`)
    }
  }
  return { n, d, data }
}

/**
 * A checked feature matrix as a float64 tensor (see {@link matrixView} for the checks).
 *
 * @param x - The features.
 * @param who - The caller (for errors).
 * @param options - The checks.
 * @returns The `[n, d]` float64 tensor.
 * @example
 * ```ts
 * import { asFloatMatrix } from 'whitebox-engine/learn'
 * asFloatMatrix([[1, 2]], 'demo').dtype // => 'float64'
 * ```
 */
export function asFloatMatrix(x: TensorLike, who: string, options: MatrixCheckOptions = {}): Tensor<'float64'> {
  const m = matrixView(x, who, options)
  return fromStorage(m.data, [m.n, m.d], 'float64')
}

/**
 * A matrix view as a tensor (no copy: the view's storage is handed over).
 *
 * @param m - The view.
 * @returns The `[n, d]` tensor.
 * @example
 * ```ts
 * import { fromView } from 'whitebox-engine/learn'
 * fromView({ n: 1, d: 2, data: new Float64Array([1, 2]) }).shape // => [1, 2]
 * ```
 */
export function fromView(m: MatrixView): Tensor<'float64'> {
  return fromStorage(m.data, [m.n, m.d], 'float64')
}

/**
 * One column of a matrix view.
 *
 * @param m - The view.
 * @param j - The column.
 * @returns Its values.
 * @example
 * ```ts
 * import { columnOf, matrixView } from 'whitebox-engine/learn'
 * Array.from(columnOf(matrixView([[1, 2], [3, 4]], 'demo'), 1)) // => [2, 4]
 * ```
 */
export function columnOf(m: MatrixView, j: number): Float64Array {
  const out = new Float64Array(m.n)
  for (let i = 0; i < m.n; i++) out[i] = m.data[i * m.d + j] ?? 0
  return out
}

/**
 * The rows of a matrix view at some indices (in that order).
 *
 * @param m - The view.
 * @param rows - Row indices.
 * @returns A new view.
 * @example
 * ```ts
 * import { matrixView, takeRows } from 'whitebox-engine/learn'
 * Array.from(takeRows(matrixView([[1], [2], [3]], 'demo'), [2, 0]).data) // => [3, 1]
 * ```
 */
export function takeRows(m: MatrixView, rows: ArrayLike<number>): MatrixView {
  const out = new Float64Array(rows.length * m.d)
  for (let r = 0; r < rows.length; r++) {
    const i = rows[r] ?? 0
    out.set(m.data.subarray(i * m.d, (i + 1) * m.d), r * m.d)
  }
  return { n: rows.length, d: m.d, data: out }
}

/**
 * The columns of a matrix view at some indices (in that order).
 *
 * @param m - The view.
 * @param cols - Column indices.
 * @returns A new view.
 * @example
 * ```ts
 * import { matrixView, takeColumns } from 'whitebox-engine/learn'
 * Array.from(takeColumns(matrixView([[1, 2, 3]], 'demo'), [2, 0]).data) // => [3, 1]
 * ```
 */
export function takeColumns(m: MatrixView, cols: readonly number[]): MatrixView {
  const out = new Float64Array(m.n * cols.length)
  for (let i = 0; i < m.n; i++) {
    for (let c = 0; c < cols.length; c++) out[i * cols.length + c] = m.data[i * m.d + (cols[c] ?? 0)] ?? 0
  }
  return { n: m.n, d: cols.length, data: out }
}

/**
 * Join matrix views side by side (numpy.hstack).
 *
 * @param parts - Views with the same number of rows.
 * @returns The joined view.
 * @throws WhiteboxError with code `shape` when the row counts differ.
 * @example
 * ```ts
 * import { hstackViews, matrixView } from 'whitebox-engine/learn'
 * Array.from(hstackViews([matrixView([[1]], 'a'), matrixView([[2, 3]], 'b')]).data) // => [1, 2, 3]
 * ```
 */
export function hstackViews(parts: readonly MatrixView[]): MatrixView {
  const n = parts[0]?.n ?? 0
  let d = 0
  for (const p of parts) {
    if (p.n !== n) throw new WhiteboxError('shape', `hstack: ${p.n} rows where ${n} were expected`)
    d += p.d
  }
  const out = new Float64Array(n * d)
  let offset = 0
  for (const p of parts) {
    for (let i = 0; i < n; i++) out.set(p.data.subarray(i * p.d, (i + 1) * p.d), i * d + offset)
    offset += p.d
  }
  return { n, d, data: out }
}

/**
 * Read a target vector `[n]` (a column `[n, 1]` is accepted) as float64.
 *
 * @param y - The targets.
 * @param who - The caller (for errors).
 * @param n - The number of rows required, when known.
 * @returns The values.
 * @throws WhiteboxError with code `shape` for other shapes or lengths, `value` for non-finite entries.
 * @example
 * ```ts
 * import { targetVector } from 'whitebox-engine/learn'
 * Array.from(targetVector([[1], [2]], 'demo')) // => [1, 2]
 * targetVector([1, 2], 'demo', 3) // => throws shape
 * ```
 */
export function targetVector(y: TensorLike, who: string, n?: number): Float64Array {
  const t = asTensor(y)
  if (!(t.shape.length === 1 || (t.shape.length === 2 && t.shape[1] === 1))) {
    throw new WhiteboxError('shape', `${who}: expected targets [n], got shape [${t.shape.join(', ')}]`)
  }
  const out = Float64Array.from(toFlat(t))
  if (n !== undefined && out.length !== n) {
    throw new WhiteboxError('shape', `${who}: ${out.length} targets for ${n} samples`)
  }
  for (const v of out) if (!Number.isFinite(v)) throw new WhiteboxError('value', `${who}: targets must be finite`)
  return out
}

/** Class labels encoded as indices into the sorted distinct labels. */
export interface EncodedLabels {
  /** The distinct labels, ascending. */
  readonly classes: readonly number[]
  /** `codes[i]` is the index of sample i's label in `classes`. */
  readonly codes: Int32Array
}

/**
 * Encode class labels as indices into their sorted distinct values (sklearn's `np.unique(y, return_inverse=True)`).
 *
 * @param y - Labels `[n]`.
 * @param who - The caller (for errors).
 * @param n - The number of rows required, when known.
 * @returns The classes and codes.
 * @example
 * ```ts
 * import { encodeLabels } from 'whitebox-engine/learn'
 * const e = encodeLabels([5, 2, 5], 'demo')
 * e.classes // => [2, 5]
 * Array.from(e.codes) // => [1, 0, 1]
 * ```
 */
export function encodeLabels(y: TensorLike, who: string, n?: number): EncodedLabels {
  const values = targetVector(y, who, n)
  const classes = Array.from(new Set(values)).sort((a, b) => a - b)
  const index = new Map(classes.map((c, i) => [c, i]))
  return { classes, codes: Int32Array.from(values, (v) => index.get(v) ?? 0) }
}

/**
 * Codes of labels against known classes (as fitted), with an error naming an unseen label.
 *
 * @param y - Labels.
 * @param classes - The fitted classes.
 * @param who - The caller (for errors).
 * @returns The codes.
 * @throws WhiteboxError with code `value` for a label not among the classes.
 * @example
 * ```ts
 * import { codesFor } from 'whitebox-engine/learn'
 * Array.from(codesFor([5, 2], [2, 5], 'demo')) // => [1, 0]
 * codesFor([3], [2, 5], 'demo') // => throws value
 * ```
 */
export function codesFor(y: TensorLike, classes: readonly number[], who: string): Int32Array {
  const index = new Map(classes.map((c, i) => [c, i]))
  return Int32Array.from(targetVector(y, who), (v) => {
    const c = index.get(v)
    if (c === undefined) throw new WhiteboxError('value', `${who}: label ${v} was not seen in fit`)
    return c
  })
}

/**
 * Sample weights `[n]` (all ones when absent), checked for length, finiteness and sign.
 *
 * @param weights - The weights, if any.
 * @param n - The number of samples.
 * @param who - The caller (for errors).
 * @returns The weights.
 * @throws WhiteboxError with code `shape` for a wrong length, `value` for negative or non-finite weights.
 * @example
 * ```ts
 * import { sampleWeightsOf } from 'whitebox-engine/learn'
 * Array.from(sampleWeightsOf(undefined, 2, 'demo')) // => [1, 1]
 * sampleWeightsOf([1, -1], 2, 'demo') // => throws value
 * ```
 */
export function sampleWeightsOf(weights: TensorLike | undefined, n: number, who: string): Float64Array {
  if (weights === undefined) return new Float64Array(n).fill(1)
  const w = Float64Array.from(toFlat(asTensor(weights)))
  if (w.length !== n) throw new WhiteboxError('shape', `${who}: ${w.length} sample weights for ${n} samples`)
  for (const v of w) {
    if (!(Number.isFinite(v) && v >= 0)) throw new WhiteboxError('value', `${who}: weights must be finite and >= 0`)
  }
  return w
}

/**
 * The number of rows of an array-like (its first dimension).
 *
 * @param x - Any array.
 * @returns The number of rows.
 * @example
 * ```ts
 * import { rowsOf } from 'whitebox-engine/learn'
 * rowsOf([[1, 2], [3, 4], [5, 6]]) // => 3
 * ```
 */
export function rowsOf(x: TensorLike): number {
  if (Array.isArray(x)) return x.length
  if (ArrayBuffer.isView(x)) return (x as Float64Array).length
  const t = isTensor(x) ? x : asTensor(x)
  return t.shape[0] ?? 1
}

/**
 * Check that several arrays have the same number of rows.
 *
 * @param who - The caller (for errors).
 * @param lengths - Named row counts.
 * @returns The common count.
 * @throws WhiteboxError with code `shape` when they differ.
 * @example
 * ```ts
 * import { consistentLength } from 'whitebox-engine/learn'
 * consistentLength('demo', { x: 3, y: 3 }) // => 3
 * consistentLength('demo', { x: 3, y: 2 }) // => throws shape
 * ```
 */
export function consistentLength(who: string, lengths: Readonly<Record<string, number>>): number {
  const entries = Object.entries(lengths)
  const n = entries[0]?.[1] ?? 0
  for (const [name, m] of entries) {
    if (m !== n) {
      throw new WhiteboxError(
        'shape',
        `${who}: inconsistent numbers of samples (${entries.map(([k, v]) => `${k}: ${v}`).join(', ')})`,
        {
          name,
        },
      )
    }
  }
  return n
}

/**
 * A positive integer option, or a clear error.
 *
 * @param value - The value.
 * @param name - The option's name.
 * @param who - The caller.
 * @param min - The smallest allowed value (default 1).
 * @returns The value.
 * @throws WhiteboxError with code `value` otherwise.
 * @example
 * ```ts
 * import { integerOption } from 'whitebox-engine/learn'
 * integerOption(3, 'nSplits', 'demo', 2) // => 3
 * integerOption(1.5, 'nSplits', 'demo') // => throws value
 * ```
 */
export function integerOption(value: number, name: string, who: string, min = 1): number {
  if (!Number.isInteger(value) || value < min) {
    throw new WhiteboxError('value', `${who}: ${name} must be an integer >= ${min}, got ${value}`)
  }
  return value
}

/**
 * A fraction option in an interval, or a clear error.
 *
 * @param value - The value.
 * @param name - The option's name.
 * @param who - The caller.
 * @param lo - Lower bound (exclusive, default 0).
 * @param hi - Upper bound (exclusive, default 1).
 * @returns The value.
 * @throws WhiteboxError with code `value` otherwise.
 * @example
 * ```ts
 * import { fractionOption } from 'whitebox-engine/learn'
 * fractionOption(0.1, 'alpha', 'demo') // => 0.1
 * fractionOption(1, 'alpha', 'demo') // => throws value
 * ```
 */
export function fractionOption(value: number, name: string, who: string, lo = 0, hi = 1): number {
  if (!(value > lo && value < hi)) {
    throw new WhiteboxError('value', `${who}: ${name} must lie in (${lo}, ${hi}), got ${value}`)
  }
  return value
}

/**
 * The rows of a learning dataset at some indices: features, targets and weights are subset together (any other
 * fields are kept as they are).
 *
 * @param data - The data.
 * @param rows - Row indices.
 * @returns The subset.
 * @example
 * ```ts
 * import { takeData } from 'whitebox-engine/learn'
 * import { toArray } from 'whitebox-engine/tensor'
 * const d = takeData({ x: [[1], [2], [3]], y: [10, 20, 30] }, [2, 0])
 * toArray(d.y) // => [30, 10]
 * ```
 */
export function takeData<D extends LearnData>(data: D, rows: ArrayLike<number>): D {
  const take = (v: TensorLike): Tensor => {
    const t = asTensor(v)
    if (!isTensor(t)) throw new WhiteboxError('value', 'takeData: expected concrete data')
    const n = t.shape[0] ?? 0
    const width = n === 0 ? 0 : t.data.length / n
    const out = new Float64Array(rows.length * width)
    const src = t.dtype === 'float64' ? (t.data as Float64Array) : Float64Array.from(toFlat(t))
    for (let r = 0; r < rows.length; r++) {
      const i = rows[r] ?? 0
      if (!(i >= 0 && i < n)) throw new WhiteboxError('value', `takeData: row ${i} outside [0, ${n})`)
      out.set(src.subarray(i * width, (i + 1) * width), r * width)
    }
    return fromStorage(out, [rows.length, ...t.shape.slice(1)], 'float64')
  }
  return {
    ...data,
    x: take(data.x),
    ...(data.y === undefined ? {} : { y: take(data.y) }),
    ...(data.weights === undefined ? {} : { weights: take(data.weights) }),
  }
}
