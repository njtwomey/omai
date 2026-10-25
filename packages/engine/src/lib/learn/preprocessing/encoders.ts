/**
 * Encoders for categorical data, scikit-learn style: one-hot and ordinal encoders with handling of unknown and
 * infrequent categories, label encoding and binarisation of targets (including multi-label sets), and the target
 * encoder, which replaces each category by a shrunk mean of the target over its rows.
 */

import { WhiteboxError } from '../../base/index.ts'
import { choice, dict, real, type Space } from '../../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { inputNames, keyFor } from '../_shared.ts'
import { defineEstimator, type Estimator, type Fitted } from '../estimator.ts'
import { kFold, type Splitter, splitIndices, stratifiedKFold } from '../model-selection/splitters.ts'
import { columnOf, encodeLabels, fromView, matrixView, targetVector } from '../validation.ts'

/** How an encoder treats a category it did not see in `fit`. */
export type HandleUnknown = 'error' | 'ignore' | 'infrequent_if_exist'

/** Options shared by the one-hot and ordinal encoders: grouping of rare categories (sklearn's infrequent categories). */
export interface CategoryOptions {
  /** Categories seen fewer times than this (a count, or a fraction of the samples when below 1) are grouped. */
  readonly minFrequency?: number
  /** At most this many output categories per feature, the infrequent group included. */
  readonly maxCategories?: number
}

/** The categories of one feature as fitted: sorted values (NaN last, stored as null) and their grouping. */
export type FeatureCategories = {
  /** The distinct values, ascending; a missing value (NaN) is the last category, stored as `null`. */
  readonly categories: (number | null)[]
  /** The output index of each category: frequent ones keep their order, infrequent ones share the last index. */
  readonly grouped: number[]
  /** The number of output categories (frequent ones, plus one for the infrequent group when there is one). */
  readonly nGrouped: number
  /** Indices (into `categories`) of the infrequent categories. */
  readonly infrequent: number[]
}

function sameCategory(a: number | null, v: number): boolean {
  return a === null ? Number.isNaN(v) : a === v
}

/** Find the category index of a value, or -1 when unseen. */
function categoryIndex(f: FeatureCategories, v: number): number {
  const cats = f.categories
  if (Number.isNaN(v)) return cats[cats.length - 1] === null ? cats.length - 1 : -1
  let lo = 0
  let hi = cats.length - (cats[cats.length - 1] === null ? 1 : 0)
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if ((cats[mid] ?? 0) < v) lo = mid + 1
    else hi = mid
  }
  return lo < cats.length && sameCategory(cats[lo] ?? null, v) ? lo : -1
}

/**
 * The categories of each feature with sklearn's grouping of infrequent ones: counts below `minFrequency` are
 * infrequent, and when more than `maxCategories` would remain, the least frequent (stable by order) join them.
 *
 * @param x - Features `[n, d]` (categories as numbers, NaN for missing).
 * @param options - `minFrequency`, `maxCategories`, and fixed categories per feature (default: those seen).
 * @returns One {@link FeatureCategories} per feature.
 * @example
 * ```ts
 * import { preprocessing } from 'whitebox-engine/learn'
 * const [f] = preprocessing.fitCategories([[1], [1], [2], [3], [1], [3]], { minFrequency: 2 })
 * f.categories // => [1, 2, 3]
 * f.grouped // => [0, 2, 1]
 * ```
 */
export function fitCategories(
  x: TensorLike,
  options: CategoryOptions & { readonly categories?: readonly (readonly number[])[] } = {},
): FeatureCategories[] {
  const m = matrixView(x, 'fitCategories', { allowNaN: true })
  const out: FeatureCategories[] = []
  for (let j = 0; j < m.d; j++) {
    const col = columnOf(m, j)
    const given = options.categories?.[j]
    let cats: (number | null)[]
    if (given !== undefined) {
      cats = [...given].sort((a, b) => a - b).map((v) => (Number.isNaN(v) ? null : v))
    } else {
      const values = Array.from(new Set(Array.from(col).filter((v) => !Number.isNaN(v)))).sort((a, b) => a - b)
      cats = Array.from(col).some(Number.isNaN) ? [...values, null] : values
    }
    const feature0: FeatureCategories = {
      categories: cats,
      grouped: cats.map((_, i) => i),
      nGrouped: cats.length,
      infrequent: [],
    }
    const counts = cats.map(() => 0)
    for (const v of col) {
      const c = categoryIndex(feature0, v)
      if (c >= 0) counts[c] = (counts[c] ?? 0) + 1
    }
    const infrequent = cats.map(() => false)
    if (options.minFrequency !== undefined) {
      const threshold = options.minFrequency >= 1 ? options.minFrequency : options.minFrequency * m.n
      counts.forEach((c, i) => {
        if (c < threshold) infrequent[i] = true
      })
    }
    const remaining = cats.length - infrequent.filter(Boolean).length + 1
    if (options.maxCategories !== undefined && options.maxCategories < remaining) {
      const keep = options.maxCategories - 1
      const order = counts.map((c, i) => ({ c, i })).sort((a, b) => a.c - b.c || a.i - b.i)
      const drop = keep === 0 ? order : order.slice(0, order.length - keep)
      for (const o of drop) infrequent[o.i] = true
    }
    const infrequentIdx = infrequent.flatMap((b, i) => (b ? [i] : []))
    if (infrequentIdx.length === 0) out.push(feature0)
    else {
      let k = 0
      const grouped = cats.map((_, i) => (infrequent[i] === true ? -1 : k++))
      out.push({
        categories: cats,
        grouped: grouped.map((g) => (g < 0 ? k : g)),
        nGrouped: k + 1,
        infrequent: infrequentIdx,
      })
    }
  }
  return out
}

function formatCategory(c: number | null): string {
  return c === null ? 'nan' : String(c)
}

/** Hyperparameters of {@link oneHotEncoder}. */
export interface OneHotHyper extends CategoryOptions {
  /** Drop one column per feature: none, the first, or the first of binary features only. */
  readonly drop: 'none' | 'first' | 'if_binary'
  readonly handleUnknown: HandleUnknown
  /** Fixed categories per feature (default: those seen in fit). */
  readonly categories?: readonly (readonly number[])[]
}

/** Parameters of a fitted one-hot encoder. */
export type OneHotParams = { readonly features: FeatureCategories[]; readonly drop: (number | null)[] }

/** The verbs of a fitted encoder. */
export interface EncoderVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

/**
 * One column per category (sklearn.preprocessing.OneHotEncoder) for numeric category codes: sorted categories, NaN as
 * its own category, optional dropping of the first (or binary) column, unknown categories as an error, all zeros or the
 * infrequent column, and rare categories grouped by `minFrequency` and `maxCategories`. The inverse maps an all-zero
 * block to the dropped category (or NaN for an unknown), the infrequent column to NaN.
 *
 * @param hyper - drop (default `'none'`), handleUnknown (default `'error'`), categories and grouping.
 * @returns The estimator.
 * @throws WhiteboxError with code `value` in transform for an unknown category with `handleUnknown: 'error'`.
 * @concept categorical-encoding
 * @example
 * ```ts
 * import { oneHotEncoder } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const e = oneHotEncoder().fit({ x: [[1, 0], [2, 1], [3, 0]] })
 * toArray(e.transform([[2, 1]])) // => [[0, 1, 0, 0, 1]]
 * e.featureNamesOut() // => ['x0_1', 'x0_2', 'x0_3', 'x1_0', 'x1_1']
 * toArray(e.inverseTransform([[0, 0, 1, 1, 0]])) // => [[3, 0]]
 * toArray(oneHotEncoder({ drop: 'first' }).fit({ x: [[1], [2]] }).transform([[1], [2]])) // => [[0], [1]]
 * ```
 */
export function oneHotEncoder(
  hyper: Partial<OneHotHyper> = {},
): Estimator<OneHotHyper, Fitted<OneHotParams> & EncoderVerbs> {
  return defineEstimator(
    {
      name: 'one-hot-encoder',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ drop: choice(['none', 'first', 'if_binary']) }),
      fit: (data, h) => {
        const features = fitCategories(data.x, h)
        const drop = features.map((f) => {
          if (h.drop === 'first') return 0
          if (h.drop === 'if_binary') return f.nGrouped === 2 ? 0 : null
          return null
        })
        return { features, drop }
      },
      methods: (p, h) => {
        const blocks = p.features.map((f, j) => f.nGrouped - (p.drop[j] === null ? 0 : 1))
        const width = blocks.reduce((a, b) => a + b, 0)
        const d = p.features.length
        return {
          nFeaturesIn: d,
          transform: (x) => {
            const m = matrixView(x, 'oneHotEncoder', { allowNaN: true, nFeatures: d })
            const out = new Float64Array(m.n * width)
            for (let i = 0; i < m.n; i++) {
              let offset = 0
              for (let j = 0; j < d; j++) {
                const f = p.features[j] as FeatureCategories
                const v = m.data[i * d + j] ?? 0
                const c = categoryIndex(f, v)
                let g: number
                if (c >= 0) g = f.grouped[c] ?? 0
                else if (h.handleUnknown === 'error') {
                  throw new WhiteboxError('value', `oneHotEncoder: unknown category ${v} in feature ${j}`)
                } else if (h.handleUnknown === 'infrequent_if_exist' && f.infrequent.length > 0) g = f.nGrouped - 1
                else g = -1
                const dropped = p.drop[j] ?? null
                if (g >= 0 && g !== dropped) out[i * width + offset + (dropped !== null && g > dropped ? g - 1 : g)] = 1
                offset += blocks[j] ?? 0
              }
            }
            return fromStorage(out, [m.n, width], 'float64')
          },
          inverseTransform: (x) => {
            const m = matrixView(x, 'oneHotEncoder', { nFeatures: width })
            const out = new Float64Array(m.n * d)
            for (let i = 0; i < m.n; i++) {
              let offset = 0
              for (let j = 0; j < d; j++) {
                const f = p.features[j] as FeatureCategories
                const dropped = p.drop[j] ?? null
                let best = -1
                for (let k = 0; k < (blocks[j] ?? 0); k++) {
                  if (
                    (m.data[i * width + offset + k] ?? 0) > 0 &&
                    (best < 0 || (m.data[i * width + offset + k] ?? 0) > (m.data[i * width + offset + best] ?? 0))
                  )
                    best = k
                }
                const g = best < 0 ? dropped : dropped !== null && best >= dropped ? best + 1 : best
                let value = Number.NaN
                if (g !== null && !(f.infrequent.length > 0 && g === f.nGrouped - 1)) {
                  const c = f.grouped.indexOf(g)
                  value = f.categories[c] ?? Number.NaN
                }
                out[i * d + j] = value
                offset += blocks[j] ?? 0
              }
            }
            return fromStorage(out, [m.n, d], 'float64')
          },
          featureNamesOut: (input) => {
            const names = inputNames(d, input, 'oneHotEncoder')
            return p.features.flatMap((f, j) => {
              const labels: string[] = []
              for (let g = 0; g < f.nGrouped; g++) {
                if (g === p.drop[j]) continue
                if (f.infrequent.length > 0 && g === f.nGrouped - 1) labels.push(`${names[j]}_infrequent`)
                else labels.push(`${names[j]}_${formatCategory(f.categories[f.grouped.indexOf(g)] ?? null)}`)
              }
              return labels
            })
          },
        }
      },
    },
    {
      drop: hyper.drop ?? 'none',
      handleUnknown: hyper.handleUnknown ?? 'error',
      ...(hyper.categories === undefined ? {} : { categories: hyper.categories }),
      ...(hyper.minFrequency === undefined ? {} : { minFrequency: hyper.minFrequency }),
      ...(hyper.maxCategories === undefined ? {} : { maxCategories: hyper.maxCategories }),
    },
  )
}

/** Hyperparameters of {@link ordinalEncoder}. */
export interface OrdinalHyper extends CategoryOptions {
  /** Unknown categories raise (`'error'`) or become `unknownValue` (`'use_encoded_value'`). */
  readonly handleUnknown: 'error' | 'use_encoded_value'
  /** The code of unknown categories (default NaN). */
  readonly unknownValue: number
  /** The code of missing values (default NaN). */
  readonly encodedMissingValue: number
  readonly categories?: readonly (readonly number[])[]
}

/** Parameters of a fitted ordinal encoder. */
export type OrdinalParams = { readonly features: FeatureCategories[] }

/**
 * Integer codes 0..k-1 for each feature's sorted categories (sklearn.preprocessing.OrdinalEncoder), with codes for
 * unknown and missing values and grouping of rare categories into one last code.
 *
 * @param hyper - handleUnknown (default `'error'`), unknownValue and encodedMissingValue (default NaN), grouping.
 * @returns The estimator.
 * @example
 * ```ts
 * import { ordinalEncoder } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const e = ordinalEncoder({ handleUnknown: 'use_encoded_value', unknownValue: -1 }).fit({ x: [[10], [30], [20]] })
 * toArray(e.transform([[30], [99]])) // => [[2], [-1]]
 * toArray(e.inverseTransform([[1]])) // => [[20]]
 * ```
 */
export function ordinalEncoder(
  hyper: Partial<OrdinalHyper> = {},
): Estimator<OrdinalHyper, Fitted<OrdinalParams> & EncoderVerbs> {
  return defineEstimator(
    {
      name: 'ordinal-encoder',
      kind: 'transformer',
      fit: (data, h) => ({ features: fitCategories(data.x, h) }),
      methods: (p, h) => {
        const d = p.features.length
        return {
          nFeaturesIn: d,
          transform: (x) => {
            const m = matrixView(x, 'ordinalEncoder', { allowNaN: true, nFeatures: d })
            const out = new Float64Array(m.n * d)
            for (let i = 0; i < m.n; i++) {
              for (let j = 0; j < d; j++) {
                const f = p.features[j] as FeatureCategories
                const v = m.data[i * d + j] ?? 0
                const c = categoryIndex(f, v)
                if (Number.isNaN(v) && c >= 0) out[i * d + j] = h.encodedMissingValue
                else if (c >= 0) out[i * d + j] = f.grouped[c] ?? 0
                else if (h.handleUnknown === 'error') {
                  throw new WhiteboxError('value', `ordinalEncoder: unknown category ${v} in feature ${j}`)
                } else out[i * d + j] = h.unknownValue
              }
            }
            return fromStorage(out, [m.n, d], 'float64')
          },
          inverseTransform: (x) => {
            const m = matrixView(x, 'ordinalEncoder', { allowNaN: true, nFeatures: d })
            const out = new Float64Array(m.n * d)
            for (let i = 0; i < m.n; i++) {
              for (let j = 0; j < d; j++) {
                const f = p.features[j] as FeatureCategories
                const code = m.data[i * d + j] ?? Number.NaN
                const infrequent = f.infrequent.length > 0 && code === f.nGrouped - 1
                const c = Number.isInteger(code) && !infrequent ? f.grouped.indexOf(code) : -1
                out[i * d + j] = c >= 0 ? (f.categories[c] ?? Number.NaN) : Number.NaN
              }
            }
            return fromStorage(out, [m.n, d], 'float64')
          },
          featureNamesOut: (input) => inputNames(d, input, 'ordinalEncoder'),
        }
      },
    },
    {
      handleUnknown: hyper.handleUnknown ?? 'error',
      unknownValue: hyper.unknownValue ?? Number.NaN,
      encodedMissingValue: hyper.encodedMissingValue ?? Number.NaN,
      ...(hyper.categories === undefined ? {} : { categories: hyper.categories }),
      ...(hyper.minFrequency === undefined ? {} : { minFrequency: hyper.minFrequency }),
      ...(hyper.maxCategories === undefined ? {} : { maxCategories: hyper.maxCategories }),
    },
  )
}

/** Parameters of a fitted label encoder or binariser: the sorted classes. */
export type LabelParams = { readonly classes: number[] }

/** The verbs of a fitted label encoder: labels to codes and back. */
export interface LabelVerbs {
  readonly classes: readonly number[]
  readonly transform: (y: TensorLike) => Tensor
  readonly inverseTransform: (codes: TensorLike) => Tensor
}

/**
 * Encode target labels as 0..k-1 (sklearn.preprocessing.LabelEncoder). It fits `y` (the features are ignored).
 *
 * @returns The estimator.
 * @throws WhiteboxError with code `value` in transform for an unseen label.
 * @example
 * ```ts
 * import { labelEncoder } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const e = labelEncoder().fit({ x: [[0], [0], [0]], y: [10, 5, 10] })
 * e.classes // => [5, 10]
 * toArray(e.transform([10, 5])) // => [1, 0]
 * toArray(e.inverseTransform([0])) // => [5]
 * ```
 */
export function labelEncoder(): Estimator<Record<string, never>, Fitted<LabelParams> & LabelVerbs> {
  return defineEstimator(
    {
      name: 'label-encoder',
      kind: 'transformer',
      fit: (data) => ({ classes: [...encodeLabels(data.y ?? data.x, 'labelEncoder').classes] }),
      methods: (p) => {
        const index = new Map(p.classes.map((c, i) => [c, i]))
        return {
          classes: p.classes,
          transform: (y) =>
            fromStorage(
              Float64Array.from(targetVector(y, 'labelEncoder'), (v) => {
                const c = index.get(v)
                if (c === undefined) throw new WhiteboxError('value', `labelEncoder: unseen label ${v}`)
                return c
              }),
              [targetVector(y, 'labelEncoder').length],
              'float64',
            ),
          inverseTransform: (codes) => {
            const c = targetVector(codes, 'labelEncoder')
            return fromStorage(
              Float64Array.from(c, (v) => {
                const label = p.classes[v]
                if (label === undefined || !Number.isInteger(v))
                  throw new WhiteboxError('value', `labelEncoder: bad code ${v}`)
                return label
              }),
              [c.length],
              'float64',
            )
          },
        }
      },
    },
    {},
  )
}

/** Hyperparameters of {@link labelBinarizer}. */
export interface LabelBinarizerHyper {
  readonly negLabel: number
  readonly posLabel: number
}

/**
 * One-vs-all indicators of target labels (sklearn.preprocessing.LabelBinarizer): `[n, k]` with `posLabel` and
 * `negLabel`, a single column for two classes. The inverse takes the largest column (or the sign threshold midway
 * between the labels for the binary column).
 *
 * @param hyper - negLabel (default 0) and posLabel (default 1).
 * @returns The estimator.
 * @example
 * ```ts
 * import { labelBinarizer } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const b = labelBinarizer().fit({ x: [[0], [0], [0]], y: [1, 2, 3] })
 * toArray(b.transform([2])) // => [[0, 1, 0]]
 * toArray(labelBinarizer().fit({ x: [[0], [0]], y: [4, 7] }).transform([7, 4])) // => [[1], [0]]
 * toArray(b.inverseTransform([[0.1, 0.2, 0.7]])) // => [3]
 * ```
 */
export function labelBinarizer(
  hyper: Partial<LabelBinarizerHyper> = {},
): Estimator<LabelBinarizerHyper, Fitted<LabelParams> & LabelVerbs> {
  const h = { negLabel: hyper.negLabel ?? 0, posLabel: hyper.posLabel ?? 1 }
  if (!(h.negLabel < h.posLabel)) throw new WhiteboxError('value', 'labelBinarizer: negLabel must be below posLabel')
  return defineEstimator(
    {
      name: 'label-binarizer',
      kind: 'transformer',
      fit: (data) => ({ classes: [...encodeLabels(data.y ?? data.x, 'labelBinarizer').classes] }),
      methods: (p, hp) => {
        const k = p.classes.length
        const cols = k === 2 ? 1 : k
        const index = new Map(p.classes.map((c, i) => [c, i]))
        return {
          classes: p.classes,
          transform: (y) => {
            const v = targetVector(y, 'labelBinarizer')
            const out = new Float64Array(v.length * cols).fill(hp.negLabel)
            v.forEach((label, i) => {
              const c = index.get(label)
              if (c === undefined) throw new WhiteboxError('value', `labelBinarizer: unseen label ${label}`)
              if (cols === 1) out[i] = c === 1 ? hp.posLabel : hp.negLabel
              else out[i * cols + c] = hp.posLabel
            })
            return fromStorage(out, [v.length, cols], 'float64')
          },
          inverseTransform: (codes) => {
            const m = matrixView(codes, 'labelBinarizer', { nFeatures: cols })
            const threshold = (hp.negLabel + hp.posLabel) / 2
            const out = new Float64Array(m.n)
            for (let i = 0; i < m.n; i++) {
              if (cols === 1) out[i] = ((m.data[i] ?? 0) > threshold ? p.classes[1] : p.classes[0]) ?? 0
              else {
                let best = 0
                for (let c = 1; c < cols; c++)
                  if ((m.data[i * cols + c] ?? 0) > (m.data[i * cols + best] ?? 0)) best = c
                out[i] = p.classes[best] ?? 0
              }
            }
            return fromStorage(out, [m.n], 'float64')
          },
        }
      },
    },
    h,
  )
}

/** A fitted multi-label binariser. */
export interface MultiLabelBinarizer {
  /** The labels, sorted (or as given). */
  readonly classes: readonly number[]
  /** Label sets to an indicator matrix `[n, k]`; unknown labels raise unless ignored. */
  transform(sets: readonly (readonly number[])[]): Tensor
  /** An indicator matrix (entries above 0.5 are on) back to label sets. */
  inverseTransform(indicators: TensorLike): number[][]
}

/**
 * Indicator matrices for sets of labels (sklearn.preprocessing.MultiLabelBinarizer): one column per label, 1 when the
 * row's set contains it. Label sets are ragged, so this is a function pair rather than a tensor estimator.
 *
 * @param sets - The training label sets.
 * @param options - Fixed classes (in that order), and whether to ignore unknown labels in transform (default false).
 * @returns The fitted binariser.
 * @throws WhiteboxError with code `value` for an unknown label unless ignored.
 * @example
 * ```ts
 * import { multiLabelBinarizer } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const b = multiLabelBinarizer([[1, 3], [2], []])
 * b.classes // => [1, 2, 3]
 * toArray(b.transform([[3], [1, 2]])) // => [[0, 0, 1], [1, 1, 0]]
 * b.inverseTransform([[1, 0, 1]]) // => [[1, 3]]
 * ```
 */
export function multiLabelBinarizer(
  sets: readonly (readonly number[])[],
  options: { readonly classes?: readonly number[]; readonly ignoreUnknown?: boolean } = {},
): MultiLabelBinarizer {
  const classes = options.classes ?? Array.from(new Set(sets.flat())).sort((a, b) => a - b)
  const index = new Map(classes.map((c, i) => [c, i]))
  const k = classes.length
  return {
    classes,
    transform: (rows) => {
      const out = new Float64Array(rows.length * k)
      rows.forEach((row, i) => {
        for (const label of row) {
          const c = index.get(label)
          if (c === undefined) {
            if (options.ignoreUnknown === true) continue
            throw new WhiteboxError('value', `multiLabelBinarizer: unknown label ${label}`)
          }
          out[i * k + c] = 1
        }
      })
      return fromStorage(out, [rows.length, k], 'float64')
    },
    inverseTransform: (indicators) => {
      const m = matrixView(indicators, 'multiLabelBinarizer', { nFeatures: k })
      return Array.from({ length: m.n }, (_, i) => classes.filter((_, c) => (m.data[i * k + c] ?? 0) > 0.5))
    },
  }
}

/** Hyperparameters of {@link targetEncoder}. */
export interface TargetEncoderHyper {
  /** Shrinkage towards the global mean: a weight m (encoding = (sum + m mean) / (count + m)) or `'auto'`. */
  readonly smooth: number | 'auto'
  /** The target type (default inferred as sklearn: non-integers continuous, two values binary, else multiclass). */
  readonly targetType: 'auto' | 'continuous' | 'binary' | 'multiclass'
  /** The cross-fitting splitter (default 5 shuffled folds, stratified for class targets). */
  readonly cv?: Splitter
  /** The number of folds of the default splitter. */
  readonly nSplits: number
}

/** Parameters of a fitted target encoder. */
export type TargetEncoderParams = {
  readonly features: FeatureCategories[]
  readonly targetType: 'continuous' | 'binary' | 'multiclass'
  readonly classes: number[]
  readonly targetMean: number[]
  /** encodings[feature][class][category] (one class for continuous and binary targets). */
  readonly encodings: number[][][]
  /** The cross-fitted encoding of the training data (what `fitTransform` returns). */
  readonly crossFitted: number[][]
}

function inferTargetType(y: Float64Array): 'continuous' | 'binary' | 'multiclass' {
  // sklearn's type_of_target: any non-integer value is continuous, two values binary, more multiclass.
  if (!Array.from(y).every(Number.isInteger)) return 'continuous'
  return new Set(y).size <= 2 ? 'binary' : 'multiclass'
}

function encodeColumn(
  codes: Int32Array,
  y: Float64Array,
  rows: readonly number[],
  nCats: number,
  smooth: number | 'auto',
): number[] {
  let mean = 0
  for (const i of rows) mean += y[i] ?? 0
  mean /= rows.length
  const sums = new Float64Array(nCats)
  const counts = new Float64Array(nCats)
  for (const i of rows) {
    const c = codes[i] ?? -1
    if (c < 0) continue
    sums[c] = (sums[c] ?? 0) + (y[i] ?? 0)
    counts[c] = (counts[c] ?? 0) + 1
  }
  if (smooth !== 'auto')
    return Array.from(sums, (s, c) =>
      counts[c] === 0 && smooth === 0 ? mean : (s + smooth * mean) / ((counts[c] ?? 0) + smooth),
    )
  let variance = 0
  for (const i of rows) variance += ((y[i] ?? 0) - mean) ** 2
  variance /= rows.length
  const ss = new Float64Array(nCats)
  for (const i of rows) {
    const c = codes[i] ?? -1
    if (c >= 0) ss[c] = (ss[c] ?? 0) + ((y[i] ?? 0) - (sums[c] ?? 0) / (counts[c] ?? 1)) ** 2
  }
  return Array.from(sums, (s, c) => {
    const n = counts[c] ?? 0
    const lambda = (variance * n) / (variance * n + (ss[c] ?? 0) / n)
    return Number.isNaN(lambda) ? mean : lambda * (s / n) + (1 - lambda) * mean
  })
}

/**
 * Replace each category by a shrunk mean of the target over its rows (sklearn.preprocessing.TargetEncoder, after
 * Micci-Barreca): continuous targets use the mean, binary the positive rate, multiclass one column per class; `auto`
 * smoothing uses the empirical-Bayes weight. `fit` also computes the cross-fitted encoding of the training rows (each
 * fold encoded by the others), which `fitTransform` returns and pipelines use, so the training encoding never sees its
 * own target.
 *
 * @param hyper - smooth (default `'auto'`), targetType, the splitter or its number of folds (default 5).
 * @returns The estimator.
 * @example
 * ```ts
 * import { targetEncoder } from 'whitebox-engine/learn/preprocessing'
 * import { toArray } from 'whitebox-engine/tensor'
 * const e = targetEncoder({ smooth: 0, targetType: 'continuous', nSplits: 2 }).fit({ x: [[0], [0], [1], [1]], y: [1, 3, 10, 20] })
 * toArray(e.transform([[0], [1], [7]])) // => [[2], [15], [8.5]]
 * ```
 */
export function targetEncoder(
  hyper: Partial<TargetEncoderHyper> = {},
): Estimator<TargetEncoderHyper, Fitted<TargetEncoderParams> & EncoderVerbs & { readonly classes: readonly number[] }> {
  return defineEstimator(
    {
      name: 'target-encoder',
      kind: 'transformer',
      hyperSpace: (): Space => dict({ smooth: real(0, 100) }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'targetEncoder', { allowNaN: true })
        const yRaw = targetVector(data.y ?? [], 'targetEncoder', m.n)
        const targetType = h.targetType === 'auto' ? inferTargetType(yRaw) : h.targetType
        const labels = targetType === 'continuous' ? undefined : encodeLabels(yRaw, 'targetEncoder')
        const classes = labels === undefined ? [] : [...labels.classes]
        const targets: Float64Array[] =
          labels === undefined
            ? [yRaw]
            : targetType === 'binary'
              ? [Float64Array.from(labels.codes)]
              : classes.map((_, k) => Float64Array.from(labels.codes, (c) => (c === k ? 1 : 0)))
        const features = fitCategories(fromView(m))
        const codes = features.map((f, j) => Int32Array.from(columnOf(m, j), (v) => categoryIndex(f, v)))
        const all = Array.from({ length: m.n }, (_, i) => i)
        const meanOf = (t: Float64Array, rows: readonly number[]): number =>
          rows.reduce((s, i) => s + (t[i] ?? 0), 0) / rows.length
        const encodings = features.map((f, j) =>
          targets.map((t) => encodeColumn(codes[j] as Int32Array, t, all, f.categories.length, h.smooth)),
        )
        const width = m.d * targets.length
        const crossFitted = Array.from({ length: m.n }, () => new Array<number>(width).fill(0))
        const splitter =
          h.cv ??
          (targetType === 'continuous'
            ? kFold(h.nSplits, { shuffle: true })
            : stratifiedKFold(h.nSplits, { shuffle: true }))
        const folds = splitIndices(
          splitter,
          { n: m.n, ...(labels === undefined ? {} : { y: labels.codes }) },
          keyFor(options, 'targetEncoder'),
        )
        for (const fold of folds) {
          const train = Array.from(fold.train)
          for (let j = 0; j < m.d; j++) {
            const f = features[j] as FeatureCategories
            targets.forEach((t, k) => {
              const enc = encodeColumn(codes[j] as Int32Array, t, train, f.categories.length, h.smooth)
              const mean = meanOf(t, train)
              for (const i of fold.test) {
                const c = codes[j]?.[i] ?? -1
                const row = crossFitted[i] as number[]
                row[j * targets.length + k] = c >= 0 ? (enc[c] ?? mean) : mean
              }
            })
          }
        }
        return { features, targetType, classes, targetMean: targets.map((t) => meanOf(t, all)), encodings, crossFitted }
      },
      methods: (p) => {
        const d = p.features.length
        const k = p.encodings[0]?.length ?? 1
        return {
          classes: p.classes,
          nFeaturesIn: d,
          transform: (x) => {
            const m = matrixView(x, 'targetEncoder', { allowNaN: true, nFeatures: d })
            const out = new Float64Array(m.n * d * k)
            for (let i = 0; i < m.n; i++) {
              for (let j = 0; j < d; j++) {
                const c = categoryIndex(p.features[j] as FeatureCategories, m.data[i * d + j] ?? 0)
                for (let t = 0; t < k; t++) {
                  out[i * d * k + j * k + t] = c >= 0 ? (p.encodings[j]?.[t]?.[c] ?? 0) : (p.targetMean[t] ?? 0)
                }
              }
            }
            return fromStorage(out, [m.n, d * k], 'float64')
          },
          inverseTransform: () => {
            throw new WhiteboxError('not-implemented', 'targetEncoder: target encodings are not invertible')
          },
          featureNamesOut: (input) => {
            const names = inputNames(d, input, 'targetEncoder')
            return p.targetType === 'multiclass' ? names.flatMap((n) => p.classes.map((c) => `${n}_${c}`)) : names
          },
        }
      },
      fitTransform: (fitted) => {
        const rows = fitted.params.crossFitted
        return fromStorage(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0], 'float64')
      },
    },
    {
      smooth: hyper.smooth ?? 'auto',
      targetType: hyper.targetType ?? 'auto',
      nSplits: hyper.nSplits ?? 5,
      ...(hyper.cv === undefined ? {} : { cv: hyper.cv }),
    },
  )
}
