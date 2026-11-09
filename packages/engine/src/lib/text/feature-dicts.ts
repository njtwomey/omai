/**
 * Feature vectorisers over records rather than text, scikit-learn's `feature_extraction.DictVectorizer` and
 * `FeatureHasher`: `dictVectorize` turns records (`{ city: 'Paris', height: 1.8, tags: ['a', 'b'] }`) into a sparse
 * matrix, numbers as columns of their own and strings (and lists of strings) as one-hot `name=value` columns, with the
 * vocabulary kept for new records; `featureHash` maps the same records (or pairs, or bare strings) into a fixed
 * number of columns by the hashing trick, with no vocabulary, as `hashingVectorize` does for text.
 */

import { OmaiError } from '../base/index.ts'
import { type CsrMatrix, fromCoo } from '../numerics/sparse/index.ts'
import { toFlat } from '../tensor/index.ts'
import { compareCodePoints } from './_classes.ts'
import { murmurhash3 } from './hashing.ts'

/** A value of a record: a number or boolean (a column of its own), a string (one-hot), or a list of strings (each one-hot). */
export type FeatureValue = number | boolean | string | readonly string[]

/** A record of named features. */
export type FeatureRecord = Readonly<Record<string, FeatureValue>>

/** A fitted {@link dictVectorize}: the names of the columns, in order, and the separator of one-hot names. */
export interface DictVectorizer {
  readonly featureNames: readonly string[]
  readonly separator: string
}

/** Options of {@link dictVectorize}. */
export interface DictVectorizerOptions {
  /** Between a feature and its value in one-hot column names (default `=`). */
  readonly separator?: string
  /** Order the columns by name, by code point (default true; false: by first appearance). */
  readonly sort?: boolean
}

/** The (column name, value) pairs of one record. */
function entriesOf(record: FeatureRecord, separator: string): [string, number][] {
  const out: [string, number][] = []
  for (const [name, value] of Object.entries(record)) {
    if (typeof value === 'string') out.push([`${name}${separator}${value}`, 1])
    else if (typeof value === 'number') out.push([name, value])
    else if (typeof value === 'boolean') out.push([name, value ? 1 : 0])
    else
      for (const item of value) {
        if (typeof item !== 'string')
          throw new OmaiError('value', `dictVectorize: the list under "${name}" must hold strings`)
        out.push([`${name}${separator}${item}`, 1])
      }
  }
  return out
}

function matrixOf(
  vocabulary: ReadonlyMap<string, number>,
  records: readonly FeatureRecord[],
  separator: string,
): CsrMatrix {
  const rows: number[] = []
  const cols: number[] = []
  const values: number[] = []
  for (const [i, record] of records.entries())
    for (const [name, value] of entriesOf(record, separator)) {
      const j = vocabulary.get(name)
      if (j === undefined) continue
      rows.push(i)
      cols.push(j)
      values.push(value)
    }
  return fromCoo(rows, cols, values, [records.length, vocabulary.size])
}

/**
 * Fit a dictionary vectoriser and transform the records (sklearn.feature_extraction.DictVectorizer.fit_transform):
 * a number or boolean is a column named by its feature, a string `v` under `f` is the column `f=v` with value 1, and
 * each string of a list is such a column too; columns are ordered by name unless `sort` is off.
 *
 * @param records - The records.
 * @param options - The separator and whether to sort the columns.
 * @returns The vectoriser and the sparse matrix `[records, columns]`.
 * @throws OmaiError with code `value` for a list that holds something other than strings.
 * @index function Vectorise records of numbers and categories: numbers as columns, strings one-hot.
 * @example
 * ```ts
 * import { dictVectorize } from 'omai-engine/text'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * const { vectorizer, matrix } = dictVectorize([{ city: 'Paris', temp: 12 }, { city: 'Rome', temp: 20 }])
 * vectorizer.featureNames // => ['city=Paris', 'city=Rome', 'temp']
 * toArray(toDense(matrix)) // => [[1, 0, 12], [0, 1, 20]]
 * ```
 */
export function dictVectorize(
  records: readonly FeatureRecord[],
  options: DictVectorizerOptions = {},
): { readonly vectorizer: DictVectorizer; readonly matrix: CsrMatrix } {
  const separator = options.separator ?? '='
  const names: string[] = []
  const seen = new Set<string>()
  for (const record of records)
    for (const [name] of entriesOf(record, separator))
      if (!seen.has(name)) {
        seen.add(name)
        names.push(name)
      }
  if (options.sort !== false) names.sort(compareCodePoints)
  const vectorizer: DictVectorizer = { featureNames: names, separator }
  return { vectorizer, matrix: dictTransform(vectorizer, records) }
}

/**
 * The matrix of new records under a fitted vectoriser (DictVectorizer.transform): features it has not seen are ignored.
 *
 * @param vectorizer - The fitted vectoriser.
 * @param records - The records.
 * @returns The sparse matrix.
 * @example
 * ```ts
 * import { dictTransform, dictVectorize } from 'omai-engine/text'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * const { vectorizer } = dictVectorize([{ a: 1, b: 'x' }])
 * toArray(toDense(dictTransform(vectorizer, [{ a: 5, b: 'y', c: 2 }]))) // => [[5, 0]]
 * ```
 */
export function dictTransform(vectorizer: DictVectorizer, records: readonly FeatureRecord[]): CsrMatrix {
  const vocabulary = new Map(vectorizer.featureNames.map((name, j) => [name, j]))
  return matrixOf(vocabulary, records, vectorizer.separator)
}

/**
 * The records a matrix stands for (DictVectorizer.inverse_transform): for each row, its non-zero columns by name.
 *
 * @param vectorizer - The fitted vectoriser.
 * @param matrix - A sparse matrix, or dense rows `[n, columns]`.
 * @returns One record of column name to value per row (a one-hot column keeps its `name=value` key and the value 1).
 * @throws OmaiError with code `shape` unless the matrix has one column per feature name.
 * @example
 * ```ts
 * import { dictInverse, dictVectorize } from 'omai-engine/text'
 * const { vectorizer, matrix } = dictVectorize([{ city: 'Paris', temp: 12 }])
 * dictInverse(vectorizer, matrix) // => [{ 'city=Paris': 1, temp: 12 }]
 * ```
 */
export function dictInverse(
  vectorizer: DictVectorizer,
  matrix: CsrMatrix | readonly (readonly number[])[],
): Record<string, number>[] {
  const columns = vectorizer.featureNames.length
  if (Array.isArray(matrix)) {
    return (matrix as readonly (readonly number[])[]).map((row) => {
      if (row.length !== columns)
        throw new OmaiError('shape', `dictInverse: ${row.length} columns for ${columns} features`)
      return Object.fromEntries(row.flatMap((v, j) => (v === 0 ? [] : [[vectorizer.featureNames[j] as string, v]])))
    })
  }
  const m = matrix as CsrMatrix
  if (m.shape[1] !== columns) throw new OmaiError('shape', `dictInverse: ${m.shape[1]} columns for ${columns} features`)
  const values = toFlat(m.values)
  return Array.from({ length: m.shape[0] }, (_, i) =>
    Object.fromEntries(
      Array.from({ length: (m.indptr[i + 1] ?? 0) - (m.indptr[i] ?? 0) }, (__, k) => {
        const at = (m.indptr[i] ?? 0) + k
        return [vectorizer.featureNames[m.indices[at] ?? 0] as string, values[at] ?? 0] as const
      }).filter(([, v]) => v !== 0),
    ),
  )
}

/** Options of {@link featureHash}. */
export interface FeatureHasherOptions {
  /** The number of columns (default 2^20). */
  readonly nFeatures?: number
  /** Flip the sign of features whose hash is negative, so collisions cancel on average (default true). */
  readonly alternateSign?: boolean
}

/** What a sample to hash is: a record, a list of (feature, value) pairs, or a list of feature strings (value 1 each). */
export type HashedSample = FeatureRecord | readonly (readonly [string, number | string])[] | readonly string[]

function pairsOf(sample: HashedSample): [string, number | string | boolean][] {
  if (!Array.isArray(sample))
    return Object.entries(sample as FeatureRecord).flatMap(([f, v]) =>
      typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
        ? [[f, v] as [string, number | string | boolean]]
        : v.map((item) => [`${f}=${item}`, 1] as [string, number | string | boolean]),
    )
  return (sample as readonly (string | readonly [string, number | string])[]).map((e) =>
    typeof e === 'string' ? [e, 1] : [e[0], e[1]],
  )
}

/**
 * The hashing trick over records (sklearn.feature_extraction.FeatureHasher): each (feature, value) goes to column
 * `|h| mod nFeatures` of the signed 32-bit MurmurHash3 `h` of its name, with the value negated when `h < 0` (unless
 * `alternateSign` is off), and collisions summed. A string value `v` under `f` is the feature `f=v` with value 1; zero
 * values are dropped. There is no vocabulary to keep, so new records hash the same way.
 *
 * @param samples - Records, lists of pairs, or lists of feature strings.
 * @param options - The width and sign alternation.
 * @returns The sparse matrix `[samples, nFeatures]`.
 * @throws OmaiError with code `value` for a non-positive width.
 * @index function Hash records of features into a fixed-width sparse matrix.
 * @example
 * ```ts
 * import { featureHash } from 'omai-engine/text'
 * import { toDense } from 'omai-engine/numerics/sparse'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(toDense(featureHash([{ foo: 1, bar: 2 }, { foo: 3 }], { nFeatures: 4 }))) // => [[-1, 2, 0, 0], [-3, 0, 0, 0]]
 * ```
 */
export function featureHash(samples: readonly HashedSample[], options: FeatureHasherOptions = {}): CsrMatrix {
  const width = options.nFeatures ?? 2 ** 20
  if (!Number.isInteger(width) || width < 1)
    throw new OmaiError('value', 'featureHash: nFeatures must be a positive integer')
  const rows: number[] = []
  const cols: number[] = []
  const values: number[] = []
  for (const [i, sample] of samples.entries())
    for (const [feature, raw] of pairsOf(sample)) {
      const name = typeof raw === 'string' ? `${feature}=${raw}` : feature
      const value = typeof raw === 'string' ? 1 : typeof raw === 'boolean' ? (raw ? 1 : 0) : raw
      if (value === 0) continue
      const h = murmurhash3(name)
      rows.push(i)
      cols.push(Math.abs(h) % width)
      values.push(options.alternateSign === false || h >= 0 ? value : -value)
    }
  return fromCoo(rows, cols, values, [samples.length, width])
}
