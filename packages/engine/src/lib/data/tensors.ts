/**
 * Building datasets from plain arrays and moving between datasets and tensors: constructors for regression,
 * classification, multilabel and multi-output sets, `table` for mixed-type data frames and `tableToXy` to encode one as
 * a feature matrix (one-hot categoricals), `toTensors` and `fromTensors` for transforming per-row fields, and
 * `describe`, a pandas-style summary of a dataset.
 */

import { MdmlError, quantileOf } from '../base/index.ts'
import { asTensor, cast, fromStorage, isTensor, type Tensor, type TensorLike } from '../tensor/index.ts'
import { cardinality, concrete, concreteFloat, defaultNames, metaFor } from './schema/_coerce.ts'
import { type Dataset, type DatasetMeta, perRowFields, rowCount } from './schema/common.ts'
import type {
  ClassificationData,
  MultilabelData,
  MultiOutputData,
  RegressionData,
  TableColumn,
  TableData,
} from './schema/tabular.ts'

type TabularMeta = Partial<DatasetMeta> & { readonly featureNames?: readonly string[]; readonly weights?: TensorLike }

function features(x: TensorLike, name: string): Tensor {
  const xs = concreteFloat(x, name)
  return xs.shape.length === 1 ? xs.backend.wrap(xs.dtype, [xs.shape[0] ?? 0, 1], xs.data) : xs
}

function weightsOf(meta: TabularMeta, n: number, name: string): { readonly weights?: Tensor } {
  if (meta.weights === undefined) return {}
  const w = concreteFloat(meta.weights, name)
  if ((w.shape[0] ?? 0) !== n) throw new MdmlError('shape', `${name}: ${w.shape[0]} weights for ${n} rows`)
  return { weights: w }
}

function strip<T extends TabularMeta>(meta: T): Partial<DatasetMeta> {
  const { featureNames: _f, weights: _w, ...rest } = meta as TabularMeta & Record<string, unknown>
  const out: Record<string, unknown> = { ...rest }
  for (const k of ['targetNames', 'classes', 'labels', 'outputs']) delete out[k]
  return out as Partial<DatasetMeta>
}

/**
 * A regression dataset from feature and target arrays.
 *
 * @param x - Features `[n, d]` (a 1-D array is one feature).
 * @param y - Targets `[n]` or `[n, k]`.
 * @param meta - Name, feature and target names, sample weights and other metadata.
 * @returns The dataset.
 * @concept datasets
 * @example
 * ```ts
 * import { xy } from 'mdml-engine/data'
 * xy([[1, 2], [3, 4]], [0.5, 1.5]).featureNames // => ['x0', 'x1']
 * ```
 */
export function xy(
  x: TensorLike,
  y: TensorLike,
  meta: TabularMeta & { readonly targetNames?: readonly string[] } = {},
): RegressionData {
  const xs = features(x, 'xy')
  const ys = concreteFloat(y, 'xy')
  const n = xs.shape[0] ?? 0
  if ((ys.shape[0] ?? 0) !== n) throw new MdmlError('shape', `xy: ${ys.shape[0]} targets for ${n} rows`)
  const k = ys.shape.length === 2 ? (ys.shape[1] ?? 1) : 1
  return {
    kind: 'regression',
    x: xs,
    y: ys,
    ...weightsOf(meta, n, 'xy'),
    featureNames: meta.featureNames ?? defaultNames('x', xs.shape[1] ?? 0),
    targetNames: meta.targetNames ?? (k === 1 ? ['y'] : defaultNames('y', k)),
    meta: metaFor('regression', strip(meta)),
  }
}

/**
 * A classification dataset from features and integer labels.
 *
 * @param x - Features `[n, d]`.
 * @param y - Class indices `[n]`.
 * @param meta - Class names, feature names, sample weights and other metadata.
 * @returns The dataset.
 * @example
 * ```ts
 * import { xyClasses } from 'mdml-engine/data'
 * xyClasses([[0], [1]], [0, 1]).classes // => ['0', '1']
 * ```
 */
export function xyClasses(
  x: TensorLike,
  y: TensorLike,
  meta: TabularMeta & { readonly classes?: readonly string[] } = {},
): ClassificationData {
  const xs = features(x, 'xyClasses')
  const ys = concrete(y, 'int32', 'xyClasses')
  const n = xs.shape[0] ?? 0
  if ((ys.shape[0] ?? 0) !== n) throw new MdmlError('shape', `xyClasses: ${ys.shape[0]} labels for ${n} rows`)
  const k = Math.max(cardinality(ys.data), meta.classes?.length ?? 0)
  return {
    kind: 'classification',
    x: xs,
    y: ys,
    ...weightsOf(meta, n, 'xyClasses'),
    classes: meta.classes ?? Array.from({ length: k }, (_, i) => String(i)),
    featureNames: meta.featureNames ?? defaultNames('x', xs.shape[1] ?? 0),
    meta: metaFor('classification', strip(meta)),
  }
}

/**
 * A multilabel dataset from features and a 0/1 indicator matrix.
 *
 * @param x - Features `[n, d]`.
 * @param y - Label indicators `[n, L]`.
 * @param meta - Label names, feature names and other metadata.
 * @returns The dataset.
 * @example
 * ```ts
 * import { xyMultilabel } from 'mdml-engine/data'
 * xyMultilabel([[0], [1]], [[1, 0, 1], [0, 1, 0]]).labels // => ['l0', 'l1', 'l2']
 * ```
 */
export function xyMultilabel(
  x: TensorLike,
  y: TensorLike,
  meta: TabularMeta & { readonly labels?: readonly string[] } = {},
): MultilabelData {
  const xs = features(x, 'xyMultilabel')
  const ys = concrete(y, 'int32', 'xyMultilabel')
  const n = xs.shape[0] ?? 0
  if (ys.shape.length !== 2 || ys.shape[0] !== n)
    throw new MdmlError('shape', `xyMultilabel: indicators must be [${n}, L]`)
  return {
    kind: 'multilabel',
    x: xs,
    y: ys,
    ...weightsOf(meta, n, 'xyMultilabel'),
    labels: meta.labels ?? defaultNames('l', ys.shape[1] ?? 0),
    featureNames: meta.featureNames ?? defaultNames('x', xs.shape[1] ?? 0),
    meta: metaFor('multilabel', strip(meta)),
  }
}

/**
 * A multiclass-multioutput dataset from features and one class column per output.
 *
 * @param x - Features `[n, d]`.
 * @param y - Class indices `[n, k]`.
 * @param meta - Output names and classes, feature names and other metadata.
 * @returns The dataset.
 * @example
 * ```ts
 * import { xyMultiOutput } from 'mdml-engine/data'
 * xyMultiOutput([[0], [1]], [[0, 2], [1, 0]]).outputs[1]?.classes // => ['0', '1', '2']
 * ```
 */
export function xyMultiOutput(
  x: TensorLike,
  y: TensorLike,
  meta: TabularMeta & {
    readonly outputs?: readonly { readonly name: string; readonly classes: readonly string[] }[]
  } = {},
): MultiOutputData {
  const xs = features(x, 'xyMultiOutput')
  const ys = concrete(y, 'int32', 'xyMultiOutput')
  const n = xs.shape[0] ?? 0
  if (ys.shape.length !== 2 || ys.shape[0] !== n)
    throw new MdmlError('shape', `xyMultiOutput: targets must be [${n}, k]`)
  const k = ys.shape[1] ?? 0
  const outputs =
    meta.outputs ??
    Array.from({ length: k }, (_, j) => {
      const col = Array.from({ length: n }, (_, i) => ys.data[i * k + j] ?? 0)
      return { name: `y${j}`, classes: Array.from({ length: cardinality(col) }, (_, c) => String(c)) }
    })
  return {
    kind: 'multioutput',
    x: xs,
    y: ys,
    ...weightsOf(meta, n, 'xyMultiOutput'),
    outputs,
    featureNames: meta.featureNames ?? defaultNames('x', xs.shape[1] ?? 0),
    meta: metaFor('multioutput', strip(meta)),
  }
}

/** A column given to {@link table}: numbers (null is missing), strings, booleans, or a ready column. */
export type ColumnInput = readonly (number | null)[] | readonly (string | null)[] | readonly boolean[] | TableColumn

function isColumn(c: ColumnInput): c is TableColumn {
  return !Array.isArray(c)
}

function toColumn(name: string, c: ColumnInput, text: boolean, levels?: readonly string[]): TableColumn {
  if (isColumn(c)) return c
  const first = c.find((v) => v !== null)
  if (typeof first === 'boolean') {
    return {
      kind: 'boolean',
      values: fromStorage(
        Uint8Array.from(c as boolean[], (v) => (v ? 1 : 0)),
        [c.length],
        'bool',
      ),
    }
  }
  if (typeof first === 'string' || (first === undefined && levels !== undefined)) {
    const values = c as readonly (string | null)[]
    if (text) return { kind: 'text', values }
    const lv = levels ?? [...new Set(values.filter((v): v is string => v !== null))].sort()
    const index = new Map(lv.map((l, i) => [l, i]))
    const codes = Int32Array.from(values, (v) => {
      if (v === null) return -1
      const i = index.get(v)
      if (i === undefined) throw new MdmlError('value', `table: ${name} has value ${v} outside its levels`)
      return i
    })
    return { kind: 'categorical', codes: fromStorage(codes, [codes.length], 'int32'), levels: lv }
  }
  const values = Float64Array.from(c as readonly (number | null)[], (v) => (v === null ? Number.NaN : v))
  return { kind: 'numeric', values: fromStorage(values, [values.length], 'float64') }
}

/**
 * A table of named mixed-type columns (a data frame): numbers become numeric columns (null as NaN), strings become
 * categorical columns with sorted levels (or text columns when listed in `text`), booleans become boolean columns.
 *
 * @param columns - The columns by name.
 * @param options - The target column, text columns, explicit category levels and metadata.
 * @returns The table.
 * @throws MdmlError with code `shape` when columns differ in length.
 * @example
 * ```ts
 * import { table } from 'mdml-engine/data'
 * const t = table({ age: [30, null], city: ['Cork', 'Bray'] })
 * t.columns['city']?.kind // => 'categorical'
 * ```
 */
export function table(
  columns: Readonly<Record<string, ColumnInput>>,
  options: {
    readonly target?: string
    readonly text?: readonly string[]
    readonly levels?: Readonly<Record<string, readonly string[]>>
    readonly meta?: Partial<DatasetMeta>
  } = {},
): TableData {
  const out: Record<string, TableColumn> = {}
  for (const [name, c] of Object.entries(columns)) {
    out[name] = toColumn(name, c, options.text?.includes(name) === true, options.levels?.[name])
  }
  if (options.target !== undefined && out[options.target] === undefined)
    throw new MdmlError('value', `table: no target column ${options.target}`)
  const data: TableData = {
    kind: 'table',
    columns: out,
    ...(options.target === undefined ? {} : { target: options.target }),
    meta: metaFor('table', options.meta),
  }
  rowCount(data)
  return data
}

/** Options of {@link tableToXy}. */
export interface TableEncoding {
  /** Encode categorical features one-hot (default) or as their integer codes. */
  readonly categorical?: 'onehot' | 'codes'
  /** Keep the first level's indicator in one-hot encoding (default true; false drops it, as `drop='first'`). */
  readonly keepFirst?: boolean
}

/**
 * A table as a feature matrix and target: numeric and boolean columns as they are, categorical columns one-hot (or as
 * codes), text columns dropped. A categorical target gives classification data; a numeric target regression data.
 *
 * @param data - The table.
 * @param options - How categorical features are encoded.
 * @returns The tabular dataset.
 * @throws MdmlError with code `value` when the table has no target.
 * @example
 * ```ts
 * import { table, tableToXy } from 'mdml-engine/data'
 * const d = tableToXy(table({ size: [1, 2, 3], colour: ['red', 'blue', 'red'], label: ['a', 'b', 'a'] }, { target: 'label' }))
 * d.featureNames // => ['size', 'colour=blue', 'colour=red']
 * ```
 */
export function tableToXy(data: TableData, options: TableEncoding = {}): RegressionData | ClassificationData {
  const target = data.target
  if (target === undefined) throw new MdmlError('value', 'tableToXy: the table has no target column')
  const n = rowCount(data)
  const cols: Float64Array[] = []
  const names: string[] = []
  for (const [name, c] of Object.entries(data.columns)) {
    if (name === target) continue
    switch (c.kind) {
      case 'numeric':
        cols.push(Float64Array.from(c.values.data))
        names.push(name)
        break
      case 'boolean':
        cols.push(Float64Array.from(c.values.data))
        names.push(name)
        break
      case 'categorical':
        if (options.categorical === 'codes') {
          cols.push(Float64Array.from(c.codes.data, (v) => (v < 0 ? Number.NaN : v)))
          names.push(name)
        } else {
          for (const [l, level] of c.levels.entries()) {
            if (l === 0 && options.keepFirst === false) continue
            cols.push(Float64Array.from(c.codes.data, (v) => (v < 0 ? Number.NaN : v === l ? 1 : 0)))
            names.push(`${name}=${level}`)
          }
        }
        break
      case 'text':
        break
    }
  }
  const x = new Float64Array(n * cols.length)
  for (const [j, col] of cols.entries()) for (let i = 0; i < n; i++) x[i * cols.length + j] = col[i] ?? 0
  const xs = fromStorage(x, [n, cols.length], 'float64')
  const t = data.columns[target] as TableColumn
  const meta = { ...data.meta, featureNames: names }
  if (t.kind === 'categorical') return xyClasses(xs, t.codes, { ...meta, classes: t.levels })
  if (t.kind === 'boolean') return xyClasses(xs, cast(t.values, 'int32'), { ...meta, classes: ['false', 'true'] })
  if (t.kind === 'numeric') return xy(xs, t.values, { ...meta, targetNames: [target] })
  throw new MdmlError('value', 'tableToXy: a text column cannot be the target')
}

/**
 * The per-row tensor fields of a dataset by name (a table is first encoded with {@link tableToXy}); array fields such
 * as raw documents are left out.
 *
 * @param data - The dataset.
 * @returns The tensors by field name.
 * @example
 * ```ts
 * import { toTensors, xy } from 'mdml-engine/data'
 * Object.keys(toTensors(xy([[1]], [2]))) // => ['x', 'y']
 * ```
 */
export function toTensors(data: Dataset): Readonly<Record<string, Tensor>> {
  if (data.kind === 'table') return toTensors(tableToXy(data))
  const record = data as unknown as Readonly<Record<string, unknown>>
  const out: Record<string, Tensor> = {}
  for (const f of perRowFields(data)) {
    const v = record[f]
    if (isTensor(v)) out[f] = v
  }
  return out
}

/**
 * A dataset like `template` with its per-row tensor fields replaced, checking row counts: the inverse of
 * {@link toTensors}, for data transformed as tensors.
 *
 * @param template - The dataset whose names and metadata are kept.
 * @param tensors - Replacement fields by name.
 * @returns The new dataset.
 * @throws MdmlError with code `value` for a field that is not a per-row tensor field of the template.
 * @example
 * ```ts
 * import { fromTensors, xy } from 'mdml-engine/data'
 * import { tensor } from 'mdml-engine/tensor'
 * fromTensors(xy([[1]], [2]), { y: tensor([5]) }).y.data // => Float64Array [5]
 * ```
 */
export function fromTensors<D extends Dataset>(template: D, tensors: Readonly<Record<string, TensorLike>>): D {
  const fields = new Set(perRowFields(template))
  const record = template as unknown as Readonly<Record<string, unknown>>
  const out: Record<string, unknown> = { ...template }
  for (const [name, value] of Object.entries(tensors)) {
    if (!fields.has(name) || !isTensor(record[name]))
      throw new MdmlError('value', `fromTensors: ${name} is not a per-row tensor field of ${template.kind} data`)
    const old = record[name] as Tensor
    const t = asTensor(value)
    out[name] = t.dtype === old.dtype ? t : cast(t, old.dtype)
  }
  const result = out as unknown as D
  rowCount(result)
  return result
}

/** Summary statistics of one numeric feature. */
export interface FeatureSummary {
  readonly name: string
  readonly count: number
  readonly missing: number
  readonly mean: number
  readonly std: number
  readonly min: number
  readonly q25: number
  readonly median: number
  readonly q75: number
  readonly max: number
}

/** What {@link describe} returns. */
export interface DatasetSummary {
  readonly name: string
  readonly kind: Dataset['kind']
  readonly rows: number
  /** Every per-row field with its shape (or length) and dtype. */
  readonly fields: readonly { readonly name: string; readonly shape: readonly number[]; readonly dtype: string }[]
  /** Statistics of each numeric feature column. */
  readonly features: readonly FeatureSummary[]
  /** Counts per class, for labelled data. */
  readonly classCounts?: Readonly<Record<string, number>>
}

function summarise(name: string, values: ArrayLike<number>): FeatureSummary {
  const present = Array.from(values).filter((v) => !Number.isNaN(v))
  const count = present.length
  const mean = count === 0 ? Number.NaN : present.reduce((s, v) => s + v, 0) / count
  const variance = count < 2 ? Number.NaN : present.reduce((s, v) => s + (v - mean) ** 2, 0) / (count - 1)
  return {
    name,
    count,
    missing: values.length - count,
    mean,
    std: Math.sqrt(variance),
    min: quantileOf(present, 0),
    q25: quantileOf(present, 0.25),
    median: quantileOf(present, 0.5),
    q75: quantileOf(present, 0.75),
    max: quantileOf(present, 1),
  }
}

function column(x: Tensor, j: number): Float64Array {
  const [n = 0, d = 1] = x.shape
  return Float64Array.from({ length: n }, (_, i) => x.data[i * d + j] ?? Number.NaN)
}

/**
 * Describe a dataset (pandas `describe`): its fields and shapes, statistics of each numeric feature (count, missing,
 * mean, standard deviation with `ddof = 1`, quartiles by linear interpolation, extremes) and class counts.
 *
 * @param data - The dataset.
 * @returns The summary.
 * @example
 * ```ts
 * import { describe, xy } from 'mdml-engine/data'
 * describe(xy([[1], [2], [3], [4]], [0, 0, 1, 1])).features[0]?.median // => 2.5
 * ```
 */
export function describe(data: Dataset): DatasetSummary {
  const record = data as unknown as Readonly<Record<string, unknown>>
  const fields = perRowFields(data).map((f) => {
    const v = record[f]
    if (isTensor(v)) return { name: f, shape: v.shape, dtype: v.dtype }
    if (Array.isArray(v)) return { name: f, shape: [v.length], dtype: 'array' }
    return { name: f, shape: [rowCount(data)], dtype: 'table' }
  })
  const feats: FeatureSummary[] = []
  if (data.kind === 'table') {
    for (const [name, c] of Object.entries(data.columns)) {
      if (c.kind === 'numeric') feats.push(summarise(name, c.values.data))
    }
  } else if ('x' in data && 'featureNames' in data) {
    const x = data.x as Tensor
    const names = data.featureNames as readonly string[]
    for (let j = 0; j < (x.shape[1] ?? 1); j++) feats.push(summarise(names[j] ?? `x${j}`, column(x, j)))
  }
  let classCounts: Record<string, number> | undefined
  const counted = (labels: ArrayLike<number>, names: readonly string[] | undefined) => {
    const out: Record<string, number> = {}
    for (let i = 0; i < labels.length; i++) {
      const c = labels[i] ?? -1
      const key = c < 0 ? 'missing' : (names?.[c] ?? String(c))
      out[key] = (out[key] ?? 0) + 1
    }
    return out
  }
  if (data.kind === 'classification') classCounts = counted(data.y.data, data.classes)
  else if (data.kind === 'image' && data.labels !== undefined) classCounts = counted(data.labels.data, data.classes)
  else if (data.kind === 'corpus' && data.labels !== undefined) classCounts = counted(data.labels.data, data.classes)
  else if (data.kind === 'table' && data.target !== undefined) {
    const t = data.columns[data.target]
    if (t?.kind === 'categorical') classCounts = counted(t.codes.data, t.levels)
  }
  return {
    name: data.meta.name,
    kind: data.kind,
    rows: rowCount(data),
    fields,
    features: feats,
    ...(classCounts === undefined ? {} : { classCounts }),
  }
}
