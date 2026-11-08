/**
 * Schemas for tabular data, the datasets estimators fit: regression, classification, multilabel and
 * multiclass-multioutput sets of a feature matrix with targets (scikit-learn's formats), and tables of named mixed-type
 * columns like a data frame. Each has a zod schema for validating data that crosses a boundary.
 *
 * @index structure Dataset schemas for regression, classification and multilabel data.
 */

import { z } from 'zod'
import { isTensor, type Tensor } from '../../tensor/index.ts'
import type { DatasetKind, DatasetMeta } from './common.ts'

/** Features with real-valued targets. */
export interface RegressionData {
  readonly kind: 'regression'
  /** Features `[n, d]`. */
  readonly x: Tensor
  /** Targets `[n]` (or `[n, k]` for several outputs). */
  readonly y: Tensor
  /** Optional sample weights `[n]`. */
  readonly weights?: Tensor
  readonly featureNames: readonly string[]
  readonly targetNames: readonly string[]
  readonly meta: DatasetMeta
}

/** Features with class labels. */
export interface ClassificationData {
  readonly kind: 'classification'
  /** Features `[n, d]`. */
  readonly x: Tensor
  /** Class indices `[n]` (int32). */
  readonly y: Tensor<'int32'>
  /** Optional sample weights `[n]`. */
  readonly weights?: Tensor
  readonly classes: readonly string[]
  readonly featureNames: readonly string[]
  readonly meta: DatasetMeta
}

/** Features with several binary labels per row (sklearn's multilabel indicator format). */
export interface MultilabelData {
  readonly kind: 'multilabel'
  /** Features `[n, d]`. */
  readonly x: Tensor
  /** Label indicators `[n, L]` (int32, 0 or 1). */
  readonly y: Tensor<'int32'>
  readonly weights?: Tensor
  readonly labels: readonly string[]
  readonly featureNames: readonly string[]
  readonly meta: DatasetMeta
}

/** Features with several categorical outputs per row (sklearn's multiclass-multioutput format). */
export interface MultiOutputData {
  readonly kind: 'multioutput'
  /** Features `[n, d]`. */
  readonly x: Tensor
  /** Class indices `[n, k]` (int32), one column per output. */
  readonly y: Tensor<'int32'>
  readonly weights?: Tensor
  /** Each output's name and classes. */
  readonly outputs: readonly { readonly name: string; readonly classes: readonly string[] }[]
  readonly featureNames: readonly string[]
  readonly meta: DatasetMeta
}

/**
 * One column of a table. Missing values are `NaN` (numeric), code `-1` (categorical) or `null` (text).
 */
export type TableColumn =
  | { readonly kind: 'numeric'; readonly values: Tensor<'float64'>; readonly unit?: string }
  | {
      readonly kind: 'categorical'
      readonly codes: Tensor<'int32'>
      readonly levels: readonly string[]
      readonly ordered?: boolean
    }
  | { readonly kind: 'boolean'; readonly values: Tensor<'bool'> }
  | { readonly kind: 'text'; readonly values: readonly (string | null)[] }

/** Named columns of mixed types (a data frame), with an optional target column. */
export interface TableData {
  readonly kind: 'table'
  readonly columns: Readonly<Record<string, TableColumn>>
  /** The column to predict, when there is one. */
  readonly target?: string
  readonly meta: DatasetMeta
}

/** Datasets with a feature matrix `x` and targets `y`: what estimators fit. */
export type TabularData = RegressionData | ClassificationData | MultilabelData | MultiOutputData

const DATASET_KINDS: readonly DatasetKind[] = [
  'regression',
  'classification',
  'multilabel',
  'multioutput',
  'table',
  'interactions',
  'queries',
  'corpus',
  'sequence-labelling',
  'parallel',
  'sequence',
  'timeseries',
  'events',
  'comparisons',
  'graph',
  'triples',
  'image',
  'trajectories',
]

/**
 * Schema for a tensor (any dtype and shape), for validating datasets that cross a boundary.
 *
 * @example
 * ```ts
 * import { TensorSchema } from 'omai-engine/data'
 * import { tensor } from 'omai-engine/tensor'
 * TensorSchema.safeParse(tensor([1, 2])).success // => true
 * ```
 */
export const TensorSchema: z.ZodType<Tensor> = z.custom<Tensor>((v) => isTensor(v), { message: 'expected a tensor' })

const names = z.array(z.string()).readonly()

/**
 * Schema for dataset metadata.
 *
 * @example
 * ```ts
 * import { DatasetMetaSchema } from 'omai-engine/data'
 * DatasetMetaSchema.parse({ name: 'd', description: '', task: 'regression' }).task // => 'regression'
 * ```
 */
export const DatasetMetaSchema: z.ZodType<DatasetMeta> = z.object({
  name: z.string(),
  description: z.string(),
  task: z.enum(DATASET_KINDS as [DatasetKind, ...DatasetKind[]]),
  source: z.string().optional(),
  url: z.string().optional(),
  licence: z.string().optional(),
  citation: z.string().optional(),
  sha256: z.string().optional(),
  recipe: z
    .object({
      generator: z.string(),
      knobs: z.unknown(),
      seed: z.number().optional(),
      modifiers: z.array(z.object({ name: z.string(), knobs: z.unknown() })).optional(),
    })
    .optional(),
  key: z.tuple([z.number(), z.number()]).optional(),
}) as z.ZodType<DatasetMeta>

/**
 * Schema for a regression dataset.
 *
 * @example
 * ```ts
 * import { RegressionDataSchema, xy } from 'omai-engine/data'
 * RegressionDataSchema.safeParse(xy([[1]], [2])).success // => true
 * ```
 */
export const RegressionDataSchema: z.ZodType<RegressionData> = z.object({
  kind: z.literal('regression'),
  x: TensorSchema,
  y: TensorSchema,
  weights: TensorSchema.optional(),
  featureNames: names,
  targetNames: names,
  meta: DatasetMetaSchema,
}) as z.ZodType<RegressionData>

const IntTensorSchema = z.custom<Tensor<'int32'>>((v) => isTensor(v) && v.dtype === 'int32', {
  message: 'expected an int32 tensor',
})

/**
 * Schema for a classification dataset (labels must be int32).
 *
 * @example
 * ```ts
 * import { ClassificationDataSchema, xyClasses } from 'omai-engine/data'
 * ClassificationDataSchema.safeParse(xyClasses([[1], [2]], [0, 1])).success // => true
 * ```
 */
export const ClassificationDataSchema: z.ZodType<ClassificationData> = z.object({
  kind: z.literal('classification'),
  x: TensorSchema,
  y: IntTensorSchema,
  weights: TensorSchema.optional(),
  classes: names,
  featureNames: names,
  meta: DatasetMetaSchema,
}) as z.ZodType<ClassificationData>

/**
 * Schema for a multilabel dataset.
 *
 * @example
 * ```ts
 * import { MultilabelDataSchema } from 'omai-engine/data'
 * MultilabelDataSchema.safeParse({ kind: 'multilabel' }).success // => false
 * ```
 */
export const MultilabelDataSchema: z.ZodType<MultilabelData> = z.object({
  kind: z.literal('multilabel'),
  x: TensorSchema,
  y: IntTensorSchema,
  weights: TensorSchema.optional(),
  labels: names,
  featureNames: names,
  meta: DatasetMetaSchema,
}) as z.ZodType<MultilabelData>

/**
 * Schema for a multiclass-multioutput dataset.
 *
 * @example
 * ```ts
 * import { MultiOutputDataSchema } from 'omai-engine/data'
 * MultiOutputDataSchema.safeParse({ kind: 'multioutput' }).success // => false
 * ```
 */
export const MultiOutputDataSchema: z.ZodType<MultiOutputData> = z.object({
  kind: z.literal('multioutput'),
  x: TensorSchema,
  y: IntTensorSchema,
  weights: TensorSchema.optional(),
  outputs: z.array(z.object({ name: z.string(), classes: names })).readonly(),
  featureNames: names,
  meta: DatasetMetaSchema,
}) as z.ZodType<MultiOutputData>

/**
 * Schema for one table column.
 *
 * @example
 * ```ts
 * import { TableColumnSchema } from 'omai-engine/data'
 * TableColumnSchema.safeParse({ kind: 'text', values: ['a', null] }).success // => true
 * ```
 */
export const TableColumnSchema: z.ZodType<TableColumn> = z.discriminatedUnion('kind', [
  z.object({
    kind: z.literal('numeric'),
    values: z.custom<Tensor<'float64'>>((v) => isTensor(v)),
    unit: z.string().optional(),
  }),
  z.object({ kind: z.literal('categorical'), codes: IntTensorSchema, levels: names, ordered: z.boolean().optional() }),
  z.object({ kind: z.literal('boolean'), values: z.custom<Tensor<'bool'>>((v) => isTensor(v) && v.dtype === 'bool') }),
  z.object({ kind: z.literal('text'), values: z.array(z.string().nullable()).readonly() }),
]) as z.ZodType<TableColumn>

/**
 * Schema for a table.
 *
 * @example
 * ```ts
 * import { TableDataSchema, table } from 'omai-engine/data'
 * TableDataSchema.safeParse(table({ a: [1, 2] })).success // => true
 * ```
 */
export const TableDataSchema: z.ZodType<TableData> = z.object({
  kind: z.literal('table'),
  columns: z.record(z.string(), TableColumnSchema),
  target: z.string().optional(),
  meta: DatasetMetaSchema,
}) as z.ZodType<TableData>
