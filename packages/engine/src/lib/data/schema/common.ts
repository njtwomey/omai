/**
 * What every dataset shares: its kind (one schema per task), its metadata and provenance, and the recipe stamp that
 * records how it was produced. Also the `Dataset` union of all schemas and the helpers that find a dataset's per-row
 * fields and count its rows, which splitting, shuffling and batching build on.
 */

import { WhiteboxError } from '../../base/index.ts'
import { isTensor, type Tensor } from '../../tensor/index.ts'
import type { ComparisonData } from './comparisons.ts'
import type { CorpusData, ParallelData, SequenceLabellingData } from './corpus.ts'
import type { EventData } from './events.ts'
import type { GraphData, TripleData } from './graph.ts'
import type { ImageData } from './image.ts'
import type { InteractionData, QueryData } from './ranking.ts'
import type { SequenceData } from './sequence.ts'
import type { ClassificationData, MultilabelData, MultiOutputData, RegressionData, TableData } from './tabular.ts'
import type { TimeSeriesData } from './timeseries.ts'
import type { TrajectoryData } from './trajectories.ts'

/** The task a dataset is for: one schema per kind. */
export type DatasetKind =
  | 'regression'
  | 'classification'
  | 'multilabel'
  | 'multioutput'
  | 'table'
  | 'interactions'
  | 'queries'
  | 'corpus'
  | 'sequence-labelling'
  | 'parallel'
  | 'sequence'
  | 'timeseries'
  | 'events'
  | 'comparisons'
  | 'graph'
  | 'triples'
  | 'image'
  | 'trajectories'

/** The generator, knobs, key and modifiers that produced a dataset, recorded so it can be made again. */
export interface RecipeStamp {
  readonly generator: string
  readonly knobs: unknown
  readonly seed?: number
  readonly modifiers?: readonly { readonly name: string; readonly knobs: unknown }[]
}

/** Provenance and description of a dataset, carried with its data. */
export interface DatasetMeta {
  readonly name: string
  readonly description: string
  readonly task: DatasetKind
  readonly source?: string
  readonly url?: string
  readonly licence?: string
  readonly citation?: string
  readonly sha256?: string
  /** The generator and knobs that produced the data, when it was generated. */
  readonly recipe?: RecipeStamp
  /** The words of the key the data was generated with. */
  readonly key?: readonly [number, number]
  /**
   * Feature columns holding protected attributes (sex, an age band, a group), each with the names of its coded values
   * in order (`{ female: ['male', 'female'] }`): fairness audits group the rows by them.
   */
  readonly sensitive?: Readonly<Record<string, readonly string[]>>
}

/** What every dataset schema has: its kind and its metadata. */
export interface DatasetBase {
  readonly kind: DatasetKind
  readonly meta: DatasetMeta
}

/** Any dataset, of any schema. */
export type Dataset =
  | RegressionData
  | ClassificationData
  | MultilabelData
  | MultiOutputData
  | TableData
  | InteractionData
  | QueryData
  | CorpusData
  | SequenceLabellingData
  | ParallelData
  | SequenceData
  | TimeSeriesData
  | EventData
  | ComparisonData
  | GraphData
  | TripleData
  | ImageData
  | TrajectoryData

/** A per-row field: a tensor whose first axis is the row, or a plain array with one entry per row. */
export type RowColumn = Tensor | readonly unknown[]

const ROW_FIELDS: Readonly<Record<DatasetKind, readonly string[]>> = {
  regression: ['x', 'y', 'weights'],
  classification: ['x', 'y', 'weights'],
  multilabel: ['x', 'y', 'weights'],
  multioutput: ['x', 'y', 'weights'],
  table: ['columns'],
  interactions: ['users', 'items', 'values', 'times', 'sessions', 'positions', 'propensities'],
  queries: ['x', 'y', 'queries'],
  corpus: ['documents', 'tokens', 'labels'],
  'sequence-labelling': ['tokens', 'tags'],
  parallel: ['source', 'target'],
  sequence: ['inputs', 'targets', 'inputMask', 'targetMask'],
  timeseries: ['times', 'values', 'segments'],
  events: ['times', 'locations', 'marks', 'sequences'],
  comparisons: ['participants', 'ranks', 'times'],
  graph: ['graphs', 'graphLabels'],
  triples: ['heads', 'relations', 'tails'],
  image: ['images', 'labels', 'boxes', 'boxLabels', 'masks'],
  trajectories: [
    'observations',
    'actions',
    'rewards',
    'nextObservations',
    'terminations',
    'truncations',
    'episodes',
    'propensities',
  ],
}

/**
 * The fields of a dataset that hold one entry per row, so they are subset, shuffled, split and batched together. Each
 * schema declares its per-row fields; optional ones count only when present. A table's rows are its columns' entries.
 *
 * @param data - The dataset.
 * @returns The names of the per-row fields present.
 * @concept datasets
 * @example
 * ```ts
 * import { perRowFields, xy } from 'whitebox-engine/data'
 * perRowFields(xy([[1], [2]], [3, 4])) // => ['x', 'y']
 * ```
 */
export function perRowFields(data: DatasetBase): readonly string[] {
  const record = data as unknown as Readonly<Record<string, unknown>>
  return (ROW_FIELDS[data.kind] ?? []).filter((f) => record[f] !== undefined)
}

// The length of a per-row column (a table counts the rows of its columns).
function columnLength(column: unknown): number {
  if (isTensor(column)) return column.shape[0] ?? 0
  if (Array.isArray(column)) return column.length
  if (typeof column === 'object' && column !== null) {
    let n = -1
    for (const c of Object.values(column as Record<string, unknown>)) {
      const m = tableColumnLength(c)
      if (n >= 0 && m !== n) throw new WhiteboxError('shape', `table columns disagree in length (${n} and ${m})`)
      n = m
    }
    return Math.max(n, 0)
  }
  throw new WhiteboxError('schema', 'a per-row field must be a tensor, an array or a table of columns')
}

function tableColumnLength(c: unknown): number {
  const values = (c as { values?: unknown; codes?: unknown }).values ?? (c as { codes?: unknown }).codes
  return columnLength(values)
}

/**
 * The number of rows, checking that every per-row field agrees.
 *
 * @param data - The dataset.
 * @returns The number of rows.
 * @throws WhiteboxError with code `shape` when two per-row fields disagree.
 * @example
 * ```ts
 * import { rowCount, xy } from 'whitebox-engine/data'
 * rowCount(xy([[1], [2], [3]], [1, 2, 3])) // => 3
 * ```
 */
export function rowCount(data: DatasetBase): number {
  const record = data as unknown as Readonly<Record<string, unknown>>
  let n = -1
  let first = ''
  for (const f of perRowFields(data)) {
    const m = columnLength(record[f])
    if (n < 0) {
      n = m
      first = f
    } else if (m !== n) {
      throw new WhiteboxError('shape', `${data.meta.name}: field ${f} has ${m} rows, ${first} has ${n}`)
    }
  }
  return Math.max(n, 0)
}
