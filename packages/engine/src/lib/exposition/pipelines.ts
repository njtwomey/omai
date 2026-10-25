import { WhiteboxError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'
import type { DecisionSurface, SurfaceGrid } from './surfaces.ts'

/** Summary statistics of one column over all of a stage's rows (the standard deviation divides by n). */
export interface ColumnStats {
  readonly mean: number
  readonly std: number
  readonly min: number
  readonly max: number
}

/**
 * Data a stage takes or gives, as a view draws it: the column names, the rows (all of them when there are few, evenly
 * spaced ones beyond a few hundred), the rows' classes or targets, and per-column statistics over every row.
 */
export interface StageData {
  readonly columns: readonly string[]
  /** rows[i]: a row's values, one per column. */
  readonly rows: readonly (readonly number[])[]
  /** How many rows the data has in all (`rows` may be a subsample of them). */
  readonly n: number
  /** labels[i]: rows[i]'s class index (into the record's `classes`), for classification data. */
  readonly labels?: readonly number[]
  /** targets[i]: rows[i]'s target, for regression data. */
  readonly targets?: readonly number[]
  /** stats[j]: column j's mean, standard deviation, minimum and maximum over all n rows. */
  readonly stats?: readonly ColumnStats[]
}

/**
 * A fitted parameter of a stage worth showing: a number or a name, a vector (a scaler's means, a model's
 * coefficients) or a matrix (polynomial powers, principal axes), with the names of its entries.
 */
export interface StageParameter {
  readonly name: string
  readonly value: number | string | readonly number[] | readonly (readonly number[])[]
  /** The names of a vector's entries or a matrix's columns (the stage's input or output columns). */
  readonly columns?: readonly string[]
  /** The names of a matrix's rows. */
  readonly rows?: readonly string[]
}

/**
 * The final estimator's loss on the training rows: the mean over every row, and per row of its decision surface's
 * rows, with the rows it gets wrong and the loss it expects over the surface's grid.
 */
export interface StageLoss {
  /** What the loss is ('log loss', 'squared error'). */
  readonly name: string
  /** The mean loss over every training row. */
  readonly mean: number
  /** values[i]: the loss at the surface's row i. */
  readonly values: readonly number[]
  /** The surface's rows the model misclassifies (indices into its rows). */
  readonly misclassified?: readonly number[]
  /**
   * expected[i][j]: the loss the model expects at the surface's grid point (x[j], y[i]) were its predicted
   * distribution right (for log loss, that distribution's entropy): high where the model is unsure.
   */
  readonly expected?: SurfaceGrid
}

/** One stage of a pipeline: what went in, what came out, and what it learned. */
export interface PipelineStage {
  readonly name: string
  readonly kind: 'transformer' | 'estimator' | 'resampler' | 'selector'
  /** Column names and a sample of rows (the first few), before and after the stage. */
  readonly inputColumns: readonly string[]
  readonly outputColumns: readonly string[]
  readonly inputSample: readonly (readonly number[])[]
  readonly outputSample: readonly (readonly number[])[]
  readonly inputShape: readonly [number, number]
  readonly outputShape: readonly [number, number]
  /** The fitted parameters worth showing (means and scales, selected features, coefficients), as name-value pairs. */
  readonly learned: readonly { readonly name: string; readonly value: number | string | readonly number[] }[]
  /** The stage's output on the training rows (up to a few hundred of them), with their classes and column statistics. */
  readonly output?: StageData
  /** The fitted parameters in full (vectors and matrices with the names of their entries). */
  readonly parameters?: readonly StageParameter[]
  /** The share of the variance each output column explains, for projections (PCA, truncated SVD). */
  readonly explainedVarianceRatio?: readonly number[]
  /** The whole pipeline's predictions over the original two input features, on the final estimator's stage. */
  readonly surface?: DecisionSurface
  /** The final estimator's loss on the training rows, per row of `surface`'s rows. */
  readonly loss?: StageLoss
}

/** A fitted pipeline, stage by stage. */
export interface PipelineStages {
  readonly kind: 'pipelineStages'
  readonly stages: readonly PipelineStage[]
  /** The data the pipeline was fitted to (its first stage's input), as {@link PipelineStage.output}. */
  readonly input?: StageData
  /** Class names, by class index (the stages' `labels` index them). */
  readonly classes?: readonly string[]
}

function checkData(field: string, data: StageData, shape?: readonly [number, number]): void {
  const k = 'pipelineStages'
  rowsOf(k, `${field}.rows`, data.rows, data.columns.length)
  if (data.rows.length > data.n)
    throw new WhiteboxError('shape', `${k}: ${field} has ${data.rows.length} rows but n is ${data.n}`)
  if (data.labels !== undefined) sameLength(k, `${field}.labels`, data.labels.length, data.rows.length)
  if (data.targets !== undefined) sameLength(k, `${field}.targets`, data.targets.length, data.rows.length)
  if (data.stats !== undefined) sameLength(k, `${field}.stats`, data.stats.length, data.columns.length)
  if (shape !== undefined) {
    sameLength(k, `${field}.columns`, data.columns.length, shape[1])
    sameLength(k, `${field}.n`, data.n, shape[0])
  }
}

function checkParameter(field: string, p: StageParameter): void {
  const k = 'pipelineStages'
  const v = p.value
  if (typeof v !== 'object') return
  const matrix = v.length > 0 && Array.isArray(v[0])
  if (matrix) {
    const m = v as readonly (readonly number[])[]
    const width = m[0]?.length ?? 0
    rowsOf(k, `${field}.value`, m, width)
    if (p.columns !== undefined) sameLength(k, `${field}.columns`, p.columns.length, width)
    if (p.rows !== undefined) sameLength(k, `${field}.rows`, p.rows.length, m.length)
  } else if (p.columns !== undefined) sameLength(k, `${field}.columns`, p.columns.length, v.length)
}

function checkStage(i: number, s: PipelineStage): void {
  const k = 'pipelineStages'
  const at = `stages[${i}]`
  if (s.output !== undefined) checkData(`${at}.output`, s.output, s.outputShape)
  for (const [j, p] of (s.parameters ?? []).entries()) checkParameter(`${at}.parameters[${j}]`, p)
  if (s.explainedVarianceRatio !== undefined && s.explainedVarianceRatio.length > s.outputShape[1])
    throw new WhiteboxError('shape', `${k}: ${at}.explainedVarianceRatio is longer than the stage's output`)
  const loss = s.loss
  if (loss !== undefined) {
    const rows = s.surface?.rows?.points.length
    if (rows === undefined) throw new WhiteboxError('shape', `${k}: ${at}.loss needs a surface with rows`)
    sameLength(k, `${at}.loss.values`, loss.values.length, rows)
    for (const r of loss.misclassified ?? [])
      if (!(r >= 0 && r < rows)) throw new WhiteboxError('shape', `${k}: ${at}.loss.misclassified has row ${r}`)
    if (loss.expected !== undefined && s.surface !== undefined) {
      sameLength(k, `${at}.loss.expected`, loss.expected.length, s.surface.y.length)
      rowsOf(k, `${at}.loss.expected`, loss.expected, s.surface.x.length)
    }
  }
}

/**
 * Pipeline stages, checked to chain (each stage's output is the next stage's input) and, where a stage carries its
 * data, parameters or loss, checked against its shapes.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when consecutive stages do not chain, a stage's data does not match its shape, a
 * parameter's names do not match its entries, or a loss does not match its surface's rows.
 * @example
 * ```ts
 * import { pipelineStages } from 'whitebox-engine/exposition'
 * pipelineStages({ stages: [] }).stages // => []
 * ```
 */
export function pipelineStages(record: Omit<PipelineStages, 'kind'>): PipelineStages {
  for (let i = 1; i < record.stages.length; i++) {
    const before = record.stages[i - 1]
    const after = record.stages[i]
    if (before !== undefined && after !== undefined && before.outputShape[1] !== after.inputShape[1])
      throw new WhiteboxError(
        'shape',
        `pipelineStages: ${before.name} outputs ${before.outputShape[1]} columns but ${after.name} takes ${after.inputShape[1]}`,
      )
  }
  for (const [i, s] of record.stages.entries()) checkStage(i, s)
  if (record.input !== undefined) checkData('input', record.input, record.stages[0]?.inputShape)
  return { kind: 'pipelineStages', ...record }
}
