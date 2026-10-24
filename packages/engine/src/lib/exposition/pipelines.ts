import { WhiteboxError } from '../base/index.ts'

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
}

/** A fitted pipeline, stage by stage. */
export interface PipelineStages {
  readonly kind: 'pipelineStages'
  readonly stages: readonly PipelineStage[]
}

/**
 * Pipeline stages, checked to chain (each stage's output is the next stage's input).
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when consecutive stages do not chain.
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
  return { kind: 'pipelineStages', ...record }
}
