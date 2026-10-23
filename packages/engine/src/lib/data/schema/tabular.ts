import { WhiteboxError } from '../../base/index.ts'
import type { Tensor } from '../../tensor/index.ts'
import type { DatasetMeta } from './common.ts'

/** Features with real-valued targets. */
export interface RegressionData {
  readonly kind: 'regression'
  /** Features `[n, d]`. */
  readonly x: Tensor
  /** Targets `[n]` (or `[n, k]` for several outputs). */
  readonly y: Tensor
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
  readonly classes: readonly string[]
  readonly featureNames: readonly string[]
  readonly meta: DatasetMeta
}

/** Any tabular dataset. */
export type TabularData = RegressionData | ClassificationData

/** The fields of a dataset that hold one entry per row (and so are subset, shuffled and batched together). */
export function perRowFields(_data: TabularData): readonly ('x' | 'y')[] {
  return ['x', 'y']
}

/** The number of rows, checking that every per-row field agrees. */
export function rowCount(data: TabularData): number {
  const n = data.x.shape[0] ?? 0
  for (const f of perRowFields(data)) {
    if ((data[f].shape[0] ?? 0) !== n)
      throw new WhiteboxError('shape', `${data.meta.name}: field ${f} has ${data[f].shape[0]} rows, x has ${n}`)
  }
  return n
}
