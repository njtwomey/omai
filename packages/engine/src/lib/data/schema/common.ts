/** The task a dataset is for. */
export type DatasetKind = 'regression' | 'classification'

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
  readonly recipe?: { readonly generator: string; readonly knobs: unknown; readonly seed?: number }
}
