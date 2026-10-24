import type { EChartsOption } from 'echarts'
import type { Exposition, ExpositionKind } from 'whitebox-engine/exposition'
import type { ViewOptions } from './_view.ts'
import { crossValidationView, learningCurveView, pruningPathView, regularisationPathView } from './paths.ts'

/** The record of one kind. */
export type RecordOf<K extends ExpositionKind> = Extract<Exposition, { kind: K }>

/**
 * The standard view of a record kind: its React component's name in `whitebox-render/react`, a title, whether it plays
 * (frames for a player), what it needs beside the record to draw rows in input space, and a headless still (the
 * figure, or the last frame of one that plays) for slides, blogs and server-side export.
 */
export interface StandardView<K extends ExpositionKind = ExpositionKind> {
  readonly kind: K
  readonly component: string
  readonly title: string
  readonly plays: boolean
  /** Extra data the view draws when given: the input rows (`points`) and their labels. */
  readonly accepts: readonly ('points' | 'labels')[]
  readonly still: (record: RecordOf<K>, options?: ViewOptions) => EChartsOption
}

type Registry = { readonly [K in ExpositionKind]?: StandardView<K> }

const REGISTRY: Registry = {
  regularisationPath: {
    kind: 'regularisationPath',
    component: 'PathView',
    title: 'Regularisation path',
    plays: false,
    accepts: [],
    still: (r, o) => regularisationPathView(r, o),
  },
  crossValidationCurve: {
    kind: 'crossValidationCurve',
    component: 'CrossValidationView',
    title: 'Cross-validation curve',
    plays: false,
    accepts: [],
    still: (r, o) => crossValidationView(r, o),
  },
  learningCurve: {
    kind: 'learningCurve',
    component: 'LearningCurveView',
    title: 'Learning curve',
    plays: false,
    accepts: [],
    still: (r, o) => learningCurveView(r, o),
  },
  pruningPath: {
    kind: 'pruningPath',
    component: 'PruningPathView',
    title: 'Pruning path',
    plays: false,
    accepts: [],
    still: (r, o) => pruningPathView(r, o),
  },
}

/**
 * The standard view of a record kind, so a hub entry or a page may name only a record.
 *
 * @param kind - The record kind.
 * @returns The view, or `undefined` for a kind without one yet.
 * @example
 * ```ts
 * import { viewFor } from 'whitebox-render/views'
 * viewFor('regularisationPath')?.component // => 'PathView'
 * ```
 */
export function viewFor<K extends ExpositionKind>(kind: K): StandardView<K> | undefined {
  return REGISTRY[kind] as StandardView<K> | undefined
}

/**
 * The record kinds that have a standard view.
 *
 * @returns The kinds, in the registry's order.
 */
export function viewedKinds(): ExpositionKind[] {
  return Object.keys(REGISTRY) as ExpositionKind[]
}

/**
 * The headless still of any record through its standard view.
 *
 * @param record - The record.
 * @param options - Mode and figure.
 * @returns The ECharts option, or `undefined` when the kind has no view yet.
 */
export function stillOf(record: Exposition, options: ViewOptions = {}): EChartsOption | undefined {
  const view = viewFor(record.kind) as StandardView | undefined
  return view?.still(record as never, options)
}
