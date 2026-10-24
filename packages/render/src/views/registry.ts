import type { EChartsOption } from 'echarts'
import type { Exposition, ExpositionKind } from 'whitebox-engine/exposition'
import type { ViewOptions } from './_view.ts'
import { crossValidationView, learningCurveView, pruningPathView, regularisationPathView } from './paths.ts'
import { clusteringFrames, dendrogramView, embeddingFrames, mixtureFrames, reachabilityView } from './steps.ts'
import { decisionSurfaceView } from './surfaces.ts'

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
  decisionSurface: {
    kind: 'decisionSurface',
    component: 'DecisionSurfaceView',
    title: 'Decision surface',
    plays: false,
    accepts: ['points', 'labels'],
    still: (r, o) => decisionSurfaceView(r, o),
  },
  clusteringSteps: {
    kind: 'clusteringSteps',
    component: 'ClusteringPlayer',
    title: 'Clustering steps',
    plays: true,
    accepts: ['points'],
    still: (r, o) => {
      if (r.merges !== undefined) return dendrogramView(r, o)
      if (r.reachability !== undefined && r.assignments.length === 1) return reachabilityView(r, o)
      const frames = clusteringFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  mixtureSteps: {
    kind: 'mixtureSteps',
    component: 'MixturePlayer',
    title: 'Mixture steps',
    plays: true,
    accepts: ['points'],
    still: (r, o) => {
      const frames = mixtureFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  embeddingSteps: {
    kind: 'embeddingSteps',
    component: 'EmbeddingPlayer',
    title: 'Embedding steps',
    plays: true,
    accepts: ['labels'],
    still: (r, o) => {
      const frames = embeddingFrames(r, o)
      return frames.option(frames.count - 1)
    },
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
