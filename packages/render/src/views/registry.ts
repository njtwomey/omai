import type { EChartsOption } from 'echarts'
import type { Exposition, ExpositionKind } from 'whitebox-engine/exposition'
import type { Scene } from 'whitebox-engine/scene'
import type { ViewOptions } from './_view.ts'
import { anomalyView } from './anomaly.ts'
import { attributionFrames, beeswarmView, importanceView } from './attributions.ts'
import { boardFrames, populationFrames } from './boards.ts'
import { confusionFrames, valueGridFrames } from './cells.ts'
import { componentsView } from './components.ts'
import { curveSetView } from './curves.ts'
import { diagramScene } from './diagrams.ts'
import { episodeView } from './episodes.ts'
import { groupReportView } from './fairness.ts'
import { imageFrames } from './images.ts'
import { marginFrames } from './margins.ts'
import { crossValidationView, learningCurveView, pruningPathView, regularisationPathView } from './paths.ts'
import { stateSequenceView } from './sequences.ts'
import { ensembleFrames, pipelineScene } from './stages.ts'
import { clusteringFrames, dendrogramView, embeddingFrames, mixtureFrames, reachabilityView } from './steps.ts'
import { decisionSurfaceFrames, decisionSurfaceView, hyperparameterSurfaceView, posteriorView } from './surfaces.ts'
import { coefficientView, likelihoodView, neighbourView, searchTrialsView } from './tables.ts'
import { treeScene } from './trees.ts'
import { conformalFrames, conformalView, reliabilityView } from './trust.ts'

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
  readonly still: (record: RecordOf<K>, options?: ViewOptions) => ViewStill
}

/** A view's headless still: a chart option, or a scene for views drawn as diagrams (trees). */
export type ViewStill = EChartsOption | Scene

// Every kind has a view: a new record kind does not compile until it has one here.
type Registry = { readonly [K in ExpositionKind]: StandardView<K> }

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
    plays: true,
    accepts: ['points', 'labels'],
    still: (r, o) => {
      if (r.steps === undefined || r.steps.length < 2) return decisionSurfaceView(r, o)
      const frames = decisionSurfaceFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  tree: {
    kind: 'tree',
    component: 'TreeView',
    title: 'Tree',
    plays: true,
    accepts: [],
    still: (r) => treeScene(r),
  },
  ensembleStages: {
    kind: 'ensembleStages',
    component: 'EnsembleView',
    title: 'Ensemble stages',
    plays: true,
    accepts: ['points', 'labels'],
    still: (r, o) => {
      const frames = ensembleFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  components: {
    kind: 'components',
    component: 'ComponentsView',
    title: 'Components',
    plays: false,
    accepts: ['labels'],
    still: (r, o) => componentsView(r, o),
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
  posteriorGrid: {
    kind: 'posteriorGrid',
    component: 'PosteriorView',
    title: 'Posterior',
    plays: false,
    accepts: [],
    still: (r, o) => posteriorView(r, o),
  },
  hyperparameterSurface: {
    kind: 'hyperparameterSurface',
    component: 'HyperparameterSurfaceView',
    title: 'Hyperparameter surface',
    plays: false,
    accepts: [],
    still: (r, o) => hyperparameterSurfaceView(r, o),
  },
  anomalyScores: {
    kind: 'anomalyScores',
    component: 'AnomalyView',
    title: 'Anomaly scores',
    plays: false,
    accepts: ['points'],
    still: (r, o) => anomalyView(r, o),
  },
  margin: {
    kind: 'margin',
    component: 'MarginView',
    title: 'Margins',
    plays: true,
    accepts: ['points', 'labels'],
    still: (r, o) => {
      const frames = marginFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  coefficientTable: {
    kind: 'coefficientTable',
    component: 'CoefficientTableView',
    title: 'Coefficients',
    plays: false,
    accepts: [],
    still: (r, o) => coefficientView(r, o),
  },
  likelihoodTable: {
    kind: 'likelihoodTable',
    component: 'LikelihoodView',
    title: 'Likelihood contributions',
    plays: false,
    accepts: [],
    still: (r, o) => likelihoodView(r, o),
  },
  neighbours: {
    kind: 'neighbours',
    component: 'NeighbourView',
    title: 'Neighbours',
    plays: false,
    accepts: ['points', 'labels'],
    still: (r, o) => neighbourView(r, o),
  },
  searchTrials: {
    kind: 'searchTrials',
    component: 'SearchTrialsView',
    title: 'Search trials',
    plays: false,
    accepts: [],
    still: (r, o) => searchTrialsView(r, o),
  },
  pipelineStages: {
    kind: 'pipelineStages',
    component: 'PipelineView',
    title: 'Pipeline',
    plays: false,
    accepts: [],
    still: (r) => pipelineScene(r),
  },
  episodes: {
    kind: 'episodes',
    component: 'EpisodeView',
    title: 'Episodes',
    plays: false,
    accepts: [],
    still: (r, o) => episodeView(r, o),
  },
  boardSteps: {
    kind: 'boardSteps',
    component: 'BoardPlayer',
    title: 'Board steps',
    plays: true,
    accepts: [],
    still: (r, o) => {
      const frames = boardFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  stateSequence: {
    kind: 'stateSequence',
    component: 'StateSequenceView',
    title: 'Hidden states',
    plays: false,
    accepts: [],
    still: (r, o) => stateSequenceView(r, o),
  },
  imageSteps: {
    kind: 'imageSteps',
    component: 'ImagePlayer',
    title: 'Images',
    plays: true,
    accepts: [],
    still: (r, o) => {
      const frames = imageFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  populationSteps: {
    kind: 'populationSteps',
    component: 'PopulationPlayer',
    title: 'Population',
    plays: true,
    accepts: [],
    still: (r, o) => {
      const frames = populationFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  curves: {
    kind: 'curves',
    component: 'CurvesView',
    title: 'Curves',
    plays: true,
    accepts: [],
    still: (r, o) => curveSetView(r, o),
  },
  diagram: {
    kind: 'diagram',
    component: 'DiagramPlayer',
    title: 'Diagram',
    plays: true,
    accepts: [],
    still: (r) => diagramScene(r),
  },
  valueGrids: {
    kind: 'valueGrids',
    component: 'ValueGridView',
    title: 'Value grids',
    plays: true,
    accepts: [],
    still: (r, o) => {
      const frames = valueGridFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  groupReport: {
    kind: 'groupReport',
    component: 'GroupReportView',
    title: 'Group report',
    plays: false,
    accepts: [],
    still: (r, o) => groupReportView(r, o),
  },
  calibration: {
    kind: 'calibration',
    component: 'CalibrationView',
    title: 'Reliability diagram',
    plays: false,
    accepts: [],
    still: (r, o) => reliabilityView(r, o),
  },
  conformal: {
    kind: 'conformal',
    component: 'ConformalView',
    title: 'Conformal prediction',
    plays: true,
    accepts: [],
    still: (r, o) => {
      if (r.stream === undefined) return conformalView(r, o)
      const frames = conformalFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  confusionMatrix: {
    kind: 'confusionMatrix',
    component: 'ConfusionMatrixView',
    title: 'Confusion matrix',
    plays: true,
    accepts: [],
    still: (r, o) => {
      const frames = confusionFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  featureAttributions: {
    kind: 'featureAttributions',
    component: 'AttributionView',
    title: 'Feature attributions',
    plays: true,
    accepts: [],
    still: (r, o) => {
      if (r.values.length > 1) return beeswarmView(r, o)
      const frames = attributionFrames(r, o)
      return frames.option(frames.count - 1)
    },
  },
  importances: {
    kind: 'importances',
    component: 'ImportanceView',
    title: 'Feature importances',
    plays: false,
    accepts: [],
    still: (r, o) => importanceView(r, o),
  },
}

/**
 * The standard view of a record kind, so a hub entry or a page may name only a record.
 *
 * @param kind - The record kind.
 * @returns The view.
 * @example
 * ```ts
 * import { viewFor } from 'whitebox-render/views'
 * viewFor('regularisationPath').component // => 'PathView'
 * ```
 */
export function viewFor<K extends ExpositionKind>(kind: K): StandardView<K> {
  return REGISTRY[kind] as StandardView<K>
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
 * @returns The chart option or scene.
 */
export function stillOf(record: Exposition, options: ViewOptions = {}): ViewStill {
  return (viewFor(record.kind) as StandardView).still(record as never, options)
}
