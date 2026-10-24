import type { AnomalyScores } from './anomaly.ts'
import type { BoardSteps } from './boards.ts'
import type { ClusteringSteps, MixtureSteps } from './clustering.ts'
import type { ComponentsRecord, EmbeddingSteps } from './embeddings.ts'
import type { EnsembleStages } from './ensembles.ts'
import type { EpisodeRecord } from './episodes.ts'
import type { ImageSteps } from './images.ts'
import type { CoefficientTable, CrossValidationCurve, RegularisationPath } from './linear.ts'
import type { MarginRecord } from './margins.ts'
import type { LikelihoodTable, NeighbourRecord } from './neighbours.ts'
import type { PipelineStages } from './pipelines.ts'
import type { PopulationSteps } from './populations.ts'
import type { HyperparameterSurface, PosteriorGrid } from './posteriors.ts'
import type { LearningCurve, SearchTrials } from './searches.ts'
import type { DecisionSurface } from './surfaces.ts'
import type { PruningPath, TreeRecord } from './trees.ts'

/**
 * Any exposition record: the plain, serialisable data a method returns beside its result so a view can explain it.
 * Every record carries a `kind` tag; render has one view per kind.
 *
 * @concept exposition
 */
export type Exposition =
  | RegularisationPath
  | CoefficientTable
  | CrossValidationCurve
  | TreeRecord
  | PruningPath
  | EnsembleStages
  | MarginRecord
  | NeighbourRecord
  | LikelihoodTable
  | PosteriorGrid
  | HyperparameterSurface
  | ClusteringSteps
  | MixtureSteps
  | ComponentsRecord
  | EmbeddingSteps
  | AnomalyScores
  | DecisionSurface
  | PipelineStages
  | SearchTrials
  | LearningCurve
  | EpisodeRecord
  | BoardSteps
  | PopulationSteps
  | ImageSteps

/** The tag of an exposition record. */
export type ExpositionKind = Exposition['kind']

/** A method's result with its exposition records, as every method returns. */
export type Explained<T> = T & { readonly exposition: readonly Exposition[] }

/**
 * The records of one kind among a method's exposition, in order.
 *
 * @param records - A method's `exposition`.
 * @param kind - The kind wanted.
 * @returns The records of that kind.
 * @example
 * ```ts
 * import { recordsOf, regularisationPath } from 'whitebox-engine/exposition'
 * const path = regularisationPath({ penalties: [1, 0.1], features: ['a'], coefficients: [[0], [0.5]] })
 * recordsOf([path], 'regularisationPath').length // => 1
 * ```
 */
export function recordsOf<K extends ExpositionKind>(
  records: readonly Exposition[],
  kind: K,
): Extract<Exposition, { kind: K }>[] {
  return records.filter((r): r is Extract<Exposition, { kind: K }> => r.kind === kind)
}

/**
 * Whether a value is an exposition record (an object with a known `kind`).
 *
 * @param x - Any value.
 * @returns `true` for exposition records.
 * @example
 * ```ts
 * import { isExposition } from 'whitebox-engine/exposition'
 * isExposition({ kind: 'learningCurve', sizes: [], train: [], validation: [] }) // => true
 * isExposition({ kind: 'nope' }) // => false
 * ```
 */
export function isExposition(x: unknown): x is Exposition {
  return typeof x === 'object' && x !== null && KINDS.has((x as { kind?: unknown }).kind as string)
}

const KINDS: ReadonlySet<string> = new Set<ExpositionKind>([
  'regularisationPath',
  'coefficientTable',
  'crossValidationCurve',
  'tree',
  'pruningPath',
  'ensembleStages',
  'margin',
  'neighbours',
  'likelihoodTable',
  'posteriorGrid',
  'hyperparameterSurface',
  'clusteringSteps',
  'mixtureSteps',
  'components',
  'embeddingSteps',
  'anomalyScores',
  'decisionSurface',
  'pipelineStages',
  'searchTrials',
  'learningCurve',
  'episodes',
  'boardSteps',
  'populationSteps',
  'imageSteps',
])
