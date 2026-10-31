/**
 * The union of every exposition record: the plain, serialisable data a method returns beside its result so a view can
 * explain it. `Explained` pairs a result with its records, `recordsOf` picks the records of one kind, and
 * `isExposition` recognises a record by its `kind` tag.
 */

import type { AgentSteps } from './agents.ts'
import type { AnomalyScores } from './anomaly.ts'
import type { FeatureAttributions, Importances } from './attributions.ts'
import type { AudioClip } from './audio.ts'
import type { BoardSteps } from './boards.ts'
import type { ValueGrids } from './cells.ts'
import type { ClusteringSteps, MixtureSteps } from './clustering.ts'
import type { ComplexPlane } from './complex-plane.ts'
import type { ConceptScores } from './concepts.ts'
import type { CurveSet } from './curves.ts'
import type { DataBias } from './data-bias.ts'
import type { Dependence } from './dependence.ts'
import type { DiagramRecord } from './diagrams.ts'
import type { ComponentsRecord, EmbeddingSteps } from './embeddings.ts'
import type { EnsembleStages } from './ensembles.ts'
import type { Epicycles } from './epicycles.ts'
import type { EpisodeRecord } from './episodes.ts'
import type { EquationFit } from './equation-fit.ts'
import type { ConfusionMatrix } from './evaluation.ts'
import type { CounterfactualSet, PrototypeSet } from './examples.ts'
import type { ExplanationQuality } from './explanation-quality.ts'
import type { GroupReport } from './fairness.ts'
import type { VectorField } from './fields.ts'
import type { ParetoFront } from './frontiers.ts'
import type { ImageSteps } from './images.ts'
import type { LandscapeSlice } from './landscape-slice.ts'
import type { CoefficientTable, CrossValidationCurve, RegularisationPath } from './linear.ts'
import type { MarginRecord } from './margins.ts'
import type { LikelihoodTable, NeighbourRecord } from './neighbours.ts'
import type { PipelineStages } from './pipelines.ts'
import type { PointsSteps } from './points-steps.ts'
import type { PopulationSteps } from './populations.ts'
import type { HyperparameterSurface, PosteriorGrid } from './posteriors.ts'
import type { Reachability } from './reachability.ts'
import type { RuleSet } from './rules.ts'
import type { SceneSteps } from './scene-steps.ts'
import type { LearningCurve, SearchTrials } from './searches.ts'
import type { StateSequence } from './sequences.ts'
import type { Simplex } from './simplex.ts'
import type { StabilityRegion } from './stability-region.ts'
import type { Strip } from './strips.ts'
import type { StructureSteps } from './structure-steps.ts'
import type { SubgroupReport } from './subgroups.ts'
import type { DecisionSurface } from './surfaces.ts'
import type { MarkedTable } from './tables.ts'
import type { ThresholdSweep } from './thresholds.ts'
import type { Transcript } from './transcript.ts'
import type { PruningPath, TreeRecord } from './trees.ts'
import type { CalibrationRecord, ConformalRecord } from './trust.ts'

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
  | StateSequence
  | PopulationSteps
  | CurveSet
  | DiagramRecord
  | ValueGrids
  | ConfusionMatrix
  | ImageSteps
  | Epicycles
  | AgentSteps
  | FeatureAttributions
  | Importances
  | Dependence
  | CounterfactualSet
  | PrototypeSet
  | ConceptScores
  | RuleSet
  | ExplanationQuality
  | GroupReport
  | CalibrationRecord
  | ConformalRecord
  | ThresholdSweep
  | SubgroupReport
  | DataBias
  | ParetoFront
  | VectorField
  | Strip
  | MarkedTable
  | StructureSteps
  | PointsSteps
  | ComplexPlane
  | Simplex
  | Reachability
  | SceneSteps
  | LandscapeSlice
  | StabilityRegion
  | EquationFit
  | AudioClip
  | Transcript

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
 * import { recordsOf, regularisationPath } from 'mdml-engine/exposition'
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
 * import { isExposition } from 'mdml-engine/exposition'
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
  'curves',
  'diagram',
  'valueGrids',
  'confusionMatrix',
  'stateSequence',
  'imageSteps',
  'epicycles',
  'agentSteps',
  'featureAttributions',
  'importances',
  'dependence',
  'counterfactuals',
  'prototypes',
  'concepts',
  'rules',
  'explanationQuality',
  'groupReport',
  'calibration',
  'conformal',
  'thresholdSweep',
  'subgroupReport',
  'dataBias',
  'paretoFront',
  'vectorField',
  'strip',
  'markedTable',
  'structureSteps',
  'pointsSteps',
  'complexPlane',
  'simplex',
  'reachability',
  'sceneSteps',
  'landscapeSlice',
  'stabilityRegion',
  'equationFit',
  'audio',
  'transcript',
])
