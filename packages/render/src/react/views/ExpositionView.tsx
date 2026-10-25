/**
 * The standard view of any exposition record: `ExpositionView` picks the component `viewFor(record.kind)` names, so a
 * page can show a method's whole exposition without knowing its kinds.
 */

import type { ReactElement } from 'react'
import type { Exposition } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { AnomalyView } from './anomaly.tsx'
import { AttributionView, ImportanceView } from './attributions.tsx'
import { BoardPlayer, PopulationPlayer } from './boards.tsx'
import { ConfusionMatrixView, ValueGridView } from './cells.tsx'
import { ComponentsView } from './components.tsx'
import { CurvesView } from './curves.tsx'
import { DependenceView } from './dependence.tsx'
import { DiagramPlayer } from './diagrams.tsx'
import { EpisodeView } from './episodes.tsx'
import { CounterfactualView, PrototypeView } from './examples.tsx'
import { ExplanationQualityView } from './explanation-quality.tsx'
import { GroupReportView } from './fairness.tsx'
import { ImagePlayer } from './images.tsx'
import { MarginView } from './margins.tsx'
import { CrossValidationView, LearningCurveView, PathView, PruningPathView } from './paths.tsx'
import { ConceptView, RuleListView } from './rules.tsx'
import { StateSequenceView } from './sequences.tsx'
import { EnsembleView, PipelineView } from './stages.tsx'
import { ClusteringPlayer, DendrogramView, EmbeddingPlayer, MixturePlayer, ReachabilityView } from './steps.tsx'
import { DataBiasView, ParetoFrontView, SubgroupView } from './subgroups.tsx'
import { DecisionSurfaceView, HyperparameterSurfaceView, PosteriorView } from './surfaces.tsx'
import { CoefficientTableView, LikelihoodView, NeighbourView, SearchTrialsView } from './tables.tsx'
import { ThresholdSweepView } from './thresholds.tsx'
import { TreeView } from './trees.tsx'
import { CalibrationView, ConformalView } from './trust.tsx'

/** Props of {@link ExpositionView}. */
export interface ExpositionViewProps {
  readonly record: Exposition
  readonly figure?: Figure
  /** The input rows `[n, d]` behind the record, for views that draw rows in input space (clusters, margins). */
  readonly points?: readonly (readonly number[])[]
  /** A label per row (class or group), for views that colour rows. */
  readonly labels?: readonly (number | string)[]
}

/**
 * The standard view of any exposition record (the component `viewFor(record.kind)` names), so a page can show a
 * method's whole `exposition` without knowing its kinds.
 *
 * @param props - The record, the figure, and the input rows and labels for views that use them.
 * @returns The figure.
 */
export function ExpositionView(props: ExpositionViewProps): ReactElement {
  const { record } = props
  const common = props.figure === undefined ? {} : { figure: props.figure }
  const rows = {
    ...(props.points === undefined ? {} : { points: props.points }),
    ...(props.labels === undefined ? {} : { labels: props.labels }),
  }
  const points = props.points === undefined ? {} : { points: props.points }
  const labels = props.labels === undefined ? {} : { labels: props.labels }
  switch (record.kind) {
    case 'regularisationPath':
      return <PathView record={record} {...common} />
    case 'crossValidationCurve':
      return <CrossValidationView record={record} {...common} />
    case 'learningCurve':
      return <LearningCurveView record={record} {...common} />
    case 'pruningPath':
      return <PruningPathView record={record} {...common} />
    case 'decisionSurface':
      return <DecisionSurfaceView record={record} {...common} {...rows} />
    case 'tree':
      return <TreeView record={record} {...common} />
    case 'ensembleStages':
      return <EnsembleView record={record} {...common} {...rows} />
    case 'components':
      return <ComponentsView record={record} {...common} {...labels} />
    case 'clusteringSteps':
      if (record.merges !== undefined) return <DendrogramView record={record} {...common} {...points} />
      if (record.reachability !== undefined && record.assignments.length === 1)
        return <ReachabilityView record={record} {...common} />
      return <ClusteringPlayer record={record} {...common} {...points} />
    case 'mixtureSteps':
      return <MixturePlayer record={record} {...common} {...points} />
    case 'embeddingSteps':
      return <EmbeddingPlayer record={record} {...common} {...labels} />
    case 'posteriorGrid':
      return <PosteriorView record={record} {...common} />
    case 'hyperparameterSurface':
      return <HyperparameterSurfaceView record={record} {...common} />
    case 'anomalyScores':
      return <AnomalyView record={record} {...common} {...points} />
    case 'margin':
      return <MarginView record={record} {...common} {...rows} />
    case 'coefficientTable':
      return <CoefficientTableView record={record} {...common} />
    case 'likelihoodTable':
      return <LikelihoodView record={record} {...common} />
    case 'neighbours':
      return <NeighbourView record={record} {...common} {...rows} />
    case 'searchTrials':
      return <SearchTrialsView record={record} {...common} />
    case 'pipelineStages':
      return <PipelineView record={record} {...common} />
    case 'episodes':
      return <EpisodeView record={record} {...common} />
    case 'boardSteps':
      return <BoardPlayer record={record} {...common} />
    case 'populationSteps':
      return <PopulationPlayer record={record} {...common} />
    case 'curves':
      return <CurvesView record={record} {...common} />
    case 'diagram':
      return <DiagramPlayer record={record} {...common} />
    case 'valueGrids':
      return <ValueGridView record={record} {...common} />
    case 'confusionMatrix':
      return <ConfusionMatrixView record={record} {...common} />
    case 'stateSequence':
      return <StateSequenceView record={record} {...common} />
    case 'imageSteps':
      return <ImagePlayer record={record} {...common} />
    case 'featureAttributions':
      return <AttributionView record={record} {...common} />
    case 'importances':
      return <ImportanceView record={record} {...common} />
    case 'dependence':
      return <DependenceView record={record} {...common} />
    case 'counterfactuals':
      return <CounterfactualView record={record} {...common} />
    case 'prototypes':
      return <PrototypeView record={record} {...common} />
    case 'concepts':
      return <ConceptView record={record} {...common} />
    case 'rules':
      return <RuleListView record={record} {...common} />
    case 'explanationQuality':
      return <ExplanationQualityView record={record} {...common} />
    case 'groupReport':
      return <GroupReportView record={record} {...common} />
    case 'calibration':
      return <CalibrationView record={record} {...common} />
    case 'conformal':
      return <ConformalView record={record} {...common} />
    case 'thresholdSweep':
      return <ThresholdSweepView record={record} {...common} />
    case 'subgroupReport':
      return <SubgroupView record={record} {...common} />
    case 'dataBias':
      return <DataBiasView record={record} {...common} />
    case 'paretoFront':
      return <ParetoFrontView record={record} {...common} />
  }
}
