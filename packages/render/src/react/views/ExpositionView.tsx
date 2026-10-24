import type { ReactElement } from 'react'
import type { Exposition } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { tokens } from '../../theme/tokens.ts'
import { useColourMode } from '../theme.tsx'
import { ComponentsView } from './components.tsx'
import { CrossValidationView, LearningCurveView, PathView, PruningPathView } from './paths.tsx'
import { EnsembleView } from './stages.tsx'
import { ClusteringPlayer, DendrogramView, EmbeddingPlayer, MixturePlayer, ReachabilityView } from './steps.tsx'
import { DecisionSurfaceView } from './surfaces.tsx'
import { TreeView } from './trees.tsx'

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
    default:
      return <Pending kind={record.kind} />
  }
}

function Pending(props: { readonly kind: string }): ReactElement {
  const t = tokens(useColourMode())
  return (
    <div
      style={{
        padding: 12,
        border: `1px dashed ${t.grid}`,
        borderRadius: 6,
        color: t.muted,
        font: `12px ${t.font}`,
      }}
    >
      No standard view of <code>{props.kind}</code> records yet.
    </div>
  )
}
