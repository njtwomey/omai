import type { ReactElement } from 'react'
import type { Exposition } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { tokens } from '../../theme/tokens.ts'
import { useColourMode } from '../theme.tsx'
import { CrossValidationView, LearningCurveView, PathView, PruningPathView } from './paths.tsx'

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
  switch (record.kind) {
    case 'regularisationPath':
      return <PathView record={record} {...common} />
    case 'crossValidationCurve':
      return <CrossValidationView record={record} {...common} />
    case 'learningCurve':
      return <LearningCurveView record={record} {...common} />
    case 'pruningPath':
      return <PruningPathView record={record} {...common} />
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
