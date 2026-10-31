/**
 * The React view of a point set that grows over a run: a scatter plot with fixed axes in two dimensions, a cloud to
 * orbit (drag, arrow keys) in three, at a given step.
 */

import type { PointsSteps } from 'mdml-engine/exposition'
import { type ReactElement, useCallback, useMemo } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { pointsStepsScene3d, pointsStepsView } from '../../views/points-steps.ts'
import { FigureFrame } from '../FigureFrame.tsx'
import { Scene3DView } from '../Scene3DView.tsx'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

/** Props of {@link PointsStepsView}: the record, the tape step to show and the frame settings. */
export interface PointsStepsViewProps extends RecordViewProps<PointsSteps> {
  /** The tape step to show: the points up to it, or the latest set at or before it (default the last recorded). */
  readonly step?: number
}

const DEFAULTS = { aspect: 1.25 } as const

/**
 * Points at a step: a scatter plot of the points so far with the latest step's points set apart, or, for three
 * dimensions, the cloud in a box to orbit.
 *
 * @param props - The record, the step and the figure and frame settings.
 * @returns The figure.
 */
export function PointsStepsView(props: PointsStepsViewProps): ReactElement {
  const { record, step } = props
  const figure = useViewFigure(props.figure, DEFAULTS)
  const build = useCallback((fig: Figure) => pointsStepsView(record, step, { figure: fig }), [record, step])
  const scene3 = useMemo(() => (record.dims === 3 ? pointsStepsScene3d(record, step) : undefined), [record, step])
  if (scene3 === undefined)
    return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'points'} />
  return (
    <FigureFrame
      width={figure.width}
      height={figure.height}
      filename={props.filename ?? 'points'}
      {...(props.title === undefined ? {} : { title: props.title })}
      caption={props.caption ?? 'Drag to orbit; double-click to return to the start.'}
    >
      <Scene3DView scene={scene3} width={figure.width} height={figure.height} />
    </FigureFrame>
  )
}
