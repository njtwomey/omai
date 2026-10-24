import { type ReactElement, useCallback } from 'react'
import type { DecisionSurface } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { decisionSurfaceView, type SurfaceViewOptions } from '../../views/surfaces.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const SURFACE_FIGURE = { aspect: 1.25 } as const

/**
 * A decision surface: a classifier's smooth regions (faded where it is unsure) with boundaries and probability
 * contours, or a regressor's or density's field, with the training rows over it.
 *
 * @param props - The record, the training rows and labels, contours and aspect, and the figure and frame settings.
 * @returns The figure.
 */
export function DecisionSurfaceView(
  props: RecordViewProps<DecisionSurface> & Omit<SurfaceViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, SURFACE_FIGURE)
  const { record, points, labels, columns, contours, aspect } = props
  const build = useCallback(
    (fig: Figure) =>
      decisionSurfaceView(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(labels === undefined ? {} : { labels }),
        ...(columns === undefined ? {} : { columns }),
        ...(contours === undefined ? {} : { contours }),
        ...(aspect === undefined ? {} : { aspect }),
      }),
    [record, points, labels, columns, contours, aspect],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'decision-surface'} />
}
