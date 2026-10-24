import { type ReactElement, useCallback } from 'react'
import type { MarginRecord } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { type MarginViewOptions, marginFrames } from '../../views/margins.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, useViewFigure } from './_frame.tsx'

const MARGIN_FIGURE = { aspect: 1.75 } as const

/**
 * A margin classifier (an SVM): support vectors ringed over the decision surface, the functional margins, and the dual
 * objective, played over the solver's steps with the pair SMO updated.
 *
 * @param props - The record, rows, labels, columns and surface, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function MarginView(
  props: RecordViewProps<MarginRecord> & Omit<MarginViewOptions, 'mode' | 'figure'> & PlayerViewProps,
): ReactElement {
  const figure = useViewFigure(props.figure, MARGIN_FIGURE)
  const { record, points, labels, columns, surface, maxFrames } = props
  const frames = useCallback(
    (fig: Figure) =>
      marginFrames(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(labels === undefined ? {} : { labels }),
        ...(columns === undefined ? {} : { columns }),
        ...(surface === undefined ? {} : { surface }),
        ...(maxFrames === undefined ? {} : { maxFrames }),
      }),
    [record, points, labels, columns, surface, maxFrames],
  )
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'margins'} />
}
