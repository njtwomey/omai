/**
 * The React view of anomaly scores: their histogram with the threshold, each row's evidence, and the rows coloured by
 * score with the flagged ones ringed, in one figure frame.
 */

import { type ReactElement, useCallback } from 'react'
import type { AnomalyScores } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { type AnomalyViewOptions, anomalyView } from '../../views/anomaly.ts'
import { RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const ANOMALY_FIGURE = { aspect: 1.75 } as const

/**
 * Anomaly scores: their histogram with the threshold, the per-row evidence, and the rows coloured by score with the
 * flagged ones ringed.
 *
 * @param props - The record, the rows and columns, a threshold, and the figure and frame settings.
 * @returns The figure.
 */
export function AnomalyView(
  props: RecordViewProps<AnomalyScores> & Omit<AnomalyViewOptions, 'mode' | 'figure'>,
): ReactElement {
  const figure = useViewFigure(props.figure, ANOMALY_FIGURE)
  const { record, points, columns, threshold } = props
  const build = useCallback(
    (fig: Figure) =>
      anomalyView(record, {
        figure: fig,
        ...(points === undefined ? {} : { points }),
        ...(columns === undefined ? {} : { columns }),
        ...(threshold === undefined ? {} : { threshold }),
      }),
    [record, points, columns, threshold],
  )
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'anomaly-scores'} />
}
