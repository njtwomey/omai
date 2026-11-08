/**
 * React views of trust in predictions: reliability diagrams with expected calibration errors, and conformal prediction
 * with its intervals or set sizes, calibration scores and coverage, played along a stream when there is one.
 */

import type { CalibrationRecord, ConformalRecord } from 'omai-engine/exposition'
import { type ReactElement, useCallback } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { conformalFrames, conformalView, reliabilityView } from '../../views/trust.ts'
import { FramesPlayer, type PlayerViewProps, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

const RELIABILITY_FIGURE = { aspect: 1.1 } as const
const CONFORMAL_FIGURE = { aspect: 1.75 } as const

/**
 * A reliability diagram: observed frequency against mean forecast for each calibrator or group, with the diagonal and
 * each curve's expected calibration error, and the forecasts per bin beneath.
 *
 * @param props - The record and the figure and frame settings.
 * @returns The figure.
 * @index function React component: a reliability diagram: `<CalibrationView record={calibration} />`
 */
export function CalibrationView(props: RecordViewProps<CalibrationRecord>): ReactElement {
  const figure = useViewFigure(props.figure, RELIABILITY_FIGURE)
  const { record } = props
  const build = useCallback((fig: Figure) => reliabilityView(record, { figure: fig }), [record])
  return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'reliability'} />
}

/**
 * Conformal prediction: intervals with the truth (or set sizes), the calibration scores with their quantile, and the
 * coverage as data arrive, played along the stream when there is one.
 *
 * @param props - The record, playback settings and the figure and frame settings.
 * @returns The figure, with its player when the record has a stream.
 */
export function ConformalView(props: RecordViewProps<ConformalRecord> & PlayerViewProps): ReactElement {
  const figure = useViewFigure(props.figure, CONFORMAL_FIGURE)
  const { record } = props
  const frames = useCallback((fig: Figure) => conformalFrames(record, { figure: fig }), [record])
  const build = useCallback((fig: Figure) => conformalView(record, { figure: fig }), [record])
  if (record.stream === undefined)
    return <RecordFigure {...props} figure={figure} build={build} filename={props.filename ?? 'conformal'} />
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'conformal'} />
}
