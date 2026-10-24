import { type ReactElement, useCallback } from 'react'
import type { DensityCurves } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { densityFrames } from '../../views/densities.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, useViewFigure } from './_frame.tsx'

const DENSITY_FIGURE = { aspect: 1.6 } as const

/**
 * Densities over one quantity: the prior dashed, the posterior filled with its credible interval, played as the
 * observations arrive when the record has the steps.
 *
 * @param props - The record, the playback rate, and the figure settings.
 * @returns The figure, with its player when it has steps.
 */
export function DensityView(props: RecordViewProps<DensityCurves> & PlayerViewProps): ReactElement {
  const figure = useViewFigure(props.figure, DENSITY_FIGURE)
  const { record } = props
  const frames = useCallback((fig: Figure) => densityFrames(record, { figure: fig }), [record])
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'densities'} />
}
