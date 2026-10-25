/**
 * The React player of epicycles: a closed curve drawn by a chain of rotating arrows, round one cycle, with the zoom on
 * the pen and the spectrum.
 */

import type { Epicycles } from 'mdml-engine/exposition'
import { type ReactElement, useCallback, useMemo } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { epicycleAspect, epicycleFrames } from '../../views/epicycles.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, useViewFigure } from './_frame.tsx'

/**
 * A closed curve drawn by rotating arrows, looping round one cycle from the start (30 frames a second by default).
 *
 * @param props - The record, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function EpicyclePlayer(props: RecordViewProps<Epicycles> & PlayerViewProps): ReactElement {
  const { record } = props
  const defaults = useMemo(() => ({ aspect: epicycleAspect(record) }), [record])
  const figure = useViewFigure(props.figure, defaults)
  const frames = useCallback((fig: Figure) => epicycleFrames(record, { figure: fig }), [record])
  return (
    <FramesPlayer
      {...props}
      fps={props.fps ?? 30}
      cycle
      figure={figure}
      frames={frames}
      filename={props.filename ?? 'epicycles'}
    />
  )
}
