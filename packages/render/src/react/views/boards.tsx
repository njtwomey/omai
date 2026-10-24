import { type ReactElement, useCallback } from 'react'
import type { BoardSteps, PopulationSteps } from 'whitebox-engine/exposition'
import type { Figure } from '../../figure/figure.ts'
import { boardFrames, populationFrames } from '../../views/boards.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, useViewFigure } from './_frame.tsx'

const BOARD_FIGURE = { aspect: 1.75 } as const

/**
 * A search, solver or simulation played on its board: cells (or graph nodes) coloured by category step by step, their
 * text, the step's caption and the metrics over the steps.
 *
 * @param props - The record, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function BoardPlayer(props: RecordViewProps<BoardSteps> & PlayerViewProps): ReactElement {
  const figure = useViewFigure(props.figure, BOARD_FIGURE)
  const { record } = props
  const frames = useCallback((fig: Figure) => boardFrames(record, { figure: fig }), [record])
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'board'} />
}

/**
 * An evolving population played generation by generation: individuals over the objective's contours (or the best
 * program over its data) and the best and mean fitness.
 *
 * @param props - The record, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function PopulationPlayer(props: RecordViewProps<PopulationSteps> & PlayerViewProps): ReactElement {
  const figure = useViewFigure(props.figure, BOARD_FIGURE)
  const { record } = props
  const frames = useCallback((fig: Figure) => populationFrames(record, { figure: fig }), [record])
  return <FramesPlayer {...props} figure={figure} frames={frames} filename={props.filename ?? 'population'} />
}
