/**
 * The React player of agent steps: agents moving in their world, frame by frame, with their series beneath.
 */

import type { AgentSteps } from 'mdml-engine/exposition'
import { type ReactElement, useCallback, useMemo } from 'react'
import type { Figure } from '../../figure/figure.ts'
import { agentAspect, agentFrames } from '../../views/agents.ts'
import { FramesPlayer, type PlayerViewProps, type RecordViewProps, useViewFigure } from './_frame.tsx'

/**
 * Agents played frame by frame, looping from the start (30 frames a second by default).
 *
 * @param props - The record, the playback rate, and the figure settings.
 * @returns The figure with its player.
 */
export function AgentPlayer(props: RecordViewProps<AgentSteps> & PlayerViewProps): ReactElement {
  const { record } = props
  const defaults = useMemo(() => ({ aspect: agentAspect(record) }), [record])
  const figure = useViewFigure(props.figure, defaults)
  const frames = useCallback((fig: Figure) => agentFrames(record, { figure: fig }), [record])
  return (
    <FramesPlayer
      {...props}
      fps={props.fps ?? 30}
      cycle
      figure={figure}
      frames={frames}
      filename={props.filename ?? 'agents'}
    />
  )
}
