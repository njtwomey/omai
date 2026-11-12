import type { ModelRun } from 'omai-engine/exposition'
import { ModelRunReplay } from 'omai-render/react'
import type { ReactNode } from 'react'
import { Figure } from './Blocks.tsx'

/**
 * The messages of a run of expectation propagation, as a figure to put below the play button once the run has produced its
 * record: the factor graph of the model with the messages of one step written on its edges and the beliefs in its nodes, the
 * same in words under it, and one play or pause button with a scrubber. It comes in with the figure fade, plays only when the
 * reader presses its button (so it needs no special case for reduced motion), and the picture scales down to a narrow screen
 * while the words under it stay at reading size.
 *
 * The record is the run's own (`modelRunFromTape` over the session's tape, or the `record()` of a library run), made in the
 * worker and sent as plain data. It is cut to at most 40 steps in the worker (`replayRecord` in `replay.ts`), which keeps the
 * first and last step and the steps in which the beliefs moved most; a run of a few sweeps is not touched.
 *
 * @param props - The record, the caption, and optionally what the replay is called for a screen reader and the line under
 *   the picture for each step.
 * @returns The figure.
 */
export function MessageReplay(props: {
  readonly record: ModelRun
  readonly caption: ReactNode
  readonly label?: string
  readonly describe?: (step: number, record: ModelRun) => ReactNode
  /** Where the replay opens: the first step (default) or the last. */
  readonly startAt?: 'first' | 'last'
  /** The direction of the graph's layout: top to bottom (default) or left to right. */
  readonly direction?: 'down' | 'right'
  /** The gap between the layout's rows (or columns), in pixels. */
  readonly rankSep?: number
}): ReactNode {
  const { record, caption, label, describe, startAt, direction, rankSep } = props
  return (
    <Figure caption={caption}>
      <ModelRunReplay
        record={record}
        tex
        {...(direction === undefined ? {} : { direction })}
        {...(rankSep === undefined ? {} : { rankSep })}
        {...(label === undefined ? {} : { label })}
        {...(describe === undefined ? {} : { describe })}
        {...(startAt === undefined ? {} : { startAt })}
      />
    </Figure>
  )
}
