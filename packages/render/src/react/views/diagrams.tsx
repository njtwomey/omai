/**
 * The React player of a diagram record: its structure (a network's architecture, a computation graph) drawn once, with
 * each step's overlay of active nodes, edges and value badges played over it.
 */

import type { DiagramRecord } from 'mdml-engine/exposition'
import { type ReactElement, useEffect, useMemo, useRef } from 'react'
import { Player } from '../Player.tsx'
import { StructureView } from '../StructureView.tsx'
import { usePlayback } from '../usePlayback.ts'
import type { PlayerViewProps, RecordViewProps } from './_frame.tsx'

/**
 * A diagram record: its structure (a network's architecture, a computation graph) drawn once, and, when the record has
 * steps, their overlays (active nodes and edges, value badges) played over it with each step's caption beneath, as
 * backpropagation's forward values and deltas are. The layout is sized for every step, so nothing jumps.
 *
 * @param props - The record, the playback rate and the frame to start on, and the frame's title and file name.
 * @returns The figure, with its player when it has steps.
 */
export function DiagramPlayer(props: RecordViewProps<DiagramRecord> & PlayerViewProps): ReactElement {
  const { record } = props
  const steps = record.steps ?? []
  const count = steps.length
  const frames = useMemo(() => steps.map((s) => s.overlay), [steps])
  const playback = usePlayback(Math.max(1, count), {
    fps: props.fps ?? 10,
  })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // A new record opens on its last step (the result), or on its first when asked.
  // biome-ignore lint/correctness/useExhaustiveDependencies: `record` changes reset the view
  useEffect(() => {
    seek.current(props.startAtFirst === true ? 0 : Math.max(0, count - 1))
  }, [record, count, props.startAtFirst])
  const step = steps[Math.min(playback.frame, Math.max(0, count - 1))]
  const stepName = record.stepName ?? 'step'
  const stamp = (f: number): string => steps[f]?.caption ?? `${stepName} ${f + 1}/${count}`
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '100%' }}>
      <StructureView
        structure={record.structure}
        {...(step === undefined ? {} : { overlay: step.overlay, frames })}
        frame={{
          filename: props.filename ?? 'diagram',
          ...(props.title === undefined ? {} : { title: props.title }),
          ...(props.caption === undefined ? {} : { caption: props.caption }),
          ...(count > 1 ? { playback } : {}),
        }}
        {...(props.figure === undefined ? {} : { width: props.figure.width })}
      />
      {count > 1 ? <Player controller={playback} stamp={stamp} /> : null}
    </div>
  )
}
