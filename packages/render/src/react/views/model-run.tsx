/**
 * The React player of a model run: the model's plate diagram with each node's belief written in it, the messages that
 * passed at the step on the edges they went along, the convergence series with a cursor beneath, and the line of
 * counters. A record of a run in progress is followed: while the viewer is on the latest step a new step moves the view
 * to it, and a viewer who has stepped back stays where they are.
 */

import type { ModelRun } from 'omai-engine/exposition'
import { type ReactElement, useCallback, useEffect, useMemo, useRef } from 'react'
import type { Figure } from '../../figure/figure.ts'
import {
  modelRunConvergence,
  modelRunDecorate,
  modelRunFrames,
  modelRunOverlay,
  modelRunStatus,
} from '../../views/model-run.ts'
import { Player } from '../Player.tsx'
import { ResponsiveChart } from '../ResponsiveChart.tsx'
import { StructureView } from '../StructureView.tsx'
import { useColourMode } from '../theme.tsx'
import { usePlayback } from '../usePlayback.ts'
import type { PlayerViewProps, RecordViewProps } from './_frame.tsx'

/**
 * Inference on a model graph: the diagram of the model with each node's belief in it (`μ 0.56 · σ 0.83`, an interval or
 * bar glyph under it), the edges that carried a message at the viewed step active with the message in words, and under
 * it the convergence charts and the status line (step, running or how it ended, improper updates). Steps play with the
 * player; a record that grows as the run goes on follows the run.
 *
 * @param props - The record, the playback rate and the frame settings.
 * @returns The figure with its player.
 * @index function React component: a model's plate diagram with live inference state: `<ModelRunPlayer record={r} />`
 */
export function ModelRunPlayer(props: RecordViewProps<ModelRun> & PlayerViewProps): ReactElement {
  const { record } = props
  const mode = useColourMode()
  const count = record.steps.length
  const playback = usePlayback(Math.max(1, count), { fps: props.fps ?? 10 })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  const last = useRef({ count: 0, frame: 0 })
  // A new record opens on its last step; more steps of the same run keep a viewer who was on the latest one on it.
  useEffect(() => {
    const was = last.current
    if (props.startAtFirst === true && was.count === 0) seek.current(0)
    else if (was.count === 0 || was.frame >= was.count - 1) seek.current(Math.max(0, count - 1))
    last.current = { count, frame: last.current.frame }
  }, [count, props.startAtFirst])
  last.current = { count: last.current.count, frame: playback.frame }
  const index = Math.min(playback.frame, Math.max(0, count - 1))
  const step = record.steps[index]?.step
  const frames = useMemo(() => modelRunFrames(record), [record])
  const decorate = useMemo(() => modelRunDecorate(record, step), [record, step])
  const overlay = useMemo(() => modelRunOverlay(record, step), [record, step])
  const hasChart = modelRunConvergence(record, step, { mode }) !== undefined
  const aspect = Math.max(1.5, 4.2 / (record.series.length + 0.6))
  const build = useCallback(
    (fig: Figure) => modelRunConvergence(record, step, { figure: fig, mode }) ?? {},
    [record, step, mode],
  )
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '100%' }}>
      <StructureView
        structure={record.structure}
        overlay={overlay}
        frames={frames}
        decorate={decorate}
        frame={{
          filename: props.filename ?? 'model-run',
          ...(props.title === undefined ? {} : { title: props.title }),
          ...(props.caption === undefined ? {} : { caption: props.caption }),
          ...(count > 1 ? { playback } : {}),
        }}
        {...(props.figure === undefined ? {} : { width: props.figure.width })}
      />
      <p className="m-0 text-xs text-(--wb-muted)" role="status">
        {modelRunStatus(record, step)}
      </p>
      {hasChart ? (
        <div style={{ width: 440, maxWidth: '100%' }}>
          <ResponsiveChart aspect={aspect} narrowAspect={aspect} build={build} />
        </div>
      ) : null}
      {count > 1 ? (
        <Player controller={playback} stamp={(f) => record.steps[f]?.caption ?? `step ${record.steps[f]?.step ?? f}`} />
      ) : null}
    </div>
  )
}
