// The generic tape viewer: whatever a session recorded, drawn by path kind with no bespoke code. Scalars become curves
// grouped by prefix, episodes a return curve you can click, 2-D tensors a grid at the scrubbed step, frames a scene at
// the scrubbed step, messages a list, and derived records their exposition view. Plus play, step and scrub controls.
import { isExposition } from 'mdml-engine/exposition'
import type { Scene } from 'mdml-engine/scene'
import { type Config, episodes, frames, messages, type TensorEvent } from 'mdml-engine/trace'
import { cartesian, heatmapChart, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  ExpositionView,
  FigureFrame,
  type PlaybackController,
  Player,
  SceneView,
  useColourMode,
} from 'mdml-render/react'
import { episodeCurves, metricCurves, metricNames } from 'mdml-render/trace'
import { type ReactElement, type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
import type { LiveSession } from './useSession.ts'

function rowsOf<T>(flat: readonly T[], width: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < flat.length; i += width) out.push(flat.slice(i, i + width))
  return out
}

const FIG = figure({ width: 520, height: 240 })
const SMALL = figure({ width: 320, height: 240 })
/** Steps per second the player's menu offers. */
const RATES = [1, 2, 5, 10, 30, 60, 120, 300, 1000]

export interface TapeViewProps<S, C extends Config> {
  readonly live: LiveSession<S, C>
  /** The step shown by the scrubber, or `null` to follow the head. */
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
  /** Called with the episode index when the return curve is clicked. */
  readonly onEpisodeClick?: (episode: number) => void
  /** Extra panels, drawn after the tape's own. */
  readonly children?: ReactNode
}

function groupByPrefix(names: readonly string[]): Map<string, string[]> {
  const out = new Map<string, string[]>()
  for (const n of names) {
    const i = n.lastIndexOf('/')
    const prefix = i < 0 ? '' : n.slice(0, i)
    out.set(prefix, [...(out.get(prefix) ?? []), n])
  }
  return out
}

export function TapeView<S, C extends Config>(props: TapeViewProps<S, C>): ReactElement {
  const { live, at, onScrub, onEpisodeClick } = props
  const mode = useColourMode()
  const s = live.session
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const run = useMemo(() => s.run(), [s, live.version])
  const head = s.t
  const viewed = at === null ? head : Math.min(at, head)
  const groups = useMemo(() => groupByPrefix(metricNames(run)), [run])
  const tensorNames = useMemo(
    () => [...new Set(run.events.flatMap((e) => (e.type === 'tensor' ? [e.name] : [])))],
    [run],
  )
  const kinds = useMemo(() => [...new Set(run.events.flatMap((e) => (e.type === 'message' ? [e.kind] : [])))], [run])
  const eps = useMemo(() => episodes(run), [run])
  const frameList = useMemo(() => frames(run), [run])
  const frameAt = frameList.filter((f) => f.step <= viewed).at(-1)
  const counts = useMemo(() => {
    const c = new Map<string, number>()
    for (const e of run.events) c.set(e.type, (c.get(e.type) ?? 0) + 1)
    return [...c].map(([k, v]) => `${v} ${k}`).join(', ')
  }, [run])
  // The standard player over the session. Play always moves the cursor forward at `fps` steps per second: through
  // the tape while there is tape ahead of it (a replay; a finished run restarts from the first step), and into live
  // computation once the cursor reaches the head of an unfinished run. Stepping forward past the head computes one
  // more step.
  const follow = (): void => onScrub(null)
  const [replaying, setReplaying] = useState(false)
  const cursor = useRef({ viewed, head, done: live.done })
  cursor.current = { viewed, head, done: live.done }
  const handlers = useRef({ onScrub, play: live.play })
  handlers.current = { onScrub, play: live.play }
  // biome-ignore lint/correctness/useExhaustiveDependencies: a remade session starts still
  useEffect(() => setReplaying(false), [s])
  useEffect(() => {
    if (!replaying) return
    let last: number | undefined
    let carry = 0
    let handle = requestAnimationFrame(function tick(now: number) {
      const { viewed: v, head: h, done } = cursor.current
      if (last !== undefined) {
        const due = carry + ((now - last) * live.fps) / 1000
        const steps = Math.floor(due)
        carry = due - steps
        if (steps > 0) {
          if (v + steps >= h) {
            // Caught up with the head: stop on a finished run, or hand over to live computation.
            setReplaying(false)
            handlers.current.onScrub(null)
            if (!done) handlers.current.play()
            return
          }
          handlers.current.onScrub(v + steps)
        }
      }
      last = now
      handle = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(handle)
  }, [replaying, live.fps])
  const playing = live.playing || replaying
  const play = (): void => {
    if (viewed < head) setReplaying(true)
    else if (live.done) {
      onScrub(0)
      setReplaying(true)
    } else {
      follow()
      live.play()
    }
  }
  const pause = (): void => {
    live.pause()
    setReplaying(false)
  }
  const controller: PlaybackController = {
    frame: viewed,
    count: head + 1,
    playing,
    fps: live.fps,
    open: !live.done,
    play,
    pause,
    toggle: () => (playing ? pause() : play()),
    step: (dir) => {
      if (dir === -1) onScrub(Math.max(0, viewed - 1))
      else if (viewed < head) onScrub(viewed + 1 >= head ? null : viewed + 1)
      else live.advance({ steps: 1 })
    },
    seek: (f) => onScrub(f >= head ? null : Math.max(0, f)),
    setFps: live.setFps,
    reset: () => {
      pause()
      onScrub(0)
    },
  }
  const status = s.status === undefined ? (live.done ? '' : '') : ` · ${s.status}`
  return (
    <div className="flex flex-col gap-3">
      <Player controller={controller} stamp={(f) => `step ${f} of ${head}${status}`} fpsChoices={RATES} />
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
        <span>{counts || 'nothing recorded'}</span>
        <span>last slice {live.lastMs.toFixed(1)} ms</span>
        {live.done ? null : (
          <button
            type="button"
            className="rounded-md border border-line px-2 py-0.5 hover:border-muted hover:text-fg"
            onClick={() => {
              follow()
              live.advance({ steps: Number.POSITIVE_INFINITY })
            }}
          >
            run to the end
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-4">
        {[...groups].map(([prefix, names]) => (
          <FigureFrame key={prefix} figure={FIG} filename={prefix || 'metrics'}>
            <Chart
              figure={FIG}
              option={cartesian([metricCurves(run, names, { mode })], {
                mode,
                figure: FIG,
                title: prefix || 'metrics',
                x: { label: 'step' },
                legend: true,
              })}
            />
          </FigureFrame>
        ))}
        {eps.length > 0 ? (
          <Chart
            figure={FIG}
            option={cartesian(episodeCurves(run, { mode, window: 10 }), {
              mode,
              figure: FIG,
              title: 'return per episode (click one)',
              x: { label: 'episode' },
            })}
            onEvents={{
              click: (p: unknown) => {
                const i = (p as { readonly dataIndex?: number }).dataIndex
                if (i !== undefined) onEpisodeClick?.(i)
              },
            }}
          />
        ) : null}
        {tensorNames.map((name) => (
          <TensorPanel
            key={name}
            name={name}
            events={run.events.filter((e): e is TensorEvent => e.type === 'tensor' && e.name === name)}
            viewed={viewed}
            mode={mode}
          />
        ))}
        {frameAt === undefined ? null : (
          <div className="flex flex-col gap-1">
            <div className="text-sm opacity-70">frame at step {frameAt.step}</div>
            <SceneView scene={frameAt.frame as unknown as Scene} width={320} mode={mode} />
          </div>
        )}
        {kinds.map((kind) => {
          const list = messages(run, kind)
          const latest = list.filter((m) => m.step <= viewed).at(-1)
          if (latest !== undefined && isExposition(latest.payload))
            return (
              <div key={kind} className="flex flex-col gap-1">
                <div className="text-sm opacity-70">
                  {kind} at step {latest.step}
                </div>
                <ExpositionView record={latest.payload} figure={SMALL} />
              </div>
            )
          return (
            <div key={kind} className="flex max-w-xs flex-col gap-1 text-xs">
              <div className="text-sm opacity-70">
                {kind} ({list.length})
              </div>
              {list
                .filter((m) => m.step <= viewed)
                .slice(-6)
                .map((m) => (
                  <div key={m.step} className="truncate font-mono">
                    {m.step}: {JSON.stringify(m.payload)}
                  </div>
                ))}
            </div>
          )
        })}
        {props.children}
      </div>
    </div>
  )
}

function TensorPanel(props: {
  readonly name: string
  readonly events: readonly TensorEvent[]
  readonly viewed: number
  readonly mode: 'light' | 'dark'
}): ReactElement | null {
  const { name, events, viewed, mode } = props
  const at = events.filter((e) => e.step <= viewed).at(-1)
  if (at === undefined) return null
  if (at.values !== undefined) {
    const width = at.shape.length >= 2 ? (at.shape[at.shape.length - 1] ?? 1) : at.values.length
    const z = rowsOf(at.values, width)
    return (
      <Chart
        figure={SMALL}
        option={heatmapChart(z, {}, { mode, figure: SMALL, title: `${name} at ${at.step} (${at.shape.join('×')})` })}
      />
    )
  }
  const steps = events.map((e) => e.step)
  const norms = events.map((e) => e.summary?.norm ?? Number.NaN)
  return (
    <Chart
      figure={SMALL}
      option={cartesian([lines([{ x: steps, y: norms, name }], { mode })], {
        mode,
        figure: SMALL,
        title: `‖${name}‖ (summary)`,
        x: { label: 'step' },
      })}
    />
  )
}
