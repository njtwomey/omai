/**
 * The generic tape viewer: whatever a session recorded, drawn by path kind with no bespoke code. Scalars become curves
 * grouped by prefix, episodes a return curve you can click, 2-D tensors a grid at the scrubbed step, frames a scene at
 * the scrubbed step, messages a list, and derived records their exposition view. Plus play, step and scrub controls.
 */

import { isExposition } from 'omai-engine/exposition'
import type { Scene } from 'omai-engine/scene'
import { type Config, episodes, frames, messages, type Run, type TensorEvent } from 'omai-engine/trace'
import { type ReactElement, type ReactNode, useMemo } from 'react'
import { lines } from '../../charts/lines.ts'
import { heatmapChart } from '../../charts/matrices.ts'
import { cartesian } from '../../charts/plot.ts'
import { figure } from '../../figure/index.ts'
import { episodeCurves } from '../../trace/episodes.ts'
import { metricCurves, metricNames } from '../../trace/metrics.ts'
import { Chart } from '../Chart.tsx'
import { FigureFrame } from '../FigureFrame.tsx'
import { SceneView } from '../SceneView.tsx'
import { useColourMode } from '../theme.tsx'
import { ExpositionView } from '../views/ExpositionView.tsx'
import { Note } from './RowsPanel.tsx'
import { SessionPlayer } from './SessionPlayer.tsx'
import { recordedSteps, stepMarker, useScrubClick } from './scrub.ts'
import type { LiveSession } from './useSession.ts'

function rowsOf<T>(flat: readonly T[], width: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < flat.length; i += width) out.push(flat.slice(i, i + width))
  return out
}

const FIG = figure({ width: 520, height: 240 })
const SMALL = figure({ width: 320, height: 240 })

/** What the tape views take. */
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

/**
 * The player over the session, then its tape's channel panels.
 *
 * @param props - The session, the scrubbed step and its setter, and extra panels.
 * @returns The player and the channel panels.
 * @index function React component: draw whatever a session logged: `<TapeView live={live} at={at} onScrub={setAt} />`
 *
 */
export function TapeView<S, C extends Config>(props: TapeViewProps<S, C>): ReactElement {
  return (
    <div className="flex flex-col gap-3">
      <SessionPlayer live={props.live} at={props.at} onScrub={props.onScrub} />
      <TapeChannels {...props} />
    </div>
  )
}

/**
 * Every channel of the tape drawn by its kind (curves, tensors, frames, messages, projections), without a player.
 *
 * @param props - The session and the scrubbed step, as for {@link TapeView}; the setter is optional here.
 * @returns The channel panels.
 */
export function TapeChannels<S, C extends Config>(
  props: Omit<TapeViewProps<S, C>, 'onScrub'> & { readonly onScrub?: (t: number | null) => void },
): ReactElement {
  const { live, at, onEpisodeClick } = props
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
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-4">
        {[...groups].map(([prefix, names]) => (
          <MetricGroup
            key={prefix}
            prefix={prefix}
            names={names}
            run={run}
            viewed={viewed}
            head={head}
            mode={mode}
            onScrub={props.onScrub}
          />
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
        {Object.keys(s.method.projections ?? {}).map((name) => {
          // A projection of the tape so far; it may have nothing to project yet, and says why when it throws.
          let record: unknown
          try {
            record = s.project(name)
          } catch (e) {
            return (
              <Note key={name}>{`projection ${name} unavailable: ${e instanceof Error ? e.message : String(e)}`}</Note>
            )
          }
          return record !== undefined && isExposition(record) ? (
            <div key={name} className="flex flex-col gap-1">
              <div className="text-sm opacity-70">{name}, from the tape so far</div>
              <ExpositionView record={record} figure={SMALL} />
            </div>
          ) : null
        })}
        {props.children}
      </div>
    </div>
  )
}

// One group of metric curves with the viewed step marked; a click on the plot scrubs to the nearest recorded step.
function MetricGroup(props: {
  readonly prefix: string
  readonly names: readonly string[]
  readonly run: Run
  readonly viewed: number
  readonly head: number
  readonly mode: 'light' | 'dark'
  readonly onScrub: ((t: number | null) => void) | undefined
}): ReactElement {
  const { prefix, names, run, viewed, head, mode } = props
  const option = useMemo(() => {
    const curves = metricCurves(run, names, { mode })
    return cartesian([curves, ...stepMarker(curves, viewed, mode)], {
      mode,
      figure: FIG,
      title: prefix || 'metrics',
      x: { label: 'step' },
      legend: true,
    })
  }, [run, names, viewed, mode, prefix])
  const steps = useMemo(() => recordedSteps(run, names), [run, names])
  const onInit = useScrubClick(steps, head, props.onScrub)
  return (
    <FigureFrame figure={FIG} filename={prefix || 'metrics'}>
      <Chart figure={FIG} option={option} onInit={onInit} />
    </FigureFrame>
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
