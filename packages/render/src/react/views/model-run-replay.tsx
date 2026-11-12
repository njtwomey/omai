/**
 * The compact replay of a model run: the model's factor graph with, at the step shown, the messages that were sent
 * written on their edges and the beliefs written in the nodes, a line of text that says the same in words, and one
 * play or pause button with a scrubber. It is the quiet counterpart of `ModelRunPlayer` (no convergence charts, no
 * rate menu, no export frame) for a page that tells the run as a story and has the record in hand.
 */

import type { ModelRun, NodeState } from 'omai-engine/exposition'
import { modelRunAt } from 'omai-engine/exposition'
import type { StructureNode } from 'omai-engine/scene'
import { type ReactElement, type ReactNode, useEffect, useId, useMemo, useRef, useState } from 'react'
import { modelRunDecorate, modelRunFrames, modelRunOverlay } from '../../views/model-run.ts'
import { cx, iconButtonClass } from '../controls/classes.ts'
import { PauseIcon, PlayIcon } from '../controls/icons.tsx'
import { StructureView } from '../StructureView.tsx'
import { usePlayback } from '../usePlayback.ts'

/** Props of {@link ModelRunReplay}. */
export interface ModelRunReplayProps {
  /** The recorded run: plain data, so it may have come from a worker. */
  readonly record: ModelRun
  /** What the replay is, for a reader who cannot see the picture (default 'The messages of the run'). */
  readonly label?: string
  /** Steps per second while playing (default 1: one step is a sweep over the whole graph, which takes a moment to read). */
  readonly fps?: number
  /** Where the replay opens: the first step (default) or the last. */
  readonly startAt?: 'first' | 'last'
  /** The line under the picture for a step, in place of the one made from the record. */
  readonly describe?: (step: number, record: ModelRun) => ReactNode
  /** The width of the picture in pixels (default its own, at most the container's). */
  readonly width?: number
  /**
   * The direction the graph is laid out in: top to bottom (`down`, the default) or left to right (`right`). A left to right
   * layout is drawn top to bottom when the replay is narrower than `narrowBelow`, where a wide picture would shrink to an
   * unreadable size.
   */
  readonly direction?: 'down' | 'right'
  /** The width in pixels below which a left to right layout turns top to bottom (default 480). */
  readonly narrowBelow?: number
  /** The gap between the rows (or columns) of the layout, in pixels. */
  readonly rankSep?: number
  readonly className?: string
}

/**
 * The record as the replay draws it: a node whose belief is improper (an improper product of messages) shows no belief at
 * that step, rather than the word `improper`. The replay is for telling a run, not for diagnosing one; `ModelRunPlayer`
 * shows improper beliefs. The node's entry is blanked, not removed: the state at a step carries the beliefs of earlier steps
 * forward (`modelRunAt`), and a removed entry would let the last proper belief show as if it still held.
 *
 * @param record - The record.
 * @returns The record with each improper belief replaced by one with no numbers and no text.
 * @example
 * ```ts
 * import { modelRun, modelRunAt } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { replayDrawing } from 'omai-render/react'
 * const s = structure('model-graph', [{ id: 'x', label: 'x', role: 'latent' }], [])
 * const steps = [
 *   { step: 1, nodes: { x: { mean: 1, sd: 2, text: '1 ± 2' } } },
 *   { step: 2, nodes: { x: { improper: true, text: 'improper' } } },
 * ]
 * const r = modelRun({ structure: s, status: 'completed', series: [], steps })
 * modelRunAt(replayDrawing(r), 2)?.nodes['x']?.mean // => undefined
 * ```
 */
export function replayDrawing(record: ModelRun): ModelRun {
  return {
    ...record,
    steps: record.steps.map((s) => ({
      ...s,
      nodes: Object.fromEntries(
        Object.entries(s.nodes).map(([id, state]): [string, NodeState] => [
          id,
          state.improper === true ? { text: '' } : state,
        ]),
      ),
    })),
  }
}

/** A name for a node of the graph: its label, or for a factor the site it belongs to. */
function nameOf(node: StructureNode | undefined, id: string): string {
  if (node === undefined) return id
  if (node.label !== '') return node.label
  const site = node.data?.['site']
  return typeof site === 'string' ? `factor of ${site}` : id
}

/**
 * The step in words, made from the record: the messages sent (`prior on x to x: N(μ = 20, σ² = 16)`) and what each node
 * with a belief was then believed to be.
 *
 * @param record - The record.
 * @param step - The tape step shown.
 * @returns The lines, messages first and then beliefs.
 * @example
 * ```ts
 * import { modelRun } from 'omai-engine/exposition'
 * import { structure } from 'omai-engine/scene'
 * import { replayLines } from 'omai-render/react'
 * const s = structure('model-graph', [{ id: 'x', label: 'x', role: 'latent' }], [])
 * const r = modelRun({ structure: s, status: 'completed', series: [], steps: [{ step: 1, nodes: { x: { mean: 1, sd: 2, text: '1 ± 2' } } }] })
 * replayLines(r, 1).beliefs // => ['x: 1 ± 2']
 * ```
 */
export function replayLines(
  record: ModelRun,
  step: number,
): { readonly messages: readonly string[]; readonly beliefs: readonly string[] } {
  const at = modelRunAt(record, step)
  if (at === undefined) return { messages: [], beliefs: [] }
  const byId = new Map(record.structure.nodes.map((n) => [n.id, n]))
  const messages = at.messages.map((m) => {
    const from = nameOf(byId.get(m.from), m.from)
    return `${from} to ${nameOf(byId.get(m.to), m.to)}: ${m.text}`
  })
  const beliefs = Object.entries(at.nodes).flatMap(([id, state]) =>
    state.observed === undefined && !state.improper ? [`${nameOf(byId.get(id), id)}: ${state.text}`] : [],
  )
  return { messages, beliefs }
}

/**
 * The replay of a model run, one step at a time: a factor graph with the messages of the step on its edges and the
 * beliefs in its nodes, the same in words beneath it, and a play or pause button with a scrubber. Nothing plays by
 * itself and the picture has no transitions, so it needs no special case for a reader who prefers reduced motion;
 * the scrubber and the button work from the keyboard and the line of words is announced when the step changes while
 * the replay is paused.
 *
 * @param props - The record and the options.
 * @returns The replay.
 * @index function React component: a compact replay of a model run's messages and beliefs: `<ModelRunReplay record={r} />`
 */
export function ModelRunReplay(props: ModelRunReplayProps): ReactElement {
  const { record, label = 'The messages of the run', fps = 1, startAt = 'first', describe, width, className } = props
  const { rankSep, narrowBelow = 480 } = props
  const root = useRef<HTMLElement | null>(null)
  const [narrow, setNarrow] = useState(false)
  useEffect(() => {
    const el = root.current
    if (el === null || typeof ResizeObserver === 'undefined') return
    const measure = (): void => setNarrow(el.clientWidth > 0 && el.clientWidth < narrowBelow)
    measure()
    const watch = new ResizeObserver(measure)
    watch.observe(el)
    return () => watch.disconnect()
  }, [narrowBelow])
  const direction = narrow ? 'down' : props.direction
  const count = record.steps.length
  const playback = usePlayback(Math.max(1, count), { fps })
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  // a new record opens where asked
  // biome-ignore lint/correctness/useExhaustiveDependencies: the record is what resets the replay
  useEffect(() => seek.current(startAt === 'last' ? Math.max(0, count - 1) : 0), [record, count, startAt])
  const index = Math.min(playback.frame, Math.max(0, count - 1))
  const step = record.steps[index]?.step ?? 0
  const drawn = useMemo(() => replayDrawing(record), [record])
  const frames = useMemo(() => modelRunFrames(drawn), [drawn])
  const overlay = useMemo(() => modelRunOverlay(drawn, step), [drawn, step])
  const decorate = useMemo(() => modelRunDecorate(drawn, step), [drawn, step])
  // biome-ignore lint/correctness/useExhaustiveDependencies: `describe` is expected to be stable or to be a pure function of the record
  const words = useMemo(
    () =>
      record.steps.map((s): ReactNode => {
        if (describe !== undefined) return describe(s.step, record)
        const lines = replayLines(record, s.step)
        return (
          <>
            <p className="m-0 text-(--wb-muted)">{s.caption ?? `step ${s.step}`}</p>
            {lines.messages.length === 0 ? null : (
              <ul className="m-0 mt-1 list-none p-0">
                {lines.messages.map((m) => (
                  <li key={m}>{m}</li>
                ))}
              </ul>
            )}
            {lines.beliefs.length === 0 ? null : <p className="m-0 mt-1">Belief: {lines.beliefs.join('; ')}</p>}
          </>
        )
      }),
    [record],
  )
  const id = useId()
  return (
    <section ref={root} className={cx('flex w-full max-w-full flex-col gap-3', className)} aria-label={label}>
      <div className="flex justify-center">
        <StructureView
          structure={record.structure}
          overlay={overlay}
          frames={frames}
          decorate={decorate}
          chrome={false}
          frame={false}
          {...(direction === undefined ? {} : { direction })}
          {...(rankSep === undefined || narrow ? {} : { rankSep })}
          {...(width === undefined ? {} : { width })}
        />
      </div>
      {/* Every step's words sit in one grid cell and only the current one is visible, so the block keeps the height of the
          longest and nothing below it moves as the replay plays. */}
      <div
        className="grid text-sm leading-relaxed text-(--wb-fg)"
        aria-live={playback.playing ? 'off' : 'polite'}
        aria-atomic="true"
      >
        {words.map((content, i) => (
          <div
            // biome-ignore lint/suspicious/noArrayIndexKey: the steps are a fixed list
            key={i}
            className={cx('col-start-1 row-start-1', i !== index && 'invisible')}
            aria-hidden={i !== index}
          >
            {content}
          </div>
        ))}
      </div>
      {count > 1 ? (
        <div className="flex items-center gap-3">
          <button
            type="button"
            className={cx(iconButtonClass, 'rounded-md')}
            aria-label={
              playback.playing ? 'Pause the replay' : index >= count - 1 ? 'Replay from the start' : 'Play the replay'
            }
            onClick={playback.toggle}
          >
            {playback.playing ? <PauseIcon size={14} /> : <PlayIcon size={14} />}
          </button>
          <input
            id={id}
            type="range"
            className="min-w-0 flex-1 accent-(--wb-accent)"
            min={0}
            max={count - 1}
            step={1}
            value={index}
            aria-label="Step of the run"
            aria-valuetext={`step ${index + 1} of ${count}`}
            onChange={(e) => {
              playback.pause()
              playback.seek(Number(e.target.value))
            }}
          />
          <span className="shrink-0 text-xs tabular-nums text-(--wb-muted)">
            {index + 1} / {count}
          </span>
        </div>
      ) : null}
    </section>
  )
}
