/**
 * The React views of a `PosteriorPairs` record: the corner plot of several methods' posterior draws with its controls
 * (a switch per method, a switch per coordinate and their order, and what the upper triangle shows), `PosteriorPairsView`
 * at a cursor given by the page, and `PosteriorPairsPlayer`, the same with a player beneath it that plays, pauses and
 * scrubs the cursor along the run, whose cursor a page can also own (`cursor` and `onCursor`) to drive it from one
 * playhead shared with its other panels.
 */

import type { PosteriorPairs } from 'omai-engine/exposition'
import { pairsFrameAt } from 'omai-engine/exposition'
import { type ReactElement, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Figure } from '../../figure/figure.ts'
import {
  type PairsUpper,
  pairsFigureOptions,
  pairsMethodColour,
  posteriorPairsView,
} from '../../views/posterior-pairs.ts'
import { Choice } from '../controls/Choice.tsx'
import { ColumnOrder } from '../controls/ColumnOrder.tsx'
import { labelClass } from '../controls/classes.ts'
import { Toggle } from '../controls/Toggle.tsx'
import { Player } from '../Player.tsx'
import { type PlaybackController, usePlayback } from '../usePlayback.ts'
import { type PlayerViewProps, RecordFigure, type RecordViewProps, useViewFigure } from './_frame.tsx'

/** Props of {@link PosteriorPairsView}. */
export interface PosteriorPairsViewProps extends RecordViewProps<PosteriorPairs> {
  /** The step the run is shown at (default its end). */
  readonly cursor?: number
  /** The methods shown at first, by name (default all); the switches change it. */
  readonly methods?: readonly string[]
  /** The coordinates shown at first, in order (default all, up to the record's); the switches and chips change it. */
  readonly coordinates?: readonly string[]
  /** The upper triangle at first (default `correlation`). */
  readonly upper?: PairsUpper
  /** Whether to offer the switches and the choice (default true); a page with controls of its own turns them off. */
  readonly controls?: boolean
  /** A footer under the figure (the player). */
  readonly footer?: ReactNode
}

/**
 * The step text of the player's stamp: whole steps as integers, others to a decimal or two.
 *
 * @param step - A step.
 * @returns The text.
 * @example
 * ```ts
 * import { stepLabel } from 'omai-render/react'
 * [stepLabel(120), stepLabel(2.5), stepLabel(0.125)] // => ['step 120', 'step 2.5', 'step 0.13']
 * ```
 */
export function stepLabel(step: number): string {
  return `step ${Number.isInteger(step) ? step : step >= 10 ? step.toFixed(0) : step >= 1 ? step.toFixed(1) : step.toFixed(2)}`
}

/** A method's label with the swatch of its colour. */
function Swatch(props: { readonly colour: string; readonly children: ReactNode }): ReactElement {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        aria-hidden="true"
        style={{
          width: 10,
          height: 10,
          borderRadius: 999,
          background: `var(--wb-${props.colour})`,
          display: 'inline-block',
        }}
      />
      {props.children}
    </span>
  )
}

/**
 * The corner plot of several methods' posterior draws at a step: a lower triangle of scatter plots, the marginal densities
 * on the diagonal and the correlations (or Gaussian ellipses) above, a colour per method, the truth marked, with a switch
 * per method, a switch and chip per coordinate (drag the chips to reorder) and the choice of the upper triangle. The
 * figure is the library's frame: responsive, themed, exportable as SVG or PNG.
 *
 * @param props - The record, the cursor and what is shown at first, and the frame settings.
 * @returns The figure with its controls.
 * @index function React component: methods' posterior draws as a corner plot at a step.
 */
export function PosteriorPairsView(props: PosteriorPairsViewProps): ReactElement {
  const { record } = props
  const names = useMemo(() => record.coordinates.map((c) => c.name), [record])
  const methodNames = useMemo(() => record.methods.map((m) => m.name), [record])
  const [hidden, setHidden] = useState<readonly string[]>(() =>
    props.methods === undefined ? [] : methodNames.filter((n) => !(props.methods ?? []).includes(n)),
  )
  const [picked, setPicked] = useState<readonly string[]>(() => props.coordinates ?? names)
  const [upper, setUpper] = useState<PairsUpper>(props.upper ?? 'correlation')
  // The coordinates the record has, in the chosen order: a record that changes may lose some.
  const coordinates = useMemo(() => {
    const kept = picked.filter((n) => names.includes(n))
    return kept.length >= 2 ? kept : names
  }, [picked, names])
  const methods = useMemo(() => methodNames.filter((n) => !hidden.includes(n)), [methodNames, hidden])
  const defaults = useMemo(() => pairsFigureOptions(coordinates.length), [coordinates.length])
  const figure = useViewFigure(props.figure, defaults)
  const cursor = props.cursor
  const build = useCallback(
    (fig: Figure) =>
      posteriorPairsView(record, {
        figure: fig,
        methods: methods.length === 0 ? methodNames : methods,
        coordinates,
        upper,
        ...(cursor === undefined ? {} : { cursor }),
      }),
    [record, methods, methodNames, coordinates, upper, cursor],
  )
  const toggleMethod = (name: string, on: boolean): void =>
    setHidden(on ? hidden.filter((n) => n !== name) : [...hidden, name])
  const toggleCoordinate = (name: string, on: boolean): void =>
    setPicked(on ? [...coordinates, name] : coordinates.filter((n) => n !== name))
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: '100%' }}>
      {props.controls === false ? null : (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className={labelClass}>methods</span>
            {record.methods.map((m, i) => (
              <Toggle
                key={m.name}
                size="sm"
                width="auto"
                label={<Swatch colour={pairsMethodColour(i)}>{m.name}</Swatch>}
                value={methods.includes(m.name)}
                disabled={methods.length === 1 && methods.includes(m.name)}
                onChange={(on) => toggleMethod(m.name, on)}
              />
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className={labelClass}>coordinates</span>
            {names.map((n) => (
              <Toggle
                key={n}
                size="sm"
                width="auto"
                label={n}
                value={coordinates.includes(n)}
                disabled={coordinates.length <= 2 && coordinates.includes(n)}
                onChange={(on) => toggleCoordinate(n, on)}
              />
            ))}
          </div>
          <div className="flex flex-wrap items-end gap-x-6 gap-y-2">
            <ColumnOrder label="order (drag)" order={coordinates} onOrder={(o) => setPicked(o)} />
            <Choice label="upper triangle" value={upper} options={['correlation', 'ellipses']} onChange={setUpper} />
          </div>
        </div>
      )}
      <RecordFigure
        {...props}
        figure={figure}
        build={build}
        filename={props.filename ?? 'posterior-pairs'}
        {...(props.footer === undefined ? {} : { footer: props.footer })}
      />
    </div>
  )
}

/** Options of {@link usePairsCursor}. */
export interface PairsCursorOptions extends PlayerViewProps {
  /** The page's cursor, when it owns the playhead. */
  readonly cursor?: number
  /** Called with the step the player moves the cursor to (playing, scrubbing, stepping). */
  readonly onCursor?: (step: number) => void
}

/**
 * The playback of a pair plot's cursor along its record's frames, optionally kept in step with a cursor the page owns:
 * the player's frame is reported through `onCursor`, and a `cursor` the page sets moves the player to the frame at or
 * before it, with no echo of either back to the other. A new record opens on its end (or its start) unless the page
 * gives a cursor.
 *
 * @param record - The record.
 * @param options - The page's cursor and handler, the playback rate and the frame to start on.
 * @returns The controller of the player and the step the figure shows (the page's cursor when it gave one).
 * @index function React hook: play a pair plot's cursor, in step with a page's.
 */
export function usePairsCursor(
  record: PosteriorPairs,
  options: PairsCursorOptions = {},
): { readonly playback: PlaybackController; readonly step: number } {
  const { cursor, onCursor } = options
  const count = record.cursors.length
  const playback = usePlayback(Math.max(1, count), { fps: options.fps ?? 5 })
  const frameOf = useCallback((step: number): number => Math.max(0, pairsFrameAt(record, step)), [record])
  const seek = useRef(playback.seek)
  seek.current = playback.seek
  const sent = useRef<number | undefined>(undefined)
  // The frame the page put the player on (the first frame at mount), which is not the viewer moving it: not reported back.
  const placed = useRef<number | undefined>(0)
  // A new record opens on its end (or its start), unless the page says where it is.
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new record resets the player
  useEffect(() => {
    if (cursor === undefined) seek.current(options.startAtFirst === true ? 0 : count - 1)
  }, [record, count, options.startAtFirst])
  // The page's cursor to the player: when the cursor changes, not when the player moves.
  const frame = useRef(playback.frame)
  frame.current = playback.frame
  useEffect(() => {
    if (cursor === undefined || cursor === sent.current) return
    const f = frameOf(cursor)
    if (f !== frame.current) {
      placed.current = f
      seek.current(f)
    }
  }, [cursor, frameOf])
  // The player's frame to the page.
  useEffect(() => {
    if (placed.current !== undefined) {
      if (placed.current === playback.frame) placed.current = undefined
      return
    }
    if (onCursor === undefined) return
    const step = record.cursors[playback.frame]
    if (step === undefined || step === sent.current) return
    if (cursor !== undefined && frameOf(cursor) === playback.frame) return
    sent.current = step
    onCursor(step)
  }, [playback.frame, record, cursor, onCursor, frameOf])
  return { playback, step: cursor ?? record.cursors[playback.frame] ?? record.stepRange[1] }
}

/** Props of {@link PosteriorPairsPlayer}. */
export interface PosteriorPairsPlayerProps extends PosteriorPairsViewProps, PairsCursorOptions {}

/**
 * The corner plot with a player: play, pause, step, scrub by dragging and a choice of speed move the cursor along the
 * run's steps, the draws arriving and the densities and correlations following. A page that owns the playhead (the
 * lab's other panels at the same step) gives `cursor` and `onCursor`: the player then moves the page's cursor, and a
 * cursor the page sets moves the player, wherever the step falls between the record's frames (the scatter shows the
 * draws up to the exact step, the densities and correlations those of the frame at or before it). With a `cursor` and
 * no `onCursor` the figure follows the page and there is no player. A new record opens on its end.
 *
 * @param props - The record, the playback rate, the optional controlled cursor and the frame settings.
 * @returns The figure with its controls and player.
 * @index function React component: the posterior corner plot with a player, optionally driven by a page's cursor.
 */
export function PosteriorPairsPlayer(props: PosteriorPairsPlayerProps): ReactElement {
  const { record, cursor, onCursor } = props
  const { playback, step } = usePairsCursor(record, {
    ...(cursor === undefined ? {} : { cursor }),
    ...(onCursor === undefined ? {} : { onCursor }),
    ...(props.fps === undefined ? {} : { fps: props.fps }),
    ...(props.startAtFirst === undefined ? {} : { startAtFirst: props.startAtFirst }),
  })
  const player = record.cursors.length > 1 && (cursor === undefined || onCursor !== undefined)
  return (
    <PosteriorPairsView
      {...props}
      cursor={step}
      footer={
        player ? (
          <Player
            controller={playback}
            stamp={(f) =>
              stepLabel(cursor !== undefined && pairsFrameAt(record, cursor) === f ? cursor : (record.cursors[f] ?? 0))
            }
          />
        ) : undefined
      }
    />
  )
}
