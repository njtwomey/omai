/**
 * Playback controls in one row for a precomputed timeline or a live algorithm: a reset of the page's controls and an
 * eager toggle when the controller offers them, seek to the start, step back, play or pause with a menu of target
 * frame rates, step forward, seek to the end, a draggable scrubber and an optional time stamp.
 */

import { type PointerEvent, type ReactNode, useCallback, useEffect, useId, useRef, useState } from 'react'
import { cx, iconButtonClass, menuItemClass } from './controls/classes.ts'
import {
  ChevronDownIcon,
  PauseIcon,
  PlayIcon,
  RewindIcon,
  SkipBackIcon,
  SkipForwardIcon,
  StepBackIcon,
  StepForwardIcon,
  ZapIcon,
} from './controls/icons.tsx'
import { Popover } from './controls/Popover.tsx'
import { FPS_CHOICES, timeStamp } from './playhead.ts'
import type { PlaybackController } from './usePlayback.ts'

/** How long after playback stops the arrow keys still step a frame from anywhere on the page, in milliseconds. */
const RECENT_MS = 10_000

// Controls that act on their own keys. Inside the player, Space and Enter belong to a focused button, a menu option or a
// field, so the player's shortcuts leave them alone ...
const OWN_KEYS =
  'button, a[href], summary, input, textarea, select, [contenteditable="true"], [role="button"], [role="option"], [role="menuitem"], [role="menu"], [role="listbox"], [role="tab"], [role="radio"], [role="checkbox"], [role="textbox"], [role="spinbutton"], [role="combobox"]'
// ... and the arrow keys that step from anywhere on the page yield to any widget that uses arrows for itself.
const OWN_ARROWS =
  'input, textarea, select, [contenteditable="true"], [role="slider"], [role="option"], [role="menuitem"], [role="listbox"], [role="menu"], [role="tab"], [role="radio"], [role="textbox"], [role="spinbutton"], [role="combobox"]'

// Whether a key event comes from a control matching `selector` (duck-typed, so it also runs where there is no DOM).
function fromControl(target: EventTarget | null, selector: string): boolean {
  const el = target as { readonly isContentEditable?: boolean; readonly closest?: (s: string) => unknown } | null
  if (el === null || typeof el.closest !== 'function') return false
  return el.isContentEditable === true || el.closest(selector) !== null
}

/** The player that played last, so that with several on a page the arrow keys belong to one. */
let lastPlayed: string | undefined

/** Props of {@link Player}. */
export interface PlayerProps {
  readonly controller: PlaybackController
  /**
   * The stamp after the scrubber: `true` for a clock at `stampFps` (default the controller's rate), a function for a
   * label of your own (`(f) => \`step ${f}\``), `false` or absent for none.
   */
  readonly stamp?: boolean | ((frame: number) => string)
  /** The clock rate for a `true` stamp: the timeline's own frames per second (default the playback rate). */
  readonly stampFps?: number
  /** The rates the caret menu offers (default 1 to 60). */
  readonly fpsChoices?: readonly number[]
  readonly className?: string
}

/**
 * Playback controls for a precomputed timeline or a live algorithm, one row: a reset of the page's controls to their
 * defaults (when the controller offers one), an eager toggle (when the controller offers one: the whole run computed
 * at once, its end shown), seek to the start, step back, play or pause with a
 * caret menu for the target frames per second, step forward, seek to the end, a scrubber to drag, and an optional
 * stamp. Space toggles play, the arrow keys step one frame and Home and End seek while the player has focus (on its
 * scrubber, not on a button or menu inside it: those keep their own keys). The arrow keys also step while the player is
 * playing and for ten seconds after, wherever the focus is except in a field, slider, tab list, radio group or menu.
 *
 * @param props - The controller (from `usePlayback`, `useTimelinePlayback`, `useLiveRun` or a session player) and the
 *   stamp.
 * @returns The player.
 * @index function React component: play, step and scrub a playback controller: `<Player controller={c} />`
 */
export function Player(props: PlayerProps): ReactNode {
  const { controller: c, stamp, stampFps, fpsChoices = FPS_CHOICES, className } = props
  const label =
    stamp === true ? timeStamp(c.frame, stampFps ?? c.fps) : typeof stamp === 'function' ? stamp(c.frame) : null
  const toStart = (): void => {
    c.pause()
    c.seek(0)
  }
  const toEnd = (): void => c.seek(c.count - 1)
  const id = useId()
  // Whether the player played within the last few seconds: the arrow keys then step, from anywhere on the page.
  const recent = useRef(false)
  useEffect(() => {
    if (c.playing) {
      recent.current = true
      lastPlayed = id
      return
    }
    const timer = setTimeout(() => {
      recent.current = false
    }, RECENT_MS)
    return () => clearTimeout(timer)
  }, [c.playing, id])
  const step = useRef(c.step)
  step.current = c.step
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      if (!recent.current || lastPlayed !== id) return
      if (fromControl(e.target, OWN_ARROWS)) return
      e.preventDefault()
      step.current(e.key === 'ArrowLeft' ? -1 : 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [id])
  const eager = c.eager === true
  const hasReset = c.resetForm !== undefined
  const hasEager = c.setEager !== undefined
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: keyboard shortcuts for the whole player, which holds the buttons
    <div
      className={cx('flex w-full min-w-64 items-center gap-2 text-(--wb-fg)', className)}
      onKeyDown={(e) => {
        if (fromControl(e.target, OWN_KEYS)) return
        if (e.key === ' ') c.toggle()
        else if (e.key === 'ArrowRight') c.step(1)
        else if (e.key === 'ArrowLeft') c.step(-1)
        else if (e.key === 'Home') toStart()
        else if (e.key === 'End') toEnd()
        else return
        e.preventDefault()
      }}
    >
      <div className="flex shrink-0 items-center gap-2">
        {/* The run: reset the controls, and compute eagerly (the whole run at once) or live. */}
        {hasReset || hasEager ? (
          <div className="flex">
            {c.resetForm === undefined ? null : (
              <button
                type="button"
                className={cx(iconButtonClass, 'rounded-l-md', !hasEager && 'rounded-r-md')}
                aria-label="Reset to defaults"
                title="Reset the controls to their defaults"
                onClick={c.resetForm}
              >
                <RewindIcon size={14} />
              </button>
            )}
            {c.setEager === undefined ? null : (
              <button
                type="button"
                className={cx(
                  iconButtonClass,
                  'rounded-r-md',
                  hasReset ? '-ml-px' : 'rounded-l-md',
                  eager && 'bg-(--wb-surface) text-(--wb-accent)',
                )}
                aria-label="Eager"
                aria-pressed={eager}
                title={
                  eager
                    ? 'Eager: the whole run computed at once and its end shown (click for live)'
                    : 'Live: computed in the background as it plays (click for eager)'
                }
                onClick={() => c.setEager?.(!eager)}
              >
                <ZapIcon size={14} />
              </button>
            )}
          </div>
        ) : null}
        {/* The playback: to the start, play or pause, to the end, and the speed. */}
        <div className="flex">
          <button
            type="button"
            className={cx(iconButtonClass, 'rounded-l-md')}
            aria-label="Go to the start"
            disabled={c.frame <= 0}
            onClick={toStart}
          >
            <SkipBackIcon size={14} />
          </button>
          <button
            type="button"
            className={cx(iconButtonClass, '-ml-px')}
            aria-label={c.playing ? 'Pause' : 'Play'}
            disabled={c.count <= 1 && !c.open}
            onClick={c.toggle}
          >
            {c.playing ? <PauseIcon size={14} /> : <PlayIcon size={14} />}
          </button>
          <button
            type="button"
            className={cx(iconButtonClass, '-ml-px')}
            aria-label="Go to the end"
            disabled={c.frame >= c.count - 1 && !c.open}
            onClick={toEnd}
          >
            <SkipForwardIcon size={14} />
          </button>
          <FpsMenu fps={c.fps} choices={fpsChoices} onPick={c.setFps} />
        </div>
        {/* The frames: one back and one forward. */}
        <div className="flex">
          <button
            type="button"
            className={cx(iconButtonClass, 'rounded-l-md')}
            aria-label="Step back"
            disabled={c.frame <= 0}
            onClick={() => c.step(-1)}
          >
            <StepBackIcon size={14} />
          </button>
          <button
            type="button"
            className={cx(iconButtonClass, '-ml-px rounded-r-md')}
            aria-label="Step forward"
            disabled={c.frame >= c.count - 1 && !c.open}
            onClick={() => c.step(1)}
          >
            <StepForwardIcon size={14} />
          </button>
        </div>
      </div>
      <Scrubber frame={c.frame} count={c.count} onSeek={c.seek} onGrab={c.pause} />
      {label === null ? null : (
        <span className="shrink-0 font-mono text-xs text-(--wb-muted) tabular-nums">{label}</span>
      )}
    </div>
  )
}

function FpsMenu(props: {
  readonly fps: number
  readonly choices: readonly number[]
  readonly onPick: (fps: number) => void
}): ReactNode {
  const { fps, choices, onPick } = props
  const [open, setOpen] = useState(false)
  const trigger = useRef<HTMLButtonElement>(null)
  const close = useCallback(() => setOpen(false), [])
  return (
    <div className="-ml-px flex">
      <button
        ref={trigger}
        type="button"
        className={cx(iconButtonClass, 'w-auto gap-0.5 rounded-r-md px-1.5 font-mono text-[11px] tabular-nums')}
        aria-label={`Frames per second: ${fps}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Frames per second"
        onClick={() => setOpen((o) => !o)}
      >
        {fps}
        <ChevronDownIcon size={12} />
      </button>
      <Popover anchor={trigger} open={open} onClose={close} role="listbox" label="Frames per second">
        {choices.map((f) => (
          <button
            key={f}
            type="button"
            role="option"
            aria-selected={f === fps}
            data-selected={f === fps}
            className={cx(menuItemClass, 'justify-end font-mono tabular-nums')}
            onClick={() => {
              onPick(f)
              setOpen(false)
            }}
          >
            {f} fps
          </button>
        ))}
      </Popover>
    </div>
  )
}

function Scrubber(props: {
  readonly frame: number
  readonly count: number
  readonly onSeek: (frame: number) => void
  readonly onGrab: () => void
}): ReactNode {
  const { frame, count, onSeek, onGrab } = props
  const track = useRef<HTMLDivElement>(null)
  const last = Math.max(count - 1, 0)
  const t = last === 0 ? 0 : frame / last
  const at = (e: PointerEvent): number => {
    const r = track.current?.getBoundingClientRect()
    if (r === undefined || r.width === 0) return frame
    return Math.round(Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1) * last)
  }
  return (
    <div
      ref={track}
      role="slider"
      tabIndex={0}
      aria-label="Frame"
      aria-valuemin={0}
      aria-valuemax={last}
      aria-valuenow={frame}
      className="group relative h-8 min-w-24 flex-1 cursor-pointer touch-none outline-none select-none"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        onGrab()
        onSeek(at(e))
      }}
      onPointerMove={(e) => {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) onSeek(at(e))
      }}
      onKeyDown={(e) => {
        if (e.key === 'Home') onSeek(0)
        else if (e.key === 'End') onSeek(last)
        else return
        e.preventDefault()
        e.stopPropagation()
      }}
    >
      <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-(--wb-grid)">
        <div className="h-full rounded-full bg-(--wb-accent)" style={{ width: `${t * 100}%` }} />
      </div>
      <div
        className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-(--wb-accent) bg-(--wb-bg) group-hover:ring-3 group-hover:ring-(--wb-accent)/30 group-focus-visible:ring-3 group-focus-visible:ring-(--wb-accent)/50"
        style={{ left: `${t * 100}%` }}
      />
    </div>
  )
}
