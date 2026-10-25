/**
 * Playback controls in one row for a precomputed timeline or a live algorithm: step back, play or pause with a menu of
 * target frame rates, step forward, a draggable scrubber and an optional time stamp.
 */

import { type PointerEvent, type ReactNode, useCallback, useRef, useState } from 'react'
import { cx, iconButtonClass, menuItemClass } from './controls/classes.ts'
import { ChevronDownIcon, PauseIcon, PlayIcon, StepBackIcon, StepForwardIcon } from './controls/icons.tsx'
import { Popover } from './controls/Popover.tsx'
import { FPS_CHOICES, timeStamp } from './playhead.ts'
import type { PlaybackController } from './usePlayback.ts'

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
 * Playback controls for a precomputed timeline or a live algorithm, one row: step back, play or pause with a caret
 * menu for the target frames per second, step forward, a scrubber to drag, and an optional stamp. Space toggles play
 * and the arrow keys step while the player has focus.
 *
 * @param props - The controller (from `usePlayback`, `useTimelinePlayback` or `useLiveRun`) and the stamp.
 * @returns The player.
 */
export function Player(props: PlayerProps): ReactNode {
  const { controller: c, stamp, stampFps, fpsChoices = FPS_CHOICES, className } = props
  const label =
    stamp === true ? timeStamp(c.frame, stampFps ?? c.fps) : typeof stamp === 'function' ? stamp(c.frame) : null
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: keyboard shortcuts for the whole player, which holds the buttons
    <div
      className={cx('flex w-full min-w-64 items-center gap-2 text-(--wb-fg)', className)}
      onKeyDown={(e) => {
        if ((e.target as HTMLElement).tagName === 'INPUT') return
        if (e.key === ' ') c.toggle()
        else if (e.key === 'ArrowRight') c.step(1)
        else if (e.key === 'ArrowLeft') c.step(-1)
        else return
        e.preventDefault()
      }}
    >
      <div className="flex shrink-0">
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
          className={cx(iconButtonClass, '-ml-px')}
          aria-label={c.playing ? 'Pause' : 'Play'}
          disabled={c.count <= 1 && !c.open}
          onClick={c.toggle}
        >
          {c.playing ? <PauseIcon size={14} /> : <PlayIcon size={14} />}
        </button>
        <FpsMenu fps={c.fps} choices={fpsChoices} onPick={c.setFps} />
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
        className={cx(iconButtonClass, 'w-auto gap-0.5 px-1.5 font-mono text-[11px] tabular-nums')}
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
