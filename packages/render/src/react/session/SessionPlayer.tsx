/**
 * The player over a live session, or over a comparison of several: the consumer of the tape. Play moves the cursor
 * through the recorded steps at the chosen rate while the producer computes ahead in the background, and waits at the
 * head when the computation is slower than the rate. Where the cursor starts follows the start policy of the handle
 * (`useSession`'s `start`, or this player's own): live parks it at the start with nothing computed, primed
 * (`'first'`) computes the whole run in the background and keeps it parked at step 0 until play, eager (`'end'`, the
 * toggle) computes the whole run in the background and jumps to its end when it is done. The playhead is a clock: the
 * position advances by the display's frame times in a ref, whatever the drawing costs, and the page's state is set
 * once per displayed frame, so slow drawing drops frames instead of slowing playback. Seek to either end, step one, a rate menu and a status line; a change held while live is applied when play is
 * pressed; a reset of the page's controls when the page offers one. Over a comparison every verb acts on all the
 * variants, the cursor is shared and the stamp shows the head of the slowest. Given `pageExport`, an export menu sits in
 * the status row (the page as PNG or SVG, a recording as GIF or WebM, the tape as JSON in and out); a session computed
 * in a worker is told where the cursor stands, and its failures are shown.
 */

import type { Config } from 'omai-engine/trace'
import { type MutableRefObject, type ReactElement, useEffect, useMemo, useRef, useState } from 'react'
import { Player } from '../Player.tsx'
import { STATE_ATTRIBUTE, settledStateOf } from '../settled.ts'
import type { StartPolicy } from '../start.ts'
import { type PlaybackController, useTicker } from '../usePlayback.ts'
import { prerollDecision } from './_preroll.ts'
import { useSessionPreferences, useSetSessionPreferences } from './preferences.ts'
import { SessionExport, type SessionExportProps } from './SessionExport.tsx'
import { type LiveSession, type LiveSessions, variantsOf } from './useSession.ts'

/** Steps per second the player's menu offers. */
const RATES = [1, 2, 5, 10, 30, 60, 120, 300, 1000]

/** The longest the clock waits for the preroll to fill, in milliseconds. */
const PREROLL_MAX_MS = 250

/**
 * The player and status line of a session, or of a comparison of several. Its root element carries
 * `data-omai-state`: `preparing` until the whole tape exists (also while it plays), `ready` once it does, `failed` when
 * the run failed or was disposed (see `OmaiState`; `waitForSettled` waits for it).
 *
 * @param props - The session or comparison, the scrubbed step (`null` follows the head) and its setter, and the reset
 *   of the page's controls when it has any.
 * @returns The player.
 * @index function React component: play a session with controls: `<SessionPlayer live={live} at={at} onScrub={setAt} />`
 */
export function SessionPlayer<S, C extends Config>(props: {
  readonly live: LiveSession<S, C> | LiveSessions<S, C>
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
  /**
   * Reset the page's controls (the method's knobs, the session settings) to their defaults: a button first in the
   * group when given; pressing it also pauses and scrubs to the start.
   */
  readonly onReset?: () => void
  /** The page's export menu: the element holding the figures, the title and caption, the layout, loading a tape. */
  readonly pageExport?: Omit<SessionExportProps, 'live' | 'at' | 'onScrub'>
  /**
   * Where the cursor starts, in place of the handle's own policy (`'play'`, `'first'` or `'end'`); the player then asks
   * the producer to compute, and the eager toggle is not offered. Omitted, the handle's `start` rules and the toggle
   * switches between `'end'` and `'play'` through the stored preference.
   */
  readonly start?: StartPolicy
  /**
   * A ref the player keeps at the playhead's step on every animation frame, before the page re-renders: a figure that
   * draws on its own frame (a canvas) reads it to show the frame at the playhead.
   */
  readonly playhead?: MutableRefObject<number>
  /**
   * The rate in steps per second. Given, it rules (the rate menu asks `onFpsChange` and shows this); omitted, the player
   * starts at the stored preference (`fps` of the session preferences, 60 unless a `SessionPreferencesProvider` says
   * another) and the reader's pick from the menu is kept in the player.
   */
  readonly fps?: number
  /** Called with the rate the reader picked from the rate menu. */
  readonly onFpsChange?: (fps: number) => void
  /**
   * Offer the eager toggle (default true). `false` hides it, for a page that has its own control for the mode; it is
   * never offered with `start` given, because the start policy then rules and a toggle would do nothing.
   */
  readonly eagerToggle?: boolean
  /** Show the statistics line (the tape's event counts and the milliseconds of the last slice; default true). */
  readonly statistics?: boolean
}): ReactElement {
  const { live, at, onScrub } = props
  const statistics = props.statistics ?? true
  const variants = variantsOf(live)
  // Every variant is remade at once, so the first one's session stands for the remake.
  const first = variants[0]?.live.session
  const policy = props.start ?? live.start
  const eager = policy === 'end'
  const parked = policy !== 'play'
  const head = live.head
  const viewed = at === null ? head : Math.min(at, head)
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const counts = useMemo(() => {
    if (!statistics) return ''
    const c = new Map<string, number>()
    for (const e of first?.run().events ?? []) c.set(e.type, (c.get(e.type) ?? 0) + 1)
    const text = [...c].map(([k, v]) => `${v} ${k}`).join(', ')
    return variants.length > 1 && text !== '' ? `${text} × ${variants.length} variants` : text
  }, [first, live.version, variants.length, statistics])
  // A producer with a lookahead computes up to the cursor and no more: tell it where the cursor stands (and again after
  // a remake, which made a new producer).
  const follow = live.follow
  const preferences = useSessionPreferences()
  const setPreferences = useSetSessionPreferences()
  const [picked, setPicked] = useState<number | undefined>(undefined)
  const fps = props.fps ?? picked ?? preferences.fps
  const setFps = (rate: number): void => {
    setPicked(rate)
    props.onFpsChange?.(rate)
  }
  // Told the step and the rate: a worker with a lookahead computes up to the step, and the main-thread producer paces itself
  // by both (half a second of playback ahead, by default).
  // biome-ignore lint/correctness/useExhaustiveDependencies: once per step, rate and remake; `follow` reads the latest producer
  useEffect(() => follow?.(viewed, fps), [viewed, fps, first])
  // The cursor ticks through the tape at `fps` steps per second: it waits at the head of an unfinished run for the
  // producer, and stops at the end of a finished one.
  const [tickingState, setTickingState] = useState(false)
  // The clock's switch, set at the moment of the call (the render that follows it is later): a tick queued before a
  // `pause`, a `seek` or a reset must find the clock off.
  const tickingNow = useRef(false)
  const ticking = tickingState
  const setTicking = (on: boolean): void => {
    tickingNow.current = on
    setTickingState(on)
  }
  const cursor = useRef({ viewed, head, done: live.done })
  cursor.current = { viewed, head, done: live.done }
  const handlers = useRef({ onScrub, onReset: props.onReset, pause: live.pause, playingNow: live.playingNow })
  handlers.current = { onScrub, onReset: props.onReset, pause: live.pause, playingNow: live.playingNow }
  // The playhead: the step the clock has reached. Ticks add to it rather than to the step the page last rendered, so
  // a slow render does not slow the clock; the page is told the step once per tick, and only when it changed. The steps
  // told but not yet rendered are remembered, to tell the page catching up from the page moving the cursor itself.
  const jumpToEnd = useRef(false)
  const pos = useRef(viewed)
  const told = useRef<number[]>([])
  const playheadRef = props.playhead
  // biome-ignore lint/correctness/useExhaustiveDependencies: the position follows the step the page renders
  useEffect(() => {
    const seen = told.current.indexOf(viewed)
    if (seen >= 0) told.current.splice(0, seen + 1)
    else {
      told.current = []
      pos.current = viewed
    }
    if (playheadRef !== undefined && !ticking) playheadRef.current = viewed
  }, [viewed])
  const tell = (step: number, ticked = false): void => {
    pos.current = step
    if (ticked) told.current.push(step)
    if (playheadRef !== undefined) playheadRef.current = step
    handlers.current.onScrub(step)
  }
  const stop = (): void => {
    setTicking(false)
    setBuffering(false)
    jumpToEnd.current = false
    told.current = []
    handlers.current.pause()
  }
  // A live run is played with a preroll: the clock waits until a quarter of a second of steps stands ahead of it, and again
  // whenever it catches up with the head, so a method slower than the rate plays in even bursts and not one jerk per step.
  const prerollSteps = Math.max(2, Math.ceil(fps * 0.25))
  const bufferingNow = useRef(false)
  // When the wait began: a method too slow to fill the preroll in a quarter of a second plays on what it has (one step is
  // enough) rather than keeping the reader waiting for the steps to pile up.
  const bufferingSince = useRef(0)
  const [buffering, setBufferingState] = useState(false)
  const setBuffering = (on: boolean): void => {
    if (on && !bufferingNow.current) bufferingSince.current = performance.now() // allow: how long the clock has waited; not on the tape
    bufferingNow.current = on
    setBufferingState(on)
  }
  useTicker(
    ticking && live.active,
    fps,
    (n) => {
      // A tick queued before the clock was stopped (here, or by the host's `live.pause()`) moves nothing.
      if (!tickingNow.current || handlers.current.playingNow?.() === false) return
      const { head: h, done } = cursor.current
      // Live and unfinished: wait for the preroll (at the start, and after catching up with the head).
      if (policy === 'play' && !done) {
        const decision = prerollDecision({
          buffering: bufferingNow.current,
          ahead: h - pos.current,
          preroll: prerollSteps,
          waitedMs: bufferingNow.current ? performance.now() - bufferingSince.current : 0, // allow: how long the clock has waited; not on the tape
          maxMs: PREROLL_MAX_MS,
        })
        if (decision === 'wait') return
        if (decision === 'starved') {
          if (pos.current !== h) tell(h, true)
          setBuffering(true)
          return
        }
        if (bufferingNow.current) setBuffering(false)
      } else if (bufferingNow.current) setBuffering(false)
      const next = pos.current + n
      if (next < h) tell(next, true)
      else if (done) {
        told.current = []
        handlers.current.onScrub(null)
        pos.current = h
        if (playheadRef !== undefined) playheadRef.current = h
        stop()
      } else if (pos.current !== h) tell(h, true)
    },
    // A tick may be a long frame (slow drawing) but not a long absence: up to half a second of steps are due at once,
    // so the clock keeps its time through slow frames, and a tab back from the background does not race ahead.
    Math.max(4, Math.ceil(fps / 2)),
  )
  // A remade session plays from its first step when play asked for it; otherwise it stands at its start (live and
  // primed: nothing shown but step 0, and the run computing in the background when primed) or, eager, waits at its start
  // until it is done and then shows its end.
  const playOnRemake = useRef(false)
  const wasDone = useRef(live.done)
  // biome-ignore lint/correctness/useExhaustiveDependencies: once per remake
  useEffect(() => {
    wasDone.current = cursor.current.done
    if (playOnRemake.current) {
      playOnRemake.current = false
      jumpToEnd.current = false
      tell(0)
      setTicking(true)
      live.play()
      return
    }
    setTicking(false)
    if (props.start !== undefined && !cursor.current.done) live.compute()
    if (!parked || cursor.current.done) {
      jumpToEnd.current = false
      told.current = []
      handlers.current.onScrub(null)
    } else {
      jumpToEnd.current = policy === 'end'
      tell(0)
    }
  }, [first])
  // Eager: the cursor jumps to the end once the whole run exists, if nothing has moved it since.
  useEffect(() => {
    const was = wasDone.current
    wasDone.current = live.done
    if (!was && live.done && jumpToEnd.current) {
      jumpToEnd.current = false
      told.current = []
      handlers.current.onScrub(null)
    }
  }, [live.done])
  // The handle's `play()` and `pause()`, called by the host, start and stop the clock.
  const hostPlaying = useRef(live.playing)
  // biome-ignore lint/correctness/useExhaustiveDependencies: on the edge only
  useEffect(() => {
    if (hostPlaying.current === live.playing) return
    hostPlaying.current = live.playing
    if (live.playing && !ticking) begin()
    else if (!live.playing && ticking) setTicking(false)
  }, [live.playing])
  const following = at === null && live.computing && !live.done
  const playing = ticking || following
  // The verb behind play, for the player's button and for the host's `live.play()`.
  function begin(): void {
    if (live.stale) {
      playOnRemake.current = true
      live.apply()
      live.compute()
      return
    }
    jumpToEnd.current = false
    if (live.done && viewed >= head) tell(0)
    // A cursor that follows the head stands wherever the head gets to, and a cheap method computes ahead within a frame:
    // play from the step being looked at (the start of a fresh run), not from wherever the first tick finds the head.
    else if (at === null) tell(viewed)
    live.compute()
    setBuffering(policy === 'play' && !live.done && head - pos.current < prerollSteps)
    setTicking(true)
  }
  const play = (): void => {
    live.play()
    begin()
  }
  const pause = (): void => {
    stop()
    if (following) onScrub(head)
  }
  const controller: PlaybackController = {
    frame: viewed,
    count: head + 1,
    playing,
    fps,
    open: !live.done,
    eager,
    ...(props.start === undefined && props.eagerToggle !== false
      ? {
          setEager: (on: boolean): void => {
            setPreferences({ mode: on ? 'eager' : 'live' })
            if (on) {
              stop()
              if (live.done) onScrub(null)
              else jumpToEnd.current = true
            }
          },
        }
      : {}),
    play,
    pause,
    toggle: () => (playing ? pause() : play()),
    step: (dir) => {
      stop()
      // From the latest position (`pos`: a seek or a tick not rendered yet counts), not from the one last rendered.
      const here = Math.min(pos.current, head)
      if (dir === -1) tell(Math.max(0, here - 1))
      else if (live.stale) live.apply()
      else if (here < head) tell(here + 1)
      else if (!live.done) {
        live.advance({ steps: 1 })
        tell(here + 1)
      }
    },
    seek: (f) => {
      jumpToEnd.current = false
      told.current = []
      if (f < head) tell(Math.max(0, f))
      else {
        if (live.stale) live.apply()
        pos.current = head
        onScrub(null)
        live.compute()
      }
    },
    setFps,
    reset: () => {
      stop()
      tell(0)
    },
    ...(props.onReset === undefined
      ? {}
      : {
          resetForm: () => {
            stop()
            tell(0)
            handlers.current.onReset?.()
          },
        }),
  }
  // The status once every variant has ended: the one status they share, or each variant's by its label.
  const ended = (): string => {
    const statuses = variants.map((v) => v.live.session.status)
    const distinct = [...new Set(statuses)]
    return distinct.length === 1 ? `${distinct[0]}` : variants.map((v, i) => `${v.label} ${statuses[i]}`).join(', ')
  }
  const where = live.producer?.kind === 'worker' ? ' · in a worker' : ''
  const status =
    live.error !== undefined
      ? ' · failed'
      : live.held
        ? ' · changes held: press play'
        : live.done
          ? ` · ${ended()}`
          : parked && live.computing && !live.playing
            ? ` · preparing ${Math.round(live.progress * 100)}%`
            : live.computing
              ? viewed < head
                ? ` · computed to ${head}`
                : ' · computing'
              : ''
  const showsRow = statistics || live.producer?.fallback !== undefined || props.pageExport !== undefined
  return (
    <div className="flex flex-col gap-3" {...{ [STATE_ATTRIBUTE]: settledStateOf(live) }}>
      <Player controller={controller} stamp={(f) => `step ${f} of ${head}${status}${where}`} fpsChoices={RATES} />
      {showsRow ? (
        <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
          {statistics ? <span>{counts || 'nothing recorded'}</span> : null}
          {statistics ? <span>last slice {live.lastMs.toFixed(1)} ms</span> : null}
          {statistics && ticking && !live.done ? (
            <span>
              {buffering ? 'buffering, ' : ''}
              {Math.max(0, head - viewed)} steps ahead
            </span>
          ) : null}
          {live.producer?.fallback === undefined ? null : <span>main thread: {live.producer.fallback}</span>}
          {props.pageExport === undefined ? null : (
            <SessionExport live={live} at={at} onScrub={onScrub} {...props.pageExport} />
          )}
        </div>
      ) : null}
      {live.error === undefined ? null : (
        <div role="alert" className="rounded border border-red-500/50 px-2 py-1 text-xs text-red-500">
          the session failed: {live.error}
        </div>
      )}
    </div>
  )
}
