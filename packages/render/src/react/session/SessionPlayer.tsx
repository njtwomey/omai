/**
 * The player over a live session, or over a comparison of several: the consumer of the tape. Play moves the cursor
 * through the recorded steps at the chosen rate while the producer computes ahead in the background, and waits at the
 * head when the computation is slower than the rate; the eager toggle computes the whole run at once and shows its
 * end. Seek to either end, step one, a rate menu and a status line; a change held while live is applied when play is
 * pressed; a reset of the page's controls when the page offers one. Over a comparison every verb acts on all the
 * variants, the cursor is shared and the stamp shows the head of the slowest. Given `pageExport`, an export menu sits in
 * the status row (the page as PNG or SVG, a recording as GIF or WebM, the tape as JSON in and out); a session computed
 * in a worker is told where the cursor stands, and its failures are shown.
 */

import type { Config } from 'mdml-engine/trace'
import { type ReactElement, useEffect, useMemo, useRef, useState } from 'react'
import { Player } from '../Player.tsx'
import { type PlaybackController, useTicker } from '../usePlayback.ts'
import { setSessionPreferences, useSessionPreferences } from './preferences.ts'
import { SessionExport, type SessionExportProps } from './SessionExport.tsx'
import { type LiveSession, type LiveSessions, variantsOf } from './useSession.ts'

/** Steps per second the player's menu offers. */
const RATES = [1, 2, 5, 10, 30, 60, 120, 300, 1000]

/**
 * The player and status line of a session, or of a comparison of several.
 *
 * @param props - The session or comparison, the scrubbed step (`null` follows the head) and its setter, and the reset
 *   of the page's controls when it has any.
 * @returns The player.
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
}): ReactElement {
  const { live, at, onScrub } = props
  const variants = variantsOf(live)
  // Every variant is remade at once, so the first one's session stands for the remake.
  const first = variants[0]?.live.session
  const eager = useSessionPreferences().mode === 'eager'
  const head = live.head
  const viewed = at === null ? head : Math.min(at, head)
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const counts = useMemo(() => {
    const c = new Map<string, number>()
    for (const e of first?.run().events ?? []) c.set(e.type, (c.get(e.type) ?? 0) + 1)
    const text = [...c].map(([k, v]) => `${v} ${k}`).join(', ')
    return variants.length > 1 && text !== '' ? `${text} × ${variants.length} variants` : text
  }, [first, live.version, variants.length])
  // A producer with a lookahead computes up to the cursor and no more: tell it where the cursor stands (and again after
  // a remake, which made a new producer).
  const follow = live.follow
  // biome-ignore lint/correctness/useExhaustiveDependencies: once per step and per remake; `follow` reads the latest producer
  useEffect(() => follow?.(viewed), [viewed, first])
  const [fps, setFps] = useState(60)
  // The cursor ticks through the tape at `fps` steps per second: it waits at the head of an unfinished run for the
  // producer, and stops at the end of a finished one.
  const [ticking, setTicking] = useState(false)
  const cursor = useRef({ viewed, head, done: live.done })
  cursor.current = { viewed, head, done: live.done }
  const handlers = useRef({ onScrub, onReset: props.onReset })
  handlers.current = { onScrub, onReset: props.onReset }
  useTicker(
    ticking,
    fps,
    (n) => {
      const { viewed: v, head: h, done } = cursor.current
      const next = v + n
      if (next < h) handlers.current.onScrub(next)
      else if (done) {
        handlers.current.onScrub(null)
        setTicking(false)
      } else handlers.current.onScrub(h)
    },
    Math.max(4, Math.ceil(fps / 15)),
  )
  // A remade session plays from its first step when play asked for it, and otherwise stands at its end (eager) or at
  // its start (live, nothing computed yet).
  const playOnRemake = useRef(false)
  // biome-ignore lint/correctness/useExhaustiveDependencies: once per remake
  useEffect(() => {
    if (playOnRemake.current) {
      playOnRemake.current = false
      handlers.current.onScrub(0)
      setTicking(true)
    } else {
      setTicking(false)
      handlers.current.onScrub(null)
    }
  }, [first])
  const following = at === null && live.computing && !live.done
  const playing = ticking || following
  const play = (): void => {
    if (live.stale) {
      playOnRemake.current = true
      live.apply()
      live.compute()
      return
    }
    if (live.done && viewed >= head) onScrub(0)
    // A cursor that follows the head stands wherever the head gets to, and a cheap method computes ahead within a frame:
    // play from the step being looked at (the start of a fresh run), not from wherever the first tick finds the head.
    else if (at === null) onScrub(viewed)
    live.compute()
    setTicking(true)
  }
  const pause = (): void => {
    setTicking(false)
    if (following) onScrub(head)
  }
  const controller: PlaybackController = {
    frame: viewed,
    count: head + 1,
    playing,
    fps,
    open: !live.done,
    eager,
    setEager: (on) => {
      setSessionPreferences({ mode: on ? 'eager' : 'live' })
      if (on) {
        setTicking(false)
        onScrub(null)
      }
    },
    play,
    pause,
    toggle: () => (playing ? pause() : play()),
    step: (dir) => {
      setTicking(false)
      if (dir === -1) onScrub(Math.max(0, viewed - 1))
      else if (live.stale) live.apply()
      else if (viewed < head) onScrub(viewed + 1)
      else if (!live.done) {
        live.advance({ steps: 1 })
        onScrub(viewed + 1)
      }
    },
    seek: (f) => {
      if (f < head) onScrub(Math.max(0, f))
      else {
        if (live.stale) live.apply()
        onScrub(null)
        live.compute()
      }
    },
    setFps,
    reset: () => {
      setTicking(false)
      onScrub(0)
    },
    ...(props.onReset === undefined
      ? {}
      : {
          resetForm: () => {
            setTicking(false)
            onScrub(0)
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
          : live.computing
            ? viewed < head
              ? ` · computed to ${head}`
              : ' · computing'
            : ''
  return (
    <div className="flex flex-col gap-3">
      <Player controller={controller} stamp={(f) => `step ${f} of ${head}${status}${where}`} fpsChoices={RATES} />
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
        <span>{counts || 'nothing recorded'}</span>
        <span>last slice {live.lastMs.toFixed(1)} ms</span>
        {live.producer?.fallback === undefined ? null : <span>main thread: {live.producer.fallback}</span>}
        {props.pageExport === undefined ? null : (
          <SessionExport live={live} at={at} onScrub={onScrub} {...props.pageExport} />
        )}
      </div>
      {live.error === undefined ? null : (
        <div role="alert" className="rounded border border-red-500/50 px-2 py-1 text-xs text-red-500">
          the session failed: {live.error}
        </div>
      )}
    </div>
  )
}
