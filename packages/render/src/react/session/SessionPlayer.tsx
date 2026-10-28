/**
 * The player over a live session: the consumer of the tape. Play moves the cursor through the recorded steps at the
 * chosen rate while the producer computes ahead in the background, and waits at the head when the computation is
 * slower than the rate; the eager toggle computes the whole run at once and shows its end. Seek to either end, step
 * one, a rate menu and a status line; a change held while live is applied when play is pressed.
 */

import type { Config } from 'mdml-engine/trace'
import { type ReactElement, useEffect, useMemo, useRef, useState } from 'react'
import { Player } from '../Player.tsx'
import { type PlaybackController, useTicker } from '../usePlayback.ts'
import { setSessionPreferences, useSessionPreferences } from './preferences.ts'
import type { LiveSession } from './useSession.ts'

/** Steps per second the player's menu offers. */
const RATES = [1, 2, 5, 10, 30, 60, 120, 300, 1000]

/**
 * The player and status line of a session.
 *
 * @param props - The session, the scrubbed step (`null` follows the head) and its setter.
 * @returns The player.
 */
export function SessionPlayer<S, C extends Config>(props: {
  readonly live: LiveSession<S, C>
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
}): ReactElement {
  const { live, at, onScrub } = props
  const s = live.session
  const eager = useSessionPreferences().mode === 'eager'
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const run = useMemo(() => s.run(), [s, live.version])
  const head = live.head
  const viewed = at === null ? head : Math.min(at, head)
  const counts = useMemo(() => {
    const c = new Map<string, number>()
    for (const e of run.events) c.set(e.type, (c.get(e.type) ?? 0) + 1)
    return [...c].map(([k, v]) => `${v} ${k}`).join(', ')
  }, [run])
  const [fps, setFps] = useState(60)
  // The cursor ticks through the tape at `fps` steps per second: it waits at the head of an unfinished run for the
  // producer, and stops at the end of a finished one.
  const [ticking, setTicking] = useState(false)
  const cursor = useRef({ viewed, head, done: live.done })
  cursor.current = { viewed, head, done: live.done }
  const handlers = useRef({ onScrub })
  handlers.current = { onScrub }
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
  // biome-ignore lint/correctness/useExhaustiveDependencies: once per session
  useEffect(() => {
    if (playOnRemake.current) {
      playOnRemake.current = false
      handlers.current.onScrub(0)
      setTicking(true)
    } else {
      setTicking(false)
      handlers.current.onScrub(null)
    }
  }, [s])
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
  }
  const status = live.held
    ? ' · changes held: press play'
    : s.status !== undefined
      ? ` · ${s.status}`
      : live.computing
        ? viewed < head
          ? ` · computed to ${head}`
          : ' · computing'
        : ''
  return (
    <div className="flex flex-col gap-3">
      <Player controller={controller} stamp={(f) => `step ${f} of ${head}${status}`} fpsChoices={RATES} />
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
        <span>{counts || 'nothing recorded'}</span>
        <span>last slice {live.lastMs.toFixed(1)} ms</span>
      </div>
    </div>
  )
}
