// The player over a live session, shared by every page that runs one: play always moves the cursor forward through the
// tape and then into live computation, with step, scrub, rate and reset, and a status line under it.
import type { Config } from 'mdml-engine/trace'
import { type PlaybackController, Player } from 'mdml-render/react'
import { type ReactElement, useEffect, useMemo, useRef, useState } from 'react'
import type { LiveSession } from './useSession.ts'

/** Steps per second the player's menu offers. */
const RATES = [1, 2, 5, 10, 30, 60, 120, 300, 1000]

/** The player and status line of a session; `at` is the scrubbed step (`null` follows the head). */
export function SessionPlayer<S, C extends Config>(props: {
  readonly live: LiveSession<S, C>
  readonly at: number | null
  readonly onScrub: (t: number | null) => void
}): ReactElement {
  const { live, at, onScrub } = props
  const s = live.session
  // biome-ignore lint/correctness/useExhaustiveDependencies: the tape changes with the version
  const run = useMemo(() => s.run(), [s, live.version])
  const head = s.t
  const viewed = at === null ? head : Math.min(at, head)
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
    </div>
  )
}
