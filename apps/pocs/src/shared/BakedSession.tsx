import type { Config, Method, Session, Subscription } from 'omai-engine/trace'
import {
  type LiveSession,
  SessionPlayer,
  useRun,
  useScrubClick,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { ComputeGate } from './ComputeGate.tsx'
import { StaleVeil } from './StaleVeil.tsx'
import { type Baked, useBake } from './useBake.ts'

/** What the page's view is given at the scrubbed step. */
export interface BakedView<F> {
  // biome-ignore lint/suspicious/noExplicitAny: the session is generic over the method
  readonly live: LiveSession<any, any>
  /** The step shown: the scrubbed one, or the head. */
  readonly t: number
  /** The views prepared at the latest prepared step at or before `t`. */
  readonly frame: F | undefined
  /** Every prepared step's views (for a curve over the run, read with `baked.at` at each step). */
  readonly baked: Baked<F>
  readonly run: ReturnType<typeof useRun>
  /** Scrubbing by clicking a chart over the steps. */
  readonly seek: ReturnType<typeof useScrubClick>
  /** Sets the scrubbed step (`null` follows the head), for the panels of `SessionPanels`, which scrub their own cursor. */
  readonly scrub: (t: number | null) => void
}

/**
 * Runs a method as a session and plays it back: the training is gated behind a progress bar, the views the page draws
 * are probed once at up to `frames` steps when it has ended (so playing back is a lookup), and the player scrubs them.
 *
 * @param props - The method, its configuration and number of steps, what to probe at a step, and the view.
 * @returns The view with the player under it.
 */
export function BakedSession<F>(props: {
  // biome-ignore lint/suspicious/noExplicitAny: the method is generic over its state
  readonly method: Method<any, any>
  readonly config: Config
  readonly steps: number
  readonly frames: number
  /** The paths kept on the tape (the probes are computed from checkpoints and need no subscription). */
  readonly logs: Subscription<string>
  readonly bake: (session: Session<never, never>, t: number) => F
  readonly onReset: () => void
  readonly children: (view: BakedView<F>) => ReactNode
}): ReactNode {
  const { method, config, steps, frames, logs, bake, onReset, children } = props
  const prefs = useSessionPreferences()
  const [at, setAt] = useState<number | null>(null)
  const live = useSession(
    method,
    // A checkpoint at every step: the state at any step is read back, not rebuilt by running again.
    { config, mode: prefs.mode, steps, checkpointEvery: 1, logs },
    [method, JSON.stringify(config), steps],
    { hold: prefs.hold },
  )
  const [cancelled, setCancelled] = useState(false)
  // biome-ignore lint/correctness/useExhaustiveDependencies: a new run starts uncancelled
  useEffect(() => setCancelled(false), [live.session])
  useEffect(() => {
    if (live.computing) setCancelled(false)
  }, [live.computing])
  const ended = live.done || cancelled
  const baked = useBake({ done: ended, head: live.head, session: live.session }, frames, (t) =>
    bake(live.session as unknown as Session<never, never>, t),
  )
  const ready = baked.progress.total > 0 && !baked.baking
  const gated = live.head >= 1 && !ready
  const run = useRun(live)
  const t = Math.max(0, Math.min(at ?? live.head, live.head))
  const stepsAxis = useMemo(() => Array.from({ length: live.head + 1 }, (_, i) => i), [live.head])
  const seek = useScrubClick(stepsAxis, live.head, setAt)
  return (
    <div className="flex flex-col gap-3">
      <StaleVeil stale={live.stale && !gated}>
        <ComputeGate
          live={{ head: live.head + baked.progress.done, computing: gated, done: !gated }}
          total={steps + Math.max(baked.progress.total, frames + 1)}
          label={ended ? 'Preparing the playback' : 'Training'}
          detail={
            ended
              ? `view ${baked.progress.done} of ${Math.max(baked.progress.total, 1)}`
              : `step ${live.head} of ${steps}`
          }
          onCancel={() => {
            if (ended) baked.stop()
            else {
              live.halt()
              setCancelled(true)
            }
          }}
        >
          {children({ live, t, frame: baked.at(t), baked, run, seek, scrub: setAt })}
        </ComputeGate>
      </StaleVeil>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={onReset} />
    </div>
  )
}
