import { defaults } from 'omai-engine/space'
import type { Config } from 'omai-engine/trace'
import {
  Choice,
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useEffect, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { heavyDescent } from '../../toys/heavy.ts'
import { toyWorkerAhead, toyWorkerProducer } from '../../toys/producer.ts'

export const recipe: Recipe = {
  title: 'Worker producer',
  question: 'How do I compute a session off the main thread, so the page stays alive while a heavy method runs?',
  explain:
    '`useSession(method, options, deps, { producer })` takes a producer factory: `workerProducer({ createWorker })` hands a Web Worker the method by name (looked up in a `methodRegistry` the worker entry serves with `serveWorker`) and its JSON options, computes the run ahead of the cursor and streams the tape back; the player and the panels are the same ones. Derived paths are probed in the worker on request (the loss landscape under the trajectory is a grid of losses per step). Without Workers, or with options that cannot cross, the hook computes on the main thread as before. Here a logistic regression on 40,000 planted rows, a thousand steps of full-batch descent: a few seconds of real work. The heartbeat is driven by `requestAnimationFrame` on the page, so a long gap is a frozen main thread: compare the worker with the main thread, and scrub while it computes. "150 ahead" holds the worker to 150 steps ahead of the cursor.',
}

const STEPS = 1000

type Where = 'worker' | '150 ahead' | 'main thread'

const LAYOUT: readonly Panel[] = [
  { kind: 'curves', paths: ['loss'], title: 'loss (click to scrub)' },
  { kind: 'curves', paths: ['accuracy'], title: 'accuracy' },
  {
    kind: 'trajectory',
    paths: ['weights'],
    dims: [0, 1],
    over: { path: 'landscape' },
    title: 'the first two weights over the loss landscape at the viewed step (a probe, computed in the worker)',
  },
]

// The page's own clock: frames per second and the longest gap between frames over the last second, from
// requestAnimationFrame. A frozen main thread shows as a long gap; the elapsed time keeps counting either way.
function Heartbeat(): ReactNode {
  const [beat, setBeat] = useState({ fps: 0, worst: 0, elapsed: 0 })
  useEffect(() => {
    const start = performance.now()
    let last = start
    let windowStart = start
    let frames = 0
    let worst = 0
    let handle = 0
    const tick = (now: number): void => {
      frames += 1
      worst = Math.max(worst, now - last)
      last = now
      if (now - windowStart >= 1000) {
        setBeat({
          fps: Math.round((frames * 1000) / (now - windowStart)),
          worst: Math.round(worst),
          elapsed: (now - start) / 1000,
        })
        windowStart = now
        frames = 0
        worst = 0
      }
      handle = requestAnimationFrame(tick)
    }
    handle = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(handle)
  }, [])
  return (
    <div
      className="flex flex-wrap items-center gap-x-5 gap-y-1 rounded border border-(--wb-grid) px-3 py-1.5 font-mono text-xs tabular-nums"
      data-testid="heartbeat"
      data-fps={beat.fps}
      data-worst={beat.worst}
    >
      <span>page clock {beat.elapsed.toFixed(0)} s</span>
      <span>{beat.fps} frames/s</span>
      <span style={{ color: beat.worst > 100 ? 'var(--wb-danger, #cf222e)' : undefined }}>
        longest gap {beat.worst} ms
      </span>
    </div>
  )
}

function Page(props: { readonly where: Where }): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(heavyDescent.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const producer =
    props.where === 'worker' ? toyWorkerProducer : props.where === '150 ahead' ? toyWorkerAhead : undefined
  const live = useSession(
    heavyDescent,
    { config: params, mode: prefs.mode, steps: STEPS, checkpointEvery: 20 },
    [JSON.stringify(params)],
    { hold: prefs.hold, ...(producer === undefined ? {} : { producer }) },
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={heavyDescent.config} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(heavyDescent.config) as Config)}
      />
      <SessionPanels id="gallery/worker-producer" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}

export default function WorkerProducerRecipe(): ReactNode {
  const [where, setWhere] = useState<Where>('worker')
  return (
    <div className="flex flex-col gap-3">
      <Heartbeat />
      <Choice
        label="compute on"
        value={where}
        options={['worker', '150 ahead', 'main thread'] as const}
        onChange={setWhere}
        className="max-w-md"
      />
      <Page key={where} where={where} />
    </div>
  )
}
