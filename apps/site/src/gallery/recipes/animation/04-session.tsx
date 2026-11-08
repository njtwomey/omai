import { type DictSpace, defaults, dict, int, real } from 'omai-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from 'omai-engine/trace'
import { figure } from 'omai-render/figure'
import {
  Choice,
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  Toggle,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Session',
  question:
    'How do I run a method as a session: computed in the background, played at its own rate, probed at any step?',
  explain:
    "`useSession(method, options, deps)` is the producer: once play is pressed it computes the run ahead in the spare time of each frame, so the tape may stand at step 200 while the page shows step 20. `SessionPlayer` is the consumer: its cursor pops through the recorded steps at the rate in the menu, waits at the head when the computation is slower, and stops at the end. The bolt is eager (the whole run at once, its end shown); it and holding (a knob change waits for play while live) are remembered from page to page; the rewind button resets the knobs to their defaults. `SessionPanels` draws a layout over the tape under one step cursor: click a curve to scrub. The switches under the knobs are the player's own props: `fps` and `onFpsChange` (the host owns the rate), `eagerToggle` and `statistics` (hide the bolt and the line of counts), `active` on the hook (while off nothing computes, workers included, the tape stays and play carries on when it is back on: what a hidden slide wants), and the panels' figure (`figure` on `SessionPanels`, or a `PanelFigureProvider`: size and type size of every panel). Here, a point walking a circle, with a probe of the plane around it.",
  shows: ['decisionSurface'],
}

interface State {
  readonly k: number
  readonly x: number
  readonly y: number
}

const STEPS = 200
const GRID = Array.from({ length: 25 }, (_, i) => -1.5 + 0.125 * i)

// A method of nothing but its configuration: a point walking a wobbly circle, with its angle and radius as scalars,
// its position as a tensor kept in full (so it is drawn element by element) and a probe of the plane around it.
const walk: Method<State, Config> = defineMethod<State, Config>({
  name: 'walk',
  config: dict({
    turns: int(1, 10, { default: 2 }),
    wobble: real(0, 0.5, { default: 0.15, label: 'wobble of the radius' }),
  }),
  make: (c) => {
    const turns = c['turns'] as number
    const wobble = c['wobble'] as number
    return {
      name: 'walk',
      config: { turns, wobble },
      init: () => ({ k: 0, x: 1, y: 0 }),
      step: (s, ctx) => {
        const k = s.k + 1
        const a = (2 * Math.PI * turns * k) / STEPS
        const r = 1 + wobble * Math.sin(7 * a)
        const next = { k, x: r * Math.cos(a), y: r * Math.sin(a) }
        ctx.log('angle', a)
        ctx.log('radius', r)
        ctx.log('position', [next.x, next.y])
        return next
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      // Checkpoints (and so probes at any step) need states that serialise.
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    angle: scalar({ description: 'the angle walked' }),
    radius: scalar({ description: 'the distance from the origin' }),
    position: tensor({ role: 'state', shape: [2], description: 'where the point is' }),
    'plane/distance': derived<State>({
      description: 'the distance from the point, over the plane',
      compute: (s) => ({
        kind: 'decisionSurface',
        features: ['x', 'y'],
        x: GRID,
        y: GRID,
        values: GRID.map((y) => GRID.map((x) => Math.hypot(x - s.x, y - s.y))),
        field: { name: 'distance' },
      }),
    }),
  },
  defaults: { angle: Log(), radius: Log(), position: Log({ detail: 'full' }) },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'trajectory', paths: ['position'], over: { path: 'plane/distance' }, title: 'the point over its probe' },
  { kind: 'curves', paths: ['radius'], title: 'distance from the origin (click to scrub)' },
  { kind: 'series', path: 'position', names: ['x', 'y'], title: 'position' },
]

const SMALL = figure({ width: 440, height: 320 })
const LARGE = figure({ width: 560, height: 360, fontSize: 14 })

export default function SessionRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(walk.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const [active, setActive] = useState(true)
  const [statistics, setStatistics] = useState(true)
  const [eagerToggle, setEagerToggle] = useState(true)
  const [large, setLarge] = useState(false)
  const [fps, setFps] = useState(30)
  const prefs = useSessionPreferences()
  const live = useSession(walk, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
    active,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={walk.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <Toggle label="active (computes)" value={active} onChange={setActive} />
        <Toggle label="statistics" value={statistics} onChange={setStatistics} />
        <Toggle label="eager toggle" value={eagerToggle} onChange={setEagerToggle} />
        <Toggle label="large panels" value={large} onChange={setLarge} />
        <Choice label="rate" value={fps} options={[5, 30, 120]} onChange={setFps} />
      </div>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(walk.config) as Config)}
        fps={fps}
        onFpsChange={setFps}
        eagerToggle={eagerToggle}
        statistics={statistics}
      />
      <SessionPanels
        id="gallery/session"
        live={live}
        at={at}
        onScrub={setAt}
        layout={LAYOUT}
        figure={large ? LARGE : SMALL}
      />
    </div>
  )
}
