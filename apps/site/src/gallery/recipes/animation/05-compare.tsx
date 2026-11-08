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
import {
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  useSessionPreferences,
  useSessions,
  type Variant,
} from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Comparison',
  question:
    'How do I run one method several ways under one player, with the curves overlaid and the probes side by side?',
  explain:
    '`useSessions(method, options, variants, deps)` runs one variant per entry (the same seed, data, budget and subscription; only the knobs differ) under one producer that keeps them in step: each frame the variant furthest behind takes the next slice, so the heads never drift apart and the shared cursor, clamped to the slowest, hardly waits. `SessionPlayer` drives all of them at once (play, pause, eager, seeks, steps, held changes), its stamp showing the slowest head. `SessionPanels` overlays the variants on curves and sweep panels (one colour per variant, the step marker shared) and draws the per-step panels side by side, one column per variant. Here the walk of the session recipe with three wobbles; `turns` is shared. The sweep panel puts a logged knob on x and a metric on y, each point a step: click one to scrub. The rewind button resets the knobs to their defaults.',
  shows: ['decisionSurface'],
}

interface State {
  readonly k: number
  readonly x: number
  readonly y: number
}

const STEPS = 200
const GRID = Array.from({ length: 25 }, (_, i) => -1.5 + 0.125 * i)

// The walk of the session recipe: a point on a wobbly circle, its angle and radius as scalars, its position as a tensor
// kept in full and a probe of the plane around it.
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

// The knob the page shares between the variants; each variant sets the wobble.
const SHARED: DictSpace = dict({ turns: int(1, 10, { default: 2 }) })

const VARIANTS: readonly Variant<Config>[] = [
  { label: 'steady', config: { wobble: 0 } },
  { label: 'wobble 0.15', config: { wobble: 0.15 } },
  { label: 'wobble 0.4', config: { wobble: 0.4 } },
]

const LAYOUT: readonly Panel[] = [
  { kind: 'curves', paths: ['radius'], title: 'distance from the origin (overlaid; click to scrub)' },
  { kind: 'sweep', path: 'radius', over: 'angle', title: 'the radius against the angle (each point a step)' },
  { kind: 'probe', path: 'plane/distance', title: 'the plane around the point' },
]

export default function CompareRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(SHARED) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSessions(
    walk,
    { config: params, mode: prefs.mode, steps: STEPS },
    VARIANTS,
    [JSON.stringify(params)],
    { hold: prefs.hold },
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={SHARED} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(SHARED) as Config)} />
      <SessionPanels id="gallery/compare" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
