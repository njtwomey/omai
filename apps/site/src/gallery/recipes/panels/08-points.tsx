import { pointsSteps } from 'omai-engine/exposition'
import { choice, type DictSpace, defaults, dict } from 'omai-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
} from 'omai-engine/trace'
import {
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Points that accumulate',
  question: 'How do I show a point set as it fills in over the run, in two or three dimensions?',
  explain:
    'A probe that returns a `pointsSteps` record (the points each step adds, optionally the extent of the whole run so the axes hold still) and the panel `{ kind: "points", path }`, which draws every point up to the viewed step, the latest step\'s in the accent colour. In three dimensions the same record is a cloud to orbit by dragging. The chaos game on a triangle fills in the Sierpinski gasket; switch to the Lorenz system and the same panel draws its attractor in three dimensions, one batch of points per step.',
  shows: ['pointsSteps', 'PointsStepsView'],
}

const STEPS = 100
const PER_STEP = 30

interface State {
  readonly k: number
  readonly system: string
}

// The chaos game: jump half-way to a corner chosen by a linear congruential stream. Points are replayed from the
// start at each probe, so the state stays one number.
function gasket(count: number): number[][] {
  const corners = [
    [0, 0],
    [1, 0],
    [0.5, Math.sqrt(3) / 2],
  ]
  let seed = 12345
  let x = 0.3
  let y = 0.3
  const out: number[][] = []
  for (let i = 0; i < count + 10; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    const c = corners[seed % 3] ?? [0, 0]
    x = (x + (c[0] ?? 0)) / 2
    y = (y + (c[1] ?? 0)) / 2
    if (i >= 10) out.push([x, y])
  }
  return out
}

// The Lorenz system by Runge-Kutta, a few small steps per point.
function lorenz(count: number): number[][] {
  const f = (v: number[]): number[] => [
    10 * ((v[1] ?? 0) - (v[0] ?? 0)),
    (v[0] ?? 0) * (28 - (v[2] ?? 0)) - (v[1] ?? 0),
    (v[0] ?? 0) * (v[1] ?? 0) - (8 / 3) * (v[2] ?? 0),
  ]
  const add = (a: number[], b: number[], h: number): number[] => a.map((v, i) => v + h * (b[i] ?? 0))
  let v = [1, 1, 1]
  const h = 0.008
  const out: number[][] = []
  for (let i = 0; i < count + 100; i++) {
    for (let j = 0; j < 4; j++) {
      const k1 = f(v)
      const k2 = f(add(v, k1, h / 2))
      const k3 = f(add(v, k2, h / 2))
      const k4 = f(add(v, k3, h))
      v = v.map((a, d) => a + (h / 6) * ((k1[d] ?? 0) + 2 * (k2[d] ?? 0) + 2 * (k3[d] ?? 0) + (k4[d] ?? 0)))
    }
    if (i >= 100) out.push(v)
  }
  return out
}

const filling: Method<State, Config> = defineMethod<State, Config>({
  name: 'filling',
  config: dict({ system: choice(['sierpinski', 'lorenz'], { default: 'sierpinski', label: 'system' }) }),
  make: (c) => {
    const system = c['system'] as string
    return {
      name: 'filling',
      config: { system },
      init: () => ({ k: 0, system }),
      step: (s, ctx) => {
        ctx.log('points', (s.k + 1) * PER_STEP)
        return { k: s.k + 1, system }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    points: scalar({ description: 'points drawn so far' }),
    cloud: derived<State>({
      description: 'the points drawn so far, a batch per step, with the extent of the whole run',
      compute: (s) => {
        const three = s.system === 'lorenz'
        const all = three ? lorenz(STEPS * PER_STEP) : gasket(STEPS * PER_STEP)
        return pointsSteps({
          dims: three ? 3 : 2,
          features: three ? ['x', 'y', 'z'] : ['x', 'y'],
          range: three
            ? [
                [-22, 22],
                [-30, 30],
                [0, 50],
              ]
            : [
                [0, 1],
                [0, 0.87],
              ],
          steps: Array.from({ length: s.k + 1 }, (_, j) => ({
            step: j,
            points: all.slice(Math.max(0, (j - 1) * PER_STEP), Math.max(0, j * PER_STEP)),
            caption: `${j * PER_STEP} points`,
          })),
        })
      },
    }),
  },
  defaults: { points: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'points', path: 'cloud', title: 'the points so far' },
  { kind: 'curves', paths: ['points'], title: 'points drawn (click to scrub)' },
]

export default function PointsRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(filling.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(filling, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={filling.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(filling.config) as Config)}
      />
      <SessionPanels id="gallery/points" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
