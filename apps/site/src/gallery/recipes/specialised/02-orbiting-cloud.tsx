import { pointsSteps } from 'mdml-engine/exposition'
import { dict, int } from 'mdml-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
} from 'mdml-engine/trace'
import {
  Choice,
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  useSession,
  useSessionPreferences,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Orbiting cloud',
  question: 'How do I draw a trajectory in three dimensions as it grows, coloured by a channel, with a fading trail?',
  explain:
    'The `points` panel over a `pointsSteps` record whose steps carry a `values` array (a number per point). `colour: "values"` colours the points along a ramp with a colour bar, `"groups"` by class and `"none"` leaves them plain; `fade` makes points older than that many recorded steps fade to a tenth of their opacity, so the front of the trajectory stands out; the buttons under the figure jump to the isometric, front, side and top viewpoints, and dragging or the arrow keys orbit from there. Here the Lorenz system, a few points per step, coloured by speed.',
}

const STEPS = 120
const PER_STEP = 8

interface State {
  readonly k: number
  readonly sigma: number
}

// The Lorenz system by Runge-Kutta: each point is the state after a few small steps, with its speed and the wing it is on.
function lorenz(sigma: number, count: number): { p: number[][]; speed: number[]; wing: number[] } {
  const f = (v: number[]): number[] => [
    sigma * ((v[1] ?? 0) - (v[0] ?? 0)),
    (v[0] ?? 0) * (28 - (v[2] ?? 0)) - (v[1] ?? 0),
    (v[0] ?? 0) * (v[1] ?? 0) - (8 / 3) * (v[2] ?? 0),
  ]
  const add = (a: number[], b: number[], h: number): number[] => a.map((v, i) => v + h * (b[i] ?? 0))
  let v = [1, 1, 1]
  const h = 0.0035
  const out = { p: [] as number[][], speed: [] as number[], wing: [] as number[] }
  for (let i = 0; i < count + 100; i++) {
    for (let j = 0; j < 2; j++) {
      const k1 = f(v)
      const k2 = f(add(v, k1, h / 2))
      const k3 = f(add(v, k2, h / 2))
      const k4 = f(add(v, k3, h))
      v = v.map((a, d) => a + (h / 6) * ((k1[d] ?? 0) + 2 * (k2[d] ?? 0) + 2 * (k3[d] ?? 0) + (k4[d] ?? 0)))
    }
    if (i >= 100) {
      out.p.push(v)
      out.speed.push(Math.hypot(...f(v)))
      out.wing.push((v[0] ?? 0) > 0 ? 1 : 0)
    }
  }
  return out
}

const orbit: Method<State, Config> = defineMethod<State, Config>({
  name: 'orbit',
  config: dict({ sigma: int(6, 14, { default: 10, label: 'σ' }) }),
  make: (c) => {
    const sigma = c['sigma'] as number
    return {
      name: 'orbit',
      config: { sigma },
      init: () => ({ k: 0, sigma }),
      step: (s, ctx) => {
        ctx.log('points', (s.k + 1) * PER_STEP)
        return { k: s.k + 1, sigma }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    points: scalar({ description: 'points drawn so far' }),
    cloud: derived<State>({
      description: 'the trajectory so far, a batch of points per step, each with its speed and wing',
      compute: (s) => {
        const all = lorenz(s.sigma, STEPS * PER_STEP)
        const [lo, hi] = [Math.min(...all.speed), Math.max(...all.speed)]
        const cut = (a: readonly number[], j: number): number[] => a.slice(j * PER_STEP, (j + 1) * PER_STEP)
        return pointsSteps({
          dims: 3,
          features: ['x', 'y', 'z'],
          range: [0, 1, 2].map(
            (d) =>
              [Math.min(...all.p.map((v) => v[d] ?? 0)), Math.max(...all.p.map((v) => v[d] ?? 0))] as [number, number],
          ),
          valueName: 'speed',
          valueRange: [lo, hi],
          steps: Array.from({ length: s.k + 1 }, (_, j) => ({
            step: j,
            points: all.p.slice(j * PER_STEP, (j + 1) * PER_STEP),
            values: cut(all.speed, j),
            groups: cut(all.wing, j),
            caption: `${j * PER_STEP} points`,
          })),
        })
      },
    }),
  },
  defaults: { points: Log() },
})

export default function OrbitingCloudRecipe(): ReactNode {
  const [colour, setColour] = useState<'values' | 'groups' | 'none'>('values')
  const [fade, setFade] = useState(15)
  const [camera, setCamera] = useState<'iso' | 'front' | 'side' | 'top'>('iso')
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(orbit, { config: { sigma: 10 }, mode: prefs.mode, steps: STEPS }, [], { hold: prefs.hold })
  const layout = useMemo<readonly Panel[]>(
    () => [
      { kind: 'points', path: 'cloud', colour, fade, camera, title: 'the Lorenz trajectory (drag to orbit)' },
      { kind: 'curves', paths: ['points'], title: 'points drawn (click to scrub)' },
    ],
    [colour, fade, camera],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Choice label="colour by" value={colour} onChange={setColour} options={['values', 'groups', 'none']} />
        <Choice label="trail fades over (steps)" value={fade} onChange={setFade} options={[0, 5, 15, 40]} />
        <Choice label="starts from" value={camera} onChange={setCamera} options={['iso', 'front', 'side', 'top']} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} />
      <SessionPanels id="gallery/orbiting-cloud" live={live} at={at} onScrub={setAt} layout={layout} />
    </div>
  )
}
