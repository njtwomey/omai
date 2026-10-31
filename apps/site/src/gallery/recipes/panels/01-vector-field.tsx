import { sampleVectorField } from 'mdml-engine/exposition'
import { type DictSpace, defaults, dict, real } from 'mdml-engine/space'
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
} from 'mdml-engine/trace'
import {
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  useSession,
  useSessionPreferences,
} from 'mdml-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Vector field',
  question: 'How do I show a flow whose field changes as the run goes on, with the path of the state over it?',
  explain:
    'A probe that returns a `vectorField` record (the grid, the components, optionally nullclines and fixed points with a stability class) and the panel `{ kind: "field", path, trajectory }`, which draws the probe at the viewed step as arrows (or streamlines) and the tape\'s position path over it, ending in its current point. Here predators and prey: the prey\'s growth rate drifts over the run, so the horizontal nullcline and the centre sit higher and higher while the state circles it; scrub the cursor and the field at that step comes back.',
  shows: ['vectorField', 'VectorFieldView'],
}

interface State {
  readonly k: number
  readonly x: number
  readonly y: number
  /** The prey growth rate at this step. */
  readonly a: number
}

const STEPS = 240
const DT = 0.04
const AXIS = Array.from({ length: 21 }, (_, i) => 0.2 * i)

// The prey's growth rate at a step: it drifts from 1 by `drift` over the run.
const growth = (k: number, drift: number): number => 1 + (drift * k) / STEPS

// Lotka-Volterra with the interaction rates at 1: prey x' = x (a - y), predators y' = y (x - 1).
const flow =
  (a: number) =>
  (x: number, y: number): [number, number] => [x * (a - y), y * (x - 1)]

// A method of nothing but its configuration: the state integrated by Runge-Kutta, its position on the tape in full,
// and a probe of the field around it at the step asked for.
const predators: Method<State, Config> = defineMethod<State, Config>({
  name: 'predators',
  config: dict({ drift: real(-0.5, 1.2, { default: 0.8, label: 'drift of the prey growth rate' }) }),
  make: (c) => {
    const drift = c['drift'] as number
    return {
      name: 'predators',
      config: { drift },
      init: () => ({ k: 0, x: 1.6, y: 1, a: growth(0, drift) }),
      step: (s, ctx) => {
        const f = flow(s.a)
        const [a1, b1] = f(s.x, s.y)
        const [a2, b2] = f(s.x + (DT / 2) * a1, s.y + (DT / 2) * b1)
        const [a3, b3] = f(s.x + (DT / 2) * a2, s.y + (DT / 2) * b2)
        const [a4, b4] = f(s.x + DT * a3, s.y + DT * b3)
        const next = {
          k: s.k + 1,
          a: growth(s.k + 1, drift),
          x: s.x + (DT / 6) * (a1 + 2 * a2 + 2 * a3 + a4),
          y: s.y + (DT / 6) * (b1 + 2 * b2 + 2 * b3 + b4),
        }
        ctx.log('growth', next.a)
        ctx.log('position', [next.x, next.y])
        return next
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    growth: scalar({ description: 'the prey growth rate' }),
    position: tensor({ role: 'state', shape: [2], description: 'prey and predators' }),
    flow: derived<State>({
      description: 'the field of the system at the step, with its nullclines and fixed points',
      compute: (s) => {
        const a = s.a
        return sampleVectorField(flow(a), AXIS, AXIS, {
          features: ['prey', 'predators'],
          field: { name: 'rate of change' },
          nullclines: [
            {
              name: 'prey steady',
              paths: [
                [
                  [0, a],
                  [4, a],
                ],
              ],
            },
            {
              name: 'predators steady',
              paths: [
                [
                  [1, 0],
                  [1, 4],
                ],
              ],
            },
          ],
          fixedPoints: [
            { at: [0, 0], stability: 'saddle' },
            { at: [1, a], stability: 'centre' },
          ],
        })
      },
    }),
  },
  defaults: { growth: Log(), position: Log({ detail: 'full' }) },
})

const LAYOUT: readonly Panel[] = [
  {
    kind: 'field',
    path: 'flow',
    trajectory: { paths: ['position'], trail: 120 },
    title: 'the field at the viewed step, the state over it',
  },
  { kind: 'series', path: 'position', names: ['prey', 'predators'], title: 'the populations' },
  { kind: 'curves', paths: ['growth'], title: 'prey growth rate (click to scrub)' },
]

export default function VectorFieldRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(predators.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(predators, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={predators.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(predators.config) as Config)}
      />
      <SessionPanels id="gallery/vector-field" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
