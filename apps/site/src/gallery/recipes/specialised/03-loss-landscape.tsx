import { landscapeSlice, sampleLandscape } from 'mdml-engine/exposition'
import { choice, type DictSpace, defaults, dict, real } from 'mdml-engine/space'
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
  title: 'Loss landscape',
  question: 'How do I show the loss around the optimiser: a line search at the iterate, and the path over a plane?',
  explain:
    'A probe that returns a `landscapeSlice` record and the panel `{ kind: "landscape", path }`. On one direction it is the loss as a curve through the iterate with the iterates marked and straight lines on it (the tangent and the Armijo bound a step must stay under); on two it is contours (or a surface to orbit, from the button) with the path of the iterates drawn over them. The slice is recomputed at the viewed step and centred on the iterate, so scrubbing moves it along the run. Here gradient descent, with or without momentum, on a curved valley: too large a step overshoots the Armijo line and bounces across the valley.',
  shows: ['landscapeSlice', 'LandscapeSliceView'],
}

const STEPS = 60

// The valley: the loss is log(1 + f) for a Rosenbrock-like f, so that the floor and the walls both show in the contours.
const f = (x: number, y: number): number => (1 - x) ** 2 + 5 * (y - x * x) ** 2
const loss = (x: number, y: number): number => Math.log1p(f(x, y))
const grad = (x: number, y: number): [number, number] => {
  const s = 1 / (1 + f(x, y))
  return [s * (-2 * (1 - x) - 20 * x * (y - x * x)), s * 10 * (y - x * x)]
}

interface State {
  readonly k: number
  readonly rate: number
  readonly algorithm: string
}

// The iterates up to step k, replayed from the start (the runs are tiny).
function iterates(rate: number, algorithm: string, k: number): [number, number][] {
  let p: [number, number] = [-1.2, 1.6]
  let v: [number, number] = [0, 0]
  const out: [number, number][] = [p]
  for (let i = 0; i < k; i++) {
    const g = grad(p[0], p[1])
    const beta = algorithm === 'momentum' ? 0.85 : 0
    v = [beta * v[0] - rate * g[0], beta * v[1] - rate * g[1]]
    p = [p[0] + v[0], p[1] + v[1]]
    out.push(p)
  }
  return out
}

const AXIS_X = Array.from({ length: 41 }, (_, i) => -1.6 + 0.08 * i)
const AXIS_Y = Array.from({ length: 41 }, (_, i) => -0.6 + 0.075 * i)

const descent: Method<State, Config> = defineMethod<State, Config>({
  name: 'descent',
  config: dict({
    rate: real(0.05, 3, { default: 0.4, log: true, label: 'learning rate' }),
    algorithm: choice(['gradient descent', 'momentum'], { default: 'gradient descent', label: 'algorithm' }),
  }),
  make: (c) => {
    const rate = c['rate'] as number
    const algorithm = c['algorithm'] === 'momentum' ? 'momentum' : 'gd'
    return {
      name: 'descent',
      config: { rate, algorithm },
      init: () => ({ k: 0, rate, algorithm }),
      step: (s, ctx) => {
        const p = iterates(rate, algorithm, s.k + 1).at(-1) ?? [0, 0]
        ctx.log('loss', loss(p[0], p[1]))
        return { k: s.k + 1, rate, algorithm }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    loss: scalar({ description: 'the loss at the iterate' }),
    plane: derived<State>({
      description: 'the loss over the plane of the problem, with the path so far',
      compute: (s) => {
        const path = iterates(s.rate, s.algorithm, s.k)
        return sampleLandscape(loss, AXIS_X, AXIS_Y, {
          axes: ['x', 'y'],
          lossName: 'log(1 + f)',
          paths: [{ name: s.algorithm === 'momentum' ? 'momentum' : 'gradient descent', coords: path }],
          caption: `step ${s.k}`,
        })
      },
    }),
    line: derived<State>({
      description: 'the loss along the negative gradient through the iterate: a line search',
      compute: (s) => {
        const p = iterates(s.rate, s.algorithm, s.k).at(-1) ?? [0, 0]
        const g = grad(p[0], p[1])
        const norm = Math.hypot(g[0], g[1]) || 1
        const d = [-g[0] / norm, -g[1] / norm]
        const phi = (a: number): number => loss(p[0] + a * (d[0] ?? 0), p[1] + a * (d[1] ?? 0))
        const alphas = Array.from({ length: 81 }, (_, i) => -0.3 + 0.03 * i)
        const l0 = phi(0)
        const slope = -norm
        const hi = alphas[alphas.length - 1] ?? 2
        // The step the optimiser takes along this direction, in units of the direction.
        const taken = Math.min(hi, s.rate * norm)
        return landscapeSlice({
          dims: 1,
          axes: ['distance along the negative gradient'],
          lossName: 'log(1 + f)',
          x: alphas,
          curve: alphas.map(phi),
          paths: [{ name: 'the step taken', coords: [[0], [taken]] }],
          guides: [
            { name: 'tangent at the iterate', from: [-0.3, l0 + slope * -0.3], to: [hi, l0 + slope * hi] },
            { name: 'Armijo bound (c₁ = 0.1)', from: [0, l0], to: [hi, l0 + 0.1 * slope * hi] },
          ],
          caption: `step ${s.k}: the step lands ${taken >= hi ? 'beyond the plot' : `at ${taken.toFixed(2)}`}`,
        })
      },
    }),
  },
  defaults: { loss: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'landscape', path: 'line', title: 'line search at the iterate' },
  { kind: 'landscape', path: 'plane', title: 'the path over the valley' },
  { kind: 'curves', paths: ['loss'], title: 'loss (click to scrub)' },
]

export default function LossLandscapeRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(descent.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(descent, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={descent.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(descent.config) as Config)}
      />
      <SessionPanels id="gallery/loss-landscape" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
