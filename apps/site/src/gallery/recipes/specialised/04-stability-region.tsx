import { type StabilityFunction, sampleStabilityRegion, stabilityModulus } from 'mdml-engine/exposition'
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
  title: 'Stability region',
  question: 'How do I show where an ODE method is stable, and whether the step size in use keeps the problem inside?',
  explain:
    'A probe that returns a `stabilityRegion` record (the grid of |R(z)| of the method, the eigenvalues of the problem times the step size) and the panel `{ kind: "stability", path }`: the region where |R(z)| ≤ 1 is filled, its boundary drawn, and each value of hλ marked by a circle inside it, a cross outside, a diamond when it falls off the plot; the caption states the verdict in words. Here the step size grows over the run, carrying the two eigenvalues of a damped oscillator and one stiff mode out of the region of forward Euler (a disc about −1) much sooner than out of RK4 or the implicit methods, whose regions contain the whole left half-plane or most of it.',
}

const STEPS = 60
const H_MAX = 0.5

const FACTORIAL = [1, 1, 2, 6, 24]
const TAYLOR = (p: number): number[] => Array.from({ length: p + 1 }, (_, k) => 1 / (FACTORIAL[k] ?? 1))
const METHODS: Record<string, { readonly r: StabilityFunction; readonly order: number }> = {
  'forward Euler': { r: { numerator: [1, 1] }, order: 1 },
  Heun: { r: { numerator: TAYLOR(2) }, order: 2 },
  RK4: { r: { numerator: TAYLOR(4) }, order: 4 },
  'backward Euler': { r: { numerator: [1], denominator: [1, -1] }, order: 1 },
  trapezoid: { r: { numerator: [1, 0.5], denominator: [1, -0.5] }, order: 2 },
}

interface State {
  readonly k: number
  readonly method: string
  readonly stiff: number
}

const stepSize = (k: number): number => (H_MAX * k) / STEPS
const lambdas = (stiff: number): { at: [number, number]; label: string }[] => [
  { at: [-1, 4], label: 'λ₁' },
  { at: [-1, -4], label: 'λ₂' },
  { at: [-stiff, 0], label: 'λ₃ (stiff)' },
]

const solver: Method<State, Config> = defineMethod<State, Config>({
  name: 'solver',
  config: dict({
    method: choice(Object.keys(METHODS), { default: 'forward Euler', label: 'method' }),
    stiff: real(2, 20, { default: 8, label: 'stiff rate' }),
  }),
  make: (c) => {
    const method = c['method'] as string
    const stiff = c['stiff'] as number
    return {
      name: 'solver',
      config: { method, stiff },
      init: () => ({ k: 0, method, stiff }),
      step: (s, ctx) => {
        const h = stepSize(s.k + 1)
        const r = METHODS[method]?.r ?? { numerator: [1] }
        ctx.log('step size', h)
        ctx.log(
          'amplification',
          Math.max(...lambdas(stiff).map((l) => stabilityModulus(r, [h * l.at[0], h * l.at[1]]))),
        )
        return { ...s, k: s.k + 1 }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    'step size': scalar({ description: 'the step size h' }),
    amplification: scalar({ description: 'the largest |R(hλ)| over the eigenvalues: above one the solution grows' }),
    region: derived<State>({
      description: 'the region of absolute stability with hλ marked at the step size of the step',
      cost: 'dear',
      compute: (s) => {
        const m = METHODS[s.method] ?? METHODS['forward Euler']
        const h = stepSize(s.k)
        return sampleStabilityRegion(
          s.method,
          m?.r ?? { numerator: [1] },
          { real: [-4, 1.5], imaginary: [-3.5, 3.5] },
          {
            n: 161,
            order: m?.order ?? 1,
            stepSize: h,
            caption: `h = ${h.toFixed(2)}`,
            marks: lambdas(s.stiff).map((l) => ({
              at: [h * l.at[0], h * l.at[1]] as [number, number],
              label: l.label,
            })),
          },
        )
      },
    }),
  },
  defaults: { 'step size': Log(), amplification: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'stability', path: 'region', title: 'region of absolute stability' },
  {
    kind: 'curves',
    paths: ['amplification'],
    truth: { value: 1, label: 'growth above this line' },
    title: 'largest |R(hλ)| (click to scrub)',
  },
]

export default function StabilityRegionRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(solver.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(solver, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={solver.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(solver.config) as Config)} />
      <SessionPanels id="gallery/stability-region" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
