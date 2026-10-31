import { equationFit } from 'mdml-engine/exposition'
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
  title: 'Equation against the truth',
  question: 'How do I show a discovered equation beside the true one, with its error term by term?',
  explain:
    'A probe that returns an `equationFit` record (terms with their estimated and, when known, true coefficients, the fitted and true functions on a grid, the observations) and the panel `{ kind: "equation", path }`. The fitted expression is typeset with KaTeX, each coefficient coloured by its error and listed in a table with its status in words (matches, differs, kept though not in the law, dropped though in it), the fit is drawn against the truth and the residual sits in a strip beneath. Here the lasso by coordinate descent on a library of six powers recovers y = 0.3 + 1.5 x − 0.6 x³ from noisy samples: the powers are strongly correlated, so it converges over many sweeps, the spurious terms shrinking toward zero as the true ones grow toward their values.',
  shows: ['equationFit', 'EquationFitView', 'strip'],
}

const STEPS = 60
const N = 60
const TERMS = [
  { tex: '1', f: (_: number) => 1, truth: 0.3 },
  { tex: 'x', f: (x: number) => x, truth: 1.5 },
  { tex: 'x^2', f: (x: number) => x ** 2, truth: 0 },
  { tex: 'x^3', f: (x: number) => x ** 3, truth: -0.6 },
  { tex: 'x^4', f: (x: number) => x ** 4, truth: 0 },
  { tex: 'x^5', f: (x: number) => x ** 5, truth: 0 },
] as const

interface State {
  readonly k: number
  readonly w: readonly number[]
  readonly noise: number
  readonly penalty: number
}

const xsData = Array.from({ length: N }, (_, i) => -2 + (4 * i) / (N - 1))
// A fixed stream of noise (a linear congruential generator, scaled to roughly unit spread).
const unitNoise = (): number[] => {
  let seed = 7
  return xsData.map(() => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
    return ((seed / 2 ** 32) * 2 - 1) * Math.sqrt(3)
  })
}
const NOISE = unitNoise()
const trueY = (x: number): number => TERMS.reduce((s, t) => s + t.truth * t.f(x), 0)
const design = xsData.map((x) => TERMS.map((t) => t.f(x)))
const colNorm = TERMS.map((_, j) => design.reduce((s, row) => s + (row[j] ?? 0) ** 2, 0))
const GRID = Array.from({ length: 121 }, (_, i) => -2 + (4 * i) / 120)

const soft = (v: number, t: number): number => Math.sign(v) * Math.max(0, Math.abs(v) - t)

// One sweep of coordinate descent on the lasso: 0.5 * |y - Xw|^2 + penalty * |w|_1.
function sweep(w: readonly number[], y: readonly number[], penalty: number): number[] {
  const next = [...w]
  for (let j = 0; j < next.length; j++) {
    let rho = 0
    for (let i = 0; i < design.length; i++) {
      const pred = next.reduce((s, wk, k) => s + wk * (design[i]?.[k] ?? 0), 0)
      rho += (design[i]?.[j] ?? 0) * ((y[i] ?? 0) - pred + (next[j] ?? 0) * (design[i]?.[j] ?? 0))
    }
    next[j] = soft(rho, penalty) / (colNorm[j] || 1)
  }
  return next
}

const observed = (noise: number): number[] => xsData.map((x, i) => trueY(x) + noise * (NOISE[i] ?? 0))

const lasso: Method<State, Config> = defineMethod<State, Config>({
  name: 'lasso',
  config: dict({
    noise: real(0, 1, { default: 0.25, label: 'noise' }),
    penalty: real(0.02, 20, { default: 0.5, log: true, label: 'penalty' }),
  }),
  make: (c) => {
    const noise = c['noise'] as number
    const penalty = c['penalty'] as number
    const y = observed(noise)
    return {
      name: 'lasso',
      config: { noise, penalty },
      init: () => ({ k: 0, w: TERMS.map(() => 0), noise, penalty }),
      step: (s, ctx) => {
        const w = sweep(s.w, y, penalty)
        ctx.log('coefficient error', Math.hypot(...w.map((v, j) => v - (TERMS[j]?.truth ?? 0))))
        ctx.log('coefficients', w)
        return { ...s, k: s.k + 1, w }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    'coefficient error': scalar({ description: 'the distance of the coefficients from the true ones' }),
    coefficients: tensor({ role: 'state', shape: [TERMS.length], description: 'the coefficients' }),
    equation: derived<State>({
      description: 'the fitted equation beside the truth',
      compute: (s) => {
        const y = observed(s.noise)
        return equationFit({
          terms: TERMS.map((t, j) => ({ tex: t.tex, estimate: s.w[j] ?? 0, truth: t.truth })),
          x: GRID,
          fit: GRID.map((x) => TERMS.reduce((sum, t, j) => sum + (s.w[j] ?? 0) * t.f(x), 0)),
          truth: GRID.map(trueY),
          observed: { x: xsData, y },
          tolerance: 0.25,
          caption: `after ${s.k} sweeps`,
        })
      },
    }),
  },
  defaults: { 'coefficient error': Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'equation', path: 'equation', title: 'the equation against the truth' },
  {
    kind: 'curves',
    paths: ['terms kept'],
    truth: { value: 3, label: 'terms in the law' },
    title: 'terms kept (click to scrub)',
  },
]

export default function EquationAgainstTruthRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(lasso.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(lasso, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={lasso.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(lasso.config) as Config)} />
      <SessionPanels id="gallery/equation-against-truth" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
