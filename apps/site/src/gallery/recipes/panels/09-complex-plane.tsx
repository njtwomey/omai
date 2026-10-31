import { complexPlane } from 'mdml-engine/exposition'
import { polyroots } from 'mdml-engine/numerics/polynomial'
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
  title: 'Complex plane',
  question:
    'How do I show poles and zeros moving in the complex plane as a parameter changes, and when they leave the stable region?',
  explain:
    'A probe that returns a `complexPlane` record (poles, zeros, the branches of a locus, the domain `s` or `z`) drawn on equal axes with the stability boundary, the imaginary axis here or the unit circle for a discrete-time system. A pole outside the stable region is a larger cross in its own colour and is listed apart in the legend, and the caption says in words whether the system is stable. Here the loop gain of K / (s (s + 1) (s + 2)) rises over the run: the closed-loop poles, the roots of s³ + 3s² + 2s + K, run along the locus and cross the imaginary axis at K = 6.',
}

const STEPS = 100

interface State {
  readonly k: number
  readonly maxGain: number
}

const gainAt = (s: State): number => (s.maxGain * s.k) / STEPS
// The closed-loop poles at gain K, as [real, imaginary].
const polesAt = (gain: number): [number, number][] => {
  const r = polyroots([gain, 2, 3, 1])
  return r.re.map((re, i) => [re, r.im[i] ?? 0])
}

// The locus: the poles at a rising gain, each pole followed to its nearest successor so that branches stay whole.
const locusOf = (maxGain: number): { path: [number, number][] }[] => {
  const branches: [number, number][][] = polesAt(0).map((p) => [p])
  for (let i = 1; i <= 200; i++) {
    const next = polesAt((maxGain * i) / 200)
    const taken = new Set<number>()
    for (const branch of branches) {
      const last = branch[branch.length - 1] ?? [0, 0]
      let best = -1
      let gap = Number.POSITIVE_INFINITY
      next.forEach((p, j) => {
        const d = Math.hypot(p[0] - last[0], p[1] - last[1])
        if (!taken.has(j) && d < gap) {
          gap = d
          best = j
        }
      })
      taken.add(best)
      branch.push(next[best] ?? last)
    }
  }
  return branches.map((path) => ({ path }))
}

const loop: Method<State, Config> = defineMethod<State, Config>({
  name: 'loop',
  config: dict({ maxGain: real(2, 20, { default: 12, label: 'largest gain K' }) }),
  make: (c) => {
    const maxGain = c['maxGain'] as number
    return {
      name: 'loop',
      config: { maxGain },
      init: () => ({ k: 0, maxGain }),
      step: (s, ctx) => {
        const next = { k: s.k + 1, maxGain }
        ctx.log('gain', gainAt(next))
        ctx.log('rightmost pole', Math.max(...polesAt(gainAt(next)).map((p) => p[0])))
        return next
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    gain: scalar({ description: 'the loop gain K' }),
    'rightmost pole': scalar({ description: 'the largest real part among the closed-loop poles: negative is stable' }),
    plane: derived<State>({
      description: 'the closed-loop poles at the gain of the step, with the locus over the whole run',
      compute: (s) =>
        complexPlane({
          domain: 's',
          poles: polesAt(gainAt(s)),
          locus: locusOf(s.maxGain),
          caption: `gain K = ${gainAt(s).toFixed(2)}`,
        }),
    }),
  },
  defaults: { gain: Log(), 'rightmost pole': Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'probe', path: 'plane', title: 'closed-loop poles' },
  { kind: 'curves', paths: ['rightmost pole'], title: 'rightmost pole, below zero is stable (click to scrub)' },
]

export default function ComplexPlaneRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(loop.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(loop, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={loop.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(loop.config) as Config)} />
      <SessionPanels id="gallery/complex-plane" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
