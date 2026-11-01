/**
 * Bayesian optimisation of a one-dimensional function as a method of nothing but its configuration: each step is one
 * trial, chosen by maximising an acquisition over a Gaussian-process surrogate fitted to the trials so far (after a few
 * random ones), then evaluated, with noise if asked. The trials' positions, values and the best so far are on the tape,
 * and the surrogate with its acquisition at any step is a probe: the true function, the posterior mean with a two
 * standard deviation band, the acquisition on a second axis and a mark at every trial.
 */

import { varianceOf } from '../../lib/base/index.ts'
import { dampedSine, forrester, gramacyLee, type TestFunction } from '../../lib/datasets/index.ts'
import { curveSet } from '../../lib/exposition/index.ts'
import { foldIn, normal, uniform } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { item, neg, tensor, toFlat } from '../../lib/tensor/index.ts'
import { type Config, type Contract, defineMethod, derived, Log, type Method, scalar } from '../../lib/trace/index.ts'
import {
  type Acquisition,
  acquisitions,
  expectedImprovement,
  fitGpSurrogate,
  type GpSurrogate,
  lowerConfidenceBound,
  maximizeAcquisition,
  probabilityOfImprovement,
} from '../../lib/tune/index.ts'

/** The functions {@link bayesianOptimisation} minimises. */
export const BAYESIAN_OBJECTIVES = ['forrester', 'damped-sine', 'gramacy-lee'] as const

/** An objective of {@link bayesianOptimisation}. */
export type BayesianObjective = (typeof BAYESIAN_OBJECTIVES)[number]

/** The acquisitions {@link bayesianOptimisation} can maximise. */
export const BAYESIAN_ACQUISITIONS = [
  'expected-improvement',
  'lower-confidence-bound',
  'probability-of-improvement',
] as const

/** An acquisition of {@link bayesianOptimisation}. */
export type BayesianAcquisition = (typeof BAYESIAN_ACQUISITIONS)[number]

/** The state of the optimisation: the trials so far, as plain numbers (positions in the function's own units). */
export interface BayesianState {
  readonly xs: readonly number[]
  readonly ys: readonly number[]
}

const FUNCTIONS: Readonly<Record<BayesianObjective, () => TestFunction>> = {
  forrester,
  'damped-sine': dampedSine,
  'gramacy-lee': gramacyLee,
}

// What is maximised when choosing the next trial (the logarithms are stabler where improvement is almost impossible),
// and what the probe shows (the quantities themselves, higher meaning a better place to try).
const MAXIMISED: Readonly<Record<BayesianAcquisition, Acquisition>> = {
  'expected-improvement': acquisitions.logEI,
  'lower-confidence-bound': acquisitions.LCB,
  'probability-of-improvement': acquisitions.PI,
}
const SHOWN: Readonly<Record<BayesianAcquisition, Acquisition>> = {
  'expected-improvement': (m, s, best) => expectedImprovement(m, s, best),
  'lower-confidence-bound': (m, s) => neg(lowerConfidenceBound(m, s, 2)),
  'probability-of-improvement': (m, s, best) => probabilityOfImprovement(m, s, best),
}

const GRID = 200

/** An objective on a grid over its interval, and the spread of its values (the scale noise is measured in). */
interface Landscape {
  readonly fn: TestFunction
  readonly lo: number
  readonly hi: number
  /** The grid in the unit interval and in the function's units. */
  readonly unit: readonly number[]
  readonly x: readonly number[]
  readonly truth: readonly number[]
  readonly scale: number
}

const landscapes = new Map<BayesianObjective, Landscape>()

function landscapeOf(name: BayesianObjective): Landscape {
  const known = landscapes.get(name)
  if (known !== undefined) return known
  const fn = FUNCTIONS[name]()
  const [lo, hi] = fn.bounds[0] as readonly [number, number]
  const unit = Array.from({ length: GRID }, (_, i) => i / (GRID - 1))
  const x = unit.map((u) => lo + u * (hi - lo))
  const truth = Array.from(toFlat(fn.f(tensor(x.map((v) => [v])))))
  const landscape: Landscape = { fn, lo, hi, unit, x, truth, scale: Math.sqrt(varianceOf(truth)) || 1 }
  landscapes.set(name, landscape)
  return landscape
}

// The surrogate depends on the trials alone, so the step that chose a trial and the probe of the same step share one
// fit: the latest few are remembered.
const fits = new Map<string, GpSurrogate>()

function surrogateOf(land: Landscape, s: BayesianState): GpSurrogate {
  const id = JSON.stringify([land.lo, land.hi, s.xs, s.ys])
  const known = fits.get(id)
  if (known !== undefined) return known
  const fit = fitGpSurrogate(
    s.xs.map((x) => [(x - land.lo) / (land.hi - land.lo)]),
    s.ys,
  )
  fits.set(id, fit)
  if (fits.size > 8) fits.delete(fits.keys().next().value as string)
  return fit
}

const bestOf = (s: BayesianState): { readonly x: number; readonly value: number } | undefined => {
  let at = -1
  s.ys.forEach((y, i) => {
    if (at < 0 || y < (s.ys[at] ?? Number.POSITIVE_INFINITY)) at = i
  })
  return at < 0 ? undefined : { x: s.xs[at] ?? Number.NaN, value: s.ys[at] ?? Number.NaN }
}

/**
 * The paths of {@link bayesianOptimisation}: each trial’s point and value, the best value so far, and the derived `model/params` (the best trial) and `tune/posterior` (the surrogate over the interval).
 *
 * @example
 * ```ts
 * import type { BayesianOptimisationPath } from 'mdml-engine/methods/tune'
 * const path: BayesianOptimisationPath = 'best'
 * path // => 'best'
 * ```
 */
export type BayesianOptimisationPath = 'x' | 'value' | 'best' | 'model/params' | 'tune/posterior'

/**
 * Bayesian optimisation of a function of one variable. After a few random trials, each trial maximises an acquisition
 * over a Gaussian-process surrogate (a Matern-5/2 kernel with fitted length scale, scale and noise) of the trials so far:
 * expected improvement, the lower confidence bound (the mirror of the upper one for a minimised function) or the
 * probability of improvement. The functions are minimised: Forrester, a damped sine and Gramacy-Lee.
 *
 * Knobs: `objective`, `acquisition`, `noise` (the standard deviation of the noise added to every evaluation, as a
 * fraction of the function's spread over its interval) and `initial` (the random trials before the surrogate is used).
 * The tape has `x`, `value` and `best` per trial; `model/params` is the best trial so far and `tune/posterior` the
 * surrogate and acquisition at a step.
 *
 * @example
 * ```ts
 * import { bayesianOptimisation } from 'mdml-engine/methods/tune'
 * import { session } from 'mdml-engine/trace'
 * const s = session(bayesianOptimisation, { mode: 'eager', steps: 12 })
 * s.t // => 12
 * ```
 */
export const bayesianOptimisation: Method<BayesianState, Config, BayesianOptimisationPath> = defineMethod<
  BayesianState,
  Config,
  BayesianOptimisationPath
>({
  name: 'tune/bayesian-optimisation',
  config: dict({
    objective: choice(BAYESIAN_OBJECTIVES, { default: 'forrester', label: 'function' }),
    acquisition: choice(BAYESIAN_ACQUISITIONS, { default: 'expected-improvement', label: 'acquisition' }),
    noise: real(0, 0.5, { default: 0, label: 'noise (fraction of the spread)' }),
    initial: int(1, 8, { default: 5, label: 'random trials first' }),
  }),
  make: (c) => {
    const land = landscapeOf(c['objective'] as BayesianObjective)
    const acquisition = MAXIMISED[c['acquisition'] as BayesianAcquisition]
    const noise = (c['noise'] as number) * land.scale
    const initial = c['initial'] as number
    return {
      name: `tune/bayesian-optimisation/${c['acquisition'] as string}`,
      config: { objective: c['objective'] as string, acquisition: c['acquisition'] as string, noise, initial },
      init: () => ({ xs: [], ys: [] }),
      step: (s, ctx) => {
        // The unit interval is the surrogate's coordinate; trials are placed in the function's units.
        const u =
          s.xs.length < initial
            ? (uniform(foldIn(ctx.key, 0), [1]).data[0] ?? 0.5)
            : (maximizeAcquisition(
                surrogateOf(land, s),
                acquisition,
                bestOf(s)?.value ?? Number.NaN,
                foldIn(ctx.key, 0),
                {
                  seeds: s.ys
                    .map((y, i) => [y, i] as const)
                    .sort((a, b) => a[0] - b[0])
                    .slice(0, 3)
                    .map(([, i]) => [((s.xs[i] ?? land.lo) - land.lo) / (land.hi - land.lo)]),
                },
              ).point[0] ?? 0.5)
        const x = land.lo + u * (land.hi - land.lo)
        const y = item(land.fn.f(tensor([x]))) + noise * (normal(foldIn(ctx.key, 1), [1]).data[0] ?? 0)
        const next: BayesianState = { xs: [...s.xs, x], ys: [...s.ys, y] }
        ctx.log('x', x)
        ctx.log('value', y)
        ctx.log('best', bestOf(next)?.value ?? y)
        return next
      },
      // Plain numbers, so the probe can seek any step from a checkpoint.
      serialise: (s) => ({ xs: [...s.xs], ys: [...s.ys] }),
      deserialise: (json) => {
        const j = json as { readonly xs: number[]; readonly ys: number[] }
        return { xs: j.xs, ys: j.ys }
      },
    }
  },
  contract: (c): Contract<BayesianState, BayesianOptimisationPath> => {
    const land = landscapeOf(c['objective'] as BayesianObjective)
    const shown = SHOWN[c['acquisition'] as BayesianAcquisition]
    return {
      x: scalar({ description: 'where the trial evaluated the function' }),
      value: scalar({ description: 'the value observed (noisy when the knob says so)' }),
      best: scalar({ description: 'the lowest value observed so far', direction: 'lower' }),
      'model/params': derived<BayesianState>({
        description: 'the best trial so far: where it was and the value observed',
        cost: 'cheap',
        compute: (s) => bestOf(s) ?? null,
      }),
      'tune/posterior': derived<BayesianState>({
        description:
          'the function, the surrogate’s mean with a two standard deviation band, and the acquisition, over the interval',
        cost: 'dear',
        compute: (s) => {
          const marks = s.xs.map((x, i) => ({ x, ...(i === s.xs.length - 1 ? { label: 'latest' } : {}) }))
          const truth = { name: 'function', y: land.truth }
          if (s.xs.length === 0) return curveSet({ x: land.x, xLabel: 'x', yLabel: 'f(x)', series: [truth] })
          const gp = surrogateOf(land, s)
          const { mean, std } = gp.predict(land.unit.map((u) => [u]))
          const best = bestOf(s)?.value ?? Number.NaN
          const score = toFlat(shown(tensor(mean), tensor(std), best))
          return curveSet({
            x: land.x,
            xLabel: 'x',
            yLabel: 'f(x)',
            y2Label: c['acquisition'] as string,
            series: [
              truth,
              {
                name: 'surrogate mean',
                y: mean,
                lower: mean.map((m, i) => m - 2 * (std[i] ?? 0)),
                upper: mean.map((m, i) => m + 2 * (std[i] ?? 0)),
              },
              { name: c['acquisition'] as string, y: Array.from(score), axis: 'right' },
            ],
            marks,
          })
        },
      }),
    }
  },
  defaults: { x: Log(), value: Log(), best: Log() },
})
