import { WhiteboxError } from '../../base/index.ts'
import { metricFunction } from '../../numerics/geometry/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { Generator } from '../../random/index.ts'
import { item } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { clipInto, finiteBounds } from './_bounds.ts'
import { defaultLocal, localSearch } from './_local.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import type { LocalMethod } from './global.ts'
import type { Bounds } from './objective.ts'
import {
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

const EMPTY = new Float64Array(0)

// ---------------------------------------------------------------------------------------------------------------
// Dual annealing

/** Options of {@link dualAnnealing}. */
export interface DualAnnealingOptions extends VectorMethodOptions {
  /** Box bounds (required, finite). */
  readonly bounds: Bounds
  /** The initial temperature (default 5230). */
  readonly initialTemperature?: number
  /** Reanneal when the temperature falls to this fraction of the initial one (default 2e-5). */
  readonly restartTemperatureRatio?: number
  /** The visiting distribution parameter q_v in (1, 3] (default 2.62). */
  readonly visit?: number
  /** The acceptance parameter q_a in (-1e4, -5] (default -5). */
  readonly accept?: number
  /** Soft evaluation budget (default 1e7). */
  readonly maxEvaluations?: number
  /** Generalised simulated annealing only, with no local search. */
  readonly noLocalSearch?: boolean
  /** The local minimiser (default bounded Nelder-Mead; scipy uses L-BFGS-B). */
  readonly localMethod?: LocalMethod
  /** Options for the local minimiser (the bounds are always passed). */
  readonly localOptions?: VectorMethodOptions
  /** Start here instead of at a random point (default: the given x0 is used; set `randomStart` to ignore it). */
  readonly randomStart?: boolean
}

/** State of a dual annealing run. */
export interface DualAnnealingState extends VectorState {
  readonly current: Float64Array
  readonly currentValue: number
  readonly temperature: number
  /** Index into the temperature schedule (reset by reannealing). */
  readonly schedule: number
  readonly xmin: Float64Array
  readonly emin: number
  readonly notImproved: number
  readonly notImprovedMax: number
  readonly restarts: number
}

const TAIL_LIMIT = 1e8
const MIN_VISIT_BOUND = 1e-10

/**
 * Dual annealing (Xiang et al. 1997; scipy.optimize.dual_annealing) as a traced algorithm: generalised simulated
 * annealing with a Tsallis-Stariolo visiting distribution (all coordinates, then one at a time, in each Markov chain),
 * the generalised Metropolis acceptance, reannealing when the temperature falls below the restart level, and local
 * searches from improving points. Each step is one temperature (one Markov chain plus its local search).
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree; also fixes the structure).
 * @param options - Bounds, temperatures, distribution parameters, budget, local search.
 * @returns The algorithm.
 * @concept simulated-annealing
 * @example
 * ```ts
 * import { dualAnnealingAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * dualAnnealingAlgorithm((x) => sum(square(x as Value)), [1, 2], { bounds: [[-5, 5], [-5, 5]] }).name // => 'dualAnnealing'
 * ```
 */
export function dualAnnealingAlgorithm(
  f: Objective,
  x0: Parameters,
  options: DualAnnealingOptions,
): Algorithm<DualAnnealingState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'dualAnnealing: no parameters')
  const euclid = metricFunction('euclidean', n)
  const box = finiteBounds(options.bounds, n, 'dualAnnealing')
  const range = Float64Array.from(box.upper, (u, i) => u - (box.lower[i] ?? 0))
  const t0 = options.initialTemperature ?? 5230
  const ratio = options.restartTemperatureRatio ?? 2e-5
  if (!(ratio > 0 && ratio < 1))
    throw new WhiteboxError('value', 'dualAnnealing: restartTemperatureRatio must be in (0, 1)')
  const qv = options.visit ?? 2.62
  const qa = options.accept ?? -5
  const maxfun = options.maxEvaluations ?? 1e7
  const local = options.localMethod ?? defaultLocal
  // scipy caps L-BFGS-B at min(max(6 n, 100), 1000) iterations; a given local method keeps its own defaults.
  const localOptions = { ...options.localOptions, bounds: options.bounds }
  const tRestart = t0 * ratio
  const t1 = Math.exp((qv - 1) * Math.log(2)) - 1
  // The constant factors of the visiting distribution.
  const factor2 = Math.exp((4 - qv) * Math.log(qv - 1))
  const factor3 = Math.exp(((2 - qv) * Math.log(2)) / (qv - 1))
  const factor4p = (Math.sqrt(Math.PI) * factor2) / (factor3 * (3 - qv))
  const factor5 = 1 / (qv - 1) - 0.5
  const d1 = 2 - factor5
  const factor6 = (Math.PI * (1 - factor5)) / Math.sin(Math.PI * (1 - factor5)) / Math.exp(item(lgamma(d1)))
  const K = 100 * n

  const visitFn = (rng: Generator, temperature: number, dim: number): Float64Array => {
    const z = rng.normal([dim, 2]).data
    const factor1 = Math.exp(Math.log(temperature) / (qv - 1))
    const factor4 = factor4p * factor1
    const scale = Math.exp((-(qv - 1) * Math.log(factor6 / factor4)) / (3 - qv))
    return Float64Array.from({ length: dim }, (_, i) => {
      const x = (z[2 * i] ?? 0) * scale
      const den = Math.exp(((qv - 1) * Math.log(Math.abs(z[2 * i + 1] ?? 0))) / (3 - qv))
      return x / den
    })
  }
  const wrap = (v: number, i: number): number => {
    const a = v - (box.lower[i] ?? 0)
    const b = (a % (range[i] ?? 1)) + (range[i] ?? 1)
    return (b % (range[i] ?? 1)) + (box.lower[i] ?? 0)
  }
  const visiting = (rng: Generator, x: Float64Array, step: number, temperature: number): Float64Array => {
    if (step < n) {
      const visits = visitFn(rng, temperature, n)
      const [up, lo] = rng.uniform([2]).data
      return Float64Array.from(visits, (vi, i) => {
        let v = vi
        if (v > TAIL_LIMIT) v = TAIL_LIMIT * (up ?? 0)
        else if (v < -TAIL_LIMIT) v = -TAIL_LIMIT * (lo ?? 0)
        let y = wrap(v + (x[i] ?? 0), i)
        if (Math.abs(y - (box.lower[i] ?? 0)) < MIN_VISIT_BOUND) y += 1e-10
        return y
      })
    }
    const out = Float64Array.from(x)
    let v = visitFn(rng, temperature, 1)[0] ?? 0
    if (v > TAIL_LIMIT) v = TAIL_LIMIT * (rng.uniform([1]).data[0] ?? 0)
    else if (v < -TAIL_LIMIT) v = -TAIL_LIMIT * (rng.uniform([1]).data[0] ?? 0)
    const i = step - n
    out[i] = wrap(v + (x[i] ?? 0), i)
    if (Math.abs((out[i] ?? 0) - (box.lower[i] ?? 0)) < MIN_VISIT_BOUND) out[i] = (out[i] ?? 0) + MIN_VISIT_BOUND
    return out
  }
  const randomPoint = (rng: Generator): Float64Array =>
    Float64Array.from(rng.uniform([n]).data, (u, i) => (box.lower[i] ?? 0) + u * (range[i] ?? 0))

  return {
    name: 'dualAnnealing',
    config: configOf({
      initialTemperature: t0,
      restartTemperatureRatio: ratio,
      visit: qv,
      accept: qa,
      maxEvaluations: maxfun,
      noLocalSearch: options.noLocalSearch ?? false,
    }),
    schema: vectorSchema({
      metrics: {
        temperature: { description: 'visiting temperature' },
        current: { description: 'objective at the current point' },
        restarts: { description: 'reannealing restarts so far' },
      },
      tensors: { current: { role: 'state', description: 'current point of the chain' } },
    }),
    init: (key) => {
      const rng = new Generator(key)
      let x = options.randomStart === true ? randomPoint(rng) : clipInto(obj.x0, box)
      let e = obj.value(x)
      let nfev = 1
      for (let tries = 0; !Number.isFinite(e) && tries < 1000; tries++) {
        x = randomPoint(rng)
        e = obj.value(x)
        nfev += 1
      }
      if (!Number.isFinite(e))
        throw new WhiteboxError('value', 'dualAnnealing: the objective is not finite anywhere tried')
      return {
        x,
        value: e,
        grad: EMPTY,
        nfev,
        ngev: 0,
        nhev: 0,
        step: 0,
        current: x,
        currentValue: e,
        temperature: t0,
        schedule: 0,
        xmin: x,
        emin: e,
        notImproved: 0,
        notImprovedMax: 1000,
        restarts: 0,
      }
    },
    step: (s, ctx) => {
      const rng = new Generator(ctx.key)
      let i = s.schedule
      let temperature = t0 * (t1 / (Math.exp((qv - 1) * Math.log(i + 2)) - 1))
      let current = s.current
      let ecur = s.currentValue
      let nfev = s.nfev
      let restarts = s.restarts
      if (temperature < tRestart) {
        // Reanneal from a random point.
        current = randomPoint(rng)
        ecur = obj.value(current)
        nfev += 1
        for (let tries = 0; !Number.isFinite(ecur) && tries < 1000; tries++) {
          current = randomPoint(rng)
          ecur = obj.value(current)
          nfev += 1
        }
        i = 0
        temperature = t0
        restarts += 1
      }
      let xbest = s.x
      let ebest = s.value
      let xmin = s.xmin
      let emin = s.emin
      let notImproved = s.notImproved + 1
      let notImprovedMax = s.notImprovedMax
      const tStep = temperature / (i + 1)
      let improved = i === 0
      let budget: string | undefined
      // The Markov chain: all coordinates (n moves), then one coordinate at a time (n moves).
      for (let j = 0; j < 2 * n; j++) {
        const visit = visiting(rng, current, j, temperature)
        const e = obj.value(visit)
        nfev += 1
        if (e < ecur) {
          current = visit
          ecur = e
          if (e < ebest) {
            xbest = visit
            ebest = e
            improved = true
            notImproved = 0
          }
        } else {
          const r = rng.uniform([1]).data[0] ?? 1
          const pqvTemp = 1 - ((1 - qa) * (e - ecur)) / tStep
          const pqv = pqvTemp <= 0 ? 0 : Math.exp(Math.log(pqvTemp) / (1 - qa))
          if (r <= pqv) {
            current = visit
            ecur = e
            xmin = visit
          }
          if (notImproved >= notImprovedMax && (j === 0 || ecur < emin)) {
            emin = ecur
            xmin = current
          }
        }
        if (nfev >= maxfun) {
          budget = 'maximum number of function calls reached during annealing'
          break
        }
      }
      if (budget === undefined && options.noLocalSearch !== true) {
        if (improved) {
          const r = localSearch(obj, local, xbest, localOptions, ctx)
          nfev += r.nfev
          if (r.fun < ebest && Number.isFinite(r.fun)) {
            notImproved = 0
            xbest = r.x
            ebest = r.fun
            current = r.x
            ecur = r.fun
          }
        }
        let doLs = false
        if (K < 90 * n) {
          const pls = Math.exp((K * (ebest - ecur)) / tStep)
          if (pls >= (rng.uniform([1]).data[0] ?? 1)) doLs = true
        }
        if (notImproved >= notImprovedMax) doLs = true
        if (doLs) {
          const r = localSearch(obj, local, xmin, localOptions, ctx)
          nfev += r.nfev
          const better = Number.isFinite(r.fun) && r.fun < emin
          xmin = better ? r.x : xmin
          emin = better ? r.fun : emin
          notImproved = 0
          notImprovedMax = n
          if (emin < ebest) {
            xbest = xmin
            ebest = emin
            current = xmin
            ecur = emin
          }
        }
        if (nfev >= maxfun) budget = 'maximum number of function calls reached'
      }
      return {
        x: xbest,
        value: ebest,
        grad: EMPTY,
        nfev,
        ngev: 0,
        nhev: 0,
        step: euclid(xbest, 0, s.x, 0),
        current,
        currentValue: ecur,
        temperature,
        schedule: i + 1,
        xmin,
        emin,
        notImproved,
        notImprovedMax,
        restarts,
        ...(budget === undefined ? {} : { stop: { status: 'failed' as const, message: budget } }),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { temperature: s.temperature, current: s.currentValue, restarts: s.restarts },
        tensors: { current: { role: 'state', value: s.current } },
      }),
  }
}

/**
 * Find a global minimum by dual annealing (scipy.optimize.dual_annealing). `maxIterations` counts temperatures (default
 * 1000); the run ends 'completed' with the best point found.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree; also fixes the structure).
 * @param options - Bounds (required), temperatures, distribution parameters, budget, local search, key or seed.
 * @returns The best point found.
 * @concept simulated-annealing
 * @example
 * ```ts
 * import { dualAnnealing } from 'whitebox-engine/optim'
 * import { cos, mul, square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * // Rastrigin in 2-D: global minimum 0 at the origin
 * const f = (x: unknown) => sum(sub(square(x as Value), mul(cos(mul(x as Value, 2 * Math.PI)), 10)))
 * const r = dualAnnealing(f, [3, -2], { bounds: [[-5.12, 5.12], [-5.12, 5.12]], seed: 0, maxIterations: 200 })
 * toFlat(r.x as Value).every((v) => Math.abs(v) < 1e-6) // => true
 * ```
 */
export function dualAnnealing(f: Objective, x0: Parameters, options: DualAnnealingOptions): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(dualAnnealingAlgorithm(f, x0, options), obj, options, 1000)
}
