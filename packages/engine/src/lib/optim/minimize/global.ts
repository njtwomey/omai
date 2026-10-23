import { WhiteboxError } from '../../base/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { Generator } from '../../random/index.ts'
import { item } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { asPairs, type BoundArrays, boundArrays, clipInto, finiteBounds } from './_bounds.ts'
import { defaultLocal, localSearch } from './_local.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import type { Bounds } from './derivative-free.ts'
import {
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

/** A local minimiser used inside a global method: any vector method that accepts box bounds. */
export type LocalMethod = (
  f: Objective,
  x0: Parameters,
  options?: VectorMethodOptions & { readonly bounds?: Bounds },
) => MinimizeResult

const EMPTY = new Float64Array(0)

// ---------------------------------------------------------------------------------------------------------------
// Simulated annealing

/** Options of {@link simulatedAnnealing}. */
export interface SimulatedAnnealingOptions extends VectorMethodOptions {
  /** Box bounds: proposals are reflected into the box. */
  readonly bounds?: Bounds
  /** The starting temperature (default 1). */
  readonly initialTemperature?: number
  /**
   * The cooling schedule: `exponential` (T0 cooling^k, the default), `logarithmic` (T0 / ln(k + e), Geman and Geman),
   * `fast` (T0 / (k + 1), Szu and Hartley), or a function of the level k and T0.
   */
  readonly schedule?: 'exponential' | 'logarithmic' | 'fast' | ((k: number, t0: number) => number)
  /** The factor of the exponential schedule (default 0.95). */
  readonly cooling?: number
  /** Standard deviation of the Gaussian proposals, per coordinate (default 10% of each bound range, else 1). */
  readonly stepSize?: number | readonly number[]
  /** Proposals per temperature level (default 20 n). */
  readonly movesPerTemperature?: number
  /** Adapt the step sizes to keep the acceptance rate between 0.4 and 0.6 (Corana et al. 1987; default true). */
  readonly adaptStep?: boolean
  /** Stop once the temperature falls below this (default 1e-8 T0). */
  readonly minTemperature?: number
}

/** State of a simulated annealing run. */
export interface SimulatedAnnealingState extends VectorState {
  /** The current point of the chain and its value (x and value hold the best point found). */
  readonly current: Float64Array
  readonly currentValue: number
  readonly temperature: number
  /** The temperature level. */
  readonly level: number
  readonly steps: Float64Array
  /** Acceptance rate at the last level. */
  readonly acceptance: number
}

/** Reflect a point into a box (mirror at the walls), then clip. */
function reflectInto(x: Float64Array, box: BoundArrays | undefined): Float64Array {
  if (box === undefined) return x
  const out = Float64Array.from(x, (v, i) => {
    const lo = box.lower[i] ?? 0
    const hi = box.upper[i] ?? 0
    let y = v
    if (y < lo && Number.isFinite(lo)) y = 2 * lo - y
    if (y > hi && Number.isFinite(hi)) y = 2 * hi - y
    return y
  })
  return clipInto(out, box)
}

/**
 * Simulated annealing (Kirkpatrick, Gelatt and Vecchi 1983) as a traced algorithm: a Metropolis chain of Gaussian
 * proposals at a decreasing temperature, keeping the best point seen. Each step is one temperature level; the trace
 * records the temperature, the acceptance rate and the best and current values.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Bounds, temperature schedule, proposal sizes and their adaptation, moves per level, stopping.
 * @returns The algorithm.
 * @concept simulated-annealing
 * @example
 * ```ts
 * import { simulatedAnnealingAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * simulatedAnnealingAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'simulatedAnnealing'
 * ```
 */
export function simulatedAnnealingAlgorithm(
  f: Objective,
  x0: Parameters,
  options: SimulatedAnnealingOptions = {},
): Algorithm<SimulatedAnnealingState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'simulatedAnnealing: no parameters')
  const box = options.bounds === undefined ? undefined : boundArrays(options.bounds, n, 'simulatedAnnealing')
  const t0 = options.initialTemperature ?? 1
  const cooling = options.cooling ?? 0.95
  const schedule = options.schedule ?? 'exponential'
  const temperatureAt = (k: number): number =>
    typeof schedule === 'function'
      ? schedule(k, t0)
      : schedule === 'exponential'
        ? t0 * cooling ** k
        : schedule === 'logarithmic'
          ? t0 / Math.log(k + Math.E)
          : t0 / (k + 1)
  const moves = options.movesPerTemperature ?? 20 * n
  const adapt = options.adaptStep ?? true
  const minT = options.minTemperature ?? 1e-8 * t0
  const step0 = Float64Array.from({ length: n }, (_, i) => {
    const s = options.stepSize
    if (typeof s === 'number') return s
    if (s !== undefined) return s[i] ?? 1
    if (box !== undefined && Number.isFinite((box.upper[i] ?? 0) - (box.lower[i] ?? 0)))
      return 0.1 * ((box.upper[i] ?? 0) - (box.lower[i] ?? 0))
    return 1
  })
  return {
    name: 'simulatedAnnealing',
    config: configOf({
      initialTemperature: t0,
      cooling,
      moves,
      adapt,
      minTemperature: minT,
      schedule: typeof schedule === 'string' ? schedule : 'custom',
    }),
    schema: vectorSchema({
      metrics: {
        temperature: { description: 'temperature of the level' },
        acceptance: { description: 'fraction of proposals accepted at the level' },
        current: { description: 'objective at the current point of the chain' },
      },
      tensors: { current: { role: 'state', description: 'current point of the chain' } },
    }),
    init: () => {
      const x = reflectInto(Float64Array.from(obj.x0), box)
      const fx = obj.value(x)
      return {
        x,
        value: fx,
        grad: EMPTY,
        nfev: 1,
        ngev: 0,
        nhev: 0,
        step: 0,
        current: x,
        currentValue: fx,
        temperature: temperatureAt(0),
        level: 0,
        steps: step0,
        acceptance: 0,
      }
    },
    step: (s, ctx) => {
      const rng = new Generator(ctx.key)
      const temperature = s.temperature
      let current = s.current
      let fcur = s.currentValue
      let best = s.x
      let fbest = s.value
      let accepted = 0
      let moved = 0
      const noise = rng.normal([moves, n]).data
      const u = rng.uniform([moves]).data
      for (let k = 0; k < moves; k++) {
        const proposal = reflectInto(
          Float64Array.from(current, (v, i) => v + (s.steps[i] ?? 0) * (noise[k * n + i] ?? 0)),
          box,
        )
        const fp = obj.value(proposal)
        const delta = fp - fcur
        if (delta <= 0 || (u[k] ?? 1) < Math.exp(-delta / temperature)) {
          current = proposal
          fcur = fp
          accepted += 1
          if (fp < fbest) {
            moved = Math.sqrt(proposal.reduce((acc, v, i) => acc + (v - (best[i] ?? 0)) ** 2, 0))
            best = proposal
            fbest = fp
          }
        }
      }
      const rate = accepted / moves
      let steps = s.steps
      if (adapt) {
        const factor = rate > 0.6 ? 1 + (2 * (rate - 0.6)) / 0.4 : rate < 0.4 ? 1 / (1 + (2 * (0.4 - rate)) / 0.4) : 1
        steps = Float64Array.from(s.steps, (v) => v * factor)
      }
      const level = s.level + 1
      const next = temperatureAt(level)
      return {
        x: best,
        value: fbest,
        grad: EMPTY,
        nfev: s.nfev + moves,
        ngev: 0,
        nhev: 0,
        step: moved,
        current,
        currentValue: fcur,
        temperature: next,
        level,
        steps,
        acceptance: rate,
        ...(next < minT ? { stop: { status: 'converged' as const, message: 'temperature below minTemperature' } } : {}),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { temperature: s.temperature, acceptance: s.acceptance, current: s.currentValue },
        tensors: { current: { role: 'state', value: s.current } },
      }),
  }
}

/**
 * Minimise by simulated annealing.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Bounds, schedule, proposals, levels (maxIterations, default 1000), key or seed and run options.
 * @returns The best point found.
 * @concept simulated-annealing
 * @example
 * ```ts
 * import { simulatedAnnealing } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = simulatedAnnealing((x) => sum(square(sub(x as Value, 1))), [3], { seed: 0, maxIterations: 300 })
 * Math.abs((toFlat(r.x as Value)[0] ?? 0) - 1) < 0.05 // => true
 * ```
 */
export function simulatedAnnealing(
  f: Objective,
  x0: Parameters,
  options: SimulatedAnnealingOptions = {},
): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(simulatedAnnealingAlgorithm(f, x0, options), obj, options, 1000)
}

// ---------------------------------------------------------------------------------------------------------------
// Basin hopping

/** Options of {@link basinHopping}. */
export interface BasinHoppingOptions extends VectorMethodOptions {
  /** The Metropolis temperature for accepting a new basin (default 1; 0 accepts only improvements). */
  readonly temperature?: number
  /** Half-width of the uniform random displacement (default 0.5). */
  readonly stepSize?: number
  /** Adjust the step size every `interval` hops towards `targetAcceptRate` (default 50 and 0.5, factor 0.9). */
  readonly interval?: number
  readonly targetAcceptRate?: number
  readonly stepwiseFactor?: number
  /** Stop when the global minimum has not improved for this many hops. */
  readonly niterSuccess?: number
  /** The local minimiser (default Nelder-Mead with tight tolerances; scipy uses BFGS). */
  readonly localMethod?: LocalMethod
  /** Options for the local minimiser. */
  readonly localOptions?: VectorMethodOptions
  /** Box bounds, passed to the local minimiser; displaced points are clipped into the box. */
  readonly bounds?: Bounds
}

/** State of a basin hopping run. */
export interface BasinHoppingState extends VectorState {
  /** The current local minimum (x and value hold the best found). */
  readonly current: Float64Array
  readonly currentValue: number
  readonly stepSize: number
  readonly hops: number
  readonly accepted: number
  /** Hops since the global minimum last improved. */
  readonly sinceImprovement: number
  /** Whether the last hop was accepted. */
  readonly lastAccepted: boolean
  readonly started: boolean
}

/**
 * Basin hopping (Wales and Doye 1997) as a traced algorithm, as scipy.optimize.basinhopping: random displacement,
 * local minimisation, Metropolis acceptance of the new minimum, adaptive step size and the best minimum kept. The first
 * step minimises from x0; each later step is one hop, with its local minimisation as a child run.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Temperature, step size and its adaptation, stopping rule, local method and bounds.
 * @returns The algorithm.
 * @concept basin-hopping
 * @example
 * ```ts
 * import { basinHoppingAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * basinHoppingAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'basinHopping'
 * ```
 */
export function basinHoppingAlgorithm(
  f: Objective,
  x0: Parameters,
  options: BasinHoppingOptions = {},
): Algorithm<BasinHoppingState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'basinHopping: no parameters')
  const box = options.bounds === undefined ? undefined : boundArrays(options.bounds, n, 'basinHopping')
  const temperature = options.temperature ?? 1
  const interval = options.interval ?? 50
  const target = options.targetAcceptRate ?? 0.5
  const factor = options.stepwiseFactor ?? 0.9
  const local = options.localMethod ?? defaultLocal
  const localOptions = { ...options.localOptions, ...(box === undefined ? {} : { bounds: asPairs(box) }) }
  const niterSuccess = options.niterSuccess
  return {
    name: 'basinHopping',
    config: configOf({
      temperature,
      stepSize: options.stepSize ?? 0.5,
      interval,
      targetAcceptRate: target,
      stepwiseFactor: factor,
    }),
    schema: vectorSchema({
      metrics: {
        current: { description: 'objective at the current local minimum' },
        stepSize: { description: 'half-width of the random displacement' },
        acceptRate: { description: 'fraction of hops accepted so far' },
      },
      tensors: { current: { role: 'state', description: 'current local minimum' } },
      messages: { hop: { description: 'whether the hop was accepted and its local minimum' } },
    }),
    init: () => {
      const x = box === undefined ? Float64Array.from(obj.x0) : clipInto(obj.x0, box)
      const fx = obj.value(x)
      return {
        x,
        value: fx,
        grad: EMPTY,
        nfev: 1,
        ngev: 0,
        nhev: 0,
        step: 0,
        current: x,
        currentValue: fx,
        stepSize: options.stepSize ?? 0.5,
        hops: 0,
        accepted: 0,
        sinceImprovement: 0,
        lastAccepted: false,
        started: false,
      }
    },
    step: (s, ctx) => {
      if (!s.started) {
        const r = localSearch(obj, local, s.x, localOptions, ctx)
        return {
          ...s,
          x: r.x,
          value: r.fun,
          current: r.x,
          currentValue: r.fun,
          nfev: s.nfev + r.nfev,
          started: true,
          lastAccepted: true,
        }
      }
      const rng = new Generator(ctx.key)
      const hops = s.hops + 1
      let stepSize = s.stepSize
      if (hops % interval === 0) stepSize = s.accepted / s.hops > target ? stepSize / factor : stepSize * factor
      const u = rng.uniform([n + 1]).data
      let trial: Float64Array = Float64Array.from(s.current, (v, i) => v + stepSize * (2 * (u[i] ?? 0) - 1))
      if (box !== undefined) trial = clipInto(trial, box)
      const r = localSearch(obj, local, trial, localOptions, ctx)
      const w = Math.exp(
        Math.min(
          0,
          temperature === 0 ? (r.fun < s.currentValue ? 0 : -Infinity) : -(r.fun - s.currentValue) / temperature,
        ),
      )
      const accept = w >= (u[n] ?? 1) && Number.isFinite(r.fun)
      const improved = accept && r.fun < s.value
      const sinceImprovement = improved ? 0 : s.sinceImprovement + 1
      const stop = niterSuccess !== undefined && sinceImprovement > niterSuccess
      return {
        ...s,
        x: improved ? r.x : s.x,
        value: improved ? r.fun : s.value,
        current: accept ? r.x : s.current,
        currentValue: accept ? r.fun : s.currentValue,
        nfev: s.nfev + r.nfev,
        step: stepSize,
        stepSize,
        hops,
        accepted: s.accepted + (accept ? 1 : 0),
        sinceImprovement,
        lastAccepted: accept,
        ...(stop
          ? { stop: { status: 'converged' as const, message: 'success condition satisfied (niterSuccess)' } }
          : {}),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { current: s.currentValue, stepSize: s.stepSize, acceptRate: s.hops === 0 ? 0 : s.accepted / s.hops },
        tensors: { current: { role: 'state', value: s.current } },
        messages: [{ kind: 'hop', payload: { accepted: s.lastAccepted, value: s.currentValue } }],
      }),
  }
}

/**
 * Find a global minimum by basin hopping (scipy.optimize.basinhopping). `maxIterations` counts hops (default 100).
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Temperature, step size, adaptation, niterSuccess, local method and options, bounds, key or seed.
 * @returns The best minimum found.
 * @concept basin-hopping
 * @example
 * ```ts
 * import { basinHopping } from 'whitebox-engine/optim'
 * import { add, cos, mul, square, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * // cos(14.5 x - 0.3) + (x + 0.2) x has its global minimum near x = -0.195 (scipy's example)
 * const f = (x: unknown) => sum(add(cos(add(mul(x as Value, 14.5), -0.3)), mul(add(x as Value, 0.2), x as Value)))
 * const r = basinHopping(f, [1], { seed: 1, maxIterations: 30 })
 * Math.round((toFlat(r.x as Value)[0] ?? 0) * 1000) / 1000 // => -0.195
 * ```
 */
export function basinHopping(f: Objective, x0: Parameters, options: BasinHoppingOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  const { maxIterations, ...rest } = options
  // One extra step for the initial local minimisation.
  return runVector(basinHoppingAlgorithm(f, x0, rest), obj, { ...rest, maxIterations: (maxIterations ?? 100) + 1 }, 101)
}

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
      let moved = 0
      for (let k = 0; k < n; k++) moved += ((xbest[k] ?? 0) - (s.x[k] ?? 0)) ** 2
      return {
        x: xbest,
        value: ebest,
        grad: EMPTY,
        nfev,
        ngev: 0,
        nhev: 0,
        step: Math.sqrt(moved),
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
