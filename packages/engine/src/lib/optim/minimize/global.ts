import { WhiteboxError } from '../../base/index.ts'
import { metricFunction } from '../../numerics/geometry/index.ts'
import { Generator } from '../../random/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { asPairs, type BoundArrays, boundArrays, clipInto } from './_bounds.ts'
import { defaultLocal, localSearch } from './_local.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import type { Bounds } from './objective.ts'
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
  const euclid = metricFunction('euclidean', n)
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
            moved = euclid(proposal, 0, best, 0)
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
