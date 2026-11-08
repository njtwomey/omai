/**
 * The Nagel-Schreckenberg traffic model: cars on a ring road of cells, each with an integer speed, that accelerate by
 * one, brake to the gap ahead, dawdle with some probability and move. Jams form out of nothing and travel backwards
 * against the traffic. With no dawdling the flow has a closed form (`min(vmax rho, 1 - rho)`), which the stochastic
 * model sags below; the sweep over density is the fundamental diagram with its capacity drop.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, key as makeKey, permutation } from '../../random/index.ts'
import { type Algorithm, type Contract, contractSchema, defineContract, scalar, tensor } from '../../trace/index.ts'
import { codec, draws, positiveInt, unit } from './_core.ts'

/** Options of {@link nagelSchreckenberg}. */
export interface TrafficOptions {
  /** Cells in the ring (default 200). */
  readonly length?: number
  /** Cars as a share of the cells (default 0.2). Ignored if `cars` is given. */
  readonly density?: number
  /** Number of cars (overrides `density`). */
  readonly cars?: number
  /** The top speed, in cells per step (default 5). */
  readonly maxSpeed?: number
  /** The chance a onRoad car slows by one for no reason (default 0.3). */
  readonly dawdle?: number
}

/** The state of {@link nagelSchreckenberg}. */
export interface TrafficState {
  /** Per cell, -1 for empty or the speed of the car in it. */
  readonly cells: Int8Array
  /** Cells covered by all the cars in the last step, divided by the cells in the ring (cars passing a point per step). */
  readonly flow: number
  /** Mean speed of the cars after the last step. */
  readonly meanSpeed: number
  /** Cars standing still after the last step. */
  readonly jammed: number
}

/** The paths of {@link nagelSchreckenberg}. */
export type TrafficPath = 'road/cells' | 'road/flow' | 'road/meanSpeed' | 'road/jammed'

/** The log contract of {@link nagelSchreckenberg}. */
export const trafficContract: Contract<TrafficState, TrafficPath> = defineContract({
  'road/cells': tensor({ role: 'state', shape: ['length'], description: "-1 for an empty cell, else the car's speed" }),
  'road/flow': scalar({ description: 'cars passing a point per step (mean speed times density)' }),
  'road/meanSpeed': scalar({ description: 'the mean speed of the cars, cells per step' }),
  'road/jammed': scalar({ description: 'cars standing still' }),
})

/**
 * The flow of the deterministic model (no dawdling) once it has relaxed: `min(vmax rho, 1 - rho)`, rising with the
 * density while cars are free and falling when they are not.
 *
 * @param density - The share of cells holding a car.
 * @param maxSpeed - The top speed.
 * @returns The flow in cars per step.
 * @concept nagel-schreckenberg
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * [automata.deterministicFlow(0.1, 5), automata.deterministicFlow(0.5, 5)] // => [0.5, 0.5]
 * ```
 */
export function deterministicFlow(density: number, maxSpeed: number): number {
  return Math.min(maxSpeed * density, 1 - density)
}

/**
 * One update of the ring (all cars at once): accelerate, brake to the gap, dawdle, move.
 *
 * @param cells - The road, -1 empty or a speed.
 * @param maxSpeed - The top speed.
 * @param dawdle - The chance of slowing by one.
 * @param u - A uniform draw for every cell of the ring (the draw at a car's cell decides its dawdle).
 * @returns The road after the step.
 * @concept nagel-schreckenberg
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * // A lone car on a ring of 8 accelerates 0, 1, 2, 3...
 * Array.from(automata.trafficStep(Int8Array.of(0, -1, -1, -1, -1, -1, -1, -1), 5, 0, new Float64Array(8))) // => [-1, 1, -1, -1, -1, -1, -1, -1]
 * ```
 */
export function trafficStep(cells: Int8Array, maxSpeed: number, dawdle: number, u: ArrayLike<number>): Int8Array {
  const n = cells.length
  const out = new Int8Array(n).fill(-1)
  const at: number[] = []
  for (let i = 0; i < n; i++) if ((cells[i] ?? -1) >= 0) at.push(i)
  const m = at.length
  for (let k = 0; k < m; k++) {
    const i = at[k] as number
    const ahead = at[(k + 1) % m] as number
    const gap = m === 1 ? n - 1 : (((ahead - i - 1) % n) + n) % n
    let v = Math.min((cells[i] ?? 0) + 1, maxSpeed)
    v = Math.min(v, gap)
    if (v > 0 && (u[i] ?? 1) < dawdle) v -= 1
    out[(i + v) % n] = v
  }
  return out
}

/**
 * The Nagel-Schreckenberg ring road as a traced algorithm: one parallel update per step. The state is the road (and the
 * last step's flow, mean speed and jammed cars) and serialises; the dawdling draws come from the step's key.
 *
 * @param options - Length, density or car count, top speed and dawdling probability.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a bad length, speed, probability or more cars than cells.
 * @concept nagel-schreckenberg
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // Without dawdling, 10 cars on 100 cells reach top speed (5) and flow 10 x 5 / 100.
 * const s = runAlgorithm(automata.nagelSchreckenberg({ length: 100, cars: 10, dawdle: 0 }), { seed: 0, maxSteps: 300 }).state
 * s.flow // => 0.5
 * ```
 */
export function nagelSchreckenberg(options: TrafficOptions = {}): Algorithm<TrafficState, TrafficPath> {
  const length = options.length ?? 200
  const maxSpeed = options.maxSpeed ?? 5
  const dawdle = options.dawdle ?? 0.3
  positiveInt('nagelSchreckenberg', 'length', length)
  positiveInt('nagelSchreckenberg', 'maxSpeed', maxSpeed)
  unit('nagelSchreckenberg', 'dawdle', dawdle)
  const cars = options.cars ?? Math.round((options.density ?? 0.2) * length)
  if (!(Number.isInteger(cars) && cars >= 0 && cars <= length))
    throw new OmaiError('value', `nagelSchreckenberg: need 0 <= cars <= length, got ${cars}`)
  const measure = (cells: Int8Array): { flow: number; meanSpeed: number; jammed: number } => {
    let total = 0
    let n = 0
    let jammed = 0
    for (const v of cells) {
      if (v < 0) continue
      total += v
      n++
      if (v === 0) jammed++
    }
    return { flow: total / length, meanSpeed: n === 0 ? 0 : total / n, jammed }
  }
  return {
    name: 'dynamics/automata/nagel-schreckenberg',
    config: { length, cars, maxSpeed, dawdle },
    schema: contractSchema(trafficContract),
    init: (key) => {
      const cells = new Int8Array(length).fill(-1)
      const order = permutation(key, length).data
      for (let k = 0; k < cars; k++) cells[order[k] ?? 0] = 0
      return { cells, ...measure(cells) }
    },
    step: (s, ctx) => {
      const cells = trafficStep(s.cells, maxSpeed, dawdle, draws(foldIn(ctx.key, 0), length))
      return { cells, ...measure(cells) }
    },
    observe: (s) => ({
      metrics: { 'road/flow': s.flow, 'road/meanSpeed': s.meanSpeed, 'road/jammed': s.jammed },
      tensors: { 'road/cells': { role: 'state', value: s.cells, shape: [length] } },
    }),
    ...codec<TrafficState>(),
  }
}

/** The fundamental diagram: flow against density for a ring. */
export interface FundamentalDiagram {
  readonly density: Float64Array
  /** The mean flow over the measuring steps, per density. */
  readonly flow: Float64Array
  /** The mean speed over the measuring steps, per density. */
  readonly meanSpeed: Float64Array
}

/**
 * Sweep the density of a ring and measure the flow once the traffic has settled: the fundamental diagram, rising
 * linearly at free flow and falling in the jammed branch, lower than {@link deterministicFlow} when cars dawdle.
 *
 * @param options - Ring length, top speed, dawdling probability, the densities, steps to settle and to measure, seed.
 * @returns Density, mean flow and mean speed per density.
 * @concept nagel-schreckenberg
 * @example
 * ```ts
 * import { automata } from 'omai-engine/dynamics'
 * const d = automata.trafficDiagram({ length: 100, densities: [0.1, 0.5], dawdle: 0, settle: 200, measure: 20 })
 * Array.from(d.flow).map((v) => Math.round(v * 1e9) / 1e9) // => [0.5, 0.5]
 * ```
 */
export function trafficDiagram(options: {
  readonly densities: readonly number[]
  readonly length?: number
  readonly maxSpeed?: number
  readonly dawdle?: number
  readonly settle?: number
  readonly measure?: number
  readonly seed?: number
}): FundamentalDiagram {
  const length = options.length ?? 200
  const settle = options.settle ?? 300
  const measure = options.measure ?? 100
  const root = makeKey(options.seed ?? 0)
  const flow = new Float64Array(options.densities.length)
  const meanSpeed = new Float64Array(options.densities.length)
  const maxSpeed = options.maxSpeed ?? 5
  const dawdle = options.dawdle ?? 0.3
  options.densities.forEach((density, d) => {
    const k = foldIn(root, d)
    const cars = Math.round(density * length)
    let cells: Int8Array = new Int8Array(length).fill(-1)
    const order = permutation(foldIn(k, 0), length).data
    for (let c = 0; c < cars; c++) cells[order[c] ?? 0] = 0
    for (let t = 0; t < settle + measure; t++) {
      cells = trafficStep(cells, maxSpeed, dawdle, draws(foldIn(k, t + 1), length))
      if (t < settle) continue
      let total = 0
      let onRoad = 0
      for (const v of cells)
        if (v >= 0) {
          total += v
          onRoad++
        }
      flow[d] = (flow[d] ?? 0) + total / length / measure
      meanSpeed[d] = (meanSpeed[d] ?? 0) + (onRoad === 0 ? 0 : total / onRoad) / measure
    }
  })
  return { density: Float64Array.from(options.densities), flow, meanSpeed }
}
