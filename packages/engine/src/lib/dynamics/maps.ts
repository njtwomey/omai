/**
 * Discrete-time dynamical systems (iterated maps) and their chaos: the logistic, tent, Henon, standard and circle maps,
 * orbits as traced algorithms, cobweb and bifurcation diagrams, and Lyapunov exponents by Benettin's QR method with
 * Jacobians from forward-mode autodiff.
 */

import { jacfwd } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { qr } from '../numerics/linalg/index.ts'
import {
  abs,
  add,
  matmul,
  mul,
  primalOf,
  reshape,
  sin,
  sliceAxis,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { floatState, fromNumbers, numbersOf } from './ode/_vec.ts'

/** A map x_{n+1} = F(x_n) on states of any shape, written with operations (so it can be differentiated). */
export type IteratedMap = (x: Value) => Value

const at = (x: Value, i: number): Value => sum(sliceAxis(reshape(x, [-1]), 0, i, i + 1))

/**
 * The logistic map x -> r x (1 - x).
 *
 * @param r - The parameter (chaotic near 4).
 * @returns The map on a state `[1]` (or any shape, elementwise).
 * @concept logistic-map
 * @example
 * ```ts
 * import { logisticMap } from 'whitebox-engine/dynamics'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(logisticMap(4)(tensor([0.5]))) // => [1]
 * ```
 */
export function logisticMap(r: number): IteratedMap {
  return (x) => mul(mul(x, sub(1, x)), r)
}

/**
 * The tent map x -> mu min(x, 1 - x).
 *
 * @param mu - The slope (2 gives the fully chaotic tent map).
 * @returns The map.
 * @example
 * ```ts
 * import { tentMap } from 'whitebox-engine/dynamics'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(tentMap(2)(tensor([0.25]))) // => [0.5]
 * ```
 */
export function tentMap(mu: number): IteratedMap {
  return (x) => {
    // min(x, 1 - x) = 1/2 - |x - 1/2|
    return mul(sub(0.5, abs(sub(x, 0.5))), mu)
  }
}

/**
 * The Henon map (x, y) -> (1 - a x^2 + y, b x).
 *
 * @param a - Default 1.4.
 * @param b - Default 0.3.
 * @returns The map on states `[2]`.
 * @concept henon-map
 * @example
 * ```ts
 * import { henonMap } from 'whitebox-engine/dynamics'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(henonMap()(tensor([0, 0]))) // => [1, 0]
 * ```
 */
export function henonMap(a = 1.4, b = 0.3): IteratedMap {
  return (s) => {
    const x = at(s, 0)
    const y = at(s, 1)
    return stack([add(sub(1, mul(mul(x, x), a)), y), mul(x, b)])
  }
}

/**
 * Chirikov's standard map p -> p + K sin(theta), theta -> theta + p (unwrapped; reduce mod 2 pi for plots).
 *
 * @param k - The kick strength K.
 * @returns The map on states `[theta, p]`.
 * @concept standard-map
 * @example
 * ```ts
 * import { standardMap } from 'whitebox-engine/dynamics'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(standardMap(1)(tensor([0, 1]))) // => [1, 1]
 * ```
 */
export function standardMap(k: number): IteratedMap {
  return (s) => {
    const theta = at(s, 0)
    const p = add(at(s, 1), mul(sin(theta), k))
    return stack([add(theta, p), p])
  }
}

/**
 * The sine circle map theta -> theta + Omega - (K / 2 pi) sin(2 pi theta) (unwrapped).
 *
 * @param omega - The bare winding number.
 * @param k - The nonlinearity (the map is invertible for K <= 1).
 * @returns The map.
 * @example
 * ```ts
 * import { circleMap } from 'whitebox-engine/dynamics'
 * import { tensor, toFlat } from 'whitebox-engine/tensor'
 * toFlat(circleMap(0.5, 0)(tensor([0.25]))) // => [0.75]
 * ```
 */
export function circleMap(omega: number, k: number): IteratedMap {
  return (x) => sub(add(x, omega), mul(sin(mul(x, 2 * Math.PI)), k / (2 * Math.PI)))
}

/** The state of {@link orbitAlgorithm}. */
export interface OrbitState {
  readonly x: Value
  readonly points: Float64Array[]
  readonly diverged: boolean
}

/**
 * Iterating a map as a traced algorithm (the state as a tensor per step); points after the transient are kept.
 *
 * @param map - The map.
 * @param x0 - The initial state.
 * @param options - How many points to keep and how many transient iterations to drop.
 * @returns The algorithm and the number of steps to run.
 * @example
 * ```ts
 * import { logisticMap, orbitAlgorithm } from 'whitebox-engine/dynamics'
 * orbitAlgorithm(logisticMap(3), [0.2], { n: 10, transient: 5 }).steps // => 15
 * ```
 */
export function orbitAlgorithm(
  map: IteratedMap,
  x0: TensorLike,
  options: { readonly n: number; readonly transient?: number },
): { readonly algorithm: Algorithm<OrbitState>; readonly steps: number } {
  const start = floatState(x0)
  const transient = options.transient ?? 0
  const steps = transient + options.n
  const algorithm: Algorithm<OrbitState> = {
    name: 'orbit',
    config: { n: options.n, transient },
    schema: channels({ tensors: { x: { role: 'state', description: 'the iterate' } } }),
    init: () => ({ x: start, points: transient === 0 ? [Float64Array.from(numbersOf(start))] : [], diverged: false }),
    step: (s, ctx) => {
      const x = map(s.x)
      const v = numbersOf(x)
      const diverged = !v.every(Number.isFinite)
      if (ctx.t + 1 >= transient && s.points.length < options.n) s.points.push(Float64Array.from(v))
      return { x, points: s.points, diverged }
    },
    done: (s) => (s.diverged ? 'diverged' : undefined),
    observe: (s) => ({ tensors: { x: { role: 'state', value: numbersOf(s.x), shape: s.x.shape } } }),
  }
  return { algorithm, steps }
}

/**
 * The orbit of a map: n points after an optional transient, stacked `[n, ...stateShape]`.
 *
 * @param map - The map.
 * @param x0 - The initial state.
 * @param options - Count, transient and tracing.
 * @returns The orbit.
 * @concept orbit
 * @example
 * ```ts
 * import { logisticMap, orbit } from 'whitebox-engine/dynamics'
 * import { toFlat } from 'whitebox-engine/tensor'
 * // r = 2: the orbit settles on the fixed point 1/2.
 * Math.abs((toFlat(orbit(logisticMap(2), [0.1], { n: 3, transient: 100 })).at(-1) ?? 0) - 0.5) < 1e-12 // => true
 * ```
 */
export function orbit(
  map: IteratedMap,
  x0: TensorLike,
  options: { readonly n: number; readonly transient?: number } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>>,
): Tensor {
  const { algorithm, steps } = orbitAlgorithm(map, x0, options)
  const run = runAlgorithm(algorithm, { maxSteps: steps, ...options })
  const shape = floatState(x0).shape
  const pts = run.state.points
  const size = pts[0]?.length ?? 0
  const flat = new Float64Array(pts.length * size)
  pts.forEach((p, i) => {
    flat.set(p, i * size)
  })
  return fromNumbers(flat, [pts.length, ...shape])
}

/**
 * The cobweb diagram of a 1-D map: the polyline (x0, x0), (x0, x1), (x1, x1), (x1, x2), ... bouncing between the graph
 * and the diagonal.
 *
 * @param map - A map of one variable.
 * @param x0 - The start.
 * @param n - Iterations.
 * @returns The polyline vertices.
 * @concept cobweb-plot
 * @example
 * ```ts
 * import { cobweb, logisticMap } from 'whitebox-engine/dynamics'
 * cobweb(logisticMap(4), 0.5, 1) // => [[0.5, 0.5], [0.5, 1], [1, 1]]
 * ```
 */
export function cobweb(map: IteratedMap, x0: number, n: number): [number, number][] {
  const out: [number, number][] = [[x0, x0]]
  let x = x0
  for (let k = 0; k < n; k++) {
    const y = numbersOf(map(tensor([x])))[0] ?? Number.NaN
    out.push([x, y], [y, y])
    x = y
    if (!Number.isFinite(x)) break
  }
  return out
}

/**
 * A bifurcation diagram: for each parameter, the points visited after a transient (period-doubling cascades and
 * chaotic bands appear as the parameter varies).
 *
 * @param family - The map for a parameter.
 * @param parameters - The parameters.
 * @param x0 - The initial state for each parameter.
 * @param options - Transient (default 500) and points kept per parameter (default 100); only the first coordinate is
 * kept.
 * @returns Parallel arrays of parameter and point.
 * @concept bifurcation-diagram
 * @example
 * ```ts
 * import { bifurcationDiagram, logisticMap } from 'whitebox-engine/dynamics'
 * // r = 3.2 has a period-2 orbit: two distinct values.
 * const d = bifurcationDiagram(logisticMap, [3.2], [0.3], { keep: 8 })
 * new Set(d.x.map((v) => v.toFixed(6))).size // => 2
 * ```
 */
export function bifurcationDiagram(
  family: (parameter: number) => IteratedMap,
  parameters: readonly number[],
  x0: TensorLike,
  options: { readonly transient?: number; readonly keep?: number } = {},
): { readonly parameter: number[]; readonly x: number[] } {
  const transient = options.transient ?? 500
  const keep = options.keep ?? 100
  const parameter: number[] = []
  const x: number[] = []
  for (const p of parameters) {
    const o = orbit(family(p), x0, { n: keep, transient })
    const size = o.data.length / Math.max(1, o.shape[0] ?? 1)
    for (let i = 0; i < (o.shape[0] ?? 0); i++) {
      parameter.push(p)
      x.push(o.data[i * size] ?? Number.NaN)
    }
  }
  return { parameter, x }
}

/**
 * The Lyapunov exponents of a map by Benettin's QR method: tangent vectors are pushed through the Jacobian (by
 * forward-mode autodiff) and re-orthonormalised each step; the exponents are the averaged logs of the diagonal of R.
 * For a 1-D map this is the mean of log|F'(x_n)| (ln 2 for the logistic map at r = 4 and the tent map at mu = 2).
 *
 * @param map - The map.
 * @param x0 - The initial state.
 * @param options - Iterations (default 10000), transient (default 100) and how many exponents (default all).
 * @returns The exponents, largest first.
 * @concept lyapunov-exponent
 * @example
 * ```ts
 * import { lyapunovExponents, logisticMap } from 'whitebox-engine/dynamics'
 * Math.abs((lyapunovExponents(logisticMap(4), [0.3], { n: 4000 })[0] ?? 0) - Math.LN2) < 0.05 // => true
 * ```
 */
export function lyapunovExponents(
  map: IteratedMap,
  x0: TensorLike,
  options: { readonly n?: number; readonly transient?: number; readonly count?: number } = {},
): number[] {
  const n = options.n ?? 10000
  const transient = options.transient ?? 100
  let x = floatState(x0)
  const shape = x.shape
  const d = numbersOf(x).length
  const k = Math.min(d, options.count ?? d)
  const flat = (v: Value): Value => reshape(map(reshape(v, shape)), [d])
  for (let i = 0; i < transient; i++) x = map(x)
  let Q = fromNumbers(
    Float64Array.from({ length: d * k }, (_, idx) => (Math.floor(idx / k) === idx % k ? 1 : 0)),
    [d, k],
  )
  const sums = new Float64Array(k)
  for (let i = 0; i < n; i++) {
    const J = primalOf(reshape(jacfwd(flat)(reshape(x, [d])), [d, d]))
    const JQ = primalOf(matmul(J, Q))
    const { q, r } = qr(JQ)
    const R = primalOf(r)
    const qm = primalOf(q)
    // Fix signs so the diagonal of R is positive.
    const signs = Float64Array.from({ length: k }, (_, j) => ((R.data[j * k + j] ?? 1) < 0 ? -1 : 1))
    for (let j = 0; j < k; j++) sums[j] = (sums[j] ?? 0) + Math.log(Math.abs(R.data[j * k + j] ?? 1))
    Q = fromNumbers(
      Float64Array.from(qm.data.slice(0, d * k), (v, idx) => v * (signs[idx % k] ?? 1)),
      [d, k],
    )
    x = map(x)
    if (!numbersOf(x).every(Number.isFinite)) {
      throw new WhiteboxError('value', 'lyapunovExponents: the orbit diverged')
    }
  }
  return Array.from(sums, (s) => s / n)
}
