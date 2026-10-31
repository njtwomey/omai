/**
 * Newton fractals: Newton's method `z -> z - p(z)/p'(z)` for a polynomial given by its roots, run from every point of a
 * window of the complex plane and coloured by the root each point reaches and how fast. The basins of the roots have
 * a common, fractal boundary: every point of it is on the boundary of all of them. The Newton step, the basin of one
 * point, and the whole picture as a traced algorithm with one Newton step over every pixel per step.
 */

import { MdmlError } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { type ComplexViewport, pixelGrid } from './escape-time.ts'

/**
 * The n-th roots of unity, the roots of `z^n - 1`: the classic Newton fractal for n = 3 has three basins that meet at
 * every boundary point.
 *
 * @param n - How many roots.
 * @returns The roots, `[re, im]`, counter-clockwise from 1.
 * @concept newton-fractals
 * @example
 * ```ts
 * import { fractals } from 'mdml-engine/dynamics'
 * fractals.rootsOfUnity(4).map(([re, im]) => [Math.round(re), Math.round(im)]) // => [[1, 0], [0, 1], [-1, 0], [0, -1]]
 * ```
 */
export function rootsOfUnity(n: number): [number, number][] {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', `rootsOfUnity: need a positive integer, got ${n}`)
  return Array.from({ length: n }, (_, k) => [Math.cos((2 * Math.PI * k) / n), Math.sin((2 * Math.PI * k) / n)])
}

/**
 * One Newton step for the polynomial with the given roots: `p/p'` is `1 / sum_k 1 / (z - r_k)`, so
 * `z <- z - 1 / sum_k 1 / (z - r_k)`.
 *
 * @param z - The point, `[re, im]`.
 * @param roots - The roots of the polynomial.
 * @returns The next point (the point itself if it is a root).
 * @concept newton-fractals
 * @example
 * ```ts
 * import { fractals } from 'mdml-engine/dynamics'
 * // For z^2 - 1, from 2: 2 - (4 - 1) / 4 = 1.25.
 * fractals.newtonStep([2, 0], [[1, 0], [-1, 0]]) // => [1.25, 0]
 * ```
 */
export function newtonStep(
  z: readonly [number, number],
  roots: readonly (readonly [number, number])[],
): [number, number] {
  let sr = 0
  let si = 0
  for (const [rr, ri] of roots) {
    const dr = z[0] - rr
    const di = z[1] - ri
    const m = dr * dr + di * di
    if (m === 0) return [z[0], z[1]]
    sr += dr / m
    si -= di / m
  }
  const m = sr * sr + si * si
  if (m === 0) return [z[0], z[1]]
  // z - 1 / s, with 1 / s = conj(s) / |s|^2.
  return [z[0] - sr / m, z[1] + si / m]
}

/** Options of {@link newtonFractal}. */
export interface NewtonOptions {
  /** The roots of the polynomial (default the cube roots of unity). */
  readonly roots?: readonly (readonly [number, number])[]
  /** The window (default centred at the origin, width 4, 200 by 200 pixels). */
  readonly viewport?: ComplexViewport
  /** A point has reached a root when it is within this distance of it (default 1e-6). */
  readonly tolerance?: number
}

/** The state of {@link newtonFractal}. */
export interface NewtonState {
  readonly zr: Float64Array
  readonly zi: Float64Array
  /** Per pixel: the index of the root it reached, -1 while it has not reached one. */
  readonly root: Int8Array
  /** Per pixel: the step at which it reached its root, 0 while it has not. */
  readonly steps: Int32Array
  readonly iteration: number
  readonly converged: number
}

/** The paths of {@link newtonFractal}. */
export type NewtonPath = 'basin/root' | 'basin/steps' | 'basin/iteration' | 'basin/convergedShare'

/** The log contract of {@link newtonFractal}. */
export const newtonContract: Contract<NewtonState, NewtonPath> = defineContract({
  'basin/root': tensor({
    role: 'state',
    shape: ['rows', 'columns'],
    description: 'the index of the root each pixel reached, -1 if none yet',
  }),
  'basin/steps': tensor({
    role: 'state',
    shape: ['rows', 'columns'],
    description: 'the Newton step at which each pixel reached its root',
  }),
  'basin/iteration': scalar({ description: 'Newton steps done' }),
  'basin/convergedShare': scalar({ description: 'the share of pixels within the tolerance of a root' }),
})

/**
 * Newton's method over a whole window as a traced algorithm: each step applies one Newton step to every pixel that has
 * not yet reached a root, and marks the pixels that now lie within the tolerance of one. The state serialises.
 *
 * @param options - The roots, the window and the tolerance.
 * @returns The algorithm.
 * @throws MdmlError with code `value` for fewer than two roots (Newton would be trivial) or more than 127.
 * @concept newton-fractals
 * @example
 * ```ts
 * import { fractals } from 'mdml-engine/dynamics'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * // From 2 on the real axis, Newton on z^3 - 1 walks to the root 1 (index 0).
 * const alg = fractals.newtonFractal({ viewport: { centre: [2, 0], width: 0.1, pixels: [1, 1] } })
 * Array.from(runAlgorithm(alg, { seed: 0, maxSteps: 30 }).state.root) // => [0]
 * ```
 */
export function newtonFractal(options: NewtonOptions = {}): Algorithm<NewtonState, NewtonPath> {
  const roots = options.roots ?? rootsOfUnity(3)
  const tol = options.tolerance ?? 1e-6
  if (roots.length < 2 || roots.length > 127)
    throw new MdmlError('value', 'newtonFractal: need between 2 and 127 roots')
  const grid = pixelGrid({ centre: [0, 0], width: 4, ...options.viewport })
  const n = grid.re.length
  return {
    name: 'dynamics/fractals/newton',
    config: {
      roots: roots.map((r) => [...r]),
      tolerance: tol,
      centre: [...(options.viewport?.centre ?? [0, 0])],
      width: options.viewport?.width ?? 4,
      pixels: [grid.columns, grid.rows],
    },
    schema: contractSchema(newtonContract),
    init: () => ({
      zr: Float64Array.from(grid.re),
      zi: Float64Array.from(grid.im),
      root: new Int8Array(n).fill(-1),
      steps: new Int32Array(n),
      iteration: 0,
      converged: 0,
    }),
    step: (s) => {
      const zr = Float64Array.from(s.zr)
      const zi = Float64Array.from(s.zi)
      const root = Int8Array.from(s.root)
      const steps = Int32Array.from(s.steps)
      const iteration = s.iteration + 1
      let converged = s.converged
      for (let i = 0; i < n; i++) {
        if ((root[i] ?? -1) >= 0) continue
        const [a, b] = newtonStep([zr[i] ?? 0, zi[i] ?? 0], roots)
        zr[i] = a
        zi[i] = b
        for (let k = 0; k < roots.length; k++) {
          const r = roots[k] as readonly [number, number]
          if (Math.hypot(a - r[0], b - r[1]) < tol) {
            root[i] = k
            steps[i] = iteration
            converged++
            break
          }
        }
      }
      return { zr, zi, root, steps, iteration, converged }
    },
    observe: (s) => ({
      metrics: { 'basin/iteration': s.iteration, 'basin/convergedShare': s.converged / n },
      tensors: {
        'basin/root': { role: 'state', value: s.root, shape: [grid.rows, grid.columns] },
        'basin/steps': { role: 'state', value: s.steps, shape: [grid.rows, grid.columns] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NewtonState,
  }
}
