/**
 * Active contours (snakes; Kass, Witkin and Terzopoulos 1988) as a traced algorithm: a closed or open polygon moves
 * under its internal elasticity (alpha) and rigidity (beta) and the external force of the image's edge or line
 * energy, by semi-implicit Euler steps (gamma (x_{t+1} - x_t) = -A x_{t+1} + f(x_t)), each solved with the same
 * pentadiagonal matrix. Contours are (x, y) points.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { image } from '../signal/index.ts'
import { primalOf, type TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { greyOf, type PointsLike, pointsOf } from './_image.ts'
import { invOf } from './_refine.ts'

/** Options of {@link activeContourAlgorithm}. */
export interface ActiveContourOptions {
  /** Elasticity (default 0.01) and rigidity (default 0.1). */
  readonly alpha?: number
  readonly beta?: number
  /** The step's inverse (viscosity, default 0.01). */
  readonly gamma?: number
  /** The weights of the image intensity (line) and gradient magnitude (edge) energies (defaults 0 and 1). */
  readonly wLine?: number
  readonly wEdge?: number
  /** Gaussian smoothing of the image before the energies (default 1). */
  readonly sigma?: number
  /** The largest move of a point per step (default 1 pixel). */
  readonly maxMove?: number
  /** Closed (periodic) or free ends (default closed). */
  readonly closed?: boolean
}

/** The state of {@link activeContourAlgorithm}: the contour and its last largest move. */
export interface ActiveContourState {
  readonly points: [number, number][]
  readonly moved: number
}

/**
 * An active contour as an algorithm: the external energy E = wLine I + wEdge |grad I| of the smoothed image (the
 * snake is attracted to its minima, so for dark-on-light edges use a positive wEdge), the force -grad E sampled
 * bilinearly at the points, and each step x <- (A + gamma I)^-1 (gamma x + f) with moves clipped to maxMove.
 *
 * @param x - Grey image `[H, W]`.
 * @param init - The starting contour `[n, 2]` as (x, y), n >= 3.
 * @param options - Elasticity, rigidity, step, energy weights, smoothing, move clip and closure.
 * @returns The algorithm (stopping when no point moves more than 0.01 pixels).
 * @concept active-contours
 * @example
 * ```ts
 * import { activeContourAlgorithm } from 'mdml-engine/vision'
 * activeContourAlgorithm([[0, 1], [1, 0]], [[0, 0], [1, 0], [1, 1]]).name // => 'active-contour'
 * ```
 */
export function activeContourAlgorithm(
  x: TensorLike,
  init: PointsLike,
  options: ActiveContourOptions = {},
): Algorithm<ActiveContourState> {
  const r = greyOf(x, 'activeContour')
  const start = pointsOf(init, 'activeContour')
  const n = start.length
  if (n < 3) throw new MdmlError('value', 'activeContour: at least three points are needed')
  const alpha = options.alpha ?? 0.01
  const beta = options.beta ?? 0.1
  const gamma = options.gamma ?? 0.01
  const closed = options.closed ?? true
  const maxMove = options.maxMove ?? 1
  const sm = Float64Array.from(
    primalOf(
      image.gaussianFilter(
        [...Array.from({ length: r.h }, (_, i) => Array.from(r.data.subarray(i * r.w, (i + 1) * r.w)))],
        options.sigma ?? 1,
        { mode: 'nearest' },
      ),
    ).data,
    Number,
  )
  const plane = (y: number, xx: number): number =>
    sm[Math.min(r.h - 1, Math.max(0, y)) * r.w + Math.min(r.w - 1, Math.max(0, xx))] ?? 0
  const E = new Float64Array(r.h * r.w)
  const wl = options.wLine ?? 0
  const we = options.wEdge ?? 1
  for (let y = 0; y < r.h; y++)
    for (let xx = 0; xx < r.w; xx++) {
      const gx = (plane(y, xx + 1) - plane(y, xx - 1)) / 2
      const gy = (plane(y + 1, xx) - plane(y - 1, xx)) / 2
      // Edges attract: the energy falls with the gradient magnitude.
      E[y * r.w + xx] = wl * plane(y, xx) - we * Math.hypot(gx, gy)
    }
  const e = (y: number, xx: number): number =>
    E[Math.min(r.h - 1, Math.max(0, y)) * r.w + Math.min(r.w - 1, Math.max(0, xx))] ?? 0
  const sample = (y: number, xx: number): number => {
    const y0 = Math.floor(y)
    const x0 = Math.floor(xx)
    const fy = y - y0
    const fx = xx - x0
    return (
      (1 - fy) * ((1 - fx) * e(y0, x0) + fx * e(y0, x0 + 1)) + fy * ((1 - fx) * e(y0 + 1, x0) + fx * e(y0 + 1, x0 + 1))
    )
  }
  // The internal-force matrix: alpha times the second difference and beta times the fourth.
  const A = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  const coef = [beta, -alpha - 4 * beta, 2 * alpha + 6 * beta, -alpha - 4 * beta, beta]
  for (let i = 0; i < n; i++)
    for (let k = -2; k <= 2; k++) {
      let j = i + k
      if (closed) j = (j + n) % n
      else j = Math.min(n - 1, Math.max(0, j))
      ;(A[i] as number[])[j] = ((A[i] as number[])[j] ?? 0) + (coef[k + 2] ?? 0)
    }
  const inv = invOf(A.map((row, i) => row.map((v, j) => v + (i === j ? gamma : 0))))
  return {
    name: 'active-contour',
    config: { alpha, beta, gamma, closed, points: n } as Json,
    schema: channels({
      metrics: { moved: { description: 'largest point move of the step', direction: 'lower' } },
      tensors: { points: { role: 'state', description: 'the contour (x, y)', shape: ['n', 2] } },
    }),
    init: () => ({ points: start.map(([a, b]) => [a, b] as [number, number]), moved: Number.POSITIVE_INFINITY }),
    step: (s) => {
      const fx = s.points.map(([px, py]) => -(sample(py, px + 0.5) - sample(py, px - 0.5)))
      const fy = s.points.map(([px, py]) => -(sample(py + 0.5, px) - sample(py - 0.5, px)))
      let moved = 0
      const next = s.points.map(([px, py], i) => {
        let nx = 0
        let ny = 0
        for (let j = 0; j < n; j++) {
          const [qx, qy] = s.points[j] as [number, number]
          const w = inv[i]?.[j] ?? 0
          nx += w * (gamma * qx + (fx[j] ?? 0))
          ny += w * (gamma * qy + (fy[j] ?? 0))
        }
        const dx = Math.max(-maxMove, Math.min(maxMove, nx - px))
        const dy = Math.max(-maxMove, Math.min(maxMove, ny - py))
        moved = Math.max(moved, Math.hypot(dx, dy))
        return [Math.min(r.w - 1, Math.max(0, px + dx)), Math.min(r.h - 1, Math.max(0, py + dy))] as [number, number]
      })
      return { points: next, moved }
    },
    done: (s) => (s.moved < 0.01 ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { moved: Number.isFinite(s.moved) ? s.moved : -1 },
      tensors: { points: { role: 'state', value: s.points.flat(), shape: [n, 2] } },
    }),
  }
}

/**
 * Fit an active contour ({@link activeContourAlgorithm} run to convergence or maxIter steps).
 *
 * @param x - Grey image `[H, W]`.
 * @param init - The starting contour (x, y).
 * @param options - The snake's options, iterations (default 500) and tracing.
 * @returns The final contour.
 * @concept active-contours
 * @example
 * ```ts
 * import { activeContour } from 'mdml-engine/vision'
 * const img = Array.from({ length: 32 }, (_, i) => Array.from({ length: 32 }, (_, j) => ((i - 16) ** 2 + (j - 16) ** 2 < 64 ? 1 : 0)))
 * const init = Array.from({ length: 20 }, (_, k) => [16 + 12 * Math.cos((k * Math.PI) / 10), 16 + 12 * Math.sin((k * Math.PI) / 10)])
 * const r = activeContour(img, init, { sigma: 1, gamma: 0.05 }).map(([x, y]) => Math.hypot(x - 16, y - 16))
 * Math.max(...r) < 11 // => true
 * ```
 */
export function activeContour(
  x: TensorLike,
  init: PointsLike,
  options: ActiveContourOptions & { readonly maxIter?: number } & Partial<
      Pick<RunOptions, 'tracer' | 'policy' | 'parent'>
    > = {},
): [number, number][] {
  return runAlgorithm(activeContourAlgorithm(x, init, options), {
    maxSteps: options.maxIter ?? 500,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  }).state.points
}
