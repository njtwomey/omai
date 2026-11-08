/**
 * Flocking: Reynolds' boids, agents on a torus that each steer by three local rules (separation from close neighbours,
 * alignment with the neighbours' mean velocity, cohesion towards their centre), as a traced algorithm with one step per
 * time step, and the polarisation that measures how aligned the flock is.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { foldIn, normal, uniform } from '../random/index.ts'
import { type Algorithm, channels } from '../trace/index.ts'

/** Options of {@link boidsAlgorithm}. */
export interface BoidsOptions {
  /** How many boids (default 150). */
  readonly count?: number
  /** The world: a torus `width` by `height` (default 100 by 100). */
  readonly width?: number
  readonly height?: number
  /** How far a boid sees its neighbours (default 10). */
  readonly radius?: number
  /** Closer than this, a neighbour is pushed away (default 3). */
  readonly personalSpace?: number
  /** The weights of the three rules (defaults 1.5, 1 and 1). */
  readonly separation?: number
  readonly alignment?: number
  readonly cohesion?: number
  /** Speed limits (defaults 0.5 and 2 per unit time). */
  readonly minSpeed?: number
  readonly maxSpeed?: number
  /** The standard deviation of the random steering added each step (default 0.05). */
  readonly noise?: number
}

/** The flock: every boid's position and velocity. */
export interface BoidsState {
  readonly x: Float64Array
  readonly y: Float64Array
  readonly vx: Float64Array
  readonly vy: Float64Array
}

/**
 * The polarisation of a flock: the length of the mean unit velocity, 1 when every boid flies the same way and near 0
 * when headings are random.
 *
 * @param s - The flock.
 * @returns The polarisation, in [0, 1].
 * @concept flocking
 * @example
 * ```ts
 * import { polarisation } from 'omai-engine/dynamics'
 * const one = Float64Array.of(0, 0)
 * polarisation({ x: one, y: one, vx: Float64Array.of(1, 2), vy: Float64Array.of(0, 0) }) // => 1
 * ```
 */
export function polarisation(s: BoidsState): number {
  let ux = 0
  let uy = 0
  const n = s.vx.length
  for (let i = 0; i < n; i++) {
    const v = Math.hypot(s.vx[i] ?? 0, s.vy[i] ?? 0) || 1
    ux += (s.vx[i] ?? 0) / v
    uy += (s.vy[i] ?? 0) / v
  }
  return n === 0 ? 0 : Math.hypot(ux, uy) / n
}

/**
 * Reynolds' boids on a torus as a traced algorithm. Each step, every boid looks at the boids within `radius` (distances
 * wrap round the torus) and adds to its velocity: separation (away from neighbours closer than `personalSpace`,
 * inversely to distance), alignment (towards the neighbours' mean velocity) and cohesion (towards their centre), each
 * weighted, plus a little keyed noise; then its speed is clamped to [minSpeed, maxSpeed] and it moves. Every random
 * number comes from the keys, so a run is a pure function of its key.
 *
 * @param options - Flock size, world, sight, the three weights, speed limits and noise.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a non-positive count, world, radius or speed range.
 * @concept flocking
 * @example
 * ```ts
 * import { boidsAlgorithm } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(boidsAlgorithm({ count: 20 }), { seed: 0, maxSteps: 5 }).state.x.length // => 20
 * ```
 */
export function boidsAlgorithm(options: BoidsOptions = {}): Algorithm<BoidsState> {
  const n = options.count ?? 150
  const w = options.width ?? 100
  const h = options.height ?? 100
  const radius = options.radius ?? 10
  const personal = options.personalSpace ?? 3
  const ws = options.separation ?? 1.5
  const wa = options.alignment ?? 1
  const wc = options.cohesion ?? 1
  const lo = options.minSpeed ?? 0.5
  const hi = options.maxSpeed ?? 2
  const noise = options.noise ?? 0.05
  if (!(Number.isInteger(n) && n >= 1 && w > 0 && h > 0 && radius > 0 && hi > 0 && lo >= 0 && lo <= hi))
    throw new OmaiError('value', 'boids: need a positive count, world, radius and a speed range 0 <= min <= max')
  // The shortest offset from a to b round the torus.
  const wrap = (d: number, size: number): number => d - size * Math.round(d / size)
  return {
    name: 'dynamics/boids',
    config: { count: n, width: w, height: h, radius, personal, ws, wa, wc, lo, hi, noise } as Json,
    schema: channels({
      metrics: { polarisation: { description: 'length of the mean unit velocity: 1 when the flock is aligned' } },
    }),
    init: (key) => {
      const u = uniform(key, [3 * n]).data
      const x = new Float64Array(n)
      const y = new Float64Array(n)
      const vx = new Float64Array(n)
      const vy = new Float64Array(n)
      const speed = (lo + hi) / 2
      for (let i = 0; i < n; i++) {
        x[i] = w * (u[3 * i] ?? 0)
        y[i] = h * (u[3 * i + 1] ?? 0)
        const a = 2 * Math.PI * (u[3 * i + 2] ?? 0)
        vx[i] = speed * Math.cos(a)
        vy[i] = speed * Math.sin(a)
      }
      return { x, y, vx, vy }
    },
    step: (s, ctx) => {
      const z = noise > 0 ? normal(foldIn(ctx.key, 0), [2 * n]).data : undefined
      const x = new Float64Array(n)
      const y = new Float64Array(n)
      const vx = new Float64Array(n)
      const vy = new Float64Array(n)
      const r2 = radius * radius
      for (let i = 0; i < n; i++) {
        const xi = s.x[i] ?? 0
        const yi = s.y[i] ?? 0
        let count = 0
        let cx = 0
        let cy = 0
        let ax = 0
        let ay = 0
        let sx = 0
        let sy = 0
        for (let j = 0; j < n; j++) {
          if (j === i) continue
          const dx = wrap((s.x[j] ?? 0) - xi, w)
          const dy = wrap((s.y[j] ?? 0) - yi, h)
          const d2 = dx * dx + dy * dy
          if (d2 > r2) continue
          count++
          cx += dx
          cy += dy
          ax += s.vx[j] ?? 0
          ay += s.vy[j] ?? 0
          if (d2 < personal * personal && d2 > 0) {
            sx -= dx / d2
            sy -= dy / d2
          }
        }
        let ux = s.vx[i] ?? 0
        let uy = s.vy[i] ?? 0
        if (count > 0) {
          // Each rule is a small steering change: towards the neighbours' mean velocity and centre, away from the close.
          ux += 0.05 * wa * (ax / count - ux) + 0.01 * wc * (cx / count) + 0.5 * ws * sx
          uy += 0.05 * wa * (ay / count - uy) + 0.01 * wc * (cy / count) + 0.5 * ws * sy
        }
        ux += noise * (z?.[2 * i] ?? 0)
        uy += noise * (z?.[2 * i + 1] ?? 0)
        const v = Math.hypot(ux, uy)
        const clamped = Math.min(hi, Math.max(lo, v))
        if (v > 0) {
          ux *= clamped / v
          uy *= clamped / v
        } else ux = clamped
        vx[i] = ux
        vy[i] = uy
        x[i] = (((xi + ux) % w) + w) % w
        y[i] = (((yi + uy) % h) + h) % h
      }
      return { x, y, vx, vy }
    },
    observe: (s) => ({ metrics: { polarisation: polarisation(s) } }),
  }
}
