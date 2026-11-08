/**
 * An active contour watched, as a method of nothing but its configuration: a snake started as a rectangle just inside
 * the frame of a picture of one shape, each step pulled in by its elasticity and rigidity and held by the shape's
 * edges, through the library's algorithm. The contour is on the tape with its largest move, its area and its overlap
 * with the shape's true mask, and the picture with the contour over it is a probe.
 */

import type { DecisionSurface } from '../../lib/exposition/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  tensor,
} from '../../lib/trace/index.ts'
import { type ActiveContourState, activeContourAlgorithm } from '../../lib/vision/index.ts'
import { picture } from './_picture.ts'
import { grid, type ShapeScene, shape } from './_scene.ts'

/** A rectangle `inset` pixels inside a square frame, as n points (x, y) clockwise from the top left. */
function frameContour(size: number, inset: number, n: number): [number, number][] {
  const lo = inset
  const hi = size - 1 - inset
  const side = hi - lo
  const total = 4 * side
  return Array.from({ length: n }, (_, i) => {
    const d = (i * total) / n
    if (d < side) return [lo + d, lo]
    if (d < 2 * side) return [hi, lo + (d - side)]
    if (d < 3 * side) return [hi - (d - 2 * side), hi]
    return [lo, hi - (d - 3 * side)]
  })
}

/** The polygon's area by the shoelace formula. */
function area(points: readonly (readonly [number, number])[]): number {
  let a = 0
  for (let i = 0; i < points.length; i++) {
    const [x0, y0] = points[i] as [number, number]
    const [x1, y1] = points[(i + 1) % points.length] as [number, number]
    a += x0 * y1 - x1 * y0
  }
  return Math.abs(a) / 2
}

/** Whether (x, y) lies inside the polygon, by crossing parity. */
function inside(points: readonly (readonly [number, number])[], x: number, y: number): boolean {
  let odd = false
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i] as [number, number]
    const [xj, yj] = points[j] as [number, number]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) odd = !odd
  }
  return odd
}

/** The overlap of the contour's inside with the shape's mask: intersection over union of pixel centres. */
function overlap(sc: ShapeScene, points: readonly (readonly [number, number])[]): number {
  let both = 0
  let either = 0
  for (let y = 0; y < sc.size; y++)
    for (let x = 0; x < sc.size; x++) {
      const inMask = (sc.mask[y * sc.size + x] ?? 0) === 1
      const inSnake = inside(points, x, y)
      if (inMask && inSnake) both++
      if (inMask || inSnake) either++
    }
  return either === 0 ? 1 : both / either
}

/**
 * The paths of {@link snakes}: the contour’s points, the largest move, the area inside and the overlap with the shape, and the derived `contour/picture`.
 *
 * @example
 * ```ts
 * import type { SnakesPath } from 'omai-engine/methods/vision'
 * const path: SnakesPath = 'contour/area'
 * path // => 'contour/area'
 * ```
 */
export type SnakesPath = 'contour/points' | 'contour/moved' | 'contour/area' | 'contour/overlap' | 'contour/picture'

/**
 * An active contour closing on a shape, one step per step.
 *
 * Configuration: `alpha` (elasticity: how hard the snake shrinks), `beta` (rigidity: how much it resists bending),
 * `gamma` (the step), `sigma` (how far the edge forces reach: the smoothing of the picture before the energies),
 * `points` (the contour's points), `noise` (the picture's pixel noise) and `size` (the image side). Paths:
 * `contour/points` (the contour as (x, y), full), `contour/moved` (the largest point move of the step),
 * `contour/area`, `contour/overlap` (intersection over union with the shape's true mask) and the probe
 * `contour/picture` (the shape with the contour over it).
 */
export const snakes: Method<ActiveContourState, Config, SnakesPath> = defineMethod<
  ActiveContourState,
  Config,
  SnakesPath
>({
  name: 'vision/active-contour',
  config: dict({
    alpha: real(0.001, 0.5, { default: 0.05, log: true, label: 'elasticity α' }),
    beta: real(0.01, 2, { default: 0.1, log: true, label: 'rigidity β' }),
    gamma: real(0.005, 0.2, { default: 0.05, log: true, label: 'step γ' }),
    sigma: real(0.5, 4, { default: 1.5, label: 'edge reach σ' }),
    points: int(16, 200, { default: 80 }),
    noise: real(0, 0.3, { default: 0.05, label: 'pixel noise σ' }),
    size: int(24, 96, { default: 64, label: 'image side' }),
  }),
  make: (c, key) => {
    const size = c['size'] as number
    const sc = shape(key, { size, noise: c['noise'] as number })
    const inner: Algorithm<ActiveContourState> = activeContourAlgorithm(
      grid(sc.pixels, size),
      frameContour(size, 2, c['points'] as number),
      {
        alpha: c['alpha'] as number,
        beta: c['beta'] as number,
        gamma: c['gamma'] as number,
        sigma: c['sigma'] as number,
      },
    )
    return {
      name: 'active-contour',
      config: inner.config,
      init: (k) => inner.init(k),
      step: (s, ctx) => {
        const next = inner.step(s, ctx)
        ctx.log(
          'contour/points',
          next.points.map((p) => [...p]),
        )
        ctx.log('contour/moved', next.moved)
        ctx.log('contour/area', area(next.points))
        ctx.log('contour/overlap', overlap(sc, next.points))
        return next
      },
      ...(inner.done === undefined ? {} : { done: inner.done }),
      serialise: (s) => ({ points: s.points.map((p) => [...p]), moved: s.moved }),
      deserialise: (j) => j as unknown as ActiveContourState,
    }
  },
  contract: (c, key): Contract<ActiveContourState, SnakesPath> => {
    const size = c['size'] as number
    const sc = shape(key, { size, noise: c['noise'] as number })
    return {
      'contour/points': tensor({
        role: 'state',
        shape: [c['points'] as number, 2],
        description: 'the contour’s points as (x, y)',
      }),
      'contour/moved': scalar({ description: 'the largest point move of the step, in pixels' }),
      'contour/area': scalar({ description: 'the area inside the contour, in pixels' }),
      'contour/overlap': scalar({ description: `intersection over union of the contour’s inside with the ${sc.name}` }),
      'contour/picture': derived<ActiveContourState>({
        description: 'the picture with the contour over it',
        compute: (s): DecisionSurface =>
          picture(sc.pixels, size, size, {
            points: s.points,
            classes: ['contour'],
            labels: s.points.map(() => 0),
          }),
      }),
    }
  },
  defaults: {
    'contour/points': Log({ detail: 'full' }),
    'contour/moved': Log(),
    'contour/area': Log(),
    'contour/overlap': Log(),
  },
})
