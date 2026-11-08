/**
 * Escape-time fractals: the Mandelbrot set, Julia sets, the Burning Ship and the multibrot family, all from iterating
 * `z -> z^d + c` (the Burning Ship takes absolute values of the parts first) and recording how long a point takes to
 * leave a disc. The viewport that maps pixels to the complex plane, the escape time of one point, the period of a
 * bulb's attractor, and the whole image as a traced algorithm: one iteration over every pixel per step, so the picture
 * sharpens as the iteration cap grows.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  derived,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { positiveInt } from '../automata/_core.ts'

/** Which set: the constant `c` is the pixel in `mandelbrot`, `burning-ship` and `multibrot`, and fixed in `julia`. */
export type EscapeSet = 'mandelbrot' | 'julia' | 'burning-ship' | 'multibrot'

/** A window on the complex plane: its centre, the extent along the real axis, and the pixels it is drawn in. */
export interface ComplexViewport {
  /** The point at the middle of the picture, `[re, im]` (default `[-0.5, 0]`). */
  readonly centre?: readonly [number, number]
  /** The width of the window along the real axis (default 3); the height follows from the pixel aspect. */
  readonly width?: number
  /** Columns and rows of pixels (default 200 by 200). */
  readonly pixels?: readonly [number, number]
}

/**
 * The complex number at the centre of every pixel of a viewport, row 0 at the top (largest imaginary part).
 *
 * @param viewport - The window.
 * @returns Real and imaginary parts, row-major, and the pixel size.
 * @throws OmaiError with code `value` for a non-positive width or pixel count.
 * @concept escape-time-fractals
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * const g = fractals.pixelGrid({ centre: [0, 0], width: 4, pixels: [2, 2] })
 * [Array.from(g.re), Array.from(g.im), g.size] // => [[-1, 1, -1, 1], [1, 1, -1, -1], 2]
 * ```
 */
export function pixelGrid(viewport: ComplexViewport = {}): {
  re: Float64Array
  im: Float64Array
  size: number
  columns: number
  rows: number
} {
  const [cx, cy] = viewport.centre ?? [-0.5, 0]
  const width = viewport.width ?? 3
  const [columns, rows] = viewport.pixels ?? [200, 200]
  positiveInt('pixelGrid', 'columns', columns)
  positiveInt('pixelGrid', 'rows', rows)
  if (!(width > 0)) throw new OmaiError('value', `pixelGrid: width must be positive, got ${width}`)
  const size = width / columns
  const re = new Float64Array(columns * rows)
  const im = new Float64Array(columns * rows)
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < columns; c++) {
      re[r * columns + c] = cx + (c + 0.5 - columns / 2) * size
      im[r * columns + c] = cy - (r + 0.5 - rows / 2) * size
    }
  return { re, im, size, columns, rows }
}

/** Options shared by {@link escapeCount} and {@link escapeTimeAlgorithm}. */
export interface EscapeRule {
  /** The set (default `'mandelbrot'`). */
  readonly set?: EscapeSet
  /** The fixed constant of a Julia set, `[re, im]` (default the dragon-like `[-0.8, 0.156]`). */
  readonly c?: readonly [number, number]
  /** The power `d` in `z^d + c` (default 2); an integer of at least 2. */
  readonly power?: number
  /** The radius of the disc a point must leave (default 2; at least 2 for the Mandelbrot family to be exact). */
  readonly bailout?: number
}

function checkRule(
  name: string,
  rule: EscapeRule,
): { set: EscapeSet; d: number; R: number; c: readonly [number, number] } {
  const set = rule.set ?? 'mandelbrot'
  const d = rule.power ?? (set === 'multibrot' ? 3 : 2)
  const R = rule.bailout ?? 2
  if (!(Number.isInteger(d) && d >= 2))
    throw new OmaiError('value', `${name}: power must be an integer of at least 2, got ${d}`)
  if (!(R > 0)) throw new OmaiError('value', `${name}: bailout must be positive, got ${R}`)
  return { set, d, R, c: rule.c ?? [-0.8, 0.156] }
}

/** z^d by repeated multiplication, in place on `[re, im]`. */
function power(re: number, im: number, d: number): [number, number] {
  let pr = re
  let pi = im
  for (let k = 1; k < d; k++) {
    const t = pr * re - pi * im
    pi = pr * im + pi * re
    pr = t
  }
  return [pr, pi]
}

function advance(set: EscapeSet, zr: number, zi: number, cr: number, ci: number, d: number): [number, number] {
  const [a, b] = set === 'burning-ship' ? power(Math.abs(zr), Math.abs(zi), d) : power(zr, zi, d)
  return [a + cr, b + ci]
}

/**
 * The escape time of one point: the first `n` with `|z_n| > bailout`, where `z_0` is 0 (Mandelbrot, Burning Ship,
 * multibrot, with `c` the point) or the point itself (Julia, with a fixed `c`).
 *
 * @param point - The complex number, `[re, im]`.
 * @param maxIterations - The cap.
 * @param rule - The set and its parameters.
 * @returns `n` (at least 1), or 0 if the point has not escaped within the cap.
 * @throws OmaiError with code `value` for a power below 2 or a non-positive bailout.
 * @concept escape-time-fractals
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * [fractals.escapeCount([0, 0], 100), fractals.escapeCount([1, 0], 100), fractals.escapeCount([-1, 0], 100)] // => [0, 3, 0]
 * ```
 */
export function escapeCount(point: readonly [number, number], maxIterations: number, rule: EscapeRule = {}): number {
  const { set, d, R, c } = checkRule('escapeCount', rule)
  const julia = set === 'julia'
  let zr = julia ? point[0] : 0
  let zi = julia ? point[1] : 0
  const cr = julia ? c[0] : point[0]
  const ci = julia ? c[1] : point[1]
  const R2 = R * R
  for (let n = 1; n <= maxIterations; n++) {
    ;[zr, zi] = advance(set, zr, zi, cr, ci, d)
    if (zr * zr + zi * zi > R2) return n
  }
  return 0
}

/**
 * The first `count` points of the orbit `z_0, z_1, ...` of a point (the probe at a clicked point).
 *
 * @param point - The complex number.
 * @param count - How many points, including `z_0`.
 * @param rule - The set and its parameters.
 * @returns Flat `[re, im, ...]`.
 * @concept escape-time-fractals
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * fractals.escapeOrbit([-1, 0], 4) // => [0, 0, -1, 0, 0, 0, -1, 0]
 * ```
 */
export function escapeOrbit(point: readonly [number, number], count: number, rule: EscapeRule = {}): number[] {
  const { set, d, c } = checkRule('escapeOrbit', rule)
  const julia = set === 'julia'
  let zr = julia ? point[0] : 0
  let zi = julia ? point[1] : 0
  const cr = julia ? c[0] : point[0]
  const ci = julia ? c[1] : point[1]
  const out: number[] = []
  for (let n = 0; n < count; n++) {
    out.push(zr, zi)
    ;[zr, zi] = advance(set, zr, zi, cr, ci, d)
  }
  return out
}

/**
 * The period of the attracting cycle of `z -> z^2 + c` (the bulb `c` lies in): iterate past a transient, then find the
 * smallest `p` with `z_{n+p}` equal to `z_n`.
 *
 * @param c - The parameter, `[re, im]`.
 * @param options - Transient length (default 5000), largest period looked for (default 64) and tolerance (default 1e-7).
 * @returns The period, or 0 if there is none within the limits (chaos, a long cycle, or a point outside the set).
 * @concept escape-time-fractals
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * [[0, 0], [-1, 0], [-1.7548776, 0], [-1.3107026, 0]].map((c) => fractals.orbitPeriod(c as [number, number])) // => [1, 2, 3, 4]
 * ```
 */
export function orbitPeriod(
  c: readonly [number, number],
  options: { readonly transient?: number; readonly maxPeriod?: number; readonly tolerance?: number } = {},
): number {
  const transient = options.transient ?? 5000
  const maxPeriod = options.maxPeriod ?? 64
  const tol = options.tolerance ?? 1e-7
  let zr = 0
  let zi = 0
  for (let n = 0; n < transient; n++) {
    ;[zr, zi] = advance('mandelbrot', zr, zi, c[0], c[1], 2)
    if (zr * zr + zi * zi > 4) return 0
  }
  let wr = zr
  let wi = zi
  for (let p = 1; p <= maxPeriod; p++) {
    ;[wr, wi] = advance('mandelbrot', wr, wi, c[0], c[1], 2)
    if (Math.hypot(wr - zr, wi - zi) < tol) return p
  }
  return 0
}

/** Options of {@link escapeTimeAlgorithm}. */
export interface EscapeTimeOptions extends EscapeRule {
  /** The window and its pixels. */
  readonly viewport?: ComplexViewport
}

/** The state of {@link escapeTimeAlgorithm}. */
export interface EscapeTimeState {
  readonly zr: Float64Array
  readonly zi: Float64Array
  /** Per pixel: the iteration at which it escaped, 0 while it has not. */
  readonly count: Int32Array
  /** Per pixel: the smooth (fractional) escape time, 0 while it has not escaped. */
  readonly smooth: Float64Array
  /** Iterations done so far. */
  readonly iteration: number
  readonly escaped: number
}

/** The paths of {@link escapeTimeAlgorithm}. */
export type EscapeTimePath = 'image/escape' | 'image/smooth' | 'image/iteration' | 'image/escapedShare' | 'point/orbit'

/**
 * The log contract of {@link escapeTimeAlgorithm}. `point/orbit` is derived on demand from `{ re, im, n }` arguments.
 *
 * @param rule - The set and its parameters (the orbit probe iterates by it).
 * @returns The contract.
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * Object.keys(fractals.escapeTimeContract({})) // => ['image/escape', 'image/smooth', 'image/iteration', 'image/escapedShare', 'point/orbit']
 * ```
 */
export function escapeTimeContract(rule: EscapeRule): Contract<EscapeTimeState, EscapeTimePath> {
  return defineContract({
    'image/escape': tensor({
      role: 'state',
      shape: ['rows', 'columns'],
      description: 'iteration at which each pixel escaped, 0 if not yet',
    }),
    'image/smooth': tensor({
      role: 'state',
      shape: ['rows', 'columns'],
      description: 'the fractional escape time for smooth colour',
    }),
    'image/iteration': scalar({ description: 'iterations done' }),
    'image/escapedShare': scalar({ description: 'the share of pixels that have escaped' }),
    'point/orbit': derived<EscapeTimeState>({
      description: 'the orbit of a chosen point; arguments { re, im, n }',
      compute: (_s, args) => {
        const a = (args ?? {}) as { re?: number; im?: number; n?: number }
        return escapeOrbit([a.re ?? 0, a.im ?? 0], a.n ?? 50, rule)
      },
    }),
  })
}

/**
 * An escape-time image as a traced algorithm: one iteration of every pixel's orbit per step. A pixel that leaves the
 * bailout disc keeps the iteration it left at (and a smooth version of it); the pixels that have not left are the
 * candidates for the set, so the share escaped climbs and the picture sharpens as the run goes on. Panning and zooming
 * are a new viewport, a new run. The state serialises.
 *
 * @param options - The set, its parameters and the viewport.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a power below 2, a non-positive bailout, width or pixel count.
 * @concept escape-time-fractals
 * @example
 * ```ts
 * import { fractals } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // The point 1 (the centre of a 1 x 1 pixel window) escapes on the third iteration: 0, 1, 2, 5.
 * const alg = fractals.escapeTimeAlgorithm({ viewport: { centre: [1, 0], width: 0.1, pixels: [1, 1] } })
 * Array.from(runAlgorithm(alg, { seed: 0, maxSteps: 5 }).state.count) // => [3]
 * ```
 */
export function escapeTimeAlgorithm(options: EscapeTimeOptions = {}): Algorithm<EscapeTimeState, EscapeTimePath> {
  const rule = checkRule('escapeTimeAlgorithm', options)
  const grid = pixelGrid(options.viewport)
  const n = grid.re.length
  const julia = rule.set === 'julia'
  const R2 = rule.R * rule.R
  const logR = Math.log(rule.R)
  const config: Json = {
    set: rule.set,
    power: rule.d,
    bailout: rule.R,
    c: [...rule.c],
    centre: [...(options.viewport?.centre ?? [-0.5, 0])],
    width: options.viewport?.width ?? 3,
    pixels: [grid.columns, grid.rows],
  }
  return {
    name: 'dynamics/fractals/escape-time',
    config,
    schema: contractSchema(escapeTimeContract(options)),
    init: () => ({
      zr: julia ? Float64Array.from(grid.re) : new Float64Array(n),
      zi: julia ? Float64Array.from(grid.im) : new Float64Array(n),
      count: new Int32Array(n),
      smooth: new Float64Array(n),
      iteration: 0,
      escaped: 0,
    }),
    step: (s) => {
      const zr = Float64Array.from(s.zr)
      const zi = Float64Array.from(s.zi)
      const count = Int32Array.from(s.count)
      const smooth = Float64Array.from(s.smooth)
      const iteration = s.iteration + 1
      let escaped = s.escaped
      for (let i = 0; i < n; i++) {
        if (count[i] !== 0) continue
        const [a, b] = advance(
          rule.set,
          zr[i] ?? 0,
          zi[i] ?? 0,
          julia ? rule.c[0] : (grid.re[i] ?? 0),
          julia ? rule.c[1] : (grid.im[i] ?? 0),
          rule.d,
        )
        zr[i] = a
        zi[i] = b
        const m2 = a * a + b * b
        if (m2 > R2) {
          count[i] = iteration
          // The continuous escape time: the orbit's size grows like R^(d^(nu - n)), so a size past R means it crossed earlier.
          smooth[i] = iteration - Math.log(Math.log(Math.sqrt(m2)) / logR) / Math.log(rule.d)
          escaped++
        }
      }
      return { zr, zi, count, smooth, iteration, escaped }
    },
    observe: (s) => ({
      metrics: { 'image/iteration': s.iteration, 'image/escapedShare': s.escaped / n },
      tensors: {
        'image/escape': { role: 'state', value: s.count, shape: [grid.rows, grid.columns] },
        'image/smooth': { role: 'state', value: s.smooth, shape: [grid.rows, grid.columns] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EscapeTimeState,
  }
}
