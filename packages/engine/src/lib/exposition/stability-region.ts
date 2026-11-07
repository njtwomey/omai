/**
 * The exposition record of the region of absolute stability of a method for ODEs (`stabilityRegion`): the set of
 * complex numbers `z = h * lambda` for which the method applied to `y' = lambda * y` with step `h` does not grow,
 * `|R(z)| <= 1`, as a grid of `|R(z)|` over the complex plane, with the eigenvalues times the step size marked on it.
 * A session probes it as the step size or the problem changes, and reads whether every mark is inside. The helpers
 * build `R` for Runge-Kutta methods (from a Butcher tableau) and rational ones, and sample the grid.
 */

import { MdmlError } from '../base/index.ts'
import { seal } from './_check.ts'
import type { Complex } from './complex-plane.ts'

/** The stability function of a one-step method as a ratio of polynomials in `z`: `R(z) = P(z) / Q(z)`. */
export interface StabilityFunction {
  /** The coefficients of P, lowest power first. */
  readonly numerator: readonly number[]
  /** The coefficients of Q, lowest power first (default `[1]`: an explicit method). */
  readonly denominator?: readonly number[]
}

/** A point of the plane `h * lambda`, with a name and whether the method is stable there. */
export interface StabilityMark {
  /** The point `h * lambda` as `[real, imaginary]`. */
  readonly at: Complex
  /** What it is (`λ₁`, `the stiff mode`). */
  readonly label?: string
}

/** The region of absolute stability of a method over a grid of the complex plane. */
export interface StabilityRegion {
  readonly kind: 'stabilityRegion'
  /** The method's name (`forward Euler`, `RK4`). */
  readonly method: string
  /** The order of accuracy, when known. */
  readonly order?: number
  /** The grid's real parts (columns), increasing. */
  readonly x: readonly number[]
  /** The grid's imaginary parts (rows), increasing. */
  readonly y: readonly number[]
  /** modulus[i][j]: `|R(x[j] + i y[i])|`; the method is stable where it is at most one. */
  readonly modulus: readonly (readonly number[])[]
  /** `h * lambda` for the problem's eigenvalues at the step size in use. */
  readonly marks?: readonly StabilityMark[]
  /** The step size the marks were scaled by. */
  readonly stepSize?: number
  /** What the picture is (`h = 0.4`), for the caption. */
  readonly caption?: string
}

/**
 * A stability region, with the grid's shape and the marks checked.
 *
 * @param record - The record without its `kind`.
 * @returns The record.
 * @throws MdmlError `shape` when the grid is smaller than two by two or the modulus does not match it; `value` when a
 *   mark is not finite or a grid is not increasing.
 * @example
 * ```ts
 * import { stabilityRegion } from 'mdml-engine/exposition'
 * const r = stabilityRegion({ method: 'm', x: [-1, 0], y: [0, 1], modulus: [[0.5, 1], [2, 1]], marks: [{ at: [-0.5, 0] }] })
 * r.method // => 'm'
 * ```
 */
export function stabilityRegion(record: Omit<StabilityRegion, 'kind'>): StabilityRegion {
  const { x, y, modulus } = record
  for (const [name, a] of [
    ['x', x],
    ['y', y],
  ] as const) {
    if (a.length < 2) throw new MdmlError('shape', `stabilityRegion: ${name} needs at least two points`)
    for (let k = 1; k < a.length; k++)
      if (!((a[k] as number) > (a[k - 1] as number)))
        throw new MdmlError('value', `stabilityRegion: ${name} is not increasing at ${k}`)
  }
  if (modulus.length !== y.length) throw new MdmlError('shape', `stabilityRegion: modulus needs ${y.length} rows`)
  for (const [i, row] of modulus.entries())
    if (row.length !== x.length)
      throw new MdmlError('shape', `stabilityRegion: modulus[${i}] has ${row.length} entries, expected ${x.length}`)
  for (const [k, m] of (record.marks ?? []).entries())
    if (!(Number.isFinite(m.at[0]) && Number.isFinite(m.at[1])))
      throw new MdmlError('value', `stabilityRegion: marks[${k}] is not finite`)
  return seal({ kind: 'stabilityRegion', ...record })
}

/**
 * The stability function of an explicit Runge-Kutta method from its Butcher tableau: `R(z) = 1 + sum_k z^k b A^(k-1) 1`
 * (a polynomial of degree at most the number of stages).
 *
 * @param a - The strictly lower triangular matrix of the tableau, `a[i][j]`.
 * @param b - The weights.
 * @returns The stability function.
 * @throws MdmlError `shape` when `a` is not square of the size of `b`.
 * @example
 * ```ts
 * import { explicitRungeKutta } from 'mdml-engine/exposition'
 * explicitRungeKutta([[0, 0], [0.5, 0]], [0, 1]).numerator // => [1, 1, 0.5]
 * ```
 */
export function explicitRungeKutta(a: readonly (readonly number[])[], b: readonly number[]): StabilityFunction {
  const s = b.length
  if (a.length !== s || a.some((row) => row.length !== s))
    throw new MdmlError('shape', `explicitRungeKutta: a must be ${s} by ${s}`)
  let v: number[] = Array.from({ length: s }, () => 1)
  const numerator = [1]
  for (let k = 1; k <= s; k++) {
    numerator.push(b.reduce((sum, bi, i) => sum + bi * (v[i] ?? 0), 0))
    v = a.map((row) => row.reduce((sum, aij, j) => sum + aij * (v[j] ?? 0), 0))
  }
  while (numerator.length > 1 && Math.abs(numerator[numerator.length - 1] ?? 0) < 1e-15) numerator.pop()
  return { numerator }
}

// A polynomial at a complex point by Horner's rule.
function horner(coefficients: readonly number[], z: Complex): Complex {
  let re = 0
  let im = 0
  for (let k = coefficients.length - 1; k >= 0; k--) {
    const nr = re * z[0] - im * z[1] + (coefficients[k] ?? 0)
    im = re * z[1] + im * z[0]
    re = nr
  }
  return [re, im]
}

/**
 * The size of the stability function at a point of the plane, `|R(z)|`: below one the method damps a mode with
 * `h * lambda = z`, above one it amplifies it.
 *
 * @param r - The stability function.
 * @param z - The point as `[real, imaginary]`.
 * @returns `|P(z) / Q(z)|` (infinite where Q vanishes).
 * @example
 * ```ts
 * import { stabilityModulus } from 'mdml-engine/exposition'
 * stabilityModulus({ numerator: [1, 1] }, [-1, 0]) // => 0
 * stabilityModulus({ numerator: [1], denominator: [1, -1] }, [-1, 0]) // => 0.5
 * ```
 */
export function stabilityModulus(r: StabilityFunction, z: Complex): number {
  const p = horner(r.numerator, z)
  const q = horner(r.denominator ?? [1], z)
  return Math.hypot(p[0], p[1]) / Math.hypot(q[0], q[1])
}

/**
 * Sample a stability function over a rectangle of the plane into a record, `n` points across the real axis (and as many
 * along the imaginary axis as keep the cells square).
 *
 * @param method - The method's name.
 * @param r - The stability function.
 * @param window - The real and imaginary extents.
 * @param options - Points across (default 161), and the rest of the record (order, marks, step size, caption).
 * @returns The record.
 * @example
 * ```ts
 * import { sampleStabilityRegion, stableArea } from 'mdml-engine/exposition'
 * const euler = sampleStabilityRegion('forward Euler', { numerator: [1, 1] }, { real: [-3, 1], imaginary: [-2, 2] }, { n: 401 })
 * Math.abs(stableArea(euler) - Math.PI) < 0.05 // => true
 * ```
 */
export function sampleStabilityRegion(
  method: string,
  r: StabilityFunction,
  window: { readonly real: readonly [number, number]; readonly imaginary: readonly [number, number] },
  options: Partial<Omit<StabilityRegion, 'kind' | 'method' | 'x' | 'y' | 'modulus'>> & { readonly n?: number } = {},
): StabilityRegion {
  const { n = 161, ...rest } = options
  const [x0, x1] = window.real
  const [y0, y1] = window.imaginary
  const dx = (x1 - x0) / (n - 1)
  const m = Math.max(2, Math.round((y1 - y0) / dx) + 1)
  const x = Array.from({ length: n }, (_, j) => x0 + j * dx)
  const y = Array.from({ length: m }, (_, i) => y0 + (i * (y1 - y0)) / (m - 1))
  return stabilityRegion({
    ...rest,
    method,
    x,
    y,
    modulus: y.map((yi) => x.map((xj) => stabilityModulus(r, [xj, yi]))),
  })
}

/**
 * The area of the stable region inside the grid, by counting the cells whose `|R| <= 1` (each cell standing for its
 * own area). The region is cut off at the grid's edges, so an unbounded region (an implicit method's) has only the
 * area of what is shown.
 *
 * @param record - The record.
 * @returns The area in the plane's units.
 * @example
 * ```ts
 * import { sampleStabilityRegion, stableArea } from 'mdml-engine/exposition'
 * const r = sampleStabilityRegion('rk2', { numerator: [1, 1, 0.5] }, { real: [-3, 1], imaginary: [-2, 2] }, { n: 401 })
 * stableArea(r) > Math.PI // => true
 * ```
 */
export function stableArea(record: StabilityRegion): number {
  const dx = ((record.x[record.x.length - 1] ?? 0) - (record.x[0] ?? 0)) / (record.x.length - 1)
  const dy = ((record.y[record.y.length - 1] ?? 0) - (record.y[0] ?? 0)) / (record.y.length - 1)
  let cells = 0
  for (const row of record.modulus) for (const v of row) if (v <= 1) cells++
  return cells * dx * dy
}

/**
 * Where the stable region meets the negative real axis: the most negative `x` such that every point from it to zero
 * is stable (forward Euler -2, RK4 about -2.785), found by scanning the grid's row nearest the real axis. `undefined`
 * when the origin itself is not on the grid, and the grid's own edge when the region runs off it.
 *
 * @param record - The record.
 * @returns The real-axis stability limit.
 * @example
 * ```ts
 * import { sampleStabilityRegion, realAxisLimit } from 'mdml-engine/exposition'
 * const r = sampleStabilityRegion('euler', { numerator: [1, 1] }, { real: [-4, 1], imaginary: [-1, 1] }, { n: 501 })
 * Math.abs((realAxisLimit(r) ?? 0) + 2) < 0.02 // => true
 * ```
 */
export function realAxisLimit(record: StabilityRegion): number | undefined {
  let row = 0
  for (const [i, v] of record.y.entries()) if (Math.abs(v) < Math.abs(record.y[row] ?? 0)) row = i
  const values = record.modulus[row]
  const j0 = record.x.findIndex((v) => v >= -1e-12)
  if (values === undefined || j0 < 0) return undefined
  let j = j0
  while (j > 0 && (values[j - 1] ?? Number.POSITIVE_INFINITY) <= 1) j--
  return record.x[j]
}

/**
 * Whether a point is in the stable region, read from the nearest cell of the grid: `stable` (`|R| <= 1`), `unstable`,
 * or `off` when the point lies beyond the grid.
 *
 * @param record - The record.
 * @param at - The point as `[real, imaginary]`.
 * @returns The verdict.
 * @example
 * ```ts
 * import { markVerdict, sampleStabilityRegion } from 'mdml-engine/exposition'
 * const r = sampleStabilityRegion('euler', { numerator: [1, 1] }, { real: [-3, 1], imaginary: [-2, 2] })
 * markVerdict(r, [-1, 0.5]) // => 'stable'
 * markVerdict(r, [-2.5, 0]) // => 'unstable'
 * markVerdict(r, [-9, 0]) // => 'off'
 * ```
 */
export function markVerdict(record: StabilityRegion, at: Complex): 'stable' | 'unstable' | 'off' {
  const nearest = (axis: readonly number[], v: number): number | undefined => {
    const lo = axis[0] ?? 0
    const hi = axis[axis.length - 1] ?? lo
    const half = (hi - lo) / (axis.length - 1) / 2
    if (v < lo - half || v > hi + half) return undefined
    return Math.min(axis.length - 1, Math.max(0, Math.round(((v - lo) / (hi - lo)) * (axis.length - 1))))
  }
  const j = nearest(record.x, at[0])
  const i = nearest(record.y, at[1])
  const m = i === undefined || j === undefined ? undefined : record.modulus[i]?.[j]
  return m === undefined ? 'off' : m <= 1 ? 'stable' : 'unstable'
}
