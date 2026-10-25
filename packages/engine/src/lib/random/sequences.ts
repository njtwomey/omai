/**
 * Quasi-random and space-filling point sets: Sobol' sequences (unscrambled, and Matousek-scrambled with a digital
 * shift, matching `scipy.stats.qmc.Sobol`), Halton sequences, Latin hypercube samples as in scipy's `LatinHypercube`,
 * and Poisson disk (blue-noise) samples by Bridson's algorithm in any dimension.
 */

import { WhiteboxError } from '../base/index.ts'
import { fromStorage, type Tensor } from '../tensor/index.ts'
import { checkSobol, directions, drawSobol, scrambleDirections } from './_sobol.ts'
import { normalFrom, unitDoubles } from './_unit.ts'
import { uniformStream } from './continuous.ts'
import { foldIn, type Key, randomBits } from './key.ts'

/** Options of {@link sobol} and {@link scrambledSobol}. */
export interface SobolOptions {
  /** Index of the first point (default 0); `skip: m` is scipy's `fast_forward(m)`. */
  readonly skip?: number
  /** Bits of precision, 1 to 32 (default 30, as scipy); at most `2^bits` points exist. */
  readonly bits?: number
}

/**
 * Points of the unscrambled Sobol' sequence (Joe and Kuo's new-joe-kuo-6.21201 direction numbers, Gray-code order),
 * the same numbers as `scipy.stats.qmc.Sobol(d, scramble=False).random(n)`, starting at the origin. The first `2^m`
 * points of any two-dimensional projection onto the first dimensions form a (t, m, 2)-net, so take `n` a power of two.
 *
 * @param n - Number of points.
 * @param d - Dimension, 1 to 1111 (the dimensions embedded from the Joe-Kuo table).
 * @param options - `skip` (first index, scipy's `fast_forward`) and `bits` (default 30).
 * @returns A float64 tensor `[n, d]` in `[0, 1)`.
 * @throws WhiteboxError with code `value` for an unsupported dimension or bit count, or more than `2^bits` points.
 * @concept quasi-random
 * @see https://web.maths.unsw.edu.au/~fkuo/sobol/
 * @example
 * ```ts
 * import { sobol } from 'whitebox-engine/random'
 * Array.from(sobol(4, 2).data) // => [0, 0, 0.5, 0.5, 0.75, 0.25, 0.25, 0.75]
 * Array.from(sobol(2, 2, { skip: 3 }).data) // => [0.25, 0.75, 0.375, 0.375]
 * ```
 */
export function sobol(n: number, d: number, options: SobolOptions = {}): Tensor<'float64'> {
  const skip = options.skip ?? 0
  const bits = options.bits ?? 30
  checkSobol('sobol', n, d, skip, bits)
  const v = Array.from({ length: d }, (_, j) => directions(j, bits))
  return fromStorage(drawSobol(v, new Uint32Array(d), n, skip, bits), [n, d], 'float64')
}

/**
 * Points of a randomised Sobol' sequence: Matousek's linear matrix scramble with a digital shift (scipy's LMS+shift,
 * the default `scipy.stats.qmc.Sobol(d)`), keyed. Each dimension is scrambled from `foldIn(key, dimension)`, so the
 * first columns do not depend on `d`. The scramble keeps the net structure (every dyadic elementary interval of the
 * first `2^m` points holds its share), the points are uniform on the cube marginally, and averages over them are
 * unbiased with the low variance of quasi-Monte Carlo. The numbers differ from scipy's, whose random generator differs.
 *
 * @param k - The random key.
 * @param n - Number of points.
 * @param d - Dimension, 1 to 1111.
 * @param options - `skip` (first index) and `bits` (default 30).
 * @returns A float64 tensor `[n, d]` in `[0, 1)`.
 * @throws WhiteboxError with code `value` for an unsupported dimension or bit count, or more than `2^bits` points.
 * @concept quasi-random
 * @reference Matousek (1998), On the L2-discrepancy for anchored boxes; Owen (2003), Variance with alternative
 * scramblings of digital nets.
 * @example
 * ```ts
 * import { key, scrambledSobol } from 'whitebox-engine/random'
 * const x = scrambledSobol(key(0), 8, 1)
 * Array.from(x.data, (u) => Math.floor(u * 8)).sort() // => [0, 1, 2, 3, 4, 5, 6, 7]
 * ```
 */
export function scrambledSobol(k: Key, n: number, d: number, options: SobolOptions = {}): Tensor<'float64'> {
  const skip = options.skip ?? 0
  const bits = options.bits ?? 30
  checkSobol('scrambledSobol', n, d, skip, bits)
  const mask = bits >= 32 ? 0xffffffff : 2 ** bits - 1
  const shift = new Uint32Array(d)
  const v = Array.from({ length: d }, (_, j) => {
    const words = randomBits(foldIn(k, j), bits + 1)
    shift[j] = (words[bits] ?? 0) & mask
    return scrambleDirections(directions(j, bits), bits, words)
  })
  return fromStorage(drawSobol(v, shift, n, skip, bits), [n, d], 'float64')
}

function firstPrimes(d: number): number[] {
  const out: number[] = []
  for (let c = 2; out.length < d; c++) {
    let prime = true
    for (const p of out) {
      if (p * p > c) break
      if (c % p === 0) {
        prime = false
        break
      }
    }
    if (prime) out.push(c)
  }
  return out
}

function radicalInverse(i: number, base: number): number {
  // The digit-by-digit sum of scipy's van der Corput sequence, so the rounding matches.
  let f = 1 / base
  let r = 0
  let n = i
  while (n > 0) {
    r += f * (n % base)
    f /= base
    n = Math.floor(n / base)
  }
  return r
}

/**
 * Points of the Halton low-discrepancy sequence: coordinate j of point i is the radical inverse of i in the j-th prime.
 * By default the sequence starts at index 1, skipping the origin (the convention `numerics.quasiMonteCarlo` relies
 * on). `skip: 0` gives exactly `scipy.stats.qmc.Halton(d, scramble=False).random(n)`, which includes the origin, and
 * `skip: m` matches scipy after `fast_forward(m)`.
 *
 * @param n - Number of points.
 * @param d - Dimension.
 * @param options - `skip`: index of the first point (default 1).
 * @returns A float64 tensor `[n, d]` in the unit cube.
 * @throws WhiteboxError with code `value` for a negative or non-integer `skip`.
 * @concept quasi-random
 * @example
 * ```ts
 * import { halton } from 'whitebox-engine/random'
 * Array.from(halton(2, 2).data) // => [0.5, 0.333333, 0.25, 0.666667]
 * Array.from(halton(2, 2, { skip: 0 }).data) // => [0, 0, 0.5, 0.333333]
 * ```
 */
export function halton(n: number, d: number, options: { readonly skip?: number } = {}): Tensor<'float64'> {
  const skip = options.skip ?? 1
  if (!Number.isInteger(skip) || skip < 0) {
    throw new WhiteboxError('value', `halton: skip must be a non-negative integer, got ${skip}`)
  }
  const primes = firstPrimes(d)
  const out = new Float64Array(n * d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i * d + j] = radicalInverse(skip + i, primes[j] ?? 2)
  return fromStorage(out, [n, d], 'float64')
}

/**
 * A Latin hypercube sample: each axis is split into `n` strata and every stratum holds exactly one point, jittered
 * uniformly within it (scipy's `LatinHypercube(d)`) or at its centre with `centered` (scipy's `scramble=False`).
 *
 * @param k - The random key.
 * @param n - Number of points.
 * @param d - Dimension.
 * @param options - `centered`: put each point at the centre of its stratum (default false).
 * @returns A float64 tensor `[n, d]` in the unit cube.
 * @concept quasi-random
 * @example
 * ```ts
 * import { key, latinHypercube } from 'whitebox-engine/random'
 * const x = latinHypercube(key(0), 4, 1)
 * Array.from(x.data, (v) => Math.floor(v * 4)).sort() // => [0, 1, 2, 3]
 * Array.from(latinHypercube(key(0), 4, 1, { centered: true }).data).sort() // => [0.125, 0.375, 0.625, 0.875]
 * ```
 */
export function latinHypercube(
  k: Key,
  n: number,
  d: number,
  options: { readonly centered?: boolean } = {},
): Tensor<'float64'> {
  const centered = options.centered ?? false
  const out = new Float64Array(n * d)
  for (let j = 0; j < d; j++) {
    const kj = foldIn(k, j)
    const u = unitDoubles(foldIn(kj, 0), 2 * n)
    // Shuffle the strata with the first n uniforms, jitter within them with the rest.
    const strata = Array.from({ length: n }, (_, i) => i)
    for (let i = n - 1; i > 0; i--) {
      const r = Math.floor((u[i] ?? 0) * (i + 1))
      ;[strata[i], strata[r]] = [strata[r] as number, strata[i] as number]
    }
    for (let i = 0; i < n; i++) out[i * d + j] = ((strata[i] ?? 0) + (centered ? 0.5 : (u[n + i] ?? 0))) / n
  }
  return fromStorage(out, [n, d], 'float64')
}

/** Options of {@link poissonDisk}. */
export interface PoissonDiskOptions {
  /** Maximum number of points (default: keep adding until no candidate fits). */
  readonly n?: number
  /** Candidates tried around an active point before it is retired (default 30). */
  readonly candidates?: number
  /** `'volume'` draws candidates uniformly in the annulus `[r, 2r]` (default); `'surface'` at distance `1.001 r`. */
  readonly hypersphere?: 'volume' | 'surface'
  /** Lower corner of the box (default zeros). */
  readonly lower?: readonly number[]
  /** Upper corner of the box (default ones). */
  readonly upper?: readonly number[]
}

const MAX_CELLS = 2 ** 24

/**
 * A Poisson disk (blue-noise) sample by Bridson's algorithm in any dimension: no two points closer than `radius`, and
 * new points are added next to existing ones until no candidate fits, so the box is covered evenly without the
 * clumps of independent uniforms (scipy's `qmc.PoissonDisk(d, radius=...).fill_space()`). An active point stays
 * active until `candidates` tries in a row fail, as in Bridson's paper (scipy retires it after one round, which
 * leaves a sparser sample). A background grid of cells of side `radius / sqrt(d)` makes each check local.
 *
 * @param k - The random key.
 * @param d - Dimension (small: the grid has `(sqrt(d) / radius)^d` cells).
 * @param radius - Minimum distance between points, in the units of the box.
 * @param options - `n`, `candidates`, `hypersphere`, `lower` and `upper`.
 * @returns A float64 tensor `[m, d]` of points in the closed box.
 * @throws WhiteboxError with code `value` for invalid arguments or a grid of more than 2^24 cells.
 * @concept quasi-random
 * @reference Bridson (2007), Fast Poisson disk sampling in arbitrary dimensions, SIGGRAPH sketches.
 * @example
 * ```ts
 * import { key, poissonDisk } from 'whitebox-engine/random'
 * const x = poissonDisk(key(0), 2, 0.2)
 * const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = x.data
 * Math.hypot(x0 - x1, y0 - y1) >= 0.2 // => true
 * poissonDisk(key(0), 2, 0.2, { n: 5 }).shape // => [5, 2]
 * ```
 */
export function poissonDisk(k: Key, d: number, radius: number, options: PoissonDiskOptions = {}): Tensor<'float64'> {
  const lower = options.lower ?? new Array<number>(d).fill(0)
  const upper = options.upper ?? new Array<number>(d).fill(1)
  const limit = options.n ?? Infinity
  const candidates = options.candidates ?? 30
  const surface = options.hypersphere === 'surface'
  if (!Number.isInteger(d) || d < 1) throw new WhiteboxError('value', `poissonDisk: dimension must be positive`)
  if (!(radius > 0) || !Number.isFinite(radius)) throw new WhiteboxError('value', 'poissonDisk: radius must be > 0')
  if (lower.length !== d || upper.length !== d || lower.some((lo, j) => !((upper[j] ?? 0) > lo))) {
    throw new WhiteboxError('value', 'poissonDisk: bounds must have length d with upper > lower')
  }
  if (!(limit >= 0) || !Number.isInteger(candidates) || candidates < 1) {
    throw new WhiteboxError('value', 'poissonDisk: n must be non-negative and candidates a positive integer')
  }
  const cell = radius / Math.sqrt(d)
  const size = lower.map((lo, j) => Math.max(1, Math.ceil(((upper[j] ?? 1) - lo) / cell)))
  const total = size.reduce((a, b) => a * b, 1)
  if (total > MAX_CELLS) {
    throw new WhiteboxError(
      'value',
      `poissonDisk: the grid would have ${total} cells; use a larger radius or fewer dims`,
    )
  }
  const stride = new Array<number>(d)
  for (let j = d - 1, s = 1; j >= 0; j--) {
    stride[j] = s
    s *= size[j] ?? 1
  }
  const grid = new Int32Array(total).fill(-1)
  const points: number[] = []
  const reach = Math.ceil(Math.sqrt(d))
  const r2 = radius * radius
  const next = uniformStream(k)
  const index = new Int32Array(d)

  const cellIndex = (x: Float64Array): number => {
    let c = 0
    for (let j = 0; j < d; j++) {
      const i = Math.min((size[j] ?? 1) - 1, Math.floor(((x[j] ?? 0) - (lower[j] ?? 0)) / cell))
      index[j] = i
      c += i * (stride[j] ?? 0)
    }
    return c
  }
  // Depth-first over neighbouring cells, pruning any whose nearest face is already at least the radius away.
  const clash = (x: Float64Array, axis: number, offset: number, gap2: number): boolean => {
    if (axis === d) {
      const p = grid[offset] ?? -1
      if (p < 0) return false
      let s = 0
      for (let j = 0; j < d; j++) s += ((x[j] ?? 0) - (points[p * d + j] ?? 0)) ** 2
      return s < r2
    }
    const i0 = index[axis] ?? 0
    const lo = (lower[axis] ?? 0) + i0 * cell
    const xi = x[axis] ?? 0
    for (let di = -reach; di <= reach; di++) {
      const i = i0 + di
      if (i < 0 || i >= (size[axis] ?? 1)) continue
      const gap = di < 0 ? xi - (lo + (di + 1) * cell) : di > 0 ? lo + di * cell - xi : 0
      const g2 = gap2 + gap * gap
      if (g2 >= r2) continue
      if (clash(x, axis + 1, offset + i * (stride[axis] ?? 0), g2)) return true
    }
    return false
  }
  const add = (x: Float64Array, c: number): void => {
    grid[c] = points.length / d
    for (let j = 0; j < d; j++) points.push(x[j] ?? 0)
  }

  const active: number[] = []
  if (limit >= 1) {
    const first = Float64Array.from(lower, (lo, j) => lo + next() * ((upper[j] ?? 1) - lo))
    add(first, cellIndex(first))
    active.push(0)
  }
  const x = new Float64Array(d)
  while (active.length > 0 && points.length / d < limit) {
    const slot = Math.floor(next() * active.length)
    const centre = active[slot] ?? 0
    let placed = false
    for (let t = 0; t < candidates && !placed; t++) {
      let norm2 = 0
      for (let j = 0; j < d; j++) {
        const z = normalFrom(next)
        x[j] = z
        norm2 += z * z
      }
      const rho = surface ? 1.001 * radius : radius * (1 + next() * (2 ** d - 1)) ** (1 / d)
      const scale = rho / Math.sqrt(norm2)
      let inside = true
      for (let j = 0; j < d; j++) {
        const xj = (points[centre * d + j] ?? 0) + scale * (x[j] ?? 0)
        x[j] = xj
        if (xj < (lower[j] ?? 0) || xj > (upper[j] ?? 1)) inside = false
      }
      if (!inside) continue
      const c = cellIndex(x)
      if ((grid[c] ?? -1) >= 0 || clash(x, 0, 0, 0)) continue
      active.push(points.length / d)
      add(x, c)
      placed = true
    }
    if (!placed) {
      active[slot] = active[active.length - 1] ?? 0
      active.pop()
    }
  }
  return fromStorage(Float64Array.from(points), [points.length / d, d], 'float64')
}
