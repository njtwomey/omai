/**
 * Epicycles: a closed curve z(t) = x(t) + i y(t) as its complex Fourier series, a chain of rotating arrows (one per
 * coefficient c_k, of length |c_k|, turning k times per cycle) whose tip traces the curve. The spectrum of a curve
 * sampled round one loop, the terms kept by count or by share of energy (Parseval), the orders the arrows can be
 * chained in, the partial sums, and the joints of the chain at a time.
 */

import { MdmlError } from '../base/index.ts'
import { fft, fftfreq, ifft } from '../numerics/fourier/index.ts'
import { type Key, permutation } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'

/**
 * The complex Fourier coefficients of a closed curve sampled at N points round one loop: c_k = (1/N) sum_n z_n
 * e^{-2 pi i k n / N}, in DFT order (bin m holds frequency m below N/2 and m - N from N/2 on).
 */
export interface CurveSpectrum {
  readonly re: readonly number[]
  readonly im: readonly number[]
  /** The frequency of each bin: turns per cycle, negative for clockwise. */
  readonly k: readonly number[]
  /** |c_k| of each bin. */
  readonly magnitude: readonly number[]
  /** The bins other than 0 (the centroid), largest |c_k| first. */
  readonly byMagnitude: readonly number[]
  /** sum over k != 0 of |c_k|^2: the mean squared distance of the curve from its centroid (Parseval). */
  readonly energy: number
}

/**
 * The Fourier spectrum of a closed curve given as points round one loop (equally spaced in time, so in arc length for
 * a resampled outline).
 *
 * @param points - The curve's points, at least 2.
 * @returns The coefficients, frequencies, magnitudes, the bins by magnitude and the energy about the centroid.
 * @throws MdmlError with code `value` for fewer than 2 points.
 * @concept fourier-series
 * @example
 * ```ts
 * import { curveSpectrum } from 'mdml-engine/signal'
 * // A unit circle, anticlockwise: all its energy is at k = 1.
 * const s = curveSpectrum(Array.from({ length: 8 }, (_, n) => [Math.cos((2 * Math.PI * n) / 8), Math.sin((2 * Math.PI * n) / 8)]))
 * [s.k[s.byMagnitude[0]], Math.round(s.energy * 1e9) / 1e9] // => [1, 1]
 * ```
 */
export function curveSpectrum(points: readonly (readonly [number, number])[]): CurveSpectrum {
  const n = points.length
  if (n < 2) throw new MdmlError('value', 'curveSpectrum: need at least 2 points')
  const c = fft({ re: points.map((p) => p[0]), im: points.map((p) => p[1]) }, n, -1, 'forward')
  const re = toFlat(c.re)
  const im = toFlat(c.im)
  const k = toFlat(fftfreq(n, 1 / n)).map(Math.round)
  const magnitude = re.map((r, m) => Math.hypot(r, im[m] ?? 0))
  const byMagnitude = Array.from({ length: n - 1 }, (_, i) => i + 1).sort(
    (a, b) => (magnitude[b] ?? 0) - (magnitude[a] ?? 0),
  )
  const energy = byMagnitude.reduce((s, m) => s + (magnitude[m] ?? 0) ** 2, 0)
  return { re, im, k, magnitude, byMagnitude, energy }
}

/**
 * The bins kept for a partial sum: the `count` largest coefficients, or the fewest largest whose energy reaches a
 * share of the total.
 *
 * @param s - The spectrum.
 * @param keep - `{ count }` terms, or `{ energy }` as a fraction in (0, 1].
 * @returns The kept bins, largest first (at least one).
 * @concept fourier-series
 * @example
 * ```ts
 * import { curveSpectrum, keptTerms } from 'mdml-engine/signal'
 * const s = curveSpectrum([[1, 0], [0, 1], [-1, 0], [0, -1]])
 * [keptTerms(s, { count: 1 }), keptTerms(s, { energy: 0.99 })] // => [[1], [1]]
 * ```
 */
export function keptTerms(s: CurveSpectrum, keep: { readonly count: number } | { readonly energy: number }): number[] {
  if ('count' in keep) return s.byMagnitude.slice(0, Math.max(1, Math.floor(keep.count)))
  const goal = keep.energy * s.energy
  let acc = 0
  let kept = 0
  while (kept < s.byMagnitude.length && acc < goal) acc += (s.magnitude[s.byMagnitude[kept++] ?? 0] ?? 0) ** 2
  return s.byMagnitude.slice(0, Math.max(1, kept))
}

/** The orders the arrows can be chained in (the sum, and so the curve, is the same in every order). */
export type ChainOrder = 'magnitude' | 'speed' | 'frequency' | 'random'

/**
 * The kept bins in the order their arrows are chained: largest first, slowest first (0, 1, -1, 2, -2, ...), by
 * frequency (most negative first), or a keyed random order.
 *
 * @param s - The spectrum.
 * @param kept - The kept bins.
 * @param order - The order.
 * @param key - The key for the random order.
 * @returns The bins in chain order.
 * @concept fourier-series
 * @example
 * ```ts
 * import { chainOrder, curveSpectrum } from 'mdml-engine/signal'
 * const s = curveSpectrum([[2, 0], [0, 1], [-2, 0], [0, -1]])
 * chainOrder(s, [3, 1], 'speed').map((b) => s.k[b]) // => [1, -1]
 * ```
 */
export function chainOrder(s: CurveSpectrum, kept: readonly number[], order: ChainOrder, key?: Key): number[] {
  const bins = [...kept]
  const k = (b: number): number => s.k[b] ?? 0
  if (order === 'magnitude') return bins
  if (order === 'speed') return bins.sort((a, b) => Math.abs(k(a)) - Math.abs(k(b)) || k(b) - k(a))
  if (order === 'frequency') return bins.sort((a, b) => k(a) - k(b))
  if (key === undefined) throw new MdmlError('value', 'chainOrder: a random order needs a key')
  return Array.from(permutation(key, bins.length).data, (i) => bins[i] ?? 0)
}

/**
 * The partial sum c_0 + sum over the kept bins of c_k e^{2 pi i k t} at t = j / m, j = 0, ..., m (so the last point
 * closes the loop), by one inverse DFT of length m (which must exceed twice the largest kept |k|, so nothing aliases).
 *
 * @param s - The spectrum.
 * @param kept - The kept bins.
 * @param m - Points per cycle.
 * @returns The curve's x and y, m + 1 points.
 * @throws MdmlError with code `value` when m is too small for the kept frequencies.
 * @concept fourier-series
 * @example
 * ```ts
 * import { curveSpectrum, partialSum } from 'mdml-engine/signal'
 * const s = curveSpectrum([[1, 0], [0, 1], [-1, 0], [0, -1]])
 * partialSum(s, [1], 4).x.map((v) => Math.round(v) + 0) // => [1, 0, -1, 0, 1]
 * ```
 */
export function partialSum(
  s: CurveSpectrum,
  kept: readonly number[],
  m: number,
): { readonly x: number[]; readonly y: number[] } {
  const top = Math.max(0, ...kept.map((b) => Math.abs(s.k[b] ?? 0)))
  if (!(Number.isInteger(m) && m > 2 * top)) throw new MdmlError('value', `partialSum: m must exceed ${2 * top}`)
  const re = new Array<number>(m).fill(0)
  const im = new Array<number>(m).fill(0)
  re[0] = s.re[0] ?? 0
  im[0] = s.im[0] ?? 0
  for (const b of kept) {
    const at = ((((s.k[b] ?? 0) % m) + m) % m) as number
    re[at] = (re[at] ?? 0) + (s.re[b] ?? 0)
    im[at] = (im[at] ?? 0) + (s.im[b] ?? 0)
  }
  const z = ifft({ re, im }, m, -1, 'forward')
  const x = toFlat(z.re)
  const y = toFlat(z.im)
  return { x: [...x, x[0] ?? 0], y: [...y, y[0] ?? 0] }
}

/**
 * The joints of the chain of arrows at time t in [0, 1]: the centroid c_0, then c_0 plus the running sums of c_k
 * e^{2 pi i k t} in chain order; the last joint is the pen.
 *
 * @param s - The spectrum.
 * @param chain - The kept bins in chain order.
 * @param t - The time, as a fraction of the cycle.
 * @returns The joints' x and y (chain length + 1 each).
 * @concept fourier-series
 * @example
 * ```ts
 * import { curveSpectrum, epicycleJoints } from 'mdml-engine/signal'
 * const s = curveSpectrum([[1, 0], [0, 1], [-1, 0], [0, -1]])
 * epicycleJoints(s, [1], 0.25).y.map((v) => Math.round(v) + 0) // => [0, 1]
 * ```
 */
export function epicycleJoints(
  s: CurveSpectrum,
  chain: readonly number[],
  t: number,
): { readonly x: number[]; readonly y: number[] } {
  let x = s.re[0] ?? 0
  let y = s.im[0] ?? 0
  const xs = [x]
  const ys = [y]
  for (const b of chain) {
    const a = 2 * Math.PI * (s.k[b] ?? 0) * t
    const c = Math.cos(a)
    const sn = Math.sin(a)
    const r = s.re[b] ?? 0
    const i = s.im[b] ?? 0
    x += r * c - i * sn
    y += r * sn + i * c
    xs.push(x)
    ys.push(y)
  }
  return { x: xs, y: ys }
}
