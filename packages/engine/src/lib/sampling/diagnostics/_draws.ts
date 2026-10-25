/** Draws of one quantity as `[chain, draw]` buffers, the input of every diagnostic (ArviZ's array interface). */
import { MdmlError } from '../../base/index.ts'
import { fourier } from '../../numerics/index.ts'
import { ndtri } from '../../numerics/special/index.ts'
import { add, asTensor, fromStorage, square, type Tensor, type TensorLike, toFlat, zeros } from '../../tensor/index.ts'

/** Row-major draws `[chains, draws]`. */
export interface Draws {
  readonly data: Float64Array
  readonly chains: number
  readonly draws: number
}

/** Read `[chain, draw]` (or `[draw]`, one chain) draws. */
export function drawsOf(x: TensorLike, name: string): Draws {
  const t = asTensor(x) as Tensor
  if (t.shape.length === 1) return { data: Float64Array.from(toFlat(t)), chains: 1, draws: t.shape[0] ?? 0 }
  if (t.shape.length === 2) {
    return { data: Float64Array.from(toFlat(t)), chains: t.shape[0] ?? 0, draws: t.shape[1] ?? 0 }
  }
  throw new MdmlError('shape', `${name}: draws must be [chain, draw] or [draw], got [${t.shape.join(', ')}]`)
}

/**
 * ArviZ's validity rule: all-NaN input, any NaN, fewer than `minChains` chains or fewer than 4 draws make the
 * diagnostic NaN.
 */
export function invalid(d: Draws, minChains: number): boolean {
  if (d.data.length === 0) return true
  for (const v of d.data) if (Number.isNaN(v)) return true
  return d.chains < minChains || d.draws < 4
}

/** Split each chain into its first and last halves (dropping the middle draw of odd lengths): `[2 chains, half]`. */
export function splitHalves(d: Draws): Draws {
  const half = Math.floor(d.draws / 2)
  const out = new Float64Array(2 * d.chains * half)
  for (let c = 0; c < d.chains; c++) {
    out.set(d.data.subarray(c * d.draws, c * d.draws + half), c * half)
    out.set(d.data.subarray((c + 1) * d.draws - half, (c + 1) * d.draws), (d.chains + c) * half)
  }
  return { data: out, chains: 2 * d.chains, draws: half }
}

/** Ranks 1..n of all values, ties sharing their average rank (scipy.stats.rankdata, method `average`). */
export function averageRanks(x: Float64Array): Float64Array {
  const n = x.length
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0))
  const ranks = new Float64Array(n)
  let i = 0
  while (i < n) {
    let j = i
    while (j + 1 < n && x[order[j + 1] ?? 0] === x[order[i] ?? 0]) j++
    const r = (i + j) / 2 + 1
    for (let k = i; k <= j; k++) ranks[order[k] ?? 0] = r
    i = j + 1
  }
  return ranks
}

/** Rank-normalise: average ranks over all draws, Blom's offset (r - 3/8) / (n + 1/4), then the normal quantile. */
export function zScale(d: Draws): Draws {
  const n = d.data.length
  const p = averageRanks(d.data).map((r) => (r - 0.375) / (n - 0.75 + 1))
  return { ...d, data: Float64Array.from(toFlat(ndtri(p))) }
}

/** The median of all draws (numpy.median). */
export function medianOf(x: Float64Array): number {
  const s = Float64Array.from(x).sort()
  const n = s.length
  if (n === 0) return Number.NaN
  return n % 2 === 1 ? (s[(n - 1) / 2] ?? Number.NaN) : ((s[n / 2 - 1] ?? 0) + (s[n / 2] ?? 0)) / 2
}

/** |x - median(x)| over all draws. */
export function fold(d: Draws): Draws {
  const m = medianOf(d.data)
  return { ...d, data: d.data.map((v) => Math.abs(v - m)) }
}

/** One chain of a set of draws. */
export function chainOf(d: Draws, c: number): Float64Array {
  return d.data.subarray(c * d.draws, (c + 1) * d.draws)
}

/** Whether an input is two-dimensional (`[chain, draw]`) rather than one chain `[draw]`. */
export function isMatrix(x: TensorLike): boolean {
  return (asTensor(x) as Tensor).shape.length === 2
}

/** Autocovariances of each row of `[rows, n]` draws at lags 0..n-1 (biased, divided by n), by FFT. */
export function rowAutocovariance(d: Draws): Float64Array {
  const { chains, draws: n } = d
  if (n === 0 || chains === 0) return new Float64Array(0)
  const centred = new Float64Array(d.data.length)
  for (let c = 0; c < chains; c++) {
    let mean = 0
    for (let i = 0; i < n; i++) mean += d.data[c * n + i] ?? 0
    mean /= n
    for (let i = 0; i < n; i++) centred[c * n + i] = (d.data[c * n + i] ?? 0) - mean
  }
  // Zero-padding to at least 2n makes the circular correlation of the FFT the linear one (ArviZ's autocov).
  const m = fourier.nextFastLen(2 * n, true)
  const spectrum = fourier.rfft(fromStorage(centred, [chains, n], 'float64'), m, -1)
  const power = add(square(spectrum.re), square(spectrum.im))
  const full = toFlat(fourier.irfft({ re: power, im: zeros([...power.shape]) }, m, -1))
  const out = new Float64Array(chains * n)
  for (let c = 0; c < chains; c++) for (let k = 0; k < n; k++) out[c * n + k] = (full[c * m + k] ?? 0) / n
  return out
}
