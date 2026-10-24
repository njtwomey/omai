// The Parks-McClellan exchange algorithm, ported from the remez.c of Jake Janovetz that scipy.signal.remez wraps:
// a dense grid, the alternation theorem by barycentric Lagrange interpolation (Oppenheim and Schafer 7.131-7.133),
// extremal search with removal of surplus extrema, and the taps by frequency sampling.
import { WhiteboxError } from '../../base/index.ts'

const PI2 = 2 * Math.PI

/** Filter types of the exchange algorithm. */
export type RemezType = 'bandpass' | 'differentiator' | 'hilbert'

interface Grid {
  readonly f: Float64Array
  readonly d: Float64Array
  readonly w: Float64Array
}

function denseGrid(
  r: number,
  numtaps: number,
  bands: readonly number[],
  des: readonly number[],
  weight: readonly number[],
  negative: boolean,
  density: number,
): Grid {
  const nb = des.length
  const delf = 0.5 / (density * r)
  const grid0 = negative && delf > (bands[0] ?? 0) ? delf : (bands[0] ?? 0)
  const f: number[] = []
  const d: number[] = []
  const w: number[] = []
  for (let band = 0; band < nb; band++) {
    let low = band === 0 ? grid0 : (bands[2 * band] ?? 0)
    const high = bands[2 * band + 1] ?? 0
    const k = Math.trunc((high - low) / delf + 0.5)
    for (let i = 0; i < k; i++) {
      d.push(des[band] ?? 0)
      w.push(weight[band] ?? 1)
      f.push(low)
      low += delf
    }
    if (f.length > 0) f[f.length - 1] = high
  }
  // scipy writes every grid point (its count estimate can be one short for odd-symmetry bands).
  const n = f.length
  const out = {
    f: Float64Array.from(f.slice(0, n)),
    d: Float64Array.from(d.slice(0, n)),
    w: Float64Array.from(w.slice(0, n)),
  }
  if (negative && (out.f[n - 1] ?? 0) > 0.5 - delf && numtaps % 2 === 1) out.f[n - 1] = 0.5 - delf
  return out
}

interface Parameters {
  readonly ad: Float64Array
  readonly x: Float64Array
  readonly y: Float64Array
}

function calcParms(r: number, ext: Int32Array, g: Grid): Parameters {
  const x = Float64Array.from({ length: r + 1 }, (_, i) => Math.cos(PI2 * (g.f[ext[i] ?? 0] ?? 0)))
  const ld = Math.trunc((r - 1) / 15) + 1
  const ad = new Float64Array(r + 1)
  for (let i = 0; i <= r; i++) {
    let denom = 1
    const xi = x[i] ?? 0
    for (let j = 0; j < ld; j++) for (let k = j; k <= r; k += ld) if (k !== i) denom *= 2 * (xi - (x[k] ?? 0))
    if (Math.abs(denom) < 0.00001) denom = 0.00001
    ad[i] = 1 / denom
  }
  let numer = 0
  let denom = 0
  let sign = 1
  for (let i = 0; i <= r; i++) {
    const e = ext[i] ?? 0
    numer += (ad[i] ?? 0) * (g.d[e] ?? 0)
    denom += (sign * (ad[i] ?? 0)) / (g.w[e] ?? 1)
    sign = -sign
  }
  const delta = numer / denom
  sign = 1
  const y = new Float64Array(r + 1)
  for (let i = 0; i <= r; i++) {
    const e = ext[i] ?? 0
    y[i] = (g.d[e] ?? 0) - (sign * delta) / (g.w[e] ?? 1)
    sign = -sign
  }
  return { ad, x, y }
}

function computeA(freq: number, r: number, p: Parameters): number {
  let denom = 0
  let numer = 0
  const xc = Math.cos(PI2 * freq)
  for (let i = 0; i <= r; i++) {
    let c = xc - (p.x[i] ?? 0)
    if (Math.abs(c) < 1.0e-7) {
      numer = p.y[i] ?? 0
      denom = 1
      break
    }
    c = (p.ad[i] ?? 0) / c
    denom += c
    numer += c * (p.y[i] ?? 0)
  }
  return numer / denom
}

function search(r: number, ext: Int32Array, e: Float64Array): void {
  const n = e.length
  const found: number[] = []
  const at = (i: number): number => e[i] ?? 0
  if ((at(0) > 0 && at(0) > at(1)) || (at(0) < 0 && at(0) < at(1))) found.push(0)
  for (let i = 1; i < n - 1; i++) {
    if (
      (at(i) >= at(i - 1) && at(i) > at(i + 1) && at(i) > 0) ||
      (at(i) <= at(i - 1) && at(i) < at(i + 1) && at(i) < 0)
    ) {
      if (found.length >= 2 * r)
        throw new WhiteboxError('convergence', 'remez: too many extremals; widen the transition bands')
      found.push(i)
    }
  }
  const j = n - 1
  if ((at(j) > 0 && at(j) > at(j - 1)) || (at(j) < 0 && at(j) < at(j - 1))) found.push(j)
  let extra = found.length - (r + 1)
  if (extra < 0) throw new WhiteboxError('convergence', 'remez: too few extremals; the grid is too coarse')
  while (extra > 0) {
    const ef = (q: number): number => at(found[q] ?? 0)
    let up = ef(0) > 0
    let l = 0
    let alt = true
    for (let q = 1; q < found.length; q++) {
      if (Math.abs(ef(q)) < Math.abs(ef(l))) l = q
      if (up && ef(q) < 0) up = false
      else if (!up && ef(q) > 0) up = true
      else {
        alt = false
        break
      }
    }
    if (alt && extra === 1) l = Math.abs(ef(found.length - 1)) < Math.abs(ef(0)) ? found.length - 1 : 0
    found.splice(l, 1)
    extra--
  }
  for (let i = 0; i <= r; i++) ext[i] = found[i] ?? 0
}

function isDone(r: number, ext: Int32Array, e: Float64Array): boolean {
  let min = Math.abs(e[ext[0] ?? 0] ?? 0)
  let max = min
  for (let i = 1; i <= r; i++) {
    const c = Math.abs(e[ext[i] ?? 0] ?? 0)
    if (c < min) min = c
    if (c > max) max = c
  }
  return (max - min) / max < 0.0001
}

function freqSample(n: number, a: Float64Array, negative: boolean): Float64Array {
  const m = (n - 1) / 2
  const h = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const x = (PI2 * (i - m)) / n
    let val: number
    if (!negative) {
      val = a[0] ?? 0
      const top = n % 2 === 1 ? Math.floor(m) : n / 2 - 1
      for (let k = 1; k <= top; k++) val += 2 * (a[k] ?? 0) * Math.cos(x * k)
    } else if (n % 2 === 1) {
      val = 0
      for (let k = 1; k <= Math.floor(m); k++) val += 2 * (a[k] ?? 0) * Math.sin(x * k)
    } else {
      val = (a[n / 2] ?? 0) * Math.sin(Math.PI * (i - m))
      for (let k = 1; k <= n / 2 - 1; k++) val += 2 * (a[k] ?? 0) * Math.sin(x * k)
    }
    h[i] = (negative ? -val : val) / n
  }
  return h
}

/** The minimax FIR filter by the Parks-McClellan algorithm; bands normalised so that the sampling rate is 1. */
export function remezKernel(
  numtaps: number,
  bands: readonly number[],
  desired: readonly number[],
  weight: readonly number[],
  type: RemezType,
  maxiter: number,
  density: number,
): Float64Array {
  const negative = type !== 'bandpass'
  let r = Math.trunc(numtaps / 2)
  if (numtaps % 2 === 1 && !negative) r++
  const g = denseGrid(r, numtaps, bands, desired, weight, negative, density)
  const size = g.f.length
  if (size < r + 1) throw new WhiteboxError('value', 'remez: the bands are too narrow for this many taps')
  const ext = Int32Array.from({ length: r + 1 }, (_, i) => Math.trunc((i * (size - 1)) / r))
  if (type === 'differentiator') {
    for (let i = 0; i < size; i++) {
      g.d[i] = (g.d[i] ?? 0) * (g.f[i] ?? 0)
      if ((g.d[i] ?? 0) > 0.0001) g.w[i] = (g.w[i] ?? 0) / (g.f[i] ?? 1)
    }
  }
  for (let i = 0; i < size; i++) {
    const f = g.f[i] ?? 0
    let c = 1
    if (!negative) {
      if (numtaps % 2 === 0) c = Math.cos(Math.PI * f)
      else continue
    } else c = numtaps % 2 === 1 ? Math.sin(PI2 * f) : Math.sin(Math.PI * f)
    g.d[i] = (g.d[i] ?? 0) / c
    g.w[i] = (g.w[i] ?? 0) * c
  }
  const e = new Float64Array(size)
  let iter = 0
  for (; iter < maxiter; iter++) {
    const p = calcParms(r, ext, g)
    for (let i = 0; i < size; i++) e[i] = (g.w[i] ?? 0) * ((g.d[i] ?? 0) - computeA(g.f[i] ?? 0, r, p))
    search(r, ext, e)
    if (isDone(r, ext, e)) break
  }
  if (iter === maxiter)
    throw new WhiteboxError(
      'convergence',
      `remez: failure to converge at iteration ${maxiter}; reduce the transition widths`,
    )
  const p = calcParms(r, ext, g)
  const taps = new Float64Array(Math.trunc(numtaps / 2) + 1)
  for (let i = 0; i <= Math.trunc(numtaps / 2); i++) {
    let c: number
    if (!negative) c = numtaps % 2 === 1 ? 1 : Math.cos((Math.PI * i) / numtaps)
    else c = numtaps % 2 === 1 ? Math.sin((PI2 * i) / numtaps) : Math.sin((Math.PI * i) / numtaps)
    taps[i] = computeA(i / numtaps, r, p) * c
  }
  return freqSample(numtaps, taps, negative)
}
