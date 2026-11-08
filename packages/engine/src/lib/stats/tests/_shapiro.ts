// The Shapiro-Wilk W statistic and its p-value by Royston's algorithm AS R94 (1995), as scipy.stats.shapiro computes
// them. Shared by tests/distribution.ts (the test) and tests/result.ts (the normality assumption check).
import { OmaiError } from '../../base/index.ts'
import { normPpf, normSf, poly, sortedCopy } from '../_util.ts'

const C1 = [0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056]
const C2 = [0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633]
const C3 = [0.544, -0.39978, 0.025054, -6.714e-4]
const C4 = [1.3822, -0.77857, 0.062767, -0.0020322]
const C5 = [-1.5861, -0.31082, -0.083751, 0.0038915]
const C6 = [-0.4803, -0.082676, 0.0030302]
const G = [-2.273, 0.459]

/** The Shapiro-Wilk statistic W and its upper-tail p-value. */
export function shapiroWilk(x: ArrayLike<number>): { w: number; p: number } {
  const n = x.length
  if (n < 3) throw new OmaiError('value', 'shapiro: needs at least 3 observations')
  const y = sortedCopy(x)
  const mid = y[n >> 1] ?? 0
  for (let i = 0; i < n; i++) y[i] = (y[i] ?? 0) - mid
  const n2 = n >> 1
  const a = new Float64Array(n2 + 1)
  if (n === 3) a[1] = Math.SQRT1_2
  else {
    const m = new Float64Array(n2 + 1)
    const an25 = n + 0.25
    let summ2 = 0
    for (let i = 1; i <= n2; i++) {
      m[i] = normPpf((i - 0.375) / an25)
      summ2 += (m[i] ?? 0) ** 2
    }
    summ2 *= 2
    const ssumm2 = Math.sqrt(summ2)
    const rsn = 1 / Math.sqrt(n)
    const m1 = m[1] ?? 0
    const a1 = poly(C1, rsn) - m1 / ssumm2
    let i1: number
    let fac: number
    if (n > 5) {
      i1 = 3
      const m2 = m[2] ?? 0
      const a2 = -m2 / ssumm2 + poly(C2, rsn)
      fac = Math.sqrt((summ2 - 2 * m1 * m1 - 2 * m2 * m2) / (1 - 2 * a1 * a1 - 2 * a2 * a2))
      a[2] = a2
    } else {
      i1 = 2
      fac = Math.sqrt((summ2 - 2 * m1 * m1) / (1 - 2 * a1 * a1))
    }
    a[1] = a1
    for (let i = i1; i <= n2; i++) a[i] = -(m[i] ?? 0) / fac
  }
  const range = (y[n - 1] ?? 0) - (y[0] ?? 0)
  if (range < 1e-19) throw new OmaiError('value', 'shapiro: the sample is constant')
  const coef = (i: number, j: number): number => (i === j ? 0 : Math.sign(i - j) * (a[Math.min(i, j)] ?? 0))
  // W as the squared correlation of the data with the coefficients (1-based indices as in AS R94).
  let sa = -(a[1] ?? 0)
  let sx = (y[0] ?? 0) / range
  let j = n - 1
  for (let i = 2; i <= n; i++) {
    sx += (y[i - 1] ?? 0) / range
    if (i !== j) sa += coef(i, j)
    j--
  }
  sa /= n
  sx /= n
  let ssa = 0
  let ssx = 0
  let sax = 0
  j = n
  for (let i = 1; i <= n; i++) {
    const asa = i !== j ? coef(i, j) - sa : -sa
    const xsx = (y[i - 1] ?? 0) / range - sx
    ssa += asa * asa
    ssx += xsx * xsx
    sax += asa * xsx
    j--
  }
  const ssassx = Math.sqrt(ssa * ssx)
  const w1 = ((ssassx - sax) * (ssassx + sax)) / (ssa * ssx)
  const w = 1 - w1
  if (n === 3) {
    const p = 1.90985931710274 * (Math.asin(Math.sqrt(w)) - 1.0471975511966)
    return { w, p: Math.max(0, p) }
  }
  let yy = Math.log(w1)
  const lxx = Math.log(n)
  let mu: number
  let s: number
  if (n <= 11) {
    const gamma = poly(G, n)
    if (yy >= gamma) return { w, p: 1e-99 }
    yy = -Math.log(gamma - yy)
    mu = poly(C3, n)
    s = Math.exp(poly(C4, n))
  } else {
    mu = poly(C5, lxx)
    s = Math.exp(poly(C6, lxx))
  }
  return { w, p: normSf((yy - mu) / s) }
}
