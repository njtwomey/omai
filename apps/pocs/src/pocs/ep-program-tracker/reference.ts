// The reference EP is held against: Gibbs sampling of the same model (which readings are wild given the path; the path given
// them, by forward filtering and backward sampling of the linear-Gaussian chain), written out here because the compiled
// factor graph has no Gibbs method. Not part of the program shown.
import { keyFromPath, normal, uniform } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { PATH, WIDTH } from './program.ts'

/** The reference's posterior of the path (mean, sd and 5% and 95% quantiles at each step) and the chance each reading is wild. */
export interface Reference {
  readonly mean: number[]
  readonly sd: number[]
  readonly lower: number[]
  readonly upper: number[]
  readonly wild: number[]
  readonly sweeps: number
}

const quantile = (sorted: readonly number[], q: number): number =>
  sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] as number

/** Gibbs: indicators given the path, path given the indicators by FFBS (a Kalman filter and a backward sample). */
export function gibbsReference(
  y: readonly number[],
  noise: { sd: number; outliers: number },
  options: { sweeps?: number; burn?: number; seed?: number } = {},
): Reference {
  const { sweeps = 2000, burn = 200, seed = 5 } = options
  const T = y.length
  const draws = Array.from({ length: T }, () => [] as number[])
  const wildCount = new Array<number>(T).fill(0)
  let wild = new Array<boolean>(T).fill(false)
  for (let sweep = 0; sweep < sweeps; sweep++) {
    // Forward filter with the noise variance each reading has now.
    const m = new Array<number>(T).fill(0)
    const v = new Array<number>(T).fill(0)
    for (let t = 0; t < T; t++) {
      const pm = t === 0 ? 0 : PATH.phi * (m[t - 1] as number)
      const pv = t === 0 ? PATH.start ** 2 : PATH.phi ** 2 * (v[t - 1] as number) + PATH.sigma ** 2
      const r = (noise.sd * (wild[t] ? WIDTH : 1)) ** 2
      const gain = pv / (pv + r)
      m[t] = pm + gain * ((y[t] as number) - pm)
      v[t] = (1 - gain) * pv
    }
    // Backward sample.
    const z = toFlat(normal(keyFromPath(seed, 'x', sweep), [T]))
    const x = new Array<number>(T).fill(0)
    x[T - 1] = (m[T - 1] as number) + Math.sqrt(v[T - 1] as number) * (z[T - 1] as number)
    for (let t = T - 2; t >= 0; t--) {
      const pv = PATH.phi ** 2 * (v[t] as number) + PATH.sigma ** 2
      const gain = ((v[t] as number) * PATH.phi) / pv
      const mean = (m[t] as number) + gain * ((x[t + 1] as number) - PATH.phi * (m[t] as number))
      const variance = (v[t] as number) - gain * PATH.phi * (v[t] as number)
      x[t] = mean + Math.sqrt(Math.max(variance, 0)) * (z[t] as number)
    }
    // Which readings are wild, given the path.
    const u = toFlat(uniform(keyFromPath(seed, 'wild', sweep), [T]))
    wild = x.map((xt, t) => {
      const r = (y[t] as number) - xt
      const inlier = ((1 - noise.outliers) * Math.exp(-0.5 * (r / noise.sd) ** 2)) / noise.sd
      const outlier = (noise.outliers * Math.exp(-0.5 * (r / (WIDTH * noise.sd)) ** 2)) / (WIDTH * noise.sd)
      return (u[t] as number) < outlier / (inlier + outlier)
    })
    if (sweep >= burn) {
      for (const [t, xt] of x.entries()) (draws[t] as number[]).push(xt)
      wild.forEach((w, t) => {
        if (w) wildCount[t] = (wildCount[t] as number) + 1
      })
    }
  }
  const n = sweeps - burn
  const sorted = draws.map((d) => [...d].sort((a, b) => a - b))
  const mean = draws.map((d) => d.reduce((s, v) => s + v, 0) / n)
  return {
    mean,
    sd: draws.map((d, t) => Math.sqrt(d.reduce((s, v) => s + (v - (mean[t] as number)) ** 2, 0) / (n - 1))),
    lower: sorted.map((s) => quantile(s, 0.05)),
    upper: sorted.map((s) => quantile(s, 0.95)),
    wild: wildCount.map((c) => c / n),
    sweeps: n,
  }
}
