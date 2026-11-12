// The reference EP is held against: Gibbs sampling of the same model (which readings are wild given the path; the path given
// them, by forward filtering and backward sampling of the linear-Gaussian chain), written out here because the compiled
// factor graph has no Gibbs method. Not part of the program shown.
import { keyFromPath, normal, uniform } from 'omai-engine/random'
import {
  add,
  div,
  exp,
  fromVector,
  less,
  mul,
  sort,
  sqrt,
  square,
  stack,
  sub,
  sum,
  take,
  tensor,
  toFlat,
  type Value,
  zeros,
} from 'omai-engine/tensor'
import { type EpHooks, pacer } from '../../../../shared/ep/sweeps.ts'
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

/** Gibbs: indicators given the path, path given the indicators by FFBS (a Kalman filter and a backward sample). */
export async function gibbsReference(
  y: readonly number[],
  noise: { sd: number; outliers: number },
  options: { sweeps?: number; burn?: number; seed?: number } = {},
  hooks: EpHooks = {},
): Promise<Reference> {
  const { sweeps = 2000, burn = 200, seed = 5 } = options
  const T = y.length
  const readings = tensor(y)
  const wide = WIDTH * noise.sd
  let wild: Value = zeros([T])
  let wildCount: Value = zeros([T])
  // The paths drawn after the burn-in, one tensor per sweep.
  const draws: Value[] = []
  // The sampler goes on through a stop until it has 100 draws, so that what it returns can be read.
  const pace = pacer(hooks)
  let ran = sweeps
  for (let sweep = 0; sweep < sweeps; sweep++) {
    if ((await pace(sweep, sweeps)) && sweep >= burn + 100) {
      ran = sweep
      break
    }
    // Forward filter with the noise variance each reading has now, and backward sample: scalar recursions over the steps,
    // each needing the step before, on working buffers (a tensor op per step per sweep makes a reference many times slower).
    const isWild = toFlat(wild)
    const m = new Float64Array(T)
    const v = new Float64Array(T)
    for (let t = 0; t < T; t++) {
      const pm = t === 0 ? 0 : PATH.phi * (m[t - 1] as number)
      const pv = t === 0 ? PATH.start ** 2 : PATH.phi ** 2 * (v[t - 1] as number) + PATH.sigma ** 2
      const r = (noise.sd * (isWild[t] ? WIDTH : 1)) ** 2
      const gain = pv / (pv + r)
      m[t] = pm + gain * ((y[t] as number) - pm)
      v[t] = (1 - gain) * pv
    }
    const z = toFlat(normal(keyFromPath(seed, 'x', sweep), [T]))
    const x = new Float64Array(T)
    x[T - 1] = (m[T - 1] as number) + Math.sqrt(v[T - 1] as number) * (z[T - 1] as number)
    for (let t = T - 2; t >= 0; t--) {
      const pv = PATH.phi ** 2 * (v[t] as number) + PATH.sigma ** 2
      const gain = ((v[t] as number) * PATH.phi) / pv
      const mean = (m[t] as number) + gain * ((x[t + 1] as number) - PATH.phi * (m[t] as number))
      const variance = (v[t] as number) - gain * PATH.phi * (v[t] as number)
      x[t] = mean + Math.sqrt(Math.max(variance, 0)) * (z[t] as number)
    }
    // Which readings are wild, given the path.
    const path = fromVector(x)
    const off = sub(readings, path)
    const inlier = div(mul(1 - noise.outliers, exp(mul(-0.5, square(div(off, noise.sd))))), noise.sd)
    const outlier = div(mul(noise.outliers, exp(mul(-0.5, square(div(off, wide))))), wide)
    wild = less(uniform(keyFromPath(seed, 'wild', sweep), [T]), div(outlier, add(inlier, outlier)))
    if (sweep >= burn) {
      draws.push(path)
      wildCount = add(wildCount, wild)
    }
  }
  const n = ran - burn
  const paths = stack(draws)
  const mean = div(sum(paths, 0), n)
  // The 5% and 95% points of the sorted draws at each step.
  const sorted = sort(paths, 0)
  const at = (q: number): Value => take(sorted, Math.min(n - 1, Math.floor(q * n)), 0)
  return {
    mean: toFlat(mean),
    sd: toFlat(sqrt(div(sum(square(sub(paths, mean)), 0), n - 1))),
    lower: toFlat(at(0.05)),
    upper: toFlat(at(0.95)),
    wild: toFlat(div(wildCount, n)),
    sweeps: n,
  }
}
