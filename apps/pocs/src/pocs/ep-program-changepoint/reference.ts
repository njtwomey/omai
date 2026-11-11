// The reference EP is held against: Gibbs sampling of the same model (rates given the regimes: gamma conjugacy; regimes given
// the rates: forward filtering, backward sampling), written out here because the compiled factor graph has no Gibbs method.
// Not part of the program shown.
import { gamma, keyFromPath, uniform } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { type EpHooks, pacer } from '../../shared/ep/sweeps.ts'
import { PRIOR } from './program.ts'

/** What the reference gives back: P(z_t = k) and the posterior mean and sd of each rate. */
export interface Reference {
  readonly regimes: number[][]
  readonly rateMean: number[]
  readonly rateSd: number[]
  readonly sweeps: number
}

const logPmf = (y: number, rate: number): number => y * Math.log(rate) - rate // up to a constant of y

/** Gibbs with forward-filtering backward-sampling for the regimes. */
export async function gibbsReference(
  y: readonly number[],
  stay: number,
  options: { sweeps?: number; burn?: number; seed?: number } = {},
  hooks: EpHooks = {},
): Promise<Reference> {
  const { sweeps = 3000, burn = 300, seed = 11 } = options
  const T = y.length
  const K = 3
  const move = (1 - stay) / 2
  const trans = (i: number, j: number): number => (i === j ? stay : move)
  let rate = PRIOR.shape.map((s, k) => s / (PRIOR.rate[k] as number))
  const visits = Array.from({ length: T }, () => new Array<number>(K).fill(0))
  const sum = new Array<number>(K).fill(0)
  const sum2 = new Array<number>(K).fill(0)
  // The sampler goes on through a stop until it has 100 draws, so that what it returns can be read.
  const pace = pacer(hooks)
  let ran = sweeps
  for (let sweep = 0; sweep < sweeps; sweep++) {
    if ((await pace(sweep, sweeps)) && sweep >= burn + 100) {
      ran = sweep
      break
    }
    const u = toFlat(uniform(keyFromPath(seed, 'z', sweep), [T]))
    // Forward: filtered probabilities.
    const alpha: number[][] = []
    for (let t = 0; t < T; t++) {
      const a = Array.from({ length: K }, (_, k) => {
        const prior = t === 0 ? 1 / K : (alpha[t - 1] as number[]).reduce((s, p, i) => s + p * trans(i, k), 0)
        return prior * Math.exp(logPmf(y[t] as number, rate[k] as number))
      })
      const n = a.reduce((s, v) => s + v, 0)
      alpha.push(a.map((v) => v / n))
    }
    // Backward: sample z_T, then z_t given z_{t+1}.
    const z = new Array<number>(T).fill(0)
    for (let t = T - 1; t >= 0; t--) {
      const w = (alpha[t] as number[]).map((p, k) => (t === T - 1 ? p : p * trans(k, z[t + 1] as number)))
      const n = w.reduce((s, v) => s + v, 0)
      let c = 0
      let pick = K - 1
      for (let k = 0; k < K; k++) {
        c += (w[k] as number) / n
        if ((u[t] as number) < c) {
          pick = k
          break
        }
      }
      z[t] = pick
    }
    // Rates given the regimes.
    rate = rate.map((_, k) => {
      let n = 0
      let tot = 0
      for (let t = 0; t < T; t++)
        if (z[t] === k) {
          n++
          tot += y[t] as number
        }
      const shape = (PRIOR.shape[k] as number) + tot
      const r = (PRIOR.rate[k] as number) + n
      return (toFlat(gamma(keyFromPath(seed, 'rate', sweep * K + k), [1], shape))[0] ?? 0) / r
    })
    if (sweep >= burn) {
      z.forEach((k, t) => {
        ;(visits[t] as number[])[k] = ((visits[t] as number[])[k] as number) + 1
      })
      rate.forEach((r, k) => {
        sum[k] = (sum[k] as number) + r
        sum2[k] = (sum2[k] as number) + r * r
      })
    }
  }
  const n = ran - burn
  const rateMean = sum.map((s) => s / n)
  return {
    regimes: visits.map((v) => v.map((c) => c / n)),
    rateMean,
    rateSd: sum2.map((s, k) => Math.sqrt(Math.max(0, s / n - (rateMean[k] as number) ** 2))),
    sweeps: n,
  }
}
