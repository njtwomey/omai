// The reference EP is held against: Gibbs sampling of the same model (rates given the regimes: gamma conjugacy; regimes given
// the rates: forward filtering, backward sampling), written out here because the compiled factor graph has no Gibbs method.
// Not part of the program shown.
import { gamma, keyFromPath, uniform } from 'omai-engine/random'
import {
  add,
  arange,
  cast,
  concat,
  div,
  equal,
  fromVector,
  maximum,
  mul,
  reshape,
  sqrt,
  square,
  sub,
  sum,
  tensor,
  toArray,
  toFlat,
  type Value,
  zeros,
} from 'omai-engine/tensor'
import { type EpHooks, pacer } from '../../../../shared/ep/sweeps.ts'
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
  const counts = tensor(y)
  const regimeIds = reshape(arange(K), [1, K])
  const priorShape = tensor(PRIOR.shape)
  const priorRate = tensor(PRIOR.rate)
  let rate: Value = div(priorShape, priorRate)
  // The tallies over the sweeps are tensors: the regime counts of each step, and the sums of the rates and their squares.
  let visits: Value = zeros([T, K])
  let rateSum: Value = zeros([K])
  let rateSum2: Value = zeros([K])
  // The sampler goes on through a stop until it has 100 draws, so that what it returns can be read.
  const pace = pacer(hooks)
  let ran = sweeps
  for (let sweep = 0; sweep < sweeps; sweep++) {
    if ((await pace(sweep, sweeps)) && sweep >= burn + 100) {
      ran = sweep
      break
    }
    const u = toFlat(uniform(keyFromPath(seed, 'z', sweep), [T]))
    const r = toFlat(rate)
    // Forward filter and backward sample: scalar recursions over the steps, each needing the step before, on working
    // buffers (a tensor op per step per sweep made this reference 17 times slower, measured).
    // banned-ok: reference: the Gibbs reference written out in plain arithmetic (forward filtering, backward sampling)
    const alpha = new Float64Array(T * K)
    for (let t = 0; t < T; t++) {
      let n = 0
      for (let k = 0; k < K; k++) {
        let prior = 1 / K
        if (t > 0) {
          prior = 0
          for (let i = 0; i < K; i++) prior += (alpha[(t - 1) * K + i] as number) * trans(i, k)
        }
        const a = prior * Math.exp(logPmf(y[t] as number, r[k] as number))
        alpha[t * K + k] = a
        n += a
      }
      for (let k = 0; k < K; k++) alpha[t * K + k] = (alpha[t * K + k] as number) / n
    }
    // banned-ok: reference: the Gibbs reference written out in plain arithmetic (forward filtering, backward sampling)
    const z = new Float64Array(T)
    const w = new Float64Array(K)
    for (let t = T - 1; t >= 0; t--) {
      let n = 0
      for (let k = 0; k < K; k++) {
        const p = alpha[t * K + k] as number
        w[k] = t === T - 1 ? p : p * trans(k, z[t + 1] as number)
        n += w[k] as number
      }
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
    const member = cast(equal(reshape(fromVector(z), [T, 1]), regimeIds), 'float64')
    const total = sum(mul(member, reshape(counts, [T, 1])), 0)
    const shapes = toFlat(add(priorShape, total))
    // banned-ok: reference: one array of draws per regime in the reference sampler
    const draws = Array.from({ length: K }, (_, k) =>
      gamma(keyFromPath(seed, 'rate', sweep * K + k), [1], shapes[k] as number),
    )
    rate = div(concat(draws, 0), add(priorRate, sum(member, 0)))
    if (sweep >= burn) {
      visits = add(visits, member)
      rateSum = add(rateSum, rate)
      rateSum2 = add(rateSum2, mul(rate, rate))
    }
  }
  const n = ran - burn
  const rateMean = div(rateSum, n)
  return {
    regimes: toArray(div(visits, n)) as number[][],
    rateMean: toFlat(rateMean),
    rateSd: toFlat(sqrt(maximum(sub(div(rateSum2, n), square(rateMean)), 0))),
    sweeps: n,
  }
}
