// The reference EP is held against: Gibbs sampling of the same model (the regimes given the rates and the transition matrix:
// forward filtering, backward sampling; the rates given the regimes: gamma conjugacy; each row of the matrix given the regimes:
// Dirichlet conjugacy on the counts of its transitions), written out here because the compiled factor graph has no Gibbs method.
// Not part of the program shown.
import { gamma, keyFromPath, uniform } from 'omai-engine/random'
import {
  add,
  arange,
  cast,
  concat,
  div,
  equal,
  floor,
  fromVector,
  matmul,
  maximum,
  mul,
  reshape,
  sliceAxis,
  sort,
  sqrt,
  square,
  sub,
  sum,
  take,
  tensor,
  toArray,
  toFlat,
  transpose,
  type Value,
  zeros,
} from 'omai-engine/tensor'
import { type EpHooks, pacer } from '../../../../shared/ep/sweeps.ts'
import { PRIOR } from './program.ts'

/** What the reference gives back: P(z_t = k), and the posterior mean and sd of each rate and each entry of the matrix. */
export interface Reference {
  readonly regimes: number[][]
  readonly rateMean: number[]
  readonly rateSd: number[]
  /** Row by row: `matrixMean[i][j]` is the probability of moving from regime i to regime j. */
  readonly matrixMean: number[][]
  readonly matrixSd: number[][]
  readonly sweeps: number
}

const K = 3
const logPmf = (y: number, rate: number): number => y * Math.log(rate) - rate // up to a constant of y

/** The Gamma(shape, 1) draws at the given indices, one keyed draw each, as a vector. */
const draws = (seed: number, name: string, indices: readonly number[], shapes: readonly number[]): Value =>
  concat(
    indices.map((index, i) => gamma(keyFromPath(seed, name, index), [1], shapes[i] as number)),
    0,
  )

/** Gibbs with forward-filtering backward-sampling for the regimes. */
export async function gibbsReference(
  y: readonly number[],
  options: { sweeps?: number; burn?: number; seed?: number } = {},
  hooks: EpHooks = {},
): Promise<Reference> {
  const { sweeps = 3000, burn = 300, seed = 11 } = options
  const T = y.length
  const counts = tensor(y)
  const regimeIds = reshape(arange(K), [1, K])
  const priorShape = tensor(PRIOR.shape)
  const priorRate = tensor(PRIOR.rate)
  const priorRows = tensor(PRIOR.rows)
  // The start: a rate per regime from the low, middle and high of the counts (not the truth), so that the chain begins in the
  // labelling the priors name (low, middle, high); a chain that starts in another labelling stays there.
  const quantiles = floor(mul(tensor([1 / 6, 1 / 2, 5 / 6]), T - 1))
  let rate: Value = maximum(take(sort(counts), cast(quantiles, 'int32')), 0.5)
  let A: Value = div(priorRows, sum(priorRows, 1, true))
  // The tallies over the sweeps are tensors: the regime counts of each step, the sums of the rates and the matrix, and of their squares.
  let visits: Value = zeros([T, K])
  let rateSum: Value = zeros([K])
  let rateSum2: Value = zeros([K])
  let aSum: Value = zeros([K, K])
  let aSum2: Value = zeros([K, K])
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
    const a = toFlat(A)
    // Forward filter and backward sample: scalar recursions over the steps, each needing the step before, on working
    // buffers (a tensor op per step per sweep makes a reference many times slower).
    // banned-ok: reference: the Gibbs reference written out in plain arithmetic (forward filtering, backward sampling)
    const alpha = new Float64Array(T * K)
    for (let t = 0; t < T; t++) {
      let n = 0
      for (let k = 0; k < K; k++) {
        let prior = 1 / K
        if (t > 0) {
          prior = 0
          for (let i = 0; i < K; i++) prior += (alpha[(t - 1) * K + i] as number) * (a[i * K + k] as number)
        }
        const w = prior * Math.exp(logPmf(y[t] as number, r[k] as number))
        alpha[t * K + k] = w
        n += w
      }
      for (let k = 0; k < K; k++) alpha[t * K + k] = (alpha[t * K + k] as number) / n
    }
    // banned-ok: reference: the Gibbs reference written out in plain arithmetic (forward filtering, backward sampling)
    const z = new Float64Array(T)
    const w = new Float64Array(K)
    for (let t = T - 1; t >= 0; t--) {
      const next = z[t + 1] ?? 0
      let n = 0
      for (let k = 0; k < K; k++) {
        const p = alpha[t * K + k] as number
        w[k] = t === T - 1 ? p : p * (a[k * K + next] as number)
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
    rate = div(draws(seed, 'rate', toFlat(add(arange(K), sweep * K)), shapes), add(priorRate, sum(member, 0)))
    // Each row of the matrix given the transitions out of its regime: the counts of from-to pairs.
    const moves = matmul(transpose(sliceAxis(member, 0, 0, T - 1)), sliceAxis(member, 0, 1, T))
    const rowShapes = toFlat(add(priorRows, moves))
    const g = reshape(draws(seed, 'row', toFlat(add(arange(K * K), sweep * K * K)), rowShapes), [K, K])
    A = div(g, sum(g, 1, true))
    if (sweep >= burn) {
      visits = add(visits, member)
      rateSum = add(rateSum, rate)
      rateSum2 = add(rateSum2, mul(rate, rate))
      aSum = add(aSum, A)
      aSum2 = add(aSum2, mul(A, A))
    }
  }
  const n = ran - burn
  const sd = (s2: Value, s1: Value): Value => sqrt(maximum(sub(div(s2, n), square(div(s1, n))), 0))
  return {
    regimes: toArray(div(visits, n)) as number[][],
    rateMean: toFlat(div(rateSum, n)),
    rateSd: toFlat(sd(rateSum2, rateSum)),
    matrixMean: toArray(div(aSum, n)) as number[][],
    matrixSd: toArray(sd(aSum2, aSum)) as number[][],
    sweeps: n,
  }
}
