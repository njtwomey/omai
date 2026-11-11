// The reference EP is held against: Gibbs sampling of the same model (the regimes given the rates and the transition matrix:
// forward filtering, backward sampling; the rates given the regimes: gamma conjugacy; each row of the matrix given the regimes:
// Dirichlet conjugacy on the counts of its transitions), written out here because the compiled factor graph has no Gibbs method.
// Not part of the program shown.
import { gamma, keyFromPath, uniform } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { type EpHooks, pacer } from '../../shared/ep/sweeps.ts'
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

/** One Gamma(shape, 1) draw. */
const draw = (seed: number, name: string, index: number, shape: number): number =>
  toFlat(gamma(keyFromPath(seed, name, index), [1], shape))[0] ?? 0

/** Gibbs with forward-filtering backward-sampling for the regimes. */
export async function gibbsReference(
  y: readonly number[],
  options: { sweeps?: number; burn?: number; seed?: number } = {},
  hooks: EpHooks = {},
): Promise<Reference> {
  const { sweeps = 3000, burn = 300, seed = 11 } = options
  const T = y.length
  // The start: a rate per regime from the low, middle and high of the counts (not the truth), so that the chain begins in the
  // labelling the priors name (low, middle, high); a chain that starts in another labelling stays there.
  const sorted = [...y].sort((a, b) => a - b)
  let rate = [1 / 6, 1 / 2, 5 / 6].map((q) => Math.max(0.5, sorted[Math.floor(q * (T - 1))] as number))
  let A = PRIOR.rows.map((row) => row.map((a) => a / row.reduce((s, v) => s + v, 0)))
  const visits = Array.from({ length: T }, () => new Array<number>(K).fill(0))
  const sum = new Array<number>(K).fill(0)
  const sum2 = new Array<number>(K).fill(0)
  const aSum = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  const aSum2 = Array.from({ length: K }, () => new Array<number>(K).fill(0))
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
        const prior =
          t === 0 ? 1 / K : (alpha[t - 1] as number[]).reduce((s, p, i) => s + p * ((A[i] as number[])[k] as number), 0)
        return prior * Math.exp(logPmf(y[t] as number, rate[k] as number))
      })
      const n = a.reduce((s, v) => s + v, 0)
      alpha.push(a.map((v) => v / n))
    }
    // Backward: sample z_T, then z_t given z_{t+1}.
    const z = new Array<number>(T).fill(0)
    for (let t = T - 1; t >= 0; t--) {
      const next = z[t + 1] ?? 0
      const w = (alpha[t] as number[]).map((p, k) => (t === T - 1 ? p : p * ((A[k] as number[])[next] as number)))
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
      return draw(seed, 'rate', sweep * K + k, (PRIOR.shape[k] as number) + tot) / ((PRIOR.rate[k] as number) + n)
    })
    // Each row of the matrix given the transitions out of its regime.
    const counts = Array.from({ length: K }, () => new Array<number>(K).fill(0))
    for (let t = 1; t < T; t++) {
      const row = counts[z[t - 1] as number] as number[]
      const to = z[t] as number
      row[to] = (row[to] as number) + 1
    }
    A = A.map((_, i) => {
      const g = Array.from({ length: K }, (__, j) =>
        draw(
          seed,
          'row',
          (sweep * K + i) * K + j,
          ((PRIOR.rows[i] as number[])[j] as number) + ((counts[i] as number[])[j] as number),
        ),
      )
      const total = g.reduce((s, v) => s + v, 0)
      return g.map((v) => v / total)
    })
    if (sweep >= burn) {
      z.forEach((k, t) => {
        ;(visits[t] as number[])[k] = ((visits[t] as number[])[k] as number) + 1
      })
      rate.forEach((r, k) => {
        sum[k] = (sum[k] as number) + r
        sum2[k] = (sum2[k] as number) + r * r
      })
      for (let i = 0; i < K; i++)
        for (let j = 0; j < K; j++) {
          const p = (A[i] as number[])[j] as number
          ;(aSum[i] as number[])[j] = ((aSum[i] as number[])[j] as number) + p
          ;(aSum2[i] as number[])[j] = ((aSum2[i] as number[])[j] as number) + p * p
        }
    }
  }
  const n = ran - burn
  const sd = (s2: number, s1: number): number => Math.sqrt(Math.max(0, s2 / n - (s1 / n) ** 2))
  return {
    regimes: visits.map((v) => v.map((c) => c / n)),
    rateMean: sum.map((s) => s / n),
    rateSd: sum2.map((s2, k) => sd(s2, sum[k] as number)),
    matrixMean: aSum.map((row) => row.map((s) => s / n)),
    matrixSd: aSum2.map((row, i) => row.map((s2, j) => sd(s2, (aSum[i] as number[])[j] as number))),
    sweeps: n,
  }
}
