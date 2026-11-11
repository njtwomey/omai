// The reference the page and the test hold EP against: the same Gaussian process written in its whitened form
// (f = L z with z standard normal, which NUTS samples well), run by NUTS. Not part of the program shown.
import { cholesky } from 'omai-engine/numerics/linalg'
import { nutsMethod } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { toArray, toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import { BASE, BINS, covariance, WIDTH } from './program.ts'

const whitened = defineModel('cox-process-whitened', (b) => {
  const bin = b.range('bin')
  const col = b.range('col')
  const counts = b.data('counts', { dims: [bin], support: 'count' })
  const L = b.data('L', { dims: [bin, col] })
  const z = b.latent('z', d.normal(0, 1), { dims: [col] })
  b.observe('counts', d.poisson(ops.mul(BASE * WIDTH, ops.exp(L.matmul(z)))), counts)
})

/** What the reference gives back. */
export interface Reference {
  readonly mean: number[]
  readonly sd: number[]
  readonly draws: number
  readonly rhat: number
}

/** NUTS on the whitened model: the posterior mean and sd of f by bin (f = L z, from the draws of z). */
export function nutsReference(
  counts: readonly number[],
  prior: { amplitude: number; lengthScale: number },
  options: { chains?: number; warmup?: number; draws?: number } = {},
): Reference {
  const L = toArray(cholesky(covariance(prior))) as number[][]
  const m = nutsMethod(whitened.bind({ counts, L }))
  const s = session(m, {
    config: { numChains: options.chains ?? 2, numWarmup: options.warmup ?? 150, numDraws: options.draws ?? 300 },
    seed: 7,
    steps: 100000,
    mode: 'eager',
  })
  const post = m.posterior?.(s.state, s.config)
  const z = post?.draws?.['z']
  const zs = post?.summary().filter((r) => r.site === 'z') ?? []
  const n = z === undefined ? 0 : (z.shape[0] ?? 0)
  const zz = z === undefined ? new Float64Array(0) : toFlat(z)
  const mean = new Array<number>(BINS).fill(0)
  const sd = new Array<number>(BINS).fill(0)
  for (let i = 0; i < BINS; i++) {
    const row = L[i] as number[]
    const f = Array.from({ length: n }, (_, k) => row.reduce((acc, v, j) => acc + v * (zz[k * BINS + j] ?? 0), 0))
    const mu = f.reduce((a, v) => a + v, 0) / n
    mean[i] = mu
    sd[i] = Math.sqrt(f.reduce((a, v) => a + (v - mu) ** 2, 0) / (n - 1))
  }
  return { mean, sd, draws: n, rhat: Math.max(...zs.map((r) => r.rhat ?? 1)) }
}
