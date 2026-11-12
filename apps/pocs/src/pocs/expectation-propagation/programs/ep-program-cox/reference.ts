// The reference the page and the test hold EP against: the same Gaussian process written in its whitened form
// (f = L z with z standard normal, which NUTS samples well), run by NUTS. Not part of the program shown.
import { cholesky } from 'omai-engine/numerics/linalg'
import { nutsMethod } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { div, matmul, sqrt, square, sub, sum, type Tensor, toFlat, transpose } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import { type EpHooks, stepSession } from '../../../../shared/ep/sweeps.ts'
import { BASE, covariance, WIDTH } from './program.ts'

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

/**
 * NUTS on the whitened model: the posterior mean and sd of f by bin (f = L z, from the draws of z). The hooks report each
 * iteration and can stop it; a run stopped before it has any draws gives undefined.
 */
export async function nutsReference(
  counts: readonly number[],
  prior: { amplitude: number; lengthScale: number },
  options: { chains?: number; warmup?: number; draws?: number } = {},
  hooks: EpHooks = {},
): Promise<Reference | undefined> {
  const L = cholesky(covariance(prior))
  const m = nutsMethod(whitened.bind({ counts, L }))
  const s = session(m, {
    config: { numChains: options.chains ?? 2, numWarmup: options.warmup ?? 150, numDraws: options.draws ?? 300 },
    seed: 7,
    steps: 100000,
    mode: 'live',
  })
  await stepSession(s, (options.warmup ?? 150) + (options.draws ?? 300) + 2, hooks)
  const post = m.posterior?.(s.state, s.config)
  const z = post?.draws?.['z']
  const zs = post?.summary().filter((r) => r.site === 'z') ?? []
  const n = z === undefined ? 0 : (z.shape[0] ?? 0)
  if (n < 4) return undefined
  // f = L z for each draw: a row per draw, a column per bin.
  const f = matmul(z as Tensor, transpose(L))
  const mean = div(sum(f, 0), n)
  const sd = sqrt(div(sum(square(sub(f, mean)), 0), n - 1))
  return { mean: toFlat(mean), sd: toFlat(sd), draws: n, rhat: Math.max(...zs.map((r) => r.rhat ?? 1)) }
}
