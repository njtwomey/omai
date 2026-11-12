import { gram, rbfKernel, scaleKernel } from 'omai-engine/kernels'
import { cholesky } from 'omai-engine/numerics/linalg'
import { epSpecMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { key, normal, poissonEach } from 'omai-engine/random'
import { add, arange, exp, matmul, mul, reshape, toFlat, type Value, zeros } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { EpHooks } from '../../../../shared/ep/sweeps.ts'

export const BINS = 40
export const WIDTH = 0.25 // the interval is [0, 10], in 40 bins
export const BASE = 20 // intensity = BASE * exp(f(x)): about 5 events a bin
const x = mul(add(arange(BINS), 0.5), WIDTH)

// region simulate
type Prior = { amplitude: number; lengthScale: number }
/** The kernel matrix of the GP prior over the bin centres. */
export const covariance = ({ amplitude, lengthScale }: Prior): Value => {
  const kernel = scaleKernel(rbfKernel({ lengthScale }), amplitude ** 2)
  return gram(kernel, reshape(x, [BINS, 1]), { jitter: 1e-6 })
}

/** The truth: f drawn from the GP, and the Poisson counts it makes. */
export function simulate(seed: number, prior: Prior) {
  const L = cholesky(covariance(prior))
  const z = normal(key(seed), [BINS])
  const f = matmul(L, z)
  const rate = mul(BASE * WIDTH, exp(f))
  // Each count has a key of its own.
  const counts = poissonEach(rate, (i) => key(seed + 1000 + i))
  return { x: toFlat(x), f: toFlat(f), counts: toFlat(counts), prior }
}
// endregion

// region model
/** f ~ GP(0, K) (a dense multivariate normal, K constant), counts ~ Poisson(exposure * exp(f)). */
export const model = (prior: Prior) =>
  defineModel('cox-process', (b) => {
    const bin = b.range('bin')
    const counts = b.data('counts', { dims: [bin], support: 'count' })
    const gp = d.multivariateNormal(zeros([BINS]), { covariance: covariance(prior) })
    const f = b.latent('f', gp, { dims: [bin] })
    b.observe('counts', d.poisson(ops.mul(BASE * WIDTH, ops.exp(f))), counts)
  })
// endregion

// region infer
/**
 * Expectation propagation on the factor graph: the posterior of f by bin, with 90% intervals. `hooks` are the page's and
 * optional: each sweep is reported (the bar) and the run can be stopped, keeping the sweeps done.
 */
export async function infer(counts: readonly number[], prior: Prior, hooks: EpHooks = {}) {
  const bound = model(prior).bind({ counts })
  const ep = epSpecMethod(bound)
  const config = { tol: 1e-9 }
  const cap = 300
  const run = session(ep, { config, seed: 1, steps: cap, mode: 'live', logs: 'all' })
  // One sweep at a time: the page hears of each, and may stop the run.
  await runAnytimeAsync(run, {
    snapshot: { everySteps: 1 },
    onSnapshot: ({ t, state }) => hooks.onSweep?.({ sweep: t, cap, change: state.change, tol: config.tol }),
    shouldStop: () => hooks.stopped?.() === true,
    yieldToPage: hooks.yieldToPage ?? (() => Promise.resolve()),
  })
  const rows =
    ep
      .posterior?.(run.state, run.config)
      ?.summary({ prob: 0.9 })
      .filter((r) => r.site === 'f') ?? []
  return { bound, run, rows }
}
// endregion
