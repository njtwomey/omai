import { epSpecMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, normal, poisson, uniform } from 'omai-engine/random'
import { add, exp, less, mul, tensor, toFlat, where } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { EpHooks } from '../../../../shared/ep/sweeps.ts'

/** What the page changes about the data. */
export interface ZipSetup {
  readonly seed: number
  readonly rows: number
  /** The probability that a count is a structural zero. */
  readonly inflation: number
  readonly intercept: number
  readonly slope: number
}

export const defaultSetup: ZipSetup = { seed: 2, rows: 150, inflation: 0.3, intercept: 0.8, slope: 0.7 }

// region simulate
export function simulate({ seed, rows, inflation, intercept, slope }: ZipSetup) {
  const k = key(seed)
  const x = normal(foldIn(k, 1), [rows])
  // Which counts are structural zeros.
  const structural = less(uniform(foldIn(k, 2), [rows]), inflation)
  const rate = exp(add(mul(x, slope), intercept))
  // A count is drawn for each row from its own key, and is zero where the state is structural.
  const draws = toFlat(rate).map((r, i) => toFlat(poisson(foldIn(k, 100 + i), r, [1]))[0] ?? 0)
  const y = where(structural, 0, tensor(draws))
  return { x: toFlat(x), y: toFlat(y), structural: toFlat(structural) }
}
// endregion

// region model
export const zeroInflatedModel = defineModel('zero-inflated-poisson', (m) => {
  const row = m.range('row')
  const state = m.range('state')
  const x = m.data('x', { dims: [row] })
  const on = m.data('on', { dims: [state] })
  const y = m.data('y', { dims: [row] })
  // The weights of the two states: P(a count) and P(a structural zero). Uniform on the inflation.
  const w = m.latent('w', d.dirichlet([1, 1]))
  // The state of each observation.
  const z = m.latent('z', d.categorical({ probs: w }), { dims: [row] })
  const b0 = m.latent('b0', d.normal(0, 2))
  const b1 = m.latent('b1', d.normal(0, 2))
  const eta = ops.add(b0, ops.mul(b1, x))
  // The Poisson is on (1) in the count state and off (0) in the structural-zero state.
  const exposure = ops.at(on, z)
  m.observe('y', d.poisson(ops.mul(exposure, ops.exp(eta))), y)
})
// endregion

// region infer
export async function infer(data: ReturnType<typeof simulate>, damping = 0.5, hooks: EpHooks = {}) {
  const bound = zeroInflatedModel.bind({ x: data.x, y: data.y, on: [1, 0] })
  const method = epSpecMethod(bound)
  const config = { tol: 1e-8, damping }
  const cap = 300
  const run = session(method, { config, seed: 1, steps: cap, mode: 'live', logs: 'all' })
  // One sweep at a time: the page hears of each (the bar), and may stop the run.
  await runAnytimeAsync(run, {
    snapshot: { everySteps: 1 },
    onSnapshot: ({ t, state }) => hooks.onSweep?.({ sweep: t, cap, change: state.change, tol: config.tol }),
    shouldStop: () => hooks.stopped?.() === true,
    yieldToPage: hooks.yieldToPage ?? (() => Promise.resolve()),
  })
  // The marginals by site: w, z, b0, b1.
  const posterior = method.posterior?.(run.state, run.config)
  if (posterior === undefined) throw new Error('EP gave no posterior')
  return { bound, run, posterior }
}
// endregion

/** The inflation and the two coefficients as EP recovered them: mean, sd and a 90% interval (the inflation's from its Beta marginal). */
export function parameters(posterior: Awaited<ReturnType<typeof infer>>['posterior']) {
  const first = (t: ReturnType<typeof posterior.mean>): number => toFlat(t)[1] ?? Number.NaN
  const q = toFlat(posterior.quantiles('w', [0.05, 0.95])) // the inflation is the second component
  const gaussian = (site: string) => {
    const [mean, sd] = [Number(toFlat(posterior.mean(site))[0]), Number(toFlat(posterior.stddev(site))[0])]
    return { mean, sd, lower: mean - 1.645 * sd, upper: mean + 1.645 * sd }
  }
  return {
    inflation: {
      mean: first(posterior.mean('w')),
      sd: first(posterior.stddev('w')),
      lower: q[2] ?? Number.NaN,
      upper: q[3] ?? Number.NaN,
    },
    intercept: gaussian('b0'),
    slope: gaussian('b1'),
    /** For each count, the probability that it is a structural zero. */
    structural: Array.from(toFlat(posterior.mean('z'))),
  }
}

/** Attempts at the model that the compiler refuses, each with its reason: what blocks the other ways of writing zero inflation. */
export const refusedForms = {
  'a Bernoulli switch with a Beta probability': defineModel('zip-bernoulli-beta', (m) => {
    const row = m.range('row')
    const z = m.latent('z', d.bernoulli({ probs: m.latent('pi', d.beta(1, 1)) }), { dims: [row] })
    const eta = m.latent('b0', d.normal(0, 2))
    m.observe('y', d.poisson(ops.mul(ops.sub(1, z), ops.exp(eta))), m.data('y', { dims: [row] }))
  }),
  'a Bernoulli switch with a known probability choosing the rate (where(z, 0, exp(eta)))': defineModel(
    'zip-where',
    (m) => {
      const row = m.range('row')
      const z = m.latent('z', d.bernoulli({ probs: 0.3 }), { dims: [row] })
      const eta = m.latent('b0', d.normal(0, 2))
      m.observe('y', d.poisson(ops.where(z, 0, ops.exp(eta))), m.data('y', { dims: [row] }))
    },
  ),
}
