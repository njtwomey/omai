import { epSpecMethod } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, normal, poisson, uniform } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'

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
  const x = Array.from(toFlat(normal(foldIn(k, 1), [rows])))
  const structural = Array.from(toFlat(uniform(foldIn(k, 2), [rows])), (u) => u < inflation) // a structural zero?
  const y = x.map((xi, i) =>
    structural[i] ? 0 : Number(toFlat(poisson(foldIn(k, 100 + i), Math.exp(intercept + slope * xi), [1]))[0]),
  )
  return { x, y, structural }
}
// endregion

// region model
export const zeroInflatedModel = defineModel('zero-inflated-poisson', (m) => {
  const [row, state] = [m.range('row'), m.range('state')]
  const w = m.latent('w', d.dirichlet([1, 1])) // [P(a count), P(a structural zero)]: a uniform prior on the inflation
  const z = m.latent('z', d.categorical({ probs: w }), { dims: [row] }) // each observation's state
  const eta = ops.add(
    m.latent('b0', d.normal(0, 2)),
    ops.mul(m.latent('b1', d.normal(0, 2)), m.data('x', { dims: [row] })),
  )
  const exposure = ops.at(m.data('on', { dims: [state] }), z) // 1 in the count state, 0 in the structural zero
  m.observe('y', d.poisson(ops.mul(exposure, ops.exp(eta))), m.data('y', { dims: [row] }))
})
// endregion

// region infer
export function infer(data: ReturnType<typeof simulate>, damping = 0.5) {
  const bound = zeroInflatedModel.bind({ x: data.x, y: data.y, on: [1, 0] })
  const method = epSpecMethod(bound)
  const run = session(method, { config: { tol: 1e-8, damping }, seed: 1, steps: 300, mode: 'eager', logs: 'all' })
  const posterior = method.posterior?.(run.state, run.config) // marginals by site: w, z, b0, b1
  if (posterior === undefined) throw new Error('EP gave no posterior')
  return { bound, run, posterior }
}
// endregion

/** The inflation and the two coefficients as EP recovered them: mean, sd and a 90% interval (the inflation's from its Beta marginal). */
export function parameters(posterior: ReturnType<typeof infer>['posterior']) {
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
