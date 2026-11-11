import { type MixtureComponent, mixtureInterval, mixtureMarginal, selectedBranch } from 'omai-engine/bayesian-examples'
import { ddot } from 'omai-engine/numerics/linalg'
import { epSpecMethod } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { foldIn, key, normal, permutation, uniform } from 'omai-engine/random'
import { toFlat } from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'

/** What the page changes about the data. */
export interface SparseSetup {
  readonly seed: number
  readonly rows: number
  readonly features: number
  /** How many coefficients are non-zero. */
  readonly active: number
  /** The sd of the noise on the inliers. */
  readonly noise: number
  /** The share of the rows that are outliers (shifted by plus or minus ten noise sds). */
  readonly outliers: number
}

export const defaultSetup: SparseSetup = { seed: 3, rows: 60, features: 12, active: 4, noise: 0.5, outliers: 0.12 }

// region simulate
export function simulate({ seed, rows, features, active, noise, outliers }: SparseSetup) {
  const k = key(seed)
  const draw = (i: number, n: number, scale = 1): number[] => Array.from(toFlat(normal(foldIn(k, i), [n], { scale })))
  const X = Array.from({ length: rows }, (_, i) => draw(100 + i, features)) // the design
  const on = Array.from(toFlat(permutation(foldIn(k, 1), features))).slice(0, active) // which coefficients are non-zero
  const truth = draw(2, features).map((v, j) => (on.includes(j) ? Math.sign(v) * (1.5 + Math.abs(v)) : 0))
  const isOutlier = Array.from(toFlat(uniform(foldIn(k, 3), [rows])), (u) => u < outliers)
  const e = draw(4, rows, noise)
  const y = X.map((x, i) => ddot(x, truth) + (e[i] ?? 0) + (isOutlier[i] ? (i % 2 ? 10 : -10) * noise : 0))
  return { X, y, truth, isOutlier }
}
// endregion

// region model
export const prior = { inclusion: 0.2, slab: 3, spike: 0.05 } // P(slab), and the sd of the slab and of the spike

export function sparseModel(likelihood: 'gaussian' | 'student-t', noise: number) {
  return defineModel('sparse-regression', (m) => {
    const [row, feature] = [m.range('row'), m.range('feature')]
    const X = m.data('X', { dims: [row, feature] })
    const z = m.latent('z', d.bernoulli({ probs: prior.inclusion }), { dims: [feature] }) // is the coefficient in the slab?
    const slab = m.latent('slab', d.normal(0, prior.slab), { dims: [feature] })
    const spike = m.latent('spike', d.normal(0, prior.spike), { dims: [feature] })
    const mean = X.matmul(ops.where(z, slab, spike)) // the coefficient is whichever component z selects
    if (likelihood === 'gaussian') {
      const tau = m.latent('tau', d.gamma(2, 0.5)) // the unknown noise precision
      m.observe('y', d.normal(mean, ops.div(1, ops.sqrt(tau))), m.data('y', { dims: [row] }))
    } else m.observe('y', d.studentT(3, mean, noise), m.data('y', { dims: [row] })) // heavy tails, known scale
  })
}
// endregion

// region infer
export function infer(model: ReturnType<typeof sparseModel>, data: ReturnType<typeof simulate>, damping = 0.5) {
  const bound = model.bind({ X: data.X, y: data.y })
  const method = epSpecMethod(bound)
  const run = session(method, { config: { tol: 1e-8, damping }, seed: 1, steps: 300, mode: 'eager', logs: 'all' })
  const posterior = method.posterior?.(run.state, run.config) // marginals by site: z, slab, spike, tau
  if (posterior === undefined) throw new Error('EP gave no posterior')
  return { bound, run, posterior }
}
// endregion

/** The same prior written as the closed-form mixture (`mixture-site`: the switch summed out), with a Gaussian likelihood. */
export function sparseMixtureModel() {
  return defineModel('sparse-regression-mixture', (m) => {
    const [row, feature] = [m.range('row'), m.range('feature')]
    const w = m.latent('w', d.normalSpikeSlab(prior.inclusion, prior.slab, prior.spike), { dims: [feature] })
    const tau = m.latent('tau', d.gamma(2, 0.5))
    m.observe(
      'y',
      d.normal(m.data('X', { dims: [row, feature] }).matmul(w), ops.div(1, ops.sqrt(tau))),
      m.data('y', { dims: [row] }),
    )
  })
}

/**
 * The coefficients EP recovered: P(in the slab) and, for each coefficient, the mixture of what its two branches say. A
 * component's marginal belief is its selected branch with the switch's probability and its prior otherwise, so the branch is
 * recovered from it (`selectedBranch`) before the two are mixed.
 */
export function coefficients(posterior: ReturnType<typeof infer>['posterior']) {
  const get = (site: string, f: 'mean' | 'stddev'): number[] => Array.from(toFlat(posterior[f](site)))
  const [slabMean, slabSd, spikeMean, spikeSd] = [
    get('slab', 'mean'),
    get('slab', 'stddev'),
    get('spike', 'mean'),
    get('spike', 'stddev'),
  ]
  const pip = get('z', 'mean')
  const mixtures = pip.map((p, j): MixtureComponent[] => {
    const slab = selectedBranch({ mean: slabMean[j] ?? 0, sd: slabSd[j] ?? 0 }, p, { mean: 0, sd: prior.slab })
    const spike = selectedBranch({ mean: spikeMean[j] ?? 0, sd: spikeSd[j] ?? 0 }, 1 - p, { mean: 0, sd: prior.spike })
    return [...(slab ? [{ weight: p, ...slab }] : []), ...(spike ? [{ weight: 1 - p, ...spike }] : [])]
  })
  const moments = mixtures.map((m) => mixtureMarginal(m))
  const bounds = mixtures.map((m) => mixtureInterval(m))
  return {
    pip,
    mean: moments.map((m) => m.mean),
    sd: moments.map((m) => m.sd),
    lower: bounds.map((b) => b[0]),
    upper: bounds.map((b) => b[1]),
  }
}

/** The Student-t likelihood with an unknown scale (a gamma latent), which the compiler refuses: the page shows its reason. */
export function studentTUnknownScale() {
  return defineModel('sparse-regression-t-unknown-scale', (m) => {
    const [row, feature] = [m.range('row'), m.range('feature')]
    const w = m.latent('w', d.normalSpikeSlab(prior.inclusion, prior.slab, prior.spike), { dims: [feature] })
    const scale = m.latent('scale', d.gamma(2, 1))
    m.observe('y', d.studentT(3, m.data('X', { dims: [row, feature] }).matmul(w), scale), m.data('y', { dims: [row] }))
  })
}
