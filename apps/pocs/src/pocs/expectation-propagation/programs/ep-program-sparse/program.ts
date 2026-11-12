import { mixtureInterval } from 'omai-engine/probprog/ep'
import { epSpecMethod, runAnytimeAsync } from 'omai-engine/probprog/inference'
import { defineModel, ops } from 'omai-engine/probprog/model'
import * as d from 'omai-engine/probprog/model/dists'
import { type MixtureComponent, mixtureMarginal, selectedBranch } from 'omai-engine/probprog/validation'
import { foldIn, key, normal, normalEach, permutation, uniform } from 'omai-engine/random'
import {
  abs,
  add,
  any,
  arange,
  equal,
  less,
  matmul,
  mod,
  mul,
  reshape,
  sign,
  sliceAxis,
  toArray,
  toFlat,
  type Value,
  where,
} from 'omai-engine/tensor'
import { session } from 'omai-engine/trace'
import type { EpHooks } from '../../../../shared/ep/sweeps.ts'

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
  // The design: a row of standard normals for each reading, and which coefficients are non-zero.
  const X = normalEach(rows, [features], (i) => foldIn(k, 100 + i))
  const on = sliceAxis(permutation(foldIn(k, 1), features), 0, 0, active)
  const isOn = any(equal(reshape(arange(features), [features, 1]), reshape(on, [1, active])), 1)
  const size = (v: Value): Value => mul(sign(v), add(abs(v), 1.5))
  const draw = normal(foldIn(k, 2), [features])
  const truth = where(isOn, size(draw), 0)
  const isOutlier = less(uniform(foldIn(k, 3), [rows]), outliers)
  const e = normal(foldIn(k, 4), [rows], { scale: noise })
  // An outlier is shifted by ten noise sds, up for the odd rows and down for the even.
  const jump = where(equal(mod(arange(rows), 2), 1), 10, -10)
  const spike = where(isOutlier, mul(jump, noise), 0)
  const y = add(add(matmul(X, truth), e), spike)
  return {
    X: toArray(X) as number[][],
    y: toFlat(y),
    truth: toFlat(truth),
    isOutlier: toFlat(isOutlier),
  }
}
// endregion

// region model
// P(slab), and the sd of the slab and of the spike.
export const prior = { inclusion: 0.2, slab: 3, spike: 0.05 }

export function sparseModel(likelihood: 'gaussian' | 'student-t', noise: number) {
  return defineModel('sparse-regression', (m) => {
    const row = m.range('row')
    const feature = m.range('feature')
    const X = m.data('X', { dims: [row, feature] })
    const y = m.data('y', { dims: [row] })
    // Is the coefficient in the slab?
    const z = m.latent('z', d.bernoulli({ probs: prior.inclusion }), { dims: [feature] })
    const slab = m.latent('slab', d.normal(0, prior.slab), { dims: [feature] })
    const spike = m.latent('spike', d.normal(0, prior.spike), { dims: [feature] })
    // Each coefficient is whichever component z selects.
    const mean = X.matmul(ops.where(z, slab, spike))
    if (likelihood === 'gaussian') {
      // The unknown noise precision.
      const tau = m.latent('tau', d.gamma(2, 0.5))
      m.observe('y', d.normal(mean, ops.div(1, ops.sqrt(tau))), y)
    } else {
      // Heavy tails with a known scale.
      m.observe('y', d.studentT(3, mean, noise), y)
    }
  })
}
// endregion

// region infer
type Data = ReturnType<typeof simulate>

export async function infer(model: ReturnType<typeof sparseModel>, data: Data, damping = 0.5, hooks: EpHooks = {}) {
  const bound = model.bind({ X: data.X, y: data.y })
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
  // The marginals by site: z, slab, spike, tau.
  const posterior = method.posterior?.(run.state, run.config)
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
export function coefficients(posterior: Awaited<ReturnType<typeof infer>>['posterior']) {
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
