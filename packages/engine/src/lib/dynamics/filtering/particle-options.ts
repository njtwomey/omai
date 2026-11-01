/**
 * The slots of the particle filters as option tables: the resampling scheme (multinomial, systematic, stratified,
 * residual), the proposal (bootstrap with resampling on a low effective sample size, or auxiliary with a first stage
 * guided by a point prediction), and a bridge from the Gaussian state-space models of the Kalman filters to the particle
 * form, so one model can be filtered both ways and the particle estimate compared with the Kalman one.
 */

import { cholesky, choleskyLogdet, choSolve } from '../../numerics/linalg/index.ts'
import { normal } from '../../random/index.ts'
import { dict, type Option, type OptionTable, optionTable, real } from '../../space/index.ts'
import {
  add,
  asTensor,
  matmul,
  mul,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import type { NonlinearModel } from './nonlinear.ts'
import type { ParticleModel, Resampling } from './particle.ts'

/** What a resampling scheme does, for validity rules and selectors. */
export interface ResamplerProperties {
  /** Whether it copies the integer part of N w deterministically and draws only the remainder. */
  readonly keepsIntegerCopies: boolean
  /** The uniform variates it draws per resampling: one, or one per particle. */
  readonly draws: 'one' | 'per particle'
}

function resampler(
  description: string,
  properties: ResamplerProperties,
  value: Resampling,
): Option<Resampling, ResamplerProperties> {
  return { description, properties, space: dict({}), make: () => value }
}

/**
 * The resampling schemes as one slot: `multinomial` (N independent draws), `stratified` (one draw per stratum),
 * `systematic` (a single draw, evenly spaced) and `residual` (the integer parts deterministically, the rest by
 * multinomial draws). Systematic is the usual choice; multinomial has the most variance.
 *
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.resamplers.make({ kind: 'residual' }) // => 'residual'
 * ```
 */
export const resamplers: OptionTable<Resampling, ResamplerProperties> = optionTable(
  {
    multinomial: resampler(
      'draw each new particle independently in proportion to the weights',
      { keepsIntegerCopies: false, draws: 'per particle' },
      'multinomial',
    ),
    systematic: resampler(
      'one uniform draw, then evenly spaced positions along the cumulative weights',
      { keepsIntegerCopies: false, draws: 'one' },
      'systematic',
    ),
    stratified: resampler(
      'one uniform draw in each of N equal strata of the cumulative weights',
      { keepsIntegerCopies: false, draws: 'per particle' },
      'stratified',
    ),
    residual: resampler(
      'copy each particle floor(N w) times, then draw the remainder by multinomial resampling',
      { keepsIntegerCopies: true, draws: 'per particle' },
      'residual',
    ),
  },
  {
    default: 'systematic',
    label: 'resampling',
    description: 'How the particles are redrawn when the weights degenerate.',
  },
)

/** A proposal of the particle filter: which filter, and when it resamples. */
export interface ParticleProposal {
  /** Whether the first stage is guided by a point prediction (the auxiliary filter). */
  readonly auxiliary: boolean
  /** Resample when ESS / N falls below this (the bootstrap filter; the auxiliary one resamples every step). */
  readonly essThreshold: number
}

/** What a proposal needs, for validity rules and selectors. */
export interface ProposalProperties {
  /** Whether the model must supply a point prediction of each particle's next state. */
  readonly needsPredict: boolean
  /** Whether it resamples at every step regardless of the weights. */
  readonly alwaysResamples: boolean
}

/**
 * The proposals as one slot: `bootstrap` (propagate by the transition, weight by the likelihood, resample when the
 * effective sample size falls below a threshold) and `auxiliary` (resample first by the likelihood of a point
 * prediction of each particle, then propagate and correct).
 *
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.particleProposals.make({ kind: 'bootstrap', essThreshold: 0.3 }).essThreshold // => 0.3
 * ```
 */
export const particleProposals: OptionTable<ParticleProposal, ProposalProperties> = optionTable(
  {
    bootstrap: <Option<ParticleProposal, ProposalProperties>>{
      description: 'propagate through the transition, weight by the likelihood, resample when the ESS is low',
      properties: { needsPredict: false, alwaysResamples: false },
      space: dict({
        essThreshold: real(0.01, 1, { default: 0.5, label: 'resample below this ESS / N (1: every step)' }),
      }),
      make: (config) => ({ auxiliary: false, essThreshold: Number(config['essThreshold'] ?? 0.5) }),
    },
    auxiliary: <Option<ParticleProposal, ProposalProperties>>{
      description:
        'resample first by the likelihood of a point prediction of each particle, then propagate and correct',
      properties: { needsPredict: true, alwaysResamples: true },
      space: dict({}),
      make: () => ({ auxiliary: true, essThreshold: 1 }),
    },
  },
  {
    default: 'bootstrap',
    label: 'proposal',
    description: 'How the particles are moved and re-weighted at each observation.',
  },
)

const rows = (x: Value, n: number): Value[] =>
  Array.from({ length: n }, (_, i) => reshape(sliceAxis(x, 0, i, i + 1), [-1]))
const mapRows = (f: (x: Value) => Value, x: Value): Value => stack(rows(x, x.shape[0] ?? 0).map(f))

/**
 * The particle form of a Gaussian state-space model x_t = f(x_{t-1}) + w_t, z_t = h(x_t) + v_t: particles start from
 * N(x0, P0), move by f plus Gaussian noise of covariance Q, and are scored by the Gaussian density of z_t around
 * h(x_t) with covariance R. The transition density (for smoothing) and the point prediction f (for the auxiliary
 * filter) are included, so the same model feeds the Kalman filters and the particle filters.
 *
 * @param model - The nonlinear Gaussian model.
 * @returns The particle model.
 * @concept particle-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { key } from 'mdml-engine/random'
 * const m = filtering.gaussianParticleModel({ f: (x) => x, h: (x) => x, Q: [[0.1]], R: [[1]], x0: [0], P0: [[1]] })
 * m.initial(key(0), 4).shape // => [4, 1]
 * ```
 */
export function gaussianParticleModel(model: NonlinearModel): ParticleModel {
  const Q = asTensor(model.Q)
  const R = asTensor(model.R)
  const x0 = reshape(asTensor(model.x0), [-1])
  const n = x0.shape[0] ?? 0
  const p = R.shape[0] ?? 0
  const LQ = cholesky(Q)
  const LR = cholesky(R)
  const LP = cholesky(asTensor(model.P0))
  const logNormaliser = (L: Value, d: number): Value => mul(add(choleskyLogdet(L), d * Math.log(2 * Math.PI)), 0.5)
  const quadratic = (L: Value, d: Value): Value => {
    // d is [N, k]: the row-wise d' S^-1 d with S = L L'.
    const sol = choSolve(L, transpose(d))
    return sum(mul(transpose(sol), d), -1)
  }
  return {
    initial: (k, count) => add(reshape(x0, [1, n]), matmul(normal(k, [count, n]), transpose(LP))),
    transition: (k, particles) => {
      const moved = mapRows(model.f, particles)
      return add(moved, matmul(normal(k, [particles.shape[0] ?? 0, n]), transpose(LQ)))
    },
    logLikelihood: (z, particles) => {
      const d = sub(reshape(z, [1, -1]), mapRows(model.h, particles))
      return sub(mul(quadratic(LR, d), -0.5), logNormaliser(LR, p))
    },
    logTransition: (particles, next) => {
      const d = sub(reshape(next, [1, -1]), mapRows(model.f, particles))
      return sub(mul(quadratic(LQ, d), -0.5), logNormaliser(LQ, n))
    },
    predict: (particles) => mapRows(model.f, particles),
  }
}
