/**
 * The adaptive graphical lasso (Fan, Feng and Wu 2009): the lasso penalty on each precision entry is divided by the
 * size of an initial estimate of it, `alpha / (|theta_ab| + epsilon)^gamma`, so strong edges are penalised little (less
 * bias) and weak ones heavily (a sparser, more accurate graph); repeating with the new estimate is the reweighted
 * l1 scheme of Candes, Wakin and Boyd. Each round is a graphical lasso with a penalty per entry, solved by
 * `graphicalLasso`; the rounds are the steps of a traced `Algorithm`.
 */

import { OmaiError } from '../base/index.ts'
import { linalg } from '../numerics/index.ts'
import { asTensor, type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { graphicalLasso, graphicalLassoObjective } from './graphical-lasso.ts'

/** Options of {@link adaptiveGraphicalLassoAlgorithm}. */
export interface AdaptiveGraphicalLassoOptions {
  /** The power of the weights `(|theta| + epsilon)^-gamma` (default 0.5; 0 is the plain graphical lasso). */
  readonly gamma?: number
  /** The offset that keeps the weights of zero entries finite (default 0.01, for precisions near unit scale). */
  readonly epsilon?: number
  /** The number of reweighted fits after the initial one (default 1: Fan et al.'s one-step estimator). */
  readonly rounds?: number
  /**
   * The initial estimate the first weights come from: `'glasso'` (the plain graphical lasso at `alpha`, default),
   * `'inverse'` (the inverse of the covariance, for more samples than variables) or a precision matrix.
   */
  readonly initial?: 'glasso' | 'inverse' | readonly (readonly number[])[]
  /** The duality-gap tolerance of each fit (default 1e-4). */
  readonly tol?: number
  /** The inner lasso tolerance (default 1e-4). */
  readonly enetTol?: number
  /** The sweep cap of each fit (default 100). */
  readonly maxIter?: number
}

/** The state of {@link adaptiveGraphicalLassoAlgorithm}. */
export interface AdaptiveGraphicalLassoState {
  /** The round: 0 is the initial estimate, then one per reweighting. */
  readonly round: number
  readonly covariance: number[][]
  readonly precision: number[][]
  /** The weights the last fit used (all ones in round 0 unless the initial estimate was given). */
  readonly weights: number[][]
  readonly gap: number
  /** The weighted objective of the last fit. */
  readonly objective: number
  /** The largest change of a precision entry in the last round. */
  readonly change: number
}

function weightsOf(precision: readonly (readonly number[])[], gamma: number, epsilon: number): number[][] {
  return precision.map((r, a) => r.map((v, b) => (a === b ? 0 : (Math.abs(v) + epsilon) ** -gamma)))
}

/**
 * The adaptive graphical lasso as a traced {@link Algorithm}: round 0 is the initial estimate, each later round a
 * graphical lasso with the penalties `alpha * w_ab`, `w_ab = (|theta_ab| + epsilon)^-gamma` from the previous round's
 * estimate, warm-started from its covariance. It records the weighted objective, the duality gap, the number of edges
 * and the largest change of the precision.
 *
 * @param s - The empirical covariance `[p][p]`.
 * @param alpha - The base penalty.
 * @param options - The weights' power and offset, the rounds and the initial estimate.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for a negative penalty, power, offset or number of rounds.
 * @example
 * ```ts
 * import { adaptiveGraphicalLassoAlgorithm } from 'omai-engine/covariance'
 * adaptiveGraphicalLassoAlgorithm([[1, 0.3], [0.3, 1]], 0.05).name // => 'adaptive-graphical-lasso'
 * ```
 */
export function adaptiveGraphicalLassoAlgorithm(
  s: readonly (readonly number[])[],
  alpha: number,
  options: AdaptiveGraphicalLassoOptions = {},
): Algorithm<AdaptiveGraphicalLassoState> {
  const gamma = options.gamma ?? 0.5
  const epsilon = options.epsilon ?? 0.01
  const rounds = options.rounds ?? 1
  if (!(alpha >= 0) || !(gamma >= 0) || !(epsilon > 0) || !(rounds >= 0) || !Number.isInteger(rounds))
    throw new OmaiError('value', 'adaptiveGraphicalLasso: alpha and gamma must be non-negative, epsilon positive')
  const p = s.length
  const fit = (penalty: number | readonly (readonly number[])[], start?: readonly (readonly number[])[]) =>
    graphicalLasso(s, penalty, {
      ...(options.tol === undefined ? {} : { tol: options.tol }),
      ...(options.enetTol === undefined ? {} : { enetTol: options.enetTol }),
      maxIter: options.maxIter ?? 100,
      ...(start === undefined ? {} : { start: { covariance: start } }),
    })
  const ones = (): number[][] =>
    Array.from({ length: p }, (_, a) => Array.from({ length: p }, (__, b) => (a === b ? 0 : 1)))
  const edges = (m: readonly (readonly number[])[]): number => {
    let e = 0
    for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) if (Math.abs(m[a]?.[b] ?? 0) > 1e-10) e++
    return e
  }
  return {
    name: 'adaptive-graphical-lasso',
    config: { alpha, gamma, epsilon, rounds },
    schema: channels({
      metrics: {
        gap: { description: 'the primal-dual gap of the round', direction: 'lower' },
        objective: {
          description: 'the weighted L1-penalised negative log-likelihood of the round',
          direction: 'lower',
        },
        edges: { description: 'the number of active off-diagonal edges in the precision matrix' },
        change: { description: 'the largest change of a precision entry since the previous round' },
      },
      tensors: {
        precision: { role: 'param', description: 'the sparse precision matrix', shape: ['p', 'p'] },
        weights: { role: 'state', description: 'the penalty weights of the round', shape: ['p', 'p'] },
      },
    }),
    init: () => {
      const initial = options.initial ?? 'glasso'
      if (initial === 'inverse' || typeof initial !== 'string') {
        const precision =
          initial === 'inverse' ? (toArray(linalg.inv(s as number[][])) as number[][]) : initial.map((r) => [...r])
        return {
          round: 0,
          covariance: toArray(linalg.pinv(precision)) as number[][],
          precision,
          weights: ones(),
          gap: Number.NaN,
          objective: Number.NaN,
          change: Number.NaN,
        }
      }
      const base = fit(alpha)
      return {
        round: 0,
        covariance: base.covariance,
        precision: base.precision,
        weights: ones(),
        gap: base.gap,
        objective: base.objective,
        change: Number.NaN,
      }
    },
    step: (st) => {
      const weights = weightsOf(st.precision, gamma, epsilon)
      const penalty = weights.map((r) => r.map((w) => alpha * w))
      const next = fit(penalty, st.covariance)
      let change = 0
      for (let a = 0; a < p; a++)
        for (let b = 0; b < p; b++)
          change = Math.max(change, Math.abs((next.precision[a]?.[b] ?? 0) - (st.precision[a]?.[b] ?? 0)))
      return {
        round: st.round + 1,
        covariance: next.covariance,
        precision: next.precision,
        weights,
        gap: next.gap,
        objective: graphicalLassoObjective(s, next.precision, penalty),
        change,
      }
    },
    done: (st) => (st.round >= rounds ? 'converged' : undefined),
    observe: (st) => ({
      metrics: {
        edges: edges(st.precision),
        ...(Number.isFinite(st.gap) ? { gap: st.gap } : {}),
        ...(Number.isFinite(st.objective) ? { objective: st.objective } : {}),
        ...(Number.isFinite(st.change) ? { change: st.change } : {}),
      },
      tensors: {
        precision: { role: 'param', value: st.precision.flat(), shape: [p, p] },
        weights: { role: 'state', value: st.weights.flat(), shape: [p, p] },
      },
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as AdaptiveGraphicalLassoState,
  }
}

/**
 * The adaptive graphical lasso (Fan, Feng and Wu 2009): a graphical lasso whose penalty on each off-diagonal entry is
 * `alpha / (|theta_ab| + epsilon)^gamma` for an initial estimate `theta` (by default the plain graphical lasso at
 * `alpha`), optionally iterated. It keeps the sparsity pattern of the plain estimate tighter and shrinks the strong
 * edges less.
 *
 * @param s - The empirical covariance `[p][p]`.
 * @param alpha - The base penalty.
 * @param options - The weights' power and offset, the rounds and the initial estimate.
 * @returns The final state and the number of reweighting rounds run.
 * @index function Adaptive graphical lasso: entry-wise penalties reweighted from an initial precision estimate.
 * @example
 * ```ts
 * import { adaptiveGraphicalLasso } from 'omai-engine/covariance'
 * adaptiveGraphicalLasso([[1, 0.5], [0.5, 1]], 0.6).precision[0]?.[1] // => 0
 * ```
 */
export function adaptiveGraphicalLasso(
  s: TensorLike,
  alpha: number,
  options: AdaptiveGraphicalLassoOptions = {},
): AdaptiveGraphicalLassoState & { readonly rounds: number } {
  const rows = toArray(asTensor(s)) as number[][]
  const rounds = options.rounds ?? 1
  const out = runAlgorithm(adaptiveGraphicalLassoAlgorithm(rows, alpha, options), { maxSteps: rounds + 1 })
  return { ...out.state, rounds: out.state.round }
}
