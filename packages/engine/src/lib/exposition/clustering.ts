import { WhiteboxError } from '../base/index.ts'
import { rowsOf, sameLength } from './_check.ts'
import type { InputRows } from './rows.ts'

/** How a clustering developed: assignments and centres per iteration, merges for hierarchies, density labels. */
export interface ClusteringSteps {
  readonly kind: 'clusteringSteps'
  readonly method: string
  /** assignments[t][i]: the cluster of row i after iteration t (-1 for noise). */
  readonly assignments: readonly (readonly number[])[]
  /** centres[t][k]: the centre of cluster k after iteration t (centroid methods). */
  readonly centres?: readonly (readonly (readonly number[])[])[]
  /** The objective (inertia, cost) after each iteration. */
  readonly objective?: readonly number[]
  /** Agglomerative merges: the two clusters joined, the height and the new cluster's size (scipy's linkage rows). */
  readonly merges?: readonly (readonly [number, number, number, number])[]
  /** Density methods: core, border or noise per row; reachability distances in OPTICS order. */
  readonly roles?: readonly ('core' | 'border' | 'noise')[]
  readonly reachability?: readonly number[]
  readonly ordering?: readonly number[]
  /** The rows clustered (two coordinates each, with their names and any known classes), when the data is small. */
  readonly rows?: InputRows
}

/**
 * Clustering steps, with their shapes checked.
 *
 * @param steps - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when per-iteration columns differ in length or rows differ in count.
 * @example
 * ```ts
 * import { clusteringSteps } from 'whitebox-engine/exposition'
 * clusteringSteps({ method: 'kmeans', assignments: [[0, 0, 1], [0, 1, 1]], objective: [3, 1] }).assignments.length // => 2
 * ```
 */
export function clusteringSteps(steps: Omit<ClusteringSteps, 'kind'>): ClusteringSteps {
  const t = steps.assignments.length
  const n = steps.assignments[0]?.length ?? 0
  rowsOf('clusteringSteps', 'assignments', steps.assignments, n)
  if (steps.centres !== undefined) sameLength('clusteringSteps', 'centres', steps.centres.length, t)
  if (steps.objective !== undefined) sameLength('clusteringSteps', 'objective', steps.objective.length, t)
  if (steps.roles !== undefined) sameLength('clusteringSteps', 'roles', steps.roles.length, n)
  if (steps.reachability !== undefined && steps.ordering !== undefined)
    sameLength('clusteringSteps', 'reachability', steps.reachability.length, steps.ordering.length)
  if (steps.rows !== undefined && t > 0) sameLength('clusteringSteps', 'rows', steps.rows.points.length, n)
  return { kind: 'clusteringSteps', ...steps }
}

/** EM (or variational) steps of a mixture model: parameters, responsibilities and the objective per step. */
export interface MixtureSteps {
  readonly kind: 'mixtureSteps'
  readonly family: string
  /** weights[t][k]: mixing weight of component k after step t. */
  readonly weights: readonly (readonly number[])[]
  /** means[t][k]: the mean of component k after step t. */
  readonly means: readonly (readonly (readonly number[])[])[]
  /** covariances[t][k]: component k's covariance after step t (flattened, row-major), when Gaussian. */
  readonly covariances?: readonly (readonly (readonly number[])[])[]
  /** responsibilities[t][i][k]: the posterior of component k for row i after step t (may hold only the last step). */
  readonly responsibilities?: readonly (readonly (readonly number[])[])[]
  /** The log likelihood (or ELBO) after each step: never decreasing for EM. */
  readonly objective: readonly number[]
  /** The rows fitted (two coordinates each, the first two features), when the data is small. */
  readonly rows?: InputRows
}

/**
 * Mixture steps, with their shapes checked.
 *
 * @param steps - The record without its `kind`.
 * @returns The record.
 * @throws WhiteboxError `shape` when per-step columns differ in length.
 * @example
 * ```ts
 * import { mixtureSteps } from 'whitebox-engine/exposition'
 * mixtureSteps({ family: 'gaussian', weights: [[0.5, 0.5]], means: [[[0], [1]]], objective: [-10] }).objective // => [-10]
 * ```
 */
export function mixtureSteps(steps: Omit<MixtureSteps, 'kind'>): MixtureSteps {
  const t = steps.objective.length
  sameLength('mixtureSteps', 'weights', steps.weights.length, t)
  sameLength('mixtureSteps', 'means', steps.means.length, t)
  if (steps.covariances !== undefined) sameLength('mixtureSteps', 'covariances', steps.covariances.length, t)
  if (steps.responsibilities !== undefined && steps.responsibilities.length > t)
    throw new WhiteboxError('shape', 'mixtureSteps: more responsibility steps than objective values')
  return { kind: 'mixtureSteps', ...steps }
}
