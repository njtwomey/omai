import { WhiteboxError } from '../../base/index.ts'
import { type Key, normal } from '../../random/index.ts'
import {
  choleskyLower,
  dotVec,
  inverse,
  logdetPositive,
  type Matrix,
  matVec,
  solveVector,
  sub,
  symmetrise,
} from '../_dense.ts'
import {
  type GaussianFactor,
  type GaussianMoments,
  gaussianFactor,
  gaussianMarginalise,
  gaussianProduct,
  gaussianReduce,
  gaussianToMoments,
  linearGaussianFactor,
  permuteGaussian,
} from '../gaussian-factor.ts'
import { topologicalOrder, type UndirectedGraph, undirectedGraph } from '../graphs.ts'

/** A node of a linear-Gaussian network: x = intercept + weights . parents + N(0, variance). */
export interface LinearGaussianNode {
  readonly name: string
  readonly parents: readonly string[]
  readonly weights: readonly number[]
  readonly intercept: number
  readonly variance: number
}

/**
 * A linear-Gaussian Bayes network: every node is a linear function of its parents plus independent Gaussian noise, so
 * the joint is multivariate Gaussian (pgmpy's LinearGaussianBayesianNetwork).
 *
 * @concept linear-gaussian-models
 */
export interface LinearGaussianBayesNet {
  readonly kind: 'linearGaussianBayesNet'
  readonly nodes: readonly LinearGaussianNode[]
}

/**
 * Build and validate a linear-Gaussian network (nodes reordered topologically).
 *
 * @param nodes - The nodes.
 * @returns The network.
 * @throws WhiteboxError with code `value` for unknown parents, cycles, non-positive variances or mismatched weights.
 * @concept linear-gaussian-models
 * @example
 * ```ts
 * import { linearGaussianBayesNet } from 'whitebox-engine/pgm'
 * const g = linearGaussianBayesNet([
 *   { name: 'y', parents: ['x'], weights: [2], intercept: 1, variance: 0.5 },
 *   { name: 'x', parents: [], weights: [], intercept: 0, variance: 1 },
 * ])
 * g.nodes.map((n) => n.name) // => ['x', 'y']
 * ```
 */
export function linearGaussianBayesNet(nodes: readonly LinearGaussianNode[]): LinearGaussianBayesNet {
  if (new Set(nodes.map((n) => n.name)).size !== nodes.length)
    throw new WhiteboxError('value', 'linearGaussianBayesNet: node names must be distinct')
  for (const n of nodes) {
    if (!(n.variance > 0))
      throw new WhiteboxError('value', `linearGaussianBayesNet: ${n.name} needs a positive variance`)
    if (n.weights.length !== n.parents.length)
      throw new WhiteboxError('shape', `linearGaussianBayesNet: ${n.name} needs one weight per parent`)
  }
  try {
    return { kind: 'linearGaussianBayesNet', nodes: topologicalOrder(nodes) }
  } catch (e) {
    throw new WhiteboxError('value', `linearGaussianBayesNet: ${(e as Error).message}`)
  }
}

/**
 * The joint Gaussian of a linear-Gaussian network (pgmpy's `to_joint_gaussian`): in topological order,
 * mean_i = b_i + w_i . mean_parents and Cov = (I - W)^{-1} D (I - W)^{-T}, computed node by node.
 *
 * @param g - The network.
 * @returns The variables (in node order), mean and covariance.
 * @concept linear-gaussian-models
 * @example
 * ```ts
 * import { linearGaussianBayesNet, toJointGaussian } from 'whitebox-engine/pgm'
 * const g = linearGaussianBayesNet([
 *   { name: 'x', parents: [], weights: [], intercept: 1, variance: 1 },
 *   { name: 'y', parents: ['x'], weights: [2], intercept: 0, variance: 0.5 },
 * ])
 * toJointGaussian(g).covariance // => [[1, 2], [2, 4.5]]
 * ```
 */
export function toJointGaussian(g: LinearGaussianBayesNet): GaussianMoments {
  const names = g.nodes.map((n) => n.name)
  const n = names.length
  const mean: number[] = []
  const cov: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (const [i, node] of g.nodes.entries()) {
    const pa = node.parents.map((p) => names.indexOf(p))
    mean.push(node.intercept + node.weights.reduce((s, w, k) => s + w * (mean[pa[k] ?? 0] ?? 0), 0))
    // Cov(x_i, x_j) = sum_k w_k Cov(pa_k, x_j) for j < i; Var(x_i) = w^T Cov(pa) w + variance.
    for (let j = 0; j < i; j++) {
      const c = node.weights.reduce((s, w, k) => s + w * (cov[pa[k] ?? 0]?.[j] ?? 0), 0)
      ;(cov[i] as number[])[j] = c
      ;(cov[j] as number[])[i] = c
    }
    let v = node.variance
    for (const [a, wa] of node.weights.entries())
      for (const [b, wb] of node.weights.entries()) v += wa * wb * (cov[pa[a] ?? 0]?.[pa[b] ?? 0] ?? 0)
    ;(cov[i] as number[])[i] = v
  }
  return { variables: names, mean, covariance: cov, logMass: 0 }
}

/**
 * The conditional density of each node as a canonical Gaussian factor over [...parents, node].
 *
 * @param g - The network.
 * @returns One factor per node.
 * @concept gaussian-factors
 * @example
 * ```ts
 * import { linearGaussianBayesNet, linearGaussianFactors } from 'whitebox-engine/pgm'
 * const g = linearGaussianBayesNet([{ name: 'x', parents: [], weights: [], intercept: 0, variance: 1 }])
 * linearGaussianFactors(g)[0]?.K // => [[1]]
 * ```
 */
export function linearGaussianFactors(g: LinearGaussianBayesNet): GaussianFactor[] {
  return g.nodes.map((n) => linearGaussianFactor(n.name, n.parents, n.weights, n.intercept, n.variance))
}

/**
 * The posterior of some variables given observed values of others, by Gaussian factor elimination (multiply the
 * conditional factors, condition on the evidence, integrate out the rest).
 *
 * @param g - The network.
 * @param query - The query variables.
 * @param evidence - Observed values by variable.
 * @returns The posterior moments over the query (in query order) and log p(evidence) as `logMass`.
 * @concept linear-gaussian-models
 * @example
 * ```ts
 * import { linearGaussianBayesNet, linearGaussianPosterior } from 'whitebox-engine/pgm'
 * const g = linearGaussianBayesNet([
 *   { name: 'x', parents: [], weights: [], intercept: 0, variance: 1 },
 *   { name: 'y', parents: ['x'], weights: [1], intercept: 0, variance: 1 },
 * ])
 * linearGaussianPosterior(g, ['x'], { y: 2 }).mean.map((v) => Math.round(v * 1e12) / 1e12) // => [1]
 * ```
 */
export function linearGaussianPosterior(
  g: LinearGaussianBayesNet,
  query: readonly string[],
  evidence: Readonly<Record<string, number>> = {},
): GaussianMoments {
  for (const q of [...query, ...Object.keys(evidence)]) {
    if (!g.nodes.some((n) => n.name === q)) throw new WhiteboxError('value', `unknown variable ${q}`)
  }
  let joint = gaussianFactor([], [], [], 0)
  for (const f of linearGaussianFactors(g)) joint = gaussianProduct(joint, f)
  const reduced = gaussianReduce(joint, evidence)
  const rest = reduced.variables.filter((v) => !query.includes(v))
  return gaussianToMoments(permuteGaussian(gaussianMarginalise(reduced, rest), query))
}

/**
 * Draw samples from a linear-Gaussian network by ancestral sampling.
 *
 * @param g - The network.
 * @param key - The random key.
 * @param n - Number of samples.
 * @returns The samples as records by variable.
 * @concept sampling
 * @example
 * ```ts
 * import { linearGaussianBayesNet, linearGaussianSample } from 'whitebox-engine/pgm'
 * import { key } from 'whitebox-engine/random'
 * const g = linearGaussianBayesNet([{ name: 'x', parents: [], weights: [], intercept: 3, variance: 1e-30 }])
 * linearGaussianSample(g, key(0), 2).map((s) => Math.round(s['x'] ?? 0)) // => [3, 3]
 * ```
 */
export function linearGaussianSample(g: LinearGaussianBayesNet, key: Key, n: number): Record<string, number>[] {
  const z = normal(key, [n, g.nodes.length]).data
  return Array.from({ length: n }, (_, i) => {
    const s: Record<string, number> = {}
    for (const [j, node] of g.nodes.entries()) {
      const mu = node.intercept + node.weights.reduce((acc, w, k) => acc + w * (s[node.parents[k] ?? ''] ?? 0), 0)
      s[node.name] = mu + Math.sqrt(node.variance) * (z[i * g.nodes.length + j] ?? 0)
    }
    return s
  })
}

/**
 * The log-likelihood of complete data under a linear-Gaussian network: the sum over rows and nodes of the log normal
 * density of each value given its parents (pgmpy's `log_likelihood`).
 *
 * @param g - The network.
 * @param rows - Complete observations.
 * @returns The total log-likelihood.
 * @example
 * ```ts
 * import { linearGaussianBayesNet, linearGaussianLogLikelihood } from 'whitebox-engine/pgm'
 * const g = linearGaussianBayesNet([{ name: 'x', parents: [], weights: [], intercept: 0, variance: 1 }])
 * linearGaussianLogLikelihood(g, [{ x: 0 }]) === -0.5 * Math.log(2 * Math.PI) // => true
 * ```
 */
export function linearGaussianLogLikelihood(
  g: LinearGaussianBayesNet,
  rows: readonly Readonly<Record<string, number>>[],
): number {
  let total = 0
  for (const row of rows) {
    for (const node of g.nodes) {
      const x = row[node.name]
      if (x === undefined) throw new WhiteboxError('value', `linearGaussianLogLikelihood: a row has no ${node.name}`)
      const mu = node.intercept + node.weights.reduce((acc, w, k) => acc + w * (row[node.parents[k] ?? ''] ?? 0), 0)
      total += -0.5 * Math.log(2 * Math.PI * node.variance) - ((x - mu) * (x - mu)) / (2 * node.variance)
    }
  }
  return total
}

/**
 * A Gaussian Markov random field: a multivariate Gaussian given by its mean and (sparse) precision matrix; zeros of the
 * precision are missing edges, so its graph is the Markov network of the distribution.
 *
 * @concept gaussian-markov-random-fields
 */
export interface Gmrf {
  readonly kind: 'gmrf'
  readonly variables: readonly string[]
  readonly mean: readonly number[]
  readonly precision: readonly (readonly number[])[]
}

/**
 * Build a Gaussian Markov random field from its mean and precision (checked symmetric and positive definite).
 *
 * @param variables - The variables.
 * @param mean - The mean.
 * @param precision - The precision matrix.
 * @returns The field.
 * @throws WhiteboxError with code `value` when the precision is not symmetric positive definite.
 * @concept gaussian-markov-random-fields
 * @example
 * ```ts
 * import { gmrf } from 'whitebox-engine/pgm'
 * gmrf(['a', 'b'], [0, 0], [[2, -1], [-1, 2]]).precision[0] // => [2, -1]
 * ```
 */
export function gmrf(variables: readonly string[], mean: readonly number[], precision: Matrix): Gmrf {
  const n = variables.length
  if (mean.length !== n || precision.length !== n || precision.some((r) => r.length !== n))
    throw new WhiteboxError('shape', `gmrf: expected ${n} means and a ${n} x ${n} precision`)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < i; j++)
      if (
        Math.abs((precision[i]?.[j] ?? 0) - (precision[j]?.[i] ?? 0)) >
        1e-10 * (1 + Math.abs(precision[i]?.[j] ?? 0))
      )
        throw new WhiteboxError('value', 'gmrf: the precision must be symmetric')
  logdetPositive(precision, 'gmrf')
  return { kind: 'gmrf', variables: [...variables], mean: [...mean], precision: symmetrise(precision) }
}

/**
 * A Gaussian Markov random field in information form: p(x) proportional to exp(potential^T x - x^T precision x / 2),
 * the form Gaussian belief propagation works in (its mean, precision^{-1} potential, is what it computes).
 *
 * @concept gaussian-markov-random-fields
 */
export interface GmrfInformation {
  readonly kind: 'gmrfInformation'
  readonly variables: readonly string[]
  readonly precision: readonly (readonly number[])[]
  readonly potential: readonly number[]
}

/**
 * A Gaussian Markov random field in information form, without solving for its mean.
 *
 * @param variables - The variables.
 * @param precision - The precision matrix (symmetric).
 * @param potential - The potential vector h.
 * @returns The field.
 * @throws WhiteboxError with code `shape` for mismatched sizes, `value` for an asymmetric precision.
 * @concept gaussian-markov-random-fields
 * @example
 * ```ts
 * import { gmrfInformation } from 'whitebox-engine/pgm'
 * gmrfInformation(['a', 'b'], [[2, -1], [-1, 2]], [1, 0]).potential // => [1, 0]
 * ```
 */
export function gmrfInformation(
  variables: readonly string[],
  precision: Matrix,
  potential: readonly number[],
): GmrfInformation {
  const n = variables.length
  if (potential.length !== n || precision.length !== n || precision.some((r) => r.length !== n))
    throw new WhiteboxError('shape', `gmrfInformation: expected ${n} potentials and a ${n} x ${n} precision`)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < i; j++)
      if (
        Math.abs((precision[i]?.[j] ?? 0) - (precision[j]?.[i] ?? 0)) >
        1e-10 * (1 + Math.abs(precision[i]?.[j] ?? 0))
      )
        throw new WhiteboxError('value', 'gmrfInformation: the precision must be symmetric')
  return {
    kind: 'gmrfInformation',
    variables: [...variables],
    precision: symmetrise(precision),
    potential: [...potential],
  }
}

/**
 * The Gaussian Markov random field of a linear-Gaussian network: precision (I - W)^T D^{-1} (I - W), whose sparsity is
 * the network's moral graph.
 *
 * @param g - The network.
 * @returns The field over the network's variables.
 * @concept gaussian-markov-random-fields
 * @example
 * ```ts
 * import { linearGaussianBayesNet, gmrfFromLinearGaussian } from 'whitebox-engine/pgm'
 * const g = linearGaussianBayesNet([
 *   { name: 'x', parents: [], weights: [], intercept: 0, variance: 1 },
 *   { name: 'y', parents: ['x'], weights: [2], intercept: 0, variance: 1 },
 * ])
 * gmrfFromLinearGaussian(g).precision // => [[5, -2], [-2, 1]]
 * ```
 */
export function gmrfFromLinearGaussian(g: LinearGaussianBayesNet): Gmrf {
  const names = g.nodes.map((n) => n.name)
  const n = names.length
  const Q: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (const [i, node] of g.nodes.entries()) {
    // Row i of (I - W): 1 at i, -w_k at parent k; Q += row^T row / variance.
    const idx = [i, ...node.parents.map((p) => names.indexOf(p))]
    const coef = [1, ...node.weights.map((w) => -w)]
    for (const [a, ia] of idx.entries())
      for (const [b, ib] of idx.entries())
        (Q[ia] as number[])[ib] = (Q[ia]?.[ib] ?? 0) + ((coef[a] ?? 0) * (coef[b] ?? 0)) / node.variance
  }
  return { kind: 'gmrf', variables: names, mean: toJointGaussian(g).mean, precision: Q }
}

/**
 * The undirected graph of a Gaussian Markov random field: an edge wherever the precision is non-zero.
 *
 * @param field - The field.
 * @param tol - Entries at most this in magnitude count as zero (default 0).
 * @returns The graph.
 * @example
 * ```ts
 * import { gmrf, gmrfGraph } from 'whitebox-engine/pgm'
 * [...(gmrfGraph(gmrf(['a', 'b', 'c'], [0, 0, 0], [[2, -1, 0], [-1, 2, -1], [0, -1, 2]])).get('a') ?? [])] // => ['b']
 * ```
 */
export function gmrfGraph(field: Gmrf, tol = 0): UndirectedGraph {
  const edges: [string, string][] = []
  const v = field.variables
  for (let i = 0; i < v.length; i++)
    for (let j = i + 1; j < v.length; j++)
      if (Math.abs(field.precision[i]?.[j] ?? 0) > tol) edges.push([v[i] as string, v[j] as string])
  return undirectedGraph(edges, v)
}

/**
 * Condition a Gaussian Markov random field on observed values: the rest is again a GMRF with precision Q_AA and mean
 * mu_A - Q_AA^{-1} Q_AB (x_B - mu_B), with no matrix inverse beyond a solve.
 *
 * @param field - The field.
 * @param evidence - Observed values by variable.
 * @returns The conditional field over the unobserved variables.
 * @concept gaussian-markov-random-fields
 * @example
 * ```ts
 * import { gmrf, gmrfConditional } from 'whitebox-engine/pgm'
 * gmrfConditional(gmrf(['a', 'b'], [0, 0], [[2, -1], [-1, 2]]), { b: 2 }).mean // => [1]
 * ```
 */
export function gmrfConditional(field: Gmrf, evidence: Readonly<Record<string, number>>): Gmrf {
  const a = field.variables.map((_, i) => i).filter((i) => !((field.variables[i] ?? '') in evidence))
  const b = field.variables.map((_, i) => i).filter((i) => (field.variables[i] ?? '') in evidence)
  const Qaa = sub(field.precision, a, a)
  const Qab = sub(field.precision, a, b)
  const dev = b.map((i) => (evidence[field.variables[i] ?? ''] ?? 0) - (field.mean[i] ?? 0))
  const shift = solveVector(Qaa, matVec(Qab, dev))
  return {
    kind: 'gmrf',
    variables: a.map((i) => field.variables[i] ?? ''),
    mean: a.map((i, k) => (field.mean[i] ?? 0) - (shift[k] ?? 0)),
    precision: Qaa,
  }
}

/**
 * The marginal covariance of a Gaussian Markov random field (the inverse of its precision).
 *
 * @param field - The field.
 * @returns The covariance matrix.
 * @example
 * ```ts
 * import { gmrf, gmrfCovariance } from 'whitebox-engine/pgm'
 * gmrfCovariance(gmrf(['a'], [0], [[4]])) // => [[0.25]]
 * ```
 */
export function gmrfCovariance(field: Gmrf): number[][] {
  return symmetrise(inverse(field.precision))
}

/**
 * The log density of a Gaussian Markov random field at a point:
 * (log |Q| - n log 2 pi - (x - mu)^T Q (x - mu)) / 2.
 *
 * @param field - The field.
 * @param x - A value for every variable.
 * @returns The log density.
 * @example
 * ```ts
 * import { gmrf, gmrfLogDensity } from 'whitebox-engine/pgm'
 * gmrfLogDensity(gmrf(['a'], [0], [[1]]), { a: 0 }) === -0.5 * Math.log(2 * Math.PI) // => true
 * ```
 */
export function gmrfLogDensity(field: Gmrf, x: Readonly<Record<string, number>>): number {
  const d = field.variables.map((v, i) => (x[v] ?? 0) - (field.mean[i] ?? 0))
  const n = d.length
  return (
    (logdetPositive(field.precision, 'gmrfLogDensity') -
      n * Math.log(2 * Math.PI) -
      dotVec(d, matVec(field.precision, d))) /
    2
  )
}

/**
 * Samples from a Gaussian Markov random field through the Cholesky factor of its precision (Rue and Held, algorithm
 * 2.4): with Q = L L^T, x = mu + L^{-T} z.
 *
 * @param field - The field.
 * @param key - The random key.
 * @param n - Number of samples.
 * @returns The samples as records by variable.
 * @concept sampling
 * @example
 * ```ts
 * import { gmrf, gmrfSample } from 'whitebox-engine/pgm'
 * import { key } from 'whitebox-engine/random'
 * gmrfSample(gmrf(['a', 'b'], [0, 1], [[2, -1], [-1, 2]]), key(0), 3).length // => 3
 * ```
 */
export function gmrfSample(field: Gmrf, key: Key, n: number): Record<string, number>[] {
  const L = choleskyLower(field.precision)
  const d = field.variables.length
  const z = normal(key, [n, d]).data
  return Array.from({ length: n }, (_, s) => {
    // Back-substitution for L^T y = z.
    const y = new Array<number>(d).fill(0)
    for (let i = d - 1; i >= 0; i--) {
      let acc = z[s * d + i] ?? 0
      for (let j = i + 1; j < d; j++) acc -= (L[j]?.[i] ?? 0) * (y[j] ?? 0)
      y[i] = acc / (L[i]?.[i] ?? 1)
    }
    return Object.fromEntries(field.variables.map((v, i) => [v, (field.mean[i] ?? 0) + (y[i] ?? 0)]))
  })
}

/**
 * A Gaussian Markov random field as a canonical Gaussian factor (K = Q, h = Q mu, normalised).
 *
 * @param field - The field.
 * @returns The factor.
 * @example
 * ```ts
 * import { gmrf, gmrfToFactor } from 'whitebox-engine/pgm'
 * gmrfToFactor(gmrf(['a'], [2], [[3]])).h // => [6]
 * ```
 */
export function gmrfToFactor(field: Gmrf): GaussianFactor {
  const h = matVec(field.precision, field.mean)
  const n = field.variables.length
  const g =
    -dotVec(field.mean, h) / 2 - (n * Math.log(2 * Math.PI) - logdetPositive(field.precision, 'gmrfToFactor')) / 2
  return gaussianFactor(field.variables, field.precision, h, g)
}
