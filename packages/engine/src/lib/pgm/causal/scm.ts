/**
 * Structural causal models (Pearl 2009): each variable is a deterministic function of its parents and its own
 * independent noise, given as discrete tables or linear equations. Construction and validation, sampling, the Bayes
 * network or linear-Gaussian network the model induces, and a `Structure` for drawing it.
 */

import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, laplace, normal, uniform } from '../../random/index.ts'
import { type Structure, type StructureEdge, type StructureNode, structure } from '../../scene/index.ts'
import { topologicalOrder } from '../graphs.ts'
import { type BayesNet, type BayesNode, bayesNet } from '../models/bayes-net.ts'
import { type LinearGaussianBayesNet, linearGaussianBayesNet } from '../models/linear-gaussian.ts'

/**
 * A discrete structural assignment X := f(parents, U) with a discrete exogenous noise U: `noise` gives P(U), and
 * `assign` the state of X for every (parents, U) configuration, row-major over [...parents, U].
 */
export interface DiscreteAssignment {
  readonly kind: 'discrete'
  readonly name: string
  readonly states: readonly string[]
  readonly parents: readonly string[]
  readonly noise: readonly number[]
  readonly assign: readonly number[]
}

/** A linear structural assignment X := intercept + weights . parents + U, with U drawn from a centred noise law. */
export interface LinearAssignment {
  readonly kind: 'linear'
  readonly name: string
  readonly parents: readonly string[]
  readonly weights: readonly number[]
  readonly intercept: number
  readonly noise: { readonly distribution: 'normal' | 'uniform' | 'laplace'; readonly scale: number }
}

/**
 * A structural causal model (Pearl 2009): endogenous variables, each a deterministic function of its parents and its
 * own exogenous noise, the noises independent. Discrete models give Bayes networks by marginalising the noise and
 * support counterfactuals by abduction; linear models give linear-Gaussian networks (with Gaussian noise) and are the
 * setting of LiNGAM and NOTEARS.
 *
 * @concept structural-causal-models
 */
export interface Scm {
  readonly kind: 'scm'
  readonly variables: readonly (DiscreteAssignment | LinearAssignment)[]
}

/**
 * Build and validate a structural causal model (all assignments discrete or all linear): parents must exist, the graph
 * must be acyclic, noise distributions must be distributions and assignment tables complete.
 *
 * @param variables - The assignments, in any order.
 * @returns The model, in topological order.
 * @throws OmaiError with code `value` for invalid structure or tables.
 * @concept structural-causal-models
 * @example
 * ```ts
 * import { scm } from 'omai-engine/pgm/causal'
 * const m = scm([
 *   { kind: 'linear', name: 'y', parents: ['x'], weights: [2], intercept: 0, noise: { distribution: 'normal', scale: 1 } },
 *   { kind: 'linear', name: 'x', parents: [], weights: [], intercept: 1, noise: { distribution: 'normal', scale: 1 } },
 * ])
 * m.variables.map((v) => v.name) // => ['x', 'y']
 * ```
 */
export function scm(variables: readonly (DiscreteAssignment | LinearAssignment)[]): Scm {
  const kinds = new Set(variables.map((v) => v.kind))
  if (kinds.size > 1) throw new OmaiError('value', 'scm: assignments must be all discrete or all linear')
  const byName = new Map(variables.map((v) => [v.name, v]))
  if (byName.size !== variables.length) throw new OmaiError('value', 'scm: names must be distinct')
  for (const v of variables) {
    for (const p of v.parents)
      if (!byName.has(p)) throw new OmaiError('value', `scm: ${v.name} has unknown parent ${p}`)
    if (v.kind === 'discrete') {
      const rows = v.parents.reduce((k, p) => {
        const pv = byName.get(p)
        return k * (pv?.kind === 'discrete' ? pv.states.length : 1)
      }, 1)
      if (v.assign.length !== rows * v.noise.length)
        throw new OmaiError('value', `scm: ${v.name} needs ${rows * v.noise.length} assignment entries`)
      if (v.assign.some((s) => !(Number.isInteger(s) && s >= 0 && s < v.states.length)))
        throw new OmaiError('value', `scm: ${v.name} assigns a state outside its states`)
      if (v.noise.some((p) => !(p >= 0)) || Math.abs(v.noise.reduce((a, b) => a + b, 0) - 1) > 1e-9)
        throw new OmaiError('value', `scm: the noise of ${v.name} is not a distribution`)
    } else {
      if (v.weights.length !== v.parents.length)
        throw new OmaiError('value', `scm: ${v.name} needs one weight per parent`)
      if (!(v.noise.scale >= 0)) throw new OmaiError('value', `scm: the noise scale of ${v.name} must be non-negative`)
    }
  }
  try {
    return { kind: 'scm', variables: topologicalOrder(variables) }
  } catch (e) {
    throw new OmaiError('value', `scm: ${(e as Error).message}`)
  }
}

function discreteVar(m: Scm, name: string): DiscreteAssignment {
  const v = m.variables.find((x) => x.name === name)
  if (v === undefined || v.kind !== 'discrete') throw new OmaiError('value', `scm: ${name} is not a discrete variable`)
  return v
}

/**
 * Evaluate a discrete assignment: the state of a variable given its parents' states and its noise state.
 *
 * @param m - The model.
 * @param name - The variable.
 * @param values - State indices of (at least) its parents.
 * @param noise - Its noise state.
 * @returns The state index.
 * @example
 * ```ts
 * import { evaluateAssignment, scm } from 'omai-engine/pgm/causal'
 * const m = scm([{ kind: 'discrete', name: 'x', states: ['0', '1'], parents: [], noise: [0.5, 0.5], assign: [1, 0] }])
 * evaluateAssignment(m, 'x', {}, 0) // => 1
 * ```
 */
export function evaluateAssignment(
  m: Scm,
  name: string,
  values: Readonly<Record<string, number>>,
  noise: number,
): number {
  const v = discreteVar(m, name)
  let row = 0
  for (const p of v.parents) row = row * discreteVar(m, p).states.length + (values[p] ?? 0)
  return v.assign[row * v.noise.length + noise] ?? 0
}

/**
 * Draw samples from a structural causal model by evaluating its assignments in topological order on independent noise
 * draws. Discrete samples give state names; linear samples give numbers.
 *
 * @param m - The model.
 * @param key - The random key.
 * @param n - Number of samples.
 * @returns The samples by variable.
 * @concept structural-causal-models
 * @example
 * ```ts
 * import { scm, scmSample } from 'omai-engine/pgm/causal'
 * import { key } from 'omai-engine/random'
 * const m = scm([{ kind: 'linear', name: 'x', parents: [], weights: [], intercept: 3, noise: { distribution: 'normal', scale: 0 } }])
 * scmSample(m, key(0), 2) // => [{ x: 3 }, { x: 3 }]
 * ```
 */
export function scmSample(m: Scm, key: Key, n: number): Record<string, string | number>[] {
  const draws = m.variables.map((v, j) => {
    const k = foldIn(key, j)
    if (v.kind === 'discrete') return uniform(k, [n]).data
    const s = v.noise.scale
    if (v.noise.distribution === 'normal') return normal(k, [n]).data.map((z) => z * s)
    if (v.noise.distribution === 'laplace') return laplace(k, [n], 0, s).data
    return uniform(k, [n]).data.map((u) => (2 * u - 1) * s)
  })
  return Array.from({ length: n }, (_, i) => {
    const values: Record<string, number> = {}
    const out: Record<string, string | number> = {}
    for (const [j, v] of m.variables.entries()) {
      const u = draws[j]?.[i] ?? 0
      if (v.kind === 'discrete') {
        let acc = 0
        let noise = v.noise.length - 1
        for (const [s, p] of v.noise.entries()) {
          acc += p
          if (u < acc) {
            noise = s
            break
          }
        }
        values[v.name] = evaluateAssignment(m, v.name, values, noise)
        out[v.name] = v.states[values[v.name] ?? 0] ?? ''
      } else {
        values[v.name] = v.intercept + v.weights.reduce((s, w, k) => s + w * (values[v.parents[k] ?? ''] ?? 0), 0) + u
        out[v.name] = values[v.name] ?? 0
      }
    }
    return out
  })
}

/**
 * The Bayes network a discrete SCM induces: P(x | parents) = sum over noise states u with f(parents, u) = x of P(u).
 *
 * @param m - A discrete model.
 * @returns The network (same graph over the endogenous variables).
 * @throws OmaiError with code `value` for a linear model.
 * @concept structural-causal-models
 * @example
 * ```ts
 * import { scm, scmToBayesNet } from 'omai-engine/pgm/causal'
 * const m = scm([{ kind: 'discrete', name: 'x', states: ['0', '1'], parents: [], noise: [0.3, 0.7], assign: [1, 1] }])
 * scmToBayesNet(m).nodes[0]?.cpt // => [0, 1]
 * ```
 */
export function scmToBayesNet(m: Scm): BayesNet {
  const nodes: BayesNode[] = m.variables.map((v) => {
    if (v.kind !== 'discrete') throw new OmaiError('value', 'scmToBayesNet: needs a discrete model')
    const r = v.states.length
    const rows = v.assign.length / v.noise.length
    const cpt = new Array<number>(rows * r).fill(0)
    for (let row = 0; row < rows; row++)
      for (const [u, p] of v.noise.entries()) {
        const x = v.assign[row * v.noise.length + u] ?? 0
        cpt[row * r + x] = (cpt[row * r + x] ?? 0) + p
      }
    return { name: v.name, states: v.states, parents: v.parents, cpt }
  })
  return bayesNet(nodes)
}

/**
 * The linear-Gaussian network of a linear SCM with Gaussian noise (variance scale^2).
 *
 * @param m - A linear model with normal noise.
 * @returns The network.
 * @throws OmaiError with code `value` for discrete models or non-Gaussian noise.
 * @concept structural-causal-models
 * @example
 * ```ts
 * import { scm, scmToLinearGaussian } from 'omai-engine/pgm/causal'
 * const m = scm([{ kind: 'linear', name: 'x', parents: [], weights: [], intercept: 1, noise: { distribution: 'normal', scale: 2 } }])
 * scmToLinearGaussian(m).nodes[0]?.variance // => 4
 * ```
 */
export function scmToLinearGaussian(m: Scm): LinearGaussianBayesNet {
  return linearGaussianBayesNet(
    m.variables.map((v) => {
      if (v.kind !== 'linear' || v.noise.distribution !== 'normal')
        throw new OmaiError('value', 'scmToLinearGaussian: needs linear assignments with normal noise')
      return {
        name: v.name,
        parents: v.parents,
        weights: v.weights,
        intercept: v.intercept,
        variance: Math.max(v.noise.scale ** 2, Number.MIN_VALUE),
      }
    }),
  )
}

/**
 * A structural causal model as a {@link Structure}: endogenous variables as `deterministic` nodes (each is a function
 * of its parents and its noise) and, optionally, their exogenous noises as `latent` nodes.
 *
 * @param m - The model.
 * @param options - `noise` to draw the exogenous variables (default true).
 * @returns The structure.
 * @concept structures
 * @example
 * ```ts
 * import { scm, scmStructure } from 'omai-engine/pgm/causal'
 * const m = scm([{ kind: 'linear', name: 'x', parents: [], weights: [], intercept: 0, noise: { distribution: 'normal', scale: 1 } }])
 * scmStructure(m).nodes.map((n) => n.role) // => ['deterministic', 'latent']
 * ```
 */
export function scmStructure(m: Scm, options: { readonly noise?: boolean } = {}): Structure {
  const withNoise = options.noise ?? true
  const nodes: StructureNode[] = []
  const edges: StructureEdge[] = []
  for (const v of m.variables) {
    nodes.push({
      id: v.name,
      label: v.name,
      role: 'deterministic',
      detail: v.kind === 'discrete' ? v.states.join(', ') : 'linear',
    })
    if (withNoise) {
      nodes.push({ id: `U_${v.name}`, label: `U ${v.name}`, role: 'latent' })
      edges.push({ from: `U_${v.name}`, to: v.name, directed: true, kind: 'dependency' })
    }
    for (const p of v.parents) edges.push({ from: p, to: v.name, directed: true, kind: 'dependency' })
  }
  return structure('scm', nodes, edges, { title: 'Structural causal model', direction: 'down' })
}
