/**
 * Factor graphs, the bipartite graph of variables and the factors over them that inference algorithms run on, and the
 * shared plumbing that lets any discrete model (Bayes network, Markov network or factor graph) be used as one: its
 * variables and factors, evidence by state name or index, unnormalised log probabilities and a forest test.
 */

import { MdmlError } from '../../base/index.ts'
import { type DiscreteVariable, type Factor, logFactorAt } from '../factor.ts'
import { type BayesNet, bayesFactors } from './bayes-net.ts'
import type { MarkovNetwork } from './markov-network.ts'

/**
 * A factor graph: discrete variables and factors over them, a bipartite graph between the two. Every discrete model
 * converts to one, and the message-passing algorithms run on it.
 *
 * @concept factor-graphs
 */
export interface FactorGraph {
  readonly kind: 'factorGraph'
  readonly variables: readonly DiscreteVariable[]
  readonly factors: readonly Factor[]
}

/** Any discrete model the inference algorithms accept. */
export type DiscreteModel = BayesNet | MarkovNetwork | FactorGraph

/**
 * Build and validate a factor graph: every factor's variables must be declared, with matching cardinalities.
 *
 * @param variables - The variables.
 * @param factors - The factors.
 * @returns The factor graph.
 * @throws MdmlError with code `value` for undeclared variables or mismatched cardinalities.
 * @concept factor-graphs
 * @example
 * ```ts
 * import { factor, factorGraph } from 'mdml-engine/pgm'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }], [factor(['a'], [2], [1, 3])])
 * fg.factors.length // => 1
 * ```
 */
export function factorGraph(variables: readonly DiscreteVariable[], factors: readonly Factor[]): FactorGraph {
  const card = new Map(variables.map((v) => [v.name, v.states.length]))
  if (card.size !== variables.length) throw new MdmlError('value', 'factorGraph: variable names must be distinct')
  for (const f of factors) {
    for (const [i, v] of f.variables.entries()) {
      const c = card.get(v)
      if (c === undefined) throw new MdmlError('value', `factorGraph: factor mentions undeclared variable ${v}`)
      if (c !== f.cardinality[i])
        throw new MdmlError('value', `factorGraph: ${v} has ${c} states but a factor gives ${f.cardinality[i]}`)
    }
  }
  return { kind: 'factorGraph', variables: variables.map((v) => ({ name: v.name, states: [...v.states] })), factors }
}

/**
 * The variables of any discrete model, in model order.
 *
 * @param model - A Bayes network, Markov network or factor graph.
 * @returns The variables with their states.
 * @example
 * ```ts
 * import { bayesNet, modelVariables } from 'mdml-engine/pgm'
 * modelVariables(bayesNet([{ name: 'a', states: ['x', 'y'], parents: [], cpt: [0.5, 0.5] }])) // => [{ name: 'a', states: ['x', 'y'] }]
 * ```
 */
export function modelVariables(model: DiscreteModel): DiscreteVariable[] {
  if (model.kind === 'bayesNet') return model.nodes.map((n) => ({ name: n.name, states: n.states }))
  return [...model.variables]
}

/**
 * The factors of any discrete model: the CPTs of a Bayes network, the potentials of a Markov network.
 *
 * @param model - The model.
 * @returns Its factors.
 * @example
 * ```ts
 * import { bayesNet, modelFactors } from 'mdml-engine/pgm'
 * modelFactors(bayesNet([{ name: 'a', states: ['x', 'y'], parents: [], cpt: [0.5, 0.5] }])).length // => 1
 * ```
 */
export function modelFactors(model: DiscreteModel): Factor[] {
  if (model.kind === 'bayesNet') return bayesFactors(model)
  if (model.kind === 'markovNetwork') return [...model.potentials]
  return [...model.factors]
}

/**
 * Any discrete model as a factor graph (one factor node per CPT or potential).
 *
 * @param model - The model.
 * @returns The factor graph.
 * @concept factor-graphs
 * @example
 * ```ts
 * import { bayesNet, toFactorGraph } from 'mdml-engine/pgm'
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: [0.5, 0.5] }, { name: 'b', states: ['0', '1'], parents: ['a'], cpt: [1, 0, 0, 1] }])
 * toFactorGraph(bn).factors.map((f) => f.variables) // => [['a'], ['a', 'b']]
 * ```
 */
export function toFactorGraph(model: DiscreteModel): FactorGraph {
  if (model.kind === 'factorGraph') return model
  return factorGraph(modelVariables(model), modelFactors(model))
}

/**
 * A variable of a model by name.
 *
 * @param model - The model.
 * @param name - The variable.
 * @returns The variable.
 * @throws MdmlError with code `value` for an unknown name.
 * @example
 * ```ts
 * import { factorGraph, modelVariable } from 'mdml-engine/pgm'
 * modelVariable(factorGraph([{ name: 'a', states: ['x', 'y'] }], []), 'a').states // => ['x', 'y']
 * ```
 */
export function modelVariable(model: DiscreteModel, name: string): DiscreteVariable {
  const v = modelVariables(model).find((x) => x.name === name)
  if (v === undefined) throw new MdmlError('value', `unknown variable ${name}`)
  return v
}

/**
 * Evidence given by state names or indices, as state indices (checked against the model).
 *
 * @param model - The model.
 * @param evidence - Observed states by variable.
 * @returns State indices by variable.
 * @throws MdmlError with code `value` for unknown variables or states.
 * @example
 * ```ts
 * import { factorGraph, evidenceIndices } from 'mdml-engine/pgm'
 * evidenceIndices(factorGraph([{ name: 'a', states: ['x', 'y'] }], []), { a: 'y' }) // => { a: 1 }
 * ```
 */
export function evidenceIndices(
  model: DiscreteModel,
  evidence: Readonly<Record<string, string | number>>,
): Record<string, number> {
  const variables = new Map(modelVariables(model).map((v) => [v.name, v]))
  return Object.fromEntries(
    Object.entries(evidence).map(([k, v]) => {
      const variable = variables.get(k)
      if (variable === undefined) throw new MdmlError('value', `evidence: unknown variable ${k}`)
      const i = typeof v === 'number' ? v : variable.states.indexOf(v)
      if (!Number.isInteger(i) || i < 0 || i >= variable.states.length)
        throw new MdmlError('value', `evidence: ${k} has no state ${String(v)}`)
      return [k, i]
    }),
  )
}

/**
 * State names for an assignment of state indices.
 *
 * @param model - The model.
 * @param assignment - State indices by variable.
 * @returns State names by variable.
 * @example
 * ```ts
 * import { factorGraph, stateNames } from 'mdml-engine/pgm'
 * stateNames(factorGraph([{ name: 'a', states: ['x', 'y'] }], []), { a: 1 }) // => { a: 'y' }
 * ```
 */
export function stateNames(model: DiscreteModel, assignment: Readonly<Record<string, number>>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(assignment).map(([k, i]) => [k, modelVariable(model, k).states[i] ?? String(i)]),
  )
}

/**
 * The unnormalised log probability of a full assignment: the sum of the log factor values (for a Bayes network, the log
 * joint probability).
 *
 * @param model - The model.
 * @param assignment - A state index for every variable.
 * @returns The sum of log factor values.
 * @example
 * ```ts
 * import { factor, factorGraph, unnormalisedLogProbability } from 'mdml-engine/pgm'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }], [factor(['a'], [2], [1, Math.E])])
 * unnormalisedLogProbability(fg, { a: 1 }) // => 1
 * ```
 */
export function unnormalisedLogProbability(model: DiscreteModel, assignment: Readonly<Record<string, number>>): number {
  let s = 0
  for (const f of modelFactors(model)) s += logFactorAt(f, assignment)
  return s
}

/**
 * The factors of each variable (indices into the factor list), the adjacency of the bipartite graph.
 *
 * @param fg - The factor graph.
 * @returns Factor indices by variable.
 * @example
 * ```ts
 * import { factor, factorGraph, variableFactors } from 'mdml-engine/pgm'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }], [factor(['a'], [2], [1, 1]), factor(['a'], [2], [1, 2])])
 * variableFactors(fg).get('a') // => [0, 1]
 * ```
 */
export function variableFactors(fg: FactorGraph): Map<string, number[]> {
  const out = new Map<string, number[]>(fg.variables.map((v) => [v.name, []]))
  for (const [i, f] of fg.factors.entries()) for (const v of f.variables) out.get(v)?.push(i)
  return out
}

/**
 * Whether a factor graph is a forest (its bipartite variable-factor graph has no cycles), the case where belief
 * propagation is exact.
 *
 * @param fg - The factor graph.
 * @returns `true` for a tree or forest.
 * @concept factor-graphs
 * @example
 * ```ts
 * import { factor, factorGraph, isFactorTree } from 'mdml-engine/pgm'
 * const v = ['a', 'b'].map((name) => ({ name, states: ['0', '1'] }))
 * isFactorTree(factorGraph(v, [factor(['a', 'b'], [2, 2], [1, 1, 1, 1])])) // => true
 * isFactorTree(factorGraph(v, [factor(['a', 'b'], [2, 2], [1, 1, 1, 1]), factor(['a', 'b'], [2, 2], [1, 1, 1, 1])])) // => false
 * ```
 */
export function isFactorTree(fg: FactorGraph): boolean {
  // A forest has (vertices - components) edges; count with union-find.
  const parent = new Map<string, string>()
  const find = (x: string): string => {
    let r = x
    while ((parent.get(r) ?? r) !== r) r = parent.get(r) ?? r
    return r
  }
  for (const v of fg.variables) parent.set(`v:${v.name}`, `v:${v.name}`)
  for (const [i, f] of fg.factors.entries()) {
    parent.set(`f:${i}`, `f:${i}`)
    for (const v of f.variables) {
      const a = find(`f:${i}`)
      const b = find(`v:${v}`)
      if (a === b) return false
      parent.set(a, b)
    }
  }
  return true
}
