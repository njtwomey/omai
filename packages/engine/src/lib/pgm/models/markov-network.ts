/**
 * Markov networks (Markov random fields), undirected models given by non-negative potentials on the cliques of a graph:
 * construction, neighbours and separation, maximal cliques, moralising a Bayes network, conversion from any discrete
 * model, the Ising and Potts models of statistical physics, and the brute-force Gibbs distribution and energy of an
 * assignment.
 */

import { WhiteboxError } from '../../base/index.ts'
import {
  type DiscreteVariable,
  type Factor,
  factorProductAll,
  logFactor,
  normaliseFactor,
  permuteFactor,
  unitFactor,
} from '../factor.ts'
import { graphEdges, interactionGraph, moralGraph, separated, undirectedGraph } from '../graphs.ts'
import { maximalCliques } from '../triangulation.ts'
import { type BayesNet, bayesFactors } from './bayes-net.ts'
import { type DiscreteModel, modelFactors, modelVariables, unnormalisedLogProbability } from './factor-graph.ts'

/**
 * A Markov network (Markov random field): discrete variables, an undirected graph, and non-negative potentials whose
 * scopes are cliques of the graph. The joint is the Gibbs distribution p(x) = (1/Z) prod_c psi_c(x_c).
 *
 * @concept markov-networks
 */
export interface MarkovNetwork {
  readonly kind: 'markovNetwork'
  readonly variables: readonly DiscreteVariable[]
  /** Undirected edges, each once. */
  readonly edges: readonly (readonly [string, string])[]
  readonly potentials: readonly Factor[]
}

/**
 * Build and validate a Markov network. The graph defaults to the interaction graph of the potentials; when edges are
 * given, every potential's scope must be a clique of it.
 *
 * @param variables - The variables.
 * @param potentials - The clique potentials.
 * @param edges - The undirected edges (default: implied by the potentials).
 * @returns The network.
 * @throws WhiteboxError with code `value` for undeclared variables, mismatched cardinalities or a potential whose scope
 * is not a clique.
 * @concept markov-networks
 * @example
 * ```ts
 * import { factor, markovNetwork } from 'whitebox-engine/pgm'
 * const v = ['a', 'b'].map((name) => ({ name, states: ['0', '1'] }))
 * markovNetwork(v, [factor(['a', 'b'], [2, 2], [10, 1, 1, 10])]).edges // => [['a', 'b']]
 * ```
 */
export function markovNetwork(
  variables: readonly DiscreteVariable[],
  potentials: readonly Factor[],
  edges?: readonly (readonly [string, string])[],
): MarkovNetwork {
  const card = new Map(variables.map((v) => [v.name, v.states.length]))
  if (card.size !== variables.length) throw new WhiteboxError('value', 'markovNetwork: variable names must be distinct')
  for (const f of potentials) {
    for (const [i, v] of f.variables.entries()) {
      if (!card.has(v)) throw new WhiteboxError('value', `markovNetwork: potential mentions undeclared variable ${v}`)
      if (card.get(v) !== f.cardinality[i])
        throw new WhiteboxError(
          'value',
          `markovNetwork: ${v} has ${card.get(v)} states but a potential gives ${f.cardinality[i]}`,
        )
    }
  }
  const names = variables.map((v) => v.name)
  const graph = edges === undefined ? interactionGraph(potentials, names) : undirectedGraph(edges, names)
  for (const [a, b] of graphEdges(graph)) {
    if (!card.has(a) || !card.has(b))
      throw new WhiteboxError('value', `markovNetwork: edge ${a}-${b} has an undeclared end`)
  }
  for (const f of potentials) {
    for (let i = 0; i < f.variables.length; i++) {
      for (let j = i + 1; j < f.variables.length; j++) {
        const a = f.variables[i] as string
        const b = f.variables[j] as string
        if (!graph.get(a)?.has(b))
          throw new WhiteboxError('value', `markovNetwork: potential over [${f.variables.join(', ')}] is not a clique`)
      }
    }
  }
  return { kind: 'markovNetwork', variables: variables.map((v) => ({ ...v })), edges: graphEdges(graph), potentials }
}

/**
 * The undirected graph of a Markov network.
 *
 * @param mn - The network.
 * @returns The neighbours of each variable.
 * @example
 * ```ts
 * import { factor, markovNetwork, markovGraph } from 'whitebox-engine/pgm'
 * const v = ['a', 'b'].map((name) => ({ name, states: ['0', '1'] }))
 * [...(markovGraph(markovNetwork(v, [factor(['a', 'b'], [2, 2], [1, 1, 1, 1])])).get('a') ?? [])] // => ['b']
 * ```
 */
export function markovGraph(mn: MarkovNetwork): Map<string, Set<string>> {
  return undirectedGraph(
    mn.edges,
    mn.variables.map((v) => v.name),
  )
}

/**
 * The neighbours of a variable in a Markov network: its Markov blanket.
 *
 * @param mn - The network.
 * @param name - The variable.
 * @returns The neighbours, sorted.
 * @throws WhiteboxError with code `value` for an unknown variable.
 * @concept markov-properties
 * @example
 * ```ts
 * import { isingModel, markovNeighbours } from 'whitebox-engine/pgm'
 * markovNeighbours(isingModel({ rows: 2, cols: 2 }), 's_0_0') // => ['s_0_1', 's_1_0']
 * ```
 */
export function markovNeighbours(mn: MarkovNetwork, name: string): string[] {
  const g = markovGraph(mn)
  const ns = g.get(name)
  if (ns === undefined) throw new WhiteboxError('value', `unknown variable ${name}`)
  return [...ns].sort()
}

/**
 * Whether two sets of variables are separated by a third in a Markov network's graph, which (by the global Markov
 * property) makes them conditionally independent under every distribution that factorises over it.
 *
 * @param mn - The network.
 * @param x - First set.
 * @param y - Second set.
 * @param z - The conditioning set.
 * @returns `true` when separated.
 * @concept markov-properties
 * @example
 * ```ts
 * import { isingModel, markovSeparated } from 'whitebox-engine/pgm'
 * markovSeparated(isingModel({ rows: 1, cols: 3 }), ['s_0_0'], ['s_0_2'], ['s_0_1']) // => true
 * ```
 */
export function markovSeparated(
  mn: MarkovNetwork,
  x: readonly string[],
  y: readonly string[],
  z: readonly string[],
): boolean {
  return separated(markovGraph(mn), x, y, z)
}

/**
 * The maximal cliques of a Markov network's graph.
 *
 * @param mn - The network.
 * @returns The cliques (sorted).
 * @example
 * ```ts
 * import { isingModel, markovCliques } from 'whitebox-engine/pgm'
 * markovCliques(isingModel({ rows: 1, cols: 3 })) // => [['s_0_0', 's_0_1'], ['s_0_1', 's_0_2']]
 * ```
 */
export function markovCliques(mn: MarkovNetwork): string[][] {
  return maximalCliques(markovGraph(mn))
}

/**
 * Moralise a Bayes network: the Markov network on its moral graph whose potentials are its CPTs (the same distribution,
 * with Z = 1).
 *
 * @param bn - The network.
 * @returns The Markov network.
 * @concept moralisation
 * @example
 * ```ts
 * import { bayesNet, moralise } from 'whitebox-engine/pgm'
 * const p = [0.5, 0.5]
 * const bn = bayesNet([{ name: 'a', states: ['0', '1'], parents: [], cpt: p }, { name: 'b', states: ['0', '1'], parents: [], cpt: p }, { name: 'c', states: ['0', '1'], parents: ['a', 'b'], cpt: [1, 0, 0, 1, 0, 1, 1, 0] }])
 * moralise(bn).edges // => [['a', 'b'], ['a', 'c'], ['b', 'c']]
 * ```
 */
export function moralise(bn: BayesNet): MarkovNetwork {
  return markovNetwork(
    bn.nodes.map((n) => ({ name: n.name, states: n.states })),
    bayesFactors(bn),
    graphEdges(moralGraph(bn)),
  )
}

/**
 * Any discrete model as a Markov network on its interaction graph (a factor graph's factors become potentials).
 *
 * @param model - The model.
 * @returns The Markov network.
 * @example
 * ```ts
 * import { factor, factorGraph, toMarkovNetwork } from 'whitebox-engine/pgm'
 * const fg = factorGraph([{ name: 'a', states: ['0', '1'] }, { name: 'b', states: ['0', '1'] }], [factor(['a', 'b'], [2, 2], [1, 2, 3, 4])])
 * toMarkovNetwork(fg).edges // => [['a', 'b']]
 * ```
 */
export function toMarkovNetwork(model: DiscreteModel): MarkovNetwork {
  if (model.kind === 'markovNetwork') return model
  if (model.kind === 'bayesNet') return moralise(model)
  return markovNetwork(model.variables, model.factors)
}

/** A grid or explicit graph for {@link isingModel} and {@link pottsModel}. */
export interface LatticeOptions {
  /** Grid rows (vertices are named `s_<row>_<col>`). */
  readonly rows?: number
  /** Grid columns. */
  readonly cols?: number
  /** Wrap the grid around (a torus). */
  readonly periodic?: boolean
  /** An explicit graph instead of a grid. */
  readonly edges?: readonly (readonly [string, string])[]
  /** The vertices of an explicit graph (default: those of its edges). */
  readonly vertices?: readonly string[]
  /** Coupling J on every edge, or per edge keyed `"a|b"` (default 1). */
  readonly coupling?: number | Readonly<Record<string, number>>
  /** Inverse temperature beta, multiplying every energy (default 1). */
  readonly beta?: number
}

/** The vertices and edges of a lattice. */
function lattice(options: LatticeOptions): { vertices: string[]; edges: [string, string][] } {
  if (options.edges !== undefined) {
    const vs = options.vertices ?? [...new Set(options.edges.flat())]
    return { vertices: [...vs], edges: options.edges.map(([a, b]): [string, string] => [a, b]) }
  }
  const rows = options.rows ?? 1
  const cols = options.cols ?? 1
  const name = (r: number, c: number): string => `s_${r}_${c}`
  const vertices: string[] = []
  const edges: [string, string][] = []
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) vertices.push(name(r, c))
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (c + 1 < cols) edges.push([name(r, c), name(r, c + 1)])
      else if (options.periodic === true && cols > 2) edges.push([name(r, c), name(r, 0)])
      if (r + 1 < rows) edges.push([name(r, c), name(r + 1, c)])
      else if (options.periodic === true && rows > 2) edges.push([name(r, c), name(0, c)])
    }
  }
  return { vertices, edges }
}

function couplingOf(options: LatticeOptions, a: string, b: string): number {
  const j = options.coupling ?? 1
  if (typeof j === 'number') return j
  return j[`${a}|${b}`] ?? j[`${b}|${a}`] ?? 0
}

/**
 * The Ising model as a Markov network: spins s in {-1, +1} (states `'-1'`, `'+1'`) on a grid or graph, with
 * p(s) proportional to exp(beta (sum_edges J s_i s_j + sum_i h_i s_i)).
 *
 * @param options - The lattice, couplings, inverse temperature and field.
 * @returns The Markov network (one pairwise potential per edge, one unary potential per vertex with a field).
 * @concept ising-model
 * @example
 * ```ts
 * import { isingModel, factorValues } from 'whitebox-engine/pgm'
 * const m = isingModel({ rows: 1, cols: 2, coupling: Math.log(2) })
 * m.potentials.map((f) => factorValues(f).map((v) => Math.round(v * 100) / 100)) // => [[2, 0.5, 0.5, 2]]
 * ```
 */
export function isingModel(
  options: LatticeOptions & { readonly field?: number | Readonly<Record<string, number>> } = {},
): MarkovNetwork {
  const { vertices, edges } = lattice(options)
  const beta = options.beta ?? 1
  const spins = [-1, 1]
  const potentials: Factor[] = edges.map(([a, b]) => {
    const j = couplingOf(options, a, b)
    return logFactor(
      [a, b],
      [2, 2],
      spins.flatMap((x) => spins.map((y) => beta * j * x * y)),
    )
  })
  const h = options.field ?? 0
  for (const v of vertices) {
    const hv = typeof h === 'number' ? h : (h[v] ?? 0)
    if (hv !== 0) potentials.push(logFactor([v], [2], [-beta * hv, beta * hv]))
  }
  return markovNetwork(
    vertices.map((name) => ({ name, states: ['-1', '+1'] })),
    potentials,
    edges,
  )
}

/**
 * The q-state Potts model as a Markov network: p(s) proportional to exp(beta sum_edges J [s_i = s_j]), states `'0'` to
 * `q - 1`.
 *
 * @param options - The lattice, couplings, inverse temperature and the number of states q (default 3).
 * @returns The Markov network.
 * @concept ising-model
 * @example
 * ```ts
 * import { pottsModel } from 'whitebox-engine/pgm'
 * pottsModel({ rows: 2, cols: 2, q: 4 }).variables[0]?.states.length // => 4
 * ```
 */
export function pottsModel(options: LatticeOptions & { readonly q?: number } = {}): MarkovNetwork {
  const { vertices, edges } = lattice(options)
  const q = options.q ?? 3
  const beta = options.beta ?? 1
  const potentials = edges.map(([a, b]) => {
    const j = couplingOf(options, a, b)
    const logs: number[] = []
    for (let x = 0; x < q; x++) for (let y = 0; y < q; y++) logs.push(x === y ? beta * j : 0)
    return logFactor([a, b], [q, q], logs)
  })
  return markovNetwork(
    vertices.map((name) => ({ name, states: Array.from({ length: q }, (_, i) => String(i)) })),
    potentials,
    edges,
  )
}

/**
 * The Gibbs distribution of a discrete model by brute force: the normalised product of all its factors, over all its
 * variables in model order (exponential in the number of variables; the definition the inference algorithms
 * approximate).
 *
 * @param model - A Markov network, factor graph or Bayes network.
 * @returns The joint distribution as a normalised factor.
 * @concept gibbs-distributions
 * @example
 * ```ts
 * import { isingModel, gibbsDistribution, factorValues } from 'whitebox-engine/pgm'
 * factorValues(gibbsDistribution(isingModel({ rows: 1, cols: 2, coupling: 0 }))) // => [0.25, 0.25, 0.25, 0.25]
 * ```
 */
export function gibbsDistribution(model: DiscreteModel): Factor {
  const variables = modelVariables(model)
  const all = factorProductAll([
    unitFactor(
      variables.map((v) => v.name),
      variables.map((v) => v.states.length),
    ),
    ...modelFactors(model),
  ])
  return normaliseFactor(
    permuteFactor(
      all,
      variables.map((v) => v.name),
    ),
  )
}

/**
 * The energy of a full assignment, E(x) = -sum_c log psi_c(x_c), so that p(x) = exp(-E(x)) / Z.
 *
 * @param model - The model.
 * @param assignment - A state index for every variable.
 * @returns The energy.
 * @concept gibbs-distributions
 * @example
 * ```ts
 * import { isingModel, energy } from 'whitebox-engine/pgm'
 * energy(isingModel({ rows: 1, cols: 2, coupling: 1 }), { s_0_0: 1, s_0_1: 1 }) // => -1
 * ```
 */
export function energy(model: DiscreteModel, assignment: Readonly<Record<string, number>>): number {
  return -unnormalisedLogProbability(model, assignment)
}
