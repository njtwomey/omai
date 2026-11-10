/**
 * The structured-model oracle of a graph CRF: feature counts, expected feature counts with the log-partition function,
 * and cost-augmented decoding on an input graph of any shape. Inference is chosen per input and is named in every result:
 * exact on a forest (sum-product on the tree) and on graphs of low treewidth (a junction tree), approximate by loopy
 * belief propagation (the Bethe approximation) or tree-reweighted belief propagation (a convex bound) on the rest.
 * Decoding is a junction-tree or max-product MAP, MPLP's linear-programming relaxation, or any function from a
 * discrete model to a MAP assignment, so another MAP engine (graph cuts) plugs in without a change here.
 */

import { OmaiError, softmaxOf } from '../../base/index.ts'
import { type DiscreteVariable, type Factor, logFactor } from '../factor.ts'
import { interactionGraph } from '../graphs.ts'
import { type BeliefPropagationOptions, beliefPropagation } from '../inference/belief-propagation.ts'
import { calibrate, junctionTreeMap, junctionTreeQuery } from '../inference/junction-tree.ts'
import { type MapResult, type MplpOptions, maxProductMap, mplp } from '../inference/map.ts'
import { type TreeReweightedOptions, treeReweighted } from '../inference/variational.ts'
import { type DiscreteModel, type FactorGraph, factorGraph, isFactorTree } from '../models/factor-graph.ts'
import { eliminationOrder, inducedWidth } from '../triangulation.ts'
import {
  checkGraphInput,
  cliqueEntry,
  graphLabelIndices,
  visitClique,
  visitEdge,
  visitNode,
} from './_graph-features.ts'
import type { GraphCrf, GraphInput, GraphLabelling } from './graph-crf.ts'
import type { StructuredOracle } from './oracle.ts'

/**
 * How expected feature counts and the log-partition function are computed. Exact: `tree` (sum-product on a forest, in
 * one leaves-to-root pass and back), `junctionTree` (any graph, at a cost exponential in its treewidth) and `exact`
 * (the tree when the input is a forest, else the junction tree, and an error when the junction tree would be too
 * large). Approximate: `bp` (loopy belief propagation; the log-partition function is the Bethe approximation and the
 * gradient is exact only at a fixed point), `trw` (tree-reweighted belief propagation, pairwise graphs only; the
 * log-partition function is an upper bound and the pseudo-marginals are the gradient of that bound, so it is a
 * consistent convex surrogate) and `auto`, the default: `exact` where that is affordable, `bp` otherwise.
 */
export type GraphInference = 'auto' | 'exact' | 'tree' | 'junctionTree' | 'bp' | 'trw'

/**
 * How the best labelling is found: `auto` (a junction tree where that is affordable, which is exact, and MPLP
 * otherwise), `tree` (max-product on a forest, exact), `junctionTree` (exact at a cost exponential in the treewidth),
 * `maxProduct` (loopy max-product, a heuristic on loops), `mplp` (a linear-programming relaxation, exact when its
 * integrality gap closes), or a function from the discrete model (a `FactorGraph` whose variables are `y0`, `y1`, ...
 * with the labels as states) to a MAP assignment.
 */
export type GraphMap = 'auto' | 'tree' | 'junctionTree' | 'maxProduct' | 'mplp' | ((model: DiscreteModel) => MapResult)

/** Options of {@link graphOracle}. */
export interface GraphOracleOptions {
  /** How expectations are computed (default `auto`). */
  readonly inference?: GraphInference
  /** Belief-propagation options for `bp` and for max-product MAP (default 500 iterations, tolerance 1e-10). */
  readonly bp?: Omit<BeliefPropagationOptions, 'mode' | 'evidence' | 'tracer' | 'policy'>
  /** Options of `trw`. */
  readonly trw?: Omit<TreeReweightedOptions, 'evidence' | 'tracer' | 'policy'>
  /** How decoding finds the best labelling (default `auto`). */
  readonly map?: GraphMap
  /** Options of `mplp`. */
  readonly mplp?: Omit<MplpOptions, 'evidence' | 'tracer' | 'policy'>
  /**
   * The largest number of states a junction-tree clique may have for `auto` and `exact` to use it (labels to the power
   * of the induced width plus one; default 100000).
   */
  readonly maxStates?: number
}

/** The expectations of {@link GraphOracle}: the counts and log Z, and how they were computed. */
export interface GraphExpectations {
  readonly counts: Float64Array
  readonly logPartition: number
  /** Whether the counts and the log-partition function are exact (`tree` and `junctionTree`). */
  readonly exact: boolean
  /** The inference that ran. */
  readonly inference: 'tree' | 'junctionTree' | 'bp' | 'trw'
}

/** The node marginals of {@link GraphOracle.marginals}: the probability of each label at each node. */
export interface GraphMarginals {
  /** Row per node, entry per label. */
  readonly nodes: readonly (readonly number[])[]
  readonly logPartition: number
  readonly exact: boolean
  readonly inference: GraphExpectations['inference']
}

/** The structured oracle of a graph CRF, with the exactness of its inference reported. */
export interface GraphOracle extends StructuredOracle<GraphInput, GraphLabelling> {
  expectations(w: ArrayLike<number>, x: GraphInput): GraphExpectations
  /** Node marginals under the weights. */
  marginals(w: ArrayLike<number>, x: GraphInput): GraphMarginals
  /** The inference `expectations` will use on this input and the induced width it found (1 for a forest). */
  plan(x: GraphInput): {
    readonly inference: GraphExpectations['inference']
    readonly exact: boolean
    readonly width: number
  }
}

interface PairTerm {
  readonly s: number
  readonly t: number
  /** Log potential over (label of s, label of t), row-major. */
  readonly theta: Float64Array
  /** The input edges merged into this pair, and whether they run from t to s. */
  readonly edges: { readonly edge: number; readonly flipped: boolean }[]
}

/** The log potentials of an input under some weights: unary per node, pairwise per node pair, one per clique. */
interface Potentials {
  readonly n: number
  readonly K: number
  readonly unary: Float64Array[]
  readonly pairs: PairTerm[]
  readonly cliques: { readonly nodes: readonly number[]; readonly table: Float64Array }[]
}

function digitsOf(idx: number, K: number, arity: number): number[] {
  const out = new Array<number>(arity)
  let rest = idx
  for (let a = arity - 1; a >= 0; a--) {
    out[a] = rest % K
    rest = Math.floor(rest / K)
  }
  return out
}

function potentials(model: GraphCrf, x: GraphInput, w: ArrayLike<number>): Potentials {
  if (w.length !== model.layout.size)
    throw new OmaiError('shape', `graph oracle: ${w.length} weights for a model with ${model.layout.size} parameters`)
  checkGraphInput(model, x)
  const K = model.layout.labels
  const n = x.nodes.length
  const unary = x.nodes.map((item) => {
    const u = new Float64Array(K)
    for (let k = 0; k < K; k++)
      visitNode(model, item, k, (j, v) => {
        u[k] = (u[k] ?? 0) + v * (w[j] ?? 0)
      })
    return u
  })
  const byPair = new Map<number, PairTerm>()
  for (const [e, edge] of x.edges.entries()) {
    const s = Math.min(edge.from, edge.to)
    const t = Math.max(edge.from, edge.to)
    const flipped = edge.from > edge.to
    let term = byPair.get(s * n + t)
    if (term === undefined) {
      term = { s, t, theta: new Float64Array(K * K), edges: [] }
      byPair.set(s * n + t, term)
    }
    term.edges.push({ edge: e, flipped })
    for (let yf = 0; yf < K; yf++)
      for (let yt = 0; yt < K; yt++) {
        const at = flipped ? yt * K + yf : yf * K + yt
        visitEdge(model, edge, yf, yt, (j, v) => {
          term.theta[at] = (term.theta[at] ?? 0) + v * (w[j] ?? 0)
        })
      }
  }
  const cliques = (x.cliques ?? []).map((clique) => {
    const size = K ** clique.nodes.length
    const table = new Float64Array(size)
    for (let idx = 0; idx < size; idx++)
      visitClique(model, clique, digitsOf(idx, K, clique.nodes.length), (j, v) => {
        table[idx] = (table[idx] ?? 0) + v * (w[j] ?? 0)
      })
    return { nodes: clique.nodes, table }
  })
  return { n, K, unary, pairs: [...byPair.values()], cliques }
}

const variableName = (i: number): string => `y${i}`

function factorGraphOf(model: GraphCrf, p: Potentials): FactorGraph {
  const variables: DiscreteVariable[] = Array.from({ length: p.n }, (_, i) => ({
    name: variableName(i),
    states: model.labels,
  }))
  const factors: Factor[] = [
    ...p.unary.map((u, i) => logFactor([variableName(i)], [p.K], u)),
    ...p.pairs.map((e) => logFactor([variableName(e.s), variableName(e.t)], [p.K, p.K], e.theta)),
    ...p.cliques.map((c) => logFactor(c.nodes.map(variableName), new Array<number>(c.nodes.length).fill(p.K), c.table)),
  ]
  return factorGraph(variables, factors)
}

/** The potential of a labelling: the log of its unnormalised probability. */
function potentialScore(p: Potentials, ys: readonly number[]): number {
  let s = 0
  for (const [i, u] of p.unary.entries()) s += u[ys[i] ?? 0] ?? 0
  for (const e of p.pairs) s += e.theta[(ys[e.s] ?? 0) * p.K + (ys[e.t] ?? 0)] ?? 0
  for (const c of p.cliques)
    s +=
      c.table[
        cliqueEntry(
          p.K,
          c.nodes.map((i) => ys[i] ?? 0),
        )
      ] ?? 0
  return s
}

/**
 * The model written as a discrete factor graph (variables `y0`, `y1`, ..., the labels as states) under the weights: a
 * unary factor per node, one pairwise factor per node pair (parallel edges are multiplied together) and one factor per
 * clique. Any inference routine of `pgm` runs on it, and it is what a custom MAP function of {@link graphOracle}
 * receives.
 *
 * @param model - The model.
 * @param x - The input graph.
 * @param weights - The weights (default the model's own).
 * @returns The factor graph.
 * @throws OmaiError with code `value` for an edge or clique outside the graph or of an unknown type, `shape` for weights of the wrong length.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrf, graphFactorGraph } from 'omai-engine/pgm'
 * const fg = graphFactorGraph(graphCrf(['a', 'b'], []), { nodes: [{}, {}, {}], edges: [{ from: 0, to: 1 }, { from: 1, to: 2 }] })
 * [fg.variables.length, fg.factors.length] // => [3, 5]
 * ```
 */
export function graphFactorGraph(
  model: GraphCrf,
  x: GraphInput,
  weights: ArrayLike<number> = model.weights,
): FactorGraph {
  return factorGraphOf(model, potentials(model, x, weights))
}

/** The structure of an input as far as inference cares: is it a factor forest, and how wide is its junction tree. */
interface Plan {
  readonly forest: boolean
  readonly width: number
  readonly order: string[] | undefined
}

function planOf(fg: FactorGraph): Plan {
  if (isFactorTree(fg)) return { forest: true, width: fg.variables.length === 0 ? 0 : 1, order: undefined }
  const names = fg.variables.map((v) => v.name)
  const graph = interactionGraph(fg.factors, names)
  const order = eliminationOrder(
    graph,
    names,
    'minFill',
    Object.fromEntries(fg.variables.map((v) => [v.name, v.states.length])),
  )
  return { forest: false, width: inducedWidth(graph, order), order }
}

type Resolved = GraphExpectations['inference']

interface Beliefs {
  readonly logPartition: number
  readonly node: Float64Array[]
  readonly pair: Float64Array[]
  readonly clique: Float64Array[]
}

const probabilities = (f: Factor): Float64Array => Float64Array.from(softmaxOf(f.logValues))

/**
 * The structured oracle of a graph CRF: features, expectations and decoding over inputs of any shape, with the weights
 * passed on every call (the model's own weights are not used). Expectations use the inference of `options.inference`,
 * exact on forests and low-treewidth graphs and by loopy belief propagation or tree-reweighted belief propagation
 * otherwise (each result says which ran and whether it is exact). Decoding adds the Hamming cost of `against` to every
 * node's label scores before the MAP search, so passive-aggressive updates and the structured SVM's separation oracle
 * work on any graph.
 *
 * @param model - The model (its structure; the weights are arguments of the oracle).
 * @param options - Inference, MAP method and their options.
 * @returns The oracle.
 * @concept conditional-random-fields
 * @index estimator The structured-model oracle of a CRF on trees, grids and arbitrary graphs.
 * @example
 * ```ts
 * import { graphCrf, graphOracle } from 'omai-engine/pgm'
 * const oracle = graphOracle(graphCrf(['a', 'b'], [], { bias: false }))
 * const e = oracle.expectations([0, 0, 0, 0], { nodes: [{}, {}], edges: [{ from: 0, to: 1 }] })
 * [Math.abs(e.logPartition - Math.log(4)) < 1e-12, e.exact, e.inference] // => [true, true, 'tree']
 * ```
 */
export function graphOracle(model: GraphCrf, options: GraphOracleOptions = {}): GraphOracle {
  const K = model.layout.labels
  const maxStates = options.maxStates ?? 100000
  const bpOptions = { maxIter: 500, tol: 1e-10, ...options.bp }
  const plans = new WeakMap<GraphInput, Plan>()
  const planFor = (x: GraphInput, fg: () => FactorGraph): Plan => {
    let plan = plans.get(x)
    if (plan === undefined) {
      plan = planOf(fg())
      plans.set(x, plan)
    }
    return plan
  }
  const affordable = (plan: Plan): boolean => (plan.width + 1) * Math.log(K) <= Math.log(maxStates)

  const resolve = (plan: Plan, hasCliques: boolean): Resolved => {
    const how = options.inference ?? 'auto'
    if (how === 'tree') {
      if (!plan.forest) throw new OmaiError('value', 'graph oracle: tree inference needs a forest')
      return 'tree'
    }
    if (how === 'junctionTree' || how === 'bp') return how
    if (how === 'trw') {
      if (hasCliques) throw new OmaiError('value', 'graph oracle: tree-reweighted inference needs a pairwise graph')
      return 'trw'
    }
    if (plan.forest) return 'tree'
    if (affordable(plan)) return 'junctionTree'
    if (how === 'exact')
      throw new OmaiError(
        'value',
        `graph oracle: exact inference needs cliques of ${K ** (plan.width + 1)} states (induced width ${plan.width}); use bp or trw`,
      )
    return 'bp'
  }

  const beliefs = (p: Potentials, fg: FactorGraph, plan: Plan, how: Resolved): Beliefs => {
    const nP = p.pairs.length
    if (how === 'junctionTree') {
      const calibrated = calibrate(fg, plan.order === undefined ? {} : { order: plan.order })
      const query = (vars: readonly number[]): Float64Array =>
        Float64Array.from(junctionTreeQuery(calibrated, vars.map(variableName)).logValues, Math.exp)
      return {
        logPartition: calibrated.logPartition,
        node: p.unary.map((_, i) => query([i])),
        pair: p.pairs.map((e) => query([e.s, e.t])),
        clique: p.cliques.map((c) => query(c.nodes)),
      }
    }
    if (how === 'trw') {
      const r = treeReweighted(fg, options.trw ?? {})
      return {
        logPartition: r.logPartitionBound,
        node: p.unary.map((_, i) => probabilities(r.marginals[variableName(i)] as Factor)),
        pair: (r.edgeBeliefs ?? []).map(probabilities),
        clique: [],
      }
    }
    const r = beliefPropagation(fg, how === 'tree' ? { schedule: 'tree' } : bpOptions)
    return {
      logPartition: r.logPartition,
      node: p.unary.map((_, i) => probabilities(r.marginals[variableName(i)] as Factor)),
      pair: r.factorBeliefs.slice(p.n, p.n + nP).map(probabilities),
      clique: r.factorBeliefs.slice(p.n + nP).map(probabilities),
    }
  }

  const infer = (w: ArrayLike<number>, x: GraphInput): { p: Potentials; b: Beliefs; how: Resolved } => {
    const p = potentials(model, x, w)
    if (p.n === 0) return { p, b: { logPartition: 0, node: [], pair: [], clique: [] }, how: 'tree' }
    const fg = factorGraphOf(model, p)
    const plan = planFor(x, () => fg)
    const how = resolve(plan, p.cliques.length > 0)
    return { p, b: beliefs(p, fg, plan, how), how }
  }

  const features = (x: GraphInput, y: GraphLabelling): Float64Array => {
    checkGraphInput(model, x)
    const ys = graphLabelIndices(model, y, x.nodes.length)
    const counts = new Float64Array(model.layout.size)
    const add = (j: number, v: number): void => {
      counts[j] = (counts[j] ?? 0) + v
    }
    for (const [i, item] of x.nodes.entries()) visitNode(model, item, ys[i] ?? 0, add)
    for (const e of x.edges) visitEdge(model, e, ys[e.from] ?? 0, ys[e.to] ?? 0, add)
    for (const c of x.cliques ?? [])
      visitClique(
        model,
        c,
        c.nodes.map((i) => ys[i] ?? 0),
        add,
      )
    return counts
  }

  const expectations = (w: ArrayLike<number>, x: GraphInput): GraphExpectations => {
    const { p, b, how } = infer(w, x)
    const counts = new Float64Array(model.layout.size)
    const add =
      (prob: number): ((j: number, v: number) => void) =>
      (j, v) => {
        counts[j] = (counts[j] ?? 0) + prob * v
      }
    for (const [i, item] of x.nodes.entries())
      for (let k = 0; k < K; k++) visitNode(model, item, k, add(b.node[i]?.[k] ?? 0))
    for (const [m, term] of p.pairs.entries()) {
      const marginal = b.pair[m] as Float64Array
      for (const { edge, flipped } of term.edges)
        for (let yf = 0; yf < K; yf++)
          for (let yt = 0; yt < K; yt++)
            visitEdge(
              model,
              x.edges[edge] as GraphInput['edges'][number],
              yf,
              yt,
              add(marginal[flipped ? yt * K + yf : yf * K + yt] ?? 0),
            )
    }
    for (const [m, c] of (x.cliques ?? []).entries()) {
      const marginal = b.clique[m] as Float64Array
      for (let idx = 0; idx < marginal.length; idx++)
        visitClique(model, c, digitsOf(idx, K, c.nodes.length), add(marginal[idx] ?? 0))
    }
    return { counts, logPartition: b.logPartition, exact: how === 'tree' || how === 'junctionTree', inference: how }
  }

  const decodeWith = (fg: FactorGraph, plan: Plan): number[] => {
    const how = options.map ?? 'auto'
    const indicesOf = (r: MapResult): number[] => fg.variables.map((_, i) => r.indices[variableName(i)] ?? 0)
    if (typeof how === 'function') return indicesOf(how(fg))
    const choice = how === 'auto' ? (plan.forest || affordable(plan) ? 'junctionTree' : 'mplp') : how
    if (choice === 'junctionTree') {
      const r = junctionTreeMap(fg, plan.order === undefined ? {} : { order: plan.order })
      return fg.variables.map((_, i) => model.labels.indexOf(r.assignment[variableName(i)] ?? ''))
    }
    if (choice === 'tree') {
      if (!plan.forest) throw new OmaiError('value', 'graph oracle: tree MAP needs a forest')
      return indicesOf(maxProductMap(fg, { ...bpOptions, schedule: 'tree' }))
    }
    if (choice === 'maxProduct') return indicesOf(maxProductMap(fg, bpOptions))
    return indicesOf(mplp(fg, options.mplp ?? {}))
  }

  return {
    size: model.layout.size,
    features,
    expectations,
    marginals: (w, x) => {
      const { b, how } = infer(w, x)
      return {
        nodes: b.node.map((m) => Array.from(m)),
        logPartition: b.logPartition,
        exact: how === 'tree' || how === 'junctionTree',
        inference: how,
      }
    },
    plan: (x) => {
      const p = potentials(model, x, model.weights)
      if (p.n === 0) return { inference: 'tree', exact: true, width: 0 }
      const plan = planFor(x, () => factorGraphOf(model, p))
      const how = resolve(plan, p.cliques.length > 0)
      return { inference: how, exact: how === 'tree' || how === 'junctionTree', width: plan.width }
    },
    decode: (w, x, decodeOptions = {}) => {
      const p = potentials(model, x, w)
      if (p.n === 0) return { y: [], score: 0 }
      if (decodeOptions.against !== undefined) {
        const gold = graphLabelIndices(model, decodeOptions.against, p.n)
        for (const [i, u] of p.unary.entries()) for (let k = 0; k < K; k++) u[k] = (u[k] ?? 0) + (k === gold[i] ? 0 : 1)
      }
      const fg = factorGraphOf(model, p)
      const ys = decodeWith(
        fg,
        planFor(x, () => fg),
      )
      return { y: ys.map((k) => model.labels[k] ?? ''), score: potentialScore(p, ys) }
    },
    loss: (gold, predicted) => {
      const g = graphLabelIndices(model, gold, gold.length)
      const q = graphLabelIndices(model, predicted, gold.length)
      return g.reduce((s, k, i) => s + (k === q[i] ? 0 : 1), 0)
    },
  }
}

/**
 * The score of a labelling under the model's own weights, w . phi(x, y).
 *
 * @param model - The model.
 * @param x - The input graph.
 * @param y - The labelling.
 * @returns The score.
 * @throws OmaiError with code `shape` for a different number of labels and nodes, `value` for unknown labels, types or nodes.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrf, graphScore } from 'omai-engine/pgm'
 * const m = graphCrf(['a', 'b'], [], { bias: false, weights: [0, 2, 0, 0] })
 * graphScore(m, { nodes: [{}, {}], edges: [{ from: 0, to: 1 }] }, ['a', 'b']) // => 2
 * ```
 */
export function graphScore(model: GraphCrf, x: GraphInput, y: GraphLabelling): number {
  const phi = graphOracle(model).features(x, y)
  return phi.reduce((s, v, i) => s + v * (model.weights[i] ?? 0), 0)
}

/**
 * The marginal probability of each label at each node under the model's own weights, with the inference named in
 * {@link GraphInference} (exact where it can be, by default).
 *
 * @param model - The model.
 * @param x - The input graph.
 * @param options - Inference options.
 * @returns Probabilities by node and label, and whether they are exact.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrf, graphMarginals } from 'omai-engine/pgm'
 * graphMarginals(graphCrf(['a', 'b'], [], { bias: false }), { nodes: [{}], edges: [] }).nodes // => [[0.5, 0.5]]
 * ```
 */
export function graphMarginals(model: GraphCrf, x: GraphInput, options: GraphOracleOptions = {}): GraphMarginals {
  return graphOracle(model, options).marginals(model.weights, x)
}

/**
 * The highest-scoring labelling under the model's own weights.
 *
 * @param model - The model.
 * @param x - The input graph.
 * @param options - Inference and MAP options.
 * @returns The labels (names) and the score.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrf, graphDecode } from 'omai-engine/pgm'
 * const m = graphCrf(['a', 'b'], [], { bias: false, weights: [0, 0, 0, 3] })
 * graphDecode(m, { nodes: [{}, {}], edges: [{ from: 0, to: 1 }] }).y // => ['b', 'b']
 * ```
 */
export function graphDecode(
  model: GraphCrf,
  x: GraphInput,
  options: GraphOracleOptions = {},
): { readonly y: GraphLabelling; readonly score: number } {
  return graphOracle(model, options).decode(model.weights, x)
}
