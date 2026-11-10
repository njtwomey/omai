/**
 * The graph CRF as an estimator, a thin wrapper like `crf` in `crf-estimator.ts` but over graphs: the inputs are
 * `GraphInput`s (trees, grids, skip chains or any graph, built with `treeInput`, `gridInput`, `skipChainInput`), the
 * model is made from the labels and node attributes seen in the training graphs by a structure builder of
 * `graph-structures.ts` (`treeCrf`, `gridCrf`, `skipChainCrf`, `graphCrf`), the fit is `trainStructured` over the graph
 * oracle with any solver of the structured slot, and the fitted value predicts labellings and node marginals (exact on
 * forests and low-treewidth graphs, loopy belief propagation otherwise, as the oracle's options say), scores flat node
 * accuracy and names its weights. Like `crf` it has the shape of the estimator protocol over its own data type
 * (`graphCrfData`), because graphs are not tensors.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { resolveKey } from '../../context/index.ts'
import type { FitOptions, LearnData, SerialisedFitted } from '../../learn/index.ts'
import type { SpaceValue } from '../../space/index.ts'
import { asTensor } from '../../tensor/index.ts'
import { type GraphCrf, type GraphInput, graphCrf, graphParameterNames } from './graph-crf.ts'
import { type GraphOracleOptions, graphOracle } from './graph-oracle.ts'
import { trainStructured } from './solvers.ts'

/** What the graph CRF estimator fits: graphs and the labels of their nodes, with `x` the graph numbers as in `crfData`. */
export interface GraphCrfData extends LearnData {
  readonly graphs: readonly GraphInput[]
  /** The label of each node of each graph (omitted for data to predict on). */
  readonly labels?: readonly (readonly string[])[]
}

/**
 * Data for the graph CRF estimator.
 *
 * @param graphs - The input graphs.
 * @param labels - The node labels of each graph, one per node.
 * @returns The data.
 * @throws OmaiError with code `shape` for a different number of graphs and label lists.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrfData, treeInput } from 'omai-engine/pgm'
 * graphCrfData([treeInput([{ f: 1 }, { f: 1 }], { parents: [-1, 0] })], [['a', 'b']]).labels // => [['a', 'b']]
 * ```
 */
export function graphCrfData(graphs: readonly GraphInput[], labels?: readonly (readonly string[])[]): GraphCrfData {
  if (labels !== undefined && labels.length !== graphs.length)
    throw new OmaiError('shape', `graphCrfData: ${graphs.length} graphs and ${labels.length} label lists`)
  return { x: asTensor(graphs.map((_, i) => i)), graphs, ...(labels === undefined ? {} : { labels }) }
}

/** The hyperparameters of {@link graphCrfEstimator}. */
export interface GraphCrfHyper {
  /**
   * The model's structure from the labels and node attributes of the training graphs: `(l, a) => gridCrf(l, a)`,
   * `treeCrf`, `skipChainCrf`, `graphCrf` with its options (default: `graphCrf` with the edge types and edge attributes
   * the graphs use).
   */
  readonly structure?: (labels: readonly string[], attributes: readonly string[]) => GraphCrf
  /** The inference and MAP options of the oracle. */
  readonly oracle?: GraphOracleOptions
  /** A value of `structuredSolvers.space` (default L-BFGS, or OWL-QN when `c1` > 0). */
  readonly solver?: SpaceValue
  readonly c1: number
  readonly c2: number
  /** Stop when the largest gradient entry falls below this (default 1e-6). */
  readonly tol: number
  /** The step budget (default 200). */
  readonly maxIterations: number
  /** Fix the labels and their order (default: those of the training data in order of first appearance). */
  readonly labels?: readonly string[]
}

const defaults: GraphCrfHyper = { c1: 0, c2: 1, tol: 1e-6, maxIterations: 200 }

/** The fitted parameters of the graph CRF estimator, as plain JSON: the model's structure, its weights and the training summary. */
export type GraphCrfParams = {
  readonly labels: string[]
  readonly nodeAttributes: string[]
  readonly bias: boolean
  readonly edgeTypes: string[]
  readonly symmetric: string[]
  readonly edgeAttributes: string[]
  readonly symmetricAttributes: boolean
  readonly cliqueTypes: Record<string, number>
  readonly weights: number[]
  readonly objective: number
  readonly error: number
  readonly converged: boolean
}

/** The verbs and facts of a fitted graph CRF estimator. */
export interface GraphCrfFitted {
  readonly estimator: string
  readonly kind: 'other'
  readonly params: GraphCrfParams
  /** The class labels, in the order of the weights. */
  readonly classes: readonly string[]
  /** The fitted model, for `graphOracle`, `graphDecode` and the rest of `graph-oracle.ts`. */
  readonly model: GraphCrf
  /** The most probable labelling of each graph. */
  predict(graphs: readonly GraphInput[]): string[][]
  predictSingle(graph: GraphInput): string[]
  /** For each node of each graph, the probability of each label. */
  predictMarginals(graphs: readonly GraphInput[]): Record<string, number>[][]
  predictMarginalsSingle(graph: GraphInput): Record<string, number>[]
  /** The flat accuracy over all nodes of all graphs. */
  score(graphs: readonly GraphInput[], labels: readonly (readonly string[])[]): number
  /** The weights by name (`node:<attribute>|<label>`, `pair:<type>|<from>|<to>`, ...), the ones that are not exactly zero. */
  weights(): Record<string, number>
  serialise(): SerialisedFitted
}

/** The graph CRF estimator: the estimator protocol's shape over {@link GraphCrfData}. */
export interface GraphCrfEstimator {
  readonly name: string
  readonly kind: 'other'
  readonly hyper: GraphCrfHyper
  fit(data: GraphCrfData, options?: FitOptions): GraphCrfFitted
  fitGraphs(graphs: readonly GraphInput[], labels: readonly (readonly string[])[], options?: FitOptions): GraphCrfFitted
  revive(params: Json): GraphCrfFitted
  with(changes: Partial<GraphCrfHyper>): GraphCrfEstimator
}

const cliqueTypesOf = (model: GraphCrf): Record<string, number> =>
  Object.fromEntries(model.cliqueTypes.map((c) => [c.name, c.arity]))

const modelOf = (p: GraphCrfParams): GraphCrf =>
  graphCrf(p.labels, p.nodeAttributes, {
    edgeTypes: p.edgeTypes,
    symmetric: p.symmetric,
    edgeAttributes: p.edgeAttributes,
    symmetricAttributes: p.symmetricAttributes,
    cliqueTypes: p.cliqueTypes,
    bias: p.bias,
    weights: p.weights,
  })

function fittedOf(params: GraphCrfParams, oracleOptions: GraphOracleOptions | undefined): GraphCrfFitted {
  const model = modelOf(params)
  const oracle = graphOracle(model, oracleOptions)
  const names = graphParameterNames(model)
  const decode = (g: GraphInput): string[] =>
    g.nodes.length === 0
      ? []
      : oracle.decode(model.weights, g).y.map((k) => (typeof k === 'number' ? (model.labels[k] ?? '') : k))
  const marginals = (g: GraphInput): Record<string, number>[] =>
    g.nodes.length === 0
      ? []
      : oracle
          .marginals(model.weights, g)
          .nodes.map((row) => Object.fromEntries(model.labels.map((l, k) => [l, row[k] ?? 0])))
  return {
    estimator: 'pgm/graph-crf',
    kind: 'other',
    params,
    classes: params.labels,
    model,
    predict: (graphs) => graphs.map(decode),
    predictSingle: decode,
    predictMarginals: (graphs) => graphs.map(marginals),
    predictMarginalsSingle: marginals,
    score: (graphs, labels) => {
      if (graphs.length !== labels.length)
        throw new OmaiError('shape', `graph crf score: ${graphs.length} graphs and ${labels.length} label lists`)
      let right = 0
      let total = 0
      for (const [i, g] of graphs.entries()) {
        const predicted = decode(g)
        const gold = labels[i] ?? []
        if (predicted.length !== gold.length)
          throw new OmaiError(
            'shape',
            `graph crf score: graph ${i} has ${predicted.length} nodes and ${gold.length} labels`,
          )
        for (const [t, label] of gold.entries()) if (predicted[t] === label) right++
        total += gold.length
      }
      return total === 0 ? Number.NaN : right / total
    },
    weights: () =>
      Object.fromEntries(names.flatMap((n, i) => ((model.weights[i] ?? 0) === 0 ? [] : [[n, model.weights[i] ?? 0]]))),
    serialise: () => ({ estimator: 'pgm/graph-crf', params: params as unknown as Json }),
  }
}

/** The graph CRF the training graphs call for: the labels, the node attributes and the kinds of edge they use. */
function defaultStructure(graphs: readonly GraphInput[]): NonNullable<GraphCrfHyper['structure']> {
  const types = new Set<string>()
  const edgeAttributes = new Set<string>()
  for (const g of graphs)
    for (const e of g.edges) {
      types.add(e.type ?? 'edge')
      for (const a of Object.keys(e.attributes ?? {})) edgeAttributes.add(a)
    }
  return (labels, attributes) =>
    graphCrf(labels, attributes, {
      ...(types.size === 0 ? {} : { edgeTypes: [...types] }),
      edgeAttributes: [...edgeAttributes],
    })
}

function fit(h: GraphCrfHyper, data: LearnData, options: FitOptions): GraphCrfFitted {
  const { graphs, labels } = data as Partial<GraphCrfData>
  if (graphs === undefined) throw new OmaiError('value', 'graphCrf: needs graphs (use graphCrfData)')
  if (labels === undefined) throw new OmaiError('value', 'graphCrf: fit needs the label lists')
  const kept: { x: GraphInput; y: readonly string[] }[] = []
  for (const [i, g] of graphs.entries()) {
    const y = labels[i] ?? []
    if (y.length !== g.nodes.length)
      throw new OmaiError('shape', `graphCrf: graph ${i} has ${g.nodes.length} nodes and ${y.length} labels`)
    if (g.nodes.length > 0) kept.push({ x: g, y })
  }
  if (kept.length === 0) throw new OmaiError('value', 'graphCrf: no non-empty training graphs')
  const names = h.labels === undefined ? [...new Set(kept.flatMap((e) => e.y))] : [...h.labels]
  const attributes = [...new Set(kept.flatMap((e) => e.x.nodes.flatMap((n) => Object.keys(n))))]
  const model = (h.structure ?? defaultStructure(kept.map((e) => e.x)))(names, attributes)
  const trained = trainStructured(
    {
      oracle: graphOracle(model, h.oracle),
      data: kept,
      penalty: { c2: h.c2, ...(h.c1 > 0 ? { c1: h.c1 } : {}) },
      start: model.weights,
    },
    { ...(h.solver === undefined ? {} : { solver: h.solver }), tol: h.tol, key: resolveKey(options, 'pgm/graph-crf') },
    h.maxIterations,
  )
  const params: GraphCrfParams = {
    labels: [...model.labels],
    nodeAttributes: [...model.nodeAttributes],
    bias: model.bias,
    edgeTypes: [...model.edgeTypes],
    symmetric: [...model.symmetric],
    edgeAttributes: [...model.edgeAttributes],
    symmetricAttributes: model.symmetricAttributes,
    cliqueTypes: cliqueTypesOf(model),
    weights: Array.from(trained.weights),
    objective: trained.objective,
    error: trained.error,
    converged: trained.converged,
  }
  return fittedOf(params, h.oracle)
}

/**
 * The graph CRF as an estimator: `fit` takes {@link graphCrfData} (input graphs and the labels of their nodes), builds
 * the model with `structure` (a builder of `graph-structures.ts`; by default `graphCrf` with the edge types and edge
 * attributes of the training graphs) over the labels and node attributes seen, and trains it with `trainStructured`
 * and any solver of the structured slot (`solver`, `c1`, `c2`). The fitted value predicts labellings and node
 * marginals, scores flat node accuracy, names its weights and serialises as plain JSON.
 *
 * @param hyper - Hyperparameters, as {@link GraphCrfHyper}.
 * @returns The estimator.
 * @concept conditional-random-fields
 * @index estimator Graph CRF estimator: trees, grids and any graph, trained through the structured solver slot.
 * @example
 * ```ts
 * import { graphCrfData, graphCrfEstimator, treeCrf, treeInput } from 'omai-engine/pgm'
 * const tree = (a: number, b: number) => treeInput([{ f: a }, { f: b }], { parents: [-1, 0] })
 * const graphs = [tree(1, 0), tree(1, 1), tree(0, 0), tree(0, 1)]
 * const labels = [['a', 'b'], ['a', 'a'], ['b', 'b'], ['b', 'a']]
 * const fitted = graphCrfEstimator({ structure: treeCrf, c2: 0.1 }).fit(graphCrfData(graphs, labels))
 * fitted.predict([tree(1, 0)]) // => [['a', 'b']]
 * ```
 */
export function graphCrfEstimator(hyper: Partial<GraphCrfHyper> = {}): GraphCrfEstimator {
  const h: GraphCrfHyper = { ...defaults, ...hyper }
  const estimator: GraphCrfEstimator = {
    name: 'pgm/graph-crf',
    kind: 'other',
    hyper: h,
    fit: (data, options = {}) => fit(h, data, options),
    fitGraphs: (graphs, labels, options = {}) => fit(h, graphCrfData(graphs, labels), options),
    revive: (params) => fittedOf(params as unknown as GraphCrfParams, h.oracle),
    with: (changes) => graphCrfEstimator({ ...h, ...changes }),
  }
  return estimator
}
