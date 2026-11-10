/**
 * The bridge between the two ways of writing a CRF on a graph. `GeneralCrf` (in `crf.ts`) takes each input as cliques
 * with a feature list per joint assignment, the most general form and the one a hand-built relational model needs;
 * `GraphCrf` derives the same feature lists from node and edge attributes with weights tied by type. Here the
 * older form gets the structured-model oracle, so every solver of the structured family trains it, and any graph CRF
 * input can be written as a general CRF instance over the same weight vector, to check one against the other or to use
 * the older training functions on it.
 */

import { OmaiError } from '../../base/index.ts'
import { logFactor } from '../factor.ts'
import type { BeliefPropagationOptions } from '../inference/belief-propagation.ts'
import { junctionTreeMap } from '../inference/junction-tree.ts'
import { type MapResult, maxProductMap, mplp } from '../inference/map.ts'
import { type DiscreteModel, type FactorGraph, factorGraph } from '../models/factor-graph.ts'
import { checkGraphInput, visitClique, visitEdge, visitNode } from './_graph-features.ts'
import { type CrfInstance, crfFactorGraph, type GeneralCrf, generalCrf, generalCrfObjective } from './crf.ts'
import { type GraphCrf, type GraphInput, type GraphLabelling, graphParameterNames } from './graph-crf.ts'
import type { StructuredOracle } from './oracle.ts'

/** A labelling of a general CRF: a state (name or index) for each variable. */
export type GeneralLabelling = Readonly<Record<string, string | number>>

/** Options of {@link generalCrfOracle}. */
export interface GeneralCrfOracleOptions {
  /** `exact` (a junction tree, default) or `bp` (loopy belief propagation, the Bethe approximation). */
  readonly inference?: 'exact' | 'bp'
  readonly bp?: Omit<BeliefPropagationOptions, 'mode' | 'evidence'>
  /** `junctionTree` (exact, default), `maxProduct`, `mplp`, or a function from the discrete model to a MAP assignment. */
  readonly map?: 'junctionTree' | 'maxProduct' | 'mplp' | ((model: DiscreteModel) => MapResult)
}

/**
 * The structured-model oracle of a general CRF, so that the structured solvers train the older per-assignment-table
 * form. Features are the sums of the clique feature lists at the labelling; expectations are `generalCrfObjective`'s
 * (exact junction tree or loopy BP clique marginals; the gradient of the negative log-likelihood is the expected minus
 * the observed counts, so the oracle reads the counts and log Z off it at any labelling); decoding adds the Hamming
 * cost of `against` as a unary factor per variable and searches for the MAP.
 *
 * @param crf - The model (variables and feature names; the weights are arguments of the oracle).
 * @param options - Inference and MAP method.
 * @returns The oracle over instances and labellings by variable.
 * @throws OmaiError with code `value` for a label that is not a state of its variable.
 * @concept conditional-random-fields
 * @index estimator The structured-model oracle of a general CRF with per-assignment feature tables.
 * @example
 * ```ts
 * import { generalCrf, generalCrfOracle } from 'omai-engine/pgm'
 * const crf = generalCrf([{ name: 'y', states: ['0', '1'] }], ['f'])
 * const x = { cliques: [{ variables: ['y'], features: [[], [[0, 1]]] }] }
 * const oracle = generalCrfOracle(crf)
 * [Array.from(oracle.features(x, { y: '1' })), oracle.decode([2], x).y] // => [[1], { y: '1' }]
 * ```
 */
export function generalCrfOracle(
  crf: GeneralCrf,
  options: GeneralCrfOracleOptions = {},
): StructuredOracle<CrfInstance, GeneralLabelling> {
  const states = new Map(crf.variables.map((v) => [v.name, v.states]))
  const indexOf = (v: string, label: string | number | undefined): number => {
    const k = typeof label === 'number' ? label : (states.get(v) ?? []).indexOf(String(label))
    if (!(k >= 0 && k < (states.get(v)?.length ?? 0)))
      throw new OmaiError('value', `generalCrfOracle: no label for ${v}`)
    return k
  }
  const features = (x: CrfInstance, y: GeneralLabelling): Float64Array => {
    const counts = new Float64Array(crf.features.length)
    for (const c of x.cliques) {
      let idx = 0
      for (const v of c.variables) idx = idx * (states.get(v)?.length ?? 1) + indexOf(v, y[v])
      for (const [k, value] of c.features[idx] ?? []) counts[k] = (counts[k] ?? 0) + value
    }
    return counts
  }
  const zero: Record<string, number> = Object.fromEntries(crf.variables.map((v) => [v.name, 0]))
  const weighted = (w: ArrayLike<number>): GeneralCrf => generalCrf(crf.variables, crf.features, w)
  return {
    size: crf.features.length,
    features,
    expectations: (w, x) => {
      const model = weighted(w)
      const r = generalCrfObjective(model, [{ instance: x, y: zero }], {
        ...(options.inference === undefined ? {} : { inference: options.inference }),
        ...(options.bp === undefined ? {} : { bp: options.bp }),
      })
      // value = log Z - w.phi(y0) and gradient = E[phi] - phi(y0), at any labelling y0.
      const phi = features(x, zero)
      const score = phi.reduce((s, v, i) => s + v * (model.weights[i] ?? 0), 0)
      return { counts: r.gradient.map((g, i) => g + (phi[i] ?? 0)), logPartition: r.value + score }
    },
    decode: (w, x, decodeOptions = {}) => {
      const model = weighted(w)
      const base = crfFactorGraph(model, x)
      const gold = decodeOptions.against
      const cost = (v: string, k: number): number => (gold === undefined || indexOf(v, gold[v]) === k ? 0 : 1)
      const fg: FactorGraph =
        gold === undefined
          ? base
          : factorGraph(base.variables, [
              ...base.factors,
              ...base.variables.map((v) =>
                logFactor(
                  [v.name],
                  [v.states.length],
                  Array.from({ length: v.states.length }, (_, k) => cost(v.name, k)),
                ),
              ),
            ])
      const how = options.map ?? 'junctionTree'
      const indices: Record<string, number> =
        typeof how === 'function'
          ? how(fg).indices
          : how === 'junctionTree'
            ? Object.fromEntries(
                Object.entries(junctionTreeMap(fg).assignment).map(([v, s]) => [v, (states.get(v) ?? []).indexOf(s)]),
              )
            : how === 'maxProduct'
              ? maxProductMap(fg, options.bp ?? {}).indices
              : mplp(fg).indices
      const y = Object.fromEntries(crf.variables.map((v) => [v.name, v.states[indices[v.name] ?? 0] ?? '']))
      let score = features(x, y).reduce((s, v, i) => s + v * (model.weights[i] ?? 0), 0)
      for (const v of crf.variables) score += cost(v.name, indices[v.name] ?? 0)
      return { y, score }
    },
    loss: (gold, predicted) =>
      crf.variables.reduce(
        (s, v) => s + (indexOf(v.name, gold[v.name]) === indexOf(v.name, predicted[v.name]) ? 0 : 1),
        0,
      ),
  }
}

/**
 * A graph CRF input written as a general CRF instance over the same weight vector: variables `y0`, `y1`, ... with the
 * labels as states, a clique per node, per edge and per higher-order clique with the feature lists that `GraphCrf`'s
 * features give them, and `GeneralCrf` feature `k` is parameter `k` of the graph model. So `generalCrfObjective`,
 * `fitGeneralCrf` and `generalCrfOracle` on the result agree with the graph oracle with the model's weights, which is
 * the relation between the two forms: a `GeneralCrf` is a graph CRF whose tying of weights you wrote by hand.
 *
 * @param model - The graph model.
 * @param x - The input graph.
 * @returns The general CRF (with the model's weights), the instance, and a function turning a graph labelling into the general one.
 * @throws OmaiError with code `value` for a malformed input.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { graphCrf, graphAsGeneralCrf } from 'omai-engine/pgm'
 * const g = graphAsGeneralCrf(graphCrf(['a', 'b'], [], { bias: false }), { nodes: [{}, {}], edges: [{ from: 0, to: 1 }] })
 * [g.instance.cliques.length, g.crf.features.length, g.labelling(['a', 'b'])] // => [3, 4, { y0: 'a', y1: 'b' }]
 * ```
 */
export function graphAsGeneralCrf(
  model: GraphCrf,
  x: GraphInput,
): {
  readonly crf: GeneralCrf
  readonly instance: CrfInstance
  readonly labelling: (y: GraphLabelling) => GeneralLabelling
} {
  checkGraphInput(model, x)
  const K = model.layout.labels
  const name = (i: number): string => `y${i}`
  const table = (
    arity: number,
    visit: (ys: readonly number[], add: (j: number, v: number) => void) => void,
  ): (readonly [number, number][])[] =>
    Array.from({ length: K ** arity }, (_, idx) => {
      const ys: number[] = []
      for (let rest = idx, a = 0; a < arity; a++, rest = Math.floor(rest / K)) ys.unshift(rest % K)
      const out: [number, number][] = []
      visit(ys, (j, v) => void out.push([j, v]))
      return out
    })
  const cliques: CrfInstance['cliques'][number][] = [
    ...x.nodes.map((item, i) => ({
      variables: [name(i)],
      features: table(1, (ys, add) => visitNode(model, item, ys[0] ?? 0, add)),
    })),
    ...x.edges.map((edge) => ({
      variables: [name(edge.from), name(edge.to)],
      features: table(2, (ys, add) => visitEdge(model, edge, ys[0] ?? 0, ys[1] ?? 0, add)),
    })),
    ...(x.cliques ?? []).map((clique) => ({
      variables: clique.nodes.map(name),
      features: table(clique.nodes.length, (ys, add) => visitClique(model, clique, ys, add)),
    })),
  ]
  return {
    crf: generalCrf(
      x.nodes.map((_, i) => ({ name: name(i), states: model.labels })),
      graphParameterNames(model),
      model.weights,
    ),
    instance: { cliques },
    labelling: (y) => Object.fromEntries(x.nodes.map((_, i) => [name(i), y[i] ?? ''])),
  }
}
