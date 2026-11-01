/**
 * Kolmogorov-Arnold networks as estimators on the trainer base (`learn/bases`): `kanRegressor` (squared error) and
 * `kanClassifier` (cross-entropy over a logit per class). The network is `nn`'s `KAN`, with every choice a slot or a
 * hyperparameter: the widths, the grid and degree of the edge functions, the family of the edges (`kanBases`: B-spline,
 * radial, Chebyshev, Fourier), the residual base function (`kanBaseFunctions`), the knots (`splineKnots`: fixed, at
 * quantiles, or trained with the weights), the optimiser and its schedule (`optimisers`, `schedules`) and the strengths of
 * the paper's sparsification penalties (L1 on the edge magnitudes, their entropy, the coefficients' L1), which enter
 * the loss through the trainer base's `penalty` hook. Beyond the base's paths the tape holds, at any logged step, every
 * edge's function on a grid with its magnitude (`kan/curves`, and one path per edge), the grids and knots, the
 * penalty terms and the share of negligible edges, so a view can play the edges as they learn. A fitted network
 * can be pruned, read symbolically and turned into a formula. The paper's whole loop (train, prune, refine the grid,
 * train again) is `progressiveKan` in `kan-progressive.ts`.
 */

import type { Json } from '../base/index.ts'
import { crossEntropy, mse } from '../losses/index.ts'
import {
  edgeSparsity,
  type FixSymbolicOptions,
  fixSymbolic,
  KAN,
  type KanCurves,
  type KanGrid,
  type KanSparsity,
  type KanState,
  kanBaseFunctions,
  kanBases,
  kanCurves,
  kanGrids,
  kanPenalties,
  type Module,
  type Params,
  type Pruned,
  type PruneOptions,
  pruneKan,
  type SymbolicFormula,
  type SymbolicKan,
  symbolicFormula,
  type TrainState,
} from '../nn/index.ts'
import { optimisers } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import { defaults, dict, int, type ModeConfig, type OptionTable, real, type SpaceValue } from '../space/index.ts'
import { asTensor, item, reshape, type Tensor, toFlat, type Value } from '../tensor/index.ts'
import { type Contract, type DerivedSource, derived } from '../trace/index.ts'
import type { IterativeSession, IterativeSessionOptions } from './bases/iterative.ts'
import {
  defineTrainer,
  type TrainerEstimator,
  type TrainerFitted,
  type TrainerParams,
  type TrainerPath,
} from './bases/trainer.ts'
import type { FittedFacts, LearnData, Verbs } from './estimator.ts'

/** The paths a KAN trainer's session adds to the trainer base's. */
export type KanPath =
  | 'kan/curves'
  | 'kan/grids'
  | 'kan/penalty'
  | 'kan/l1'
  | 'kan/entropy'
  | 'kan/coef'
  | 'kan/sparsity'
  | 'kan/scales'
  | `kan/edge/${number}/${number}/${number}`

/** The hyperparameters of the KAN trainers. */
export interface KanHyper {
  /** The hidden widths (default `[5]`); the input width is the data's and the output width the targets'. */
  readonly hidden: readonly number[]
  /** The whole widths `[n_in, ..., n_out]`, instead of `hidden` (checked against the data). */
  readonly widths?: readonly number[]
  /** The grid size `G` of the edge functions (default 5): intervals of the spline. */
  readonly grid: number
  readonly degree: number
  /** The `kanBases` option: the family of the edge functions. */
  readonly basis: SpaceValue
  /** The `kanBaseFunctions` option: the residual base function of each edge. */
  readonly base: SpaceValue
  /** The `splineKnots` option of the spline family. */
  readonly knots: SpaceValue
  /** A bias per node (default false). */
  readonly bias: boolean
  /** The weights of the paper's penalties: the L1 of the edge magnitudes, their entropy per layer, the coefficients' L1. */
  readonly lambdaL1: number
  readonly lambdaEntropy: number
  readonly lambdaCoef: number
  /** The rows (spread over the data) on which the edge magnitudes of the penalty are measured (default 128). */
  readonly regRows: number
  /** The `optimisers` option; the learning rate and schedule are separate. */
  readonly optimiser: SpaceValue
  readonly learningRate: number
  /** A `schedules` option shaping the learning rate (default constant). */
  readonly schedule?: SpaceValue
  readonly epochs: number
  /** Minibatches in all, instead of epochs. */
  readonly steps?: number
  /** Minibatch size (default the whole data). */
  readonly batchSize?: number
  readonly clipNorm?: number
  readonly validationFraction?: number
  /** Points per edge curve on the tape (default 41). */
  readonly curvePoints: number
  /** An edge is negligible when its mean magnitude is below this (default 0.01). */
  readonly sparsityThreshold: number
}

/** The default hyperparameters. */
export const KAN_DEFAULTS: KanHyper = {
  hidden: [5],
  grid: 5,
  degree: 3,
  basis: { kind: 'bspline' },
  base: { kind: 'silu' },
  knots: { kind: 'uniform' },
  bias: false,
  lambdaL1: 0,
  lambdaEntropy: 0,
  lambdaCoef: 0,
  regRows: 128,
  optimiser: { kind: 'adam' },
  learningRate: 0.01,
  epochs: 100,
  curvePoints: 41,
  sparsityThreshold: 0.01,
}

/** What a fitted KAN offers beyond the verbs of its kind. */
export interface KanVerbs {
  /** The network and its trained state. */
  readonly network: KAN
  readonly state: KanState
  /** One edge on a grid (default over the layer's input range). */
  readonly edgeCurve: (layer: number, i: number, j: number, grid?: ArrayLike<number>) => Tensor
  /** Every edge on its grid with its magnitude on the training rows. */
  readonly curves: (points?: number) => KanCurves
  /** The grids: input ranges and spline knots, per layer. */
  readonly grids: () => KanGrid[]
  /** The mean magnitude of every edge on the training rows, per layer `[out, in]`. */
  readonly edgeScales: () => Tensor[]
  /** The share of negligible edges. */
  readonly sparsity: (threshold?: number) => KanSparsity
  /** The network pruned on the training rows. */
  readonly prune: (options?: PruneOptions) => Pruned
  /** Every edge fixed to its best symbolic function on the training rows. */
  readonly symbolic: (options?: FixSymbolicOptions) => SymbolicKan
  /** The closed form of the symbolic network. */
  readonly formula: (names?: readonly string[], options?: FixSymbolicOptions) => SymbolicFormula
}

/** A fitted KAN trainer: the trainer base's fitted value and the KAN verbs. */
export type KanFitted = TrainerFitted & KanVerbs

/** A session of KAN training: the trainer base's, whose `fittedAt(t)` has the KAN verbs. */
export type KanSession = IterativeSession<
  TrainState,
  TrainerParams,
  Verbs & FittedFacts & KanVerbs,
  TrainerPath<KanPath>
>

/** A KAN estimator: the trainer base's, with the KAN verbs on what it fits. */
export interface KanEstimator extends Omit<TrainerEstimator<KanHyper, KanPath>, 'fit' | 'with' | 'revive' | 'session'> {
  session(data: LearnData, options?: IterativeSessionOptions<TrainerPath<KanPath>>): KanSession
  fit(...args: Parameters<TrainerEstimator<KanHyper, KanPath>['fit']>): KanFitted
  revive(params: Json): KanFitted
  with(changes: Partial<KanHyper>): KanEstimator
}

/** A configuration of an option table with the knobs it leaves out at their defaults (a union value must be whole). */
export function completeOption<T, P>(table: OptionTable<T, P>, config: SpaceValue | undefined): SpaceValue {
  const given = (typeof config === 'object' && config !== null && !Array.isArray(config) ? config : {}) as ModeConfig
  const chosen = table.optionOf(given)
  return { ...(defaults(chosen.space) as ModeConfig), ...given, [table.tag]: chosen.name }
}

/** About `count` rows spread evenly over the data, as `[m, d]`. */
export function referenceRows(x: Value, count: number): Value {
  const [n, d] = [x.shape[0] ?? 0, x.shape[1] ?? 1]
  const flat = toFlat(x)
  const m = Math.max(1, Math.min(n, count))
  return asTensor(
    Array.from({ length: m }, (_, r) => {
      const src = m === n ? r : Math.floor((r * n) / m)
      return Array.from({ length: d }, (__, i) => flat[src * d + i] as number)
    }),
  ) as Value
}

const build = (data: LearnData, h: KanHyper, key: Key, outputs: number): KAN =>
  new KAN(h.widths === undefined ? [asTensor(data.x).shape[1] ?? 1, ...h.hidden, outputs] : [...h.widths], {
    grid: h.grid,
    degree: h.degree,
    basis: h.basis,
    base: h.base,
    knots: h.knots,
    bias: h.bias,
    sample: asTensor(data.x),
    key,
  })

const weightsOf = (h: KanHyper): { l1: number; entropy: number; coef: number } => ({
  l1: h.lambdaL1,
  entropy: h.lambdaEntropy,
  coef: h.lambdaCoef,
})

function verbsOf(
  model: Module,
  state: { readonly params: Params; readonly buffers: Params },
  data: LearnData,
  h: KanHyper,
): KanVerbs {
  const kan = model as KAN
  const s: KanState = { params: state.params, buffers: state.buffers }
  const x = asTensor(data.x) as Value
  const reference = referenceRows(x, h.regRows)
  return {
    network: kan,
    state: s,
    edgeCurve: (layer, i, j, grid) => {
      const [lo, hi] = kan.layers[layer]?.inputRange(s.buffers[String(layer)] as Params)[i] ?? [-1, 1]
      const g =
        grid ?? Array.from({ length: h.curvePoints }, (_, k) => lo + ((hi - lo) * k) / Math.max(1, h.curvePoints - 1))
      return kan.edgeCurve(layer, i, j, g, s)
    },
    curves: (points) => kanCurves(kan, s, reference, points ?? h.curvePoints),
    grids: () => kanGrids(kan, s),
    edgeScales: () => kan.edgeScales(s.params, reference, s.buffers) as Tensor[],
    sparsity: (threshold) => edgeSparsity(kan, s, reference, threshold ?? h.sparsityThreshold),
    prune: (o) => pruneKan(kan, s, x, o),
    symbolic: (o) => fixSymbolic(kan, s, x, o),
    formula: (names, o) => symbolicFormula(fixSymbolic(kan, s, x, o), names),
  }
}

const targetLike = (t: Value, shapeOf: Value): Value => reshape(t, shapeOf.shape)

function kanTrainer(kind: 'regressor' | 'classifier', changes: Partial<KanHyper> = {}): KanEstimator {
  const hyper: KanHyper = {
    ...KAN_DEFAULTS,
    ...changes,
    basis: completeOption(kanBases, changes.basis ?? KAN_DEFAULTS.basis),
    base: completeOption(kanBaseFunctions, changes.base ?? KAN_DEFAULTS.base),
    optimiser: completeOption(optimisers, changes.optimiser ?? KAN_DEFAULTS.optimiser),
  }
  const outputsOf = (data: LearnData): number => {
    if (kind === 'classifier') return Math.max(2, new Set(toFlat(asTensor(data.y ?? [0, 1]))).size)
    const y = data.y === undefined ? undefined : asTensor(data.y)
    return y === undefined || y.shape.length < 2 ? 1 : (y.shape[1] ?? 1)
  }
  const base = defineTrainer<KanHyper, KanPath>(
    {
      name: kind === 'regressor' ? 'kan-regressor' : 'kan-classifier',
      kind,
      defaults: KAN_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          grid: int(2, 200, { default: h.grid, label: 'grid size' }),
          degree: int(1, 7, { default: h.degree, label: 'spline degree' }),
          basis: kanBases.space,
          base: kanBaseFunctions.space,
          optimiser: optimisers.space,
          learningRate: real(1e-4, 1, { log: true, default: h.learningRate, label: 'learning rate' }),
          lambdaL1: real(0, 10, { default: h.lambdaL1, label: 'edge L1 weight' }),
          lambdaEntropy: real(0, 10, { default: h.lambdaEntropy, label: 'edge entropy weight' }),
          lambdaCoef: real(0, 10, { default: h.lambdaCoef, label: 'coefficient L1 weight' }),
        }),
      model: (data, h, key) => build(data, h, key, outputsOf(data)),
      objective: () => (kind === 'classifier' ? (p, t) => crossEntropy(p, t) : (p, t) => mse(p, targetLike(t, p))),
      penalty: (h, data, model) => {
        const w = weightsOf(h)
        if (w.l1 === 0 && w.entropy === 0 && w.coef === 0) return () => asTensor(0)
        const reference = referenceRows(asTensor(data.x) as Value, h.regRows)
        const kan = model as KAN
        return (params) => kanPenalties(kan, params, reference, w, kan.buffers).total
      },
      optimiser: (lr, h) => optimisers.make(h.optimiser)(lr),
      settings: (h) => ({
        learningRate: h.learningRate,
        epochs: h.epochs,
        ...(h.schedule === undefined ? {} : { schedule: h.schedule }),
        ...(h.steps === undefined ? {} : { steps: h.steps }),
        ...(h.batchSize === undefined ? {} : { batchSize: h.batchSize }),
        ...(h.clipNorm === undefined ? {} : { clipNorm: h.clipNorm }),
        ...(h.validationFraction === undefined ? {} : { validationFraction: h.validationFraction }),
      }),
      verbs: (model, state, data, h) =>
        ({ ...verbsOf(model, state, data, h) }) as KanVerbs & Readonly<Record<string, unknown>>,
      contract: (data, h): Contract<TrainState, KanPath> => {
        const kan = build(data, h, makeKey(0), outputsOf(data))
        const reference = referenceRows(asTensor(data.x) as Value, h.regRows)
        const w = weightsOf(h)
        const at = (s: TrainState): KanState => ({ params: s.params, buffers: s.buffers })
        const out: Record<string, DerivedSource<TrainState>> = {
          'kan/curves': derived<TrainState>({
            description:
              'every edge’s function on a grid over its input range, with its magnitude and whether it is on',
            cost: 'dear',
            compute: (s) => kanCurves(kan, at(s), reference, h.curvePoints) as unknown as Json,
          }),
          'kan/grids': derived<TrainState>({
            description: 'each layer’s input ranges and the interior knots of its spline edges',
            compute: (s) => kanGrids(kan, at(s)) as unknown as Json,
          }),
          'kan/penalty': derived<TrainState>({
            description: 'the sparsification penalty added to the loss (weighted)',
            compute: (s) => item(kanPenalties(kan, s.params, reference, w, s.buffers).total),
          }),
          'kan/l1': derived<TrainState>({
            description: 'the sum of the edges’ mean magnitudes (before its weight)',
            compute: (s) => item(kanPenalties(kan, s.params, reference, w, s.buffers).l1),
          }),
          'kan/entropy': derived<TrainState>({
            description: 'the per-layer entropy of the edges’ magnitudes, summed (before its weight)',
            compute: (s) => item(kanPenalties(kan, s.params, reference, w, s.buffers).entropy),
          }),
          'kan/coef': derived<TrainState>({
            description: 'the mean absolute coefficient of the edges, summed (before its weight)',
            compute: (s) => item(kanPenalties(kan, s.params, reference, { ...w, coef: 1 }, s.buffers).coef),
          }),
          'kan/sparsity': derived<TrainState>({
            description: 'the share of switched-on edges whose mean magnitude is negligible',
            compute: (s) => edgeSparsity(kan, at(s), reference, h.sparsityThreshold).fraction,
          }),
          'kan/scales': derived<TrainState>({
            description: 'the mean magnitude of every edge, per layer [out, in]',
            compute: (s) => kan.edgeScales(s.params, reference, s.buffers).map((m) => Array.from(toFlat(m))) as Json,
          }),
        }
        kan.layers.forEach((layer, l) => {
          for (let i = 0; i < layer.inFeatures; i++)
            for (let j = 0; j < layer.outFeatures; j++)
              out[`kan/edge/${l}/${i}/${j}`] = derived<TrainState>({
                description: `the function on the edge from node ${i} of layer ${l} to node ${j} of layer ${l + 1}`,
                compute: (s) =>
                  (kanCurves(kan, at(s), reference, h.curvePoints).edges.find(
                    (e) => e.layer === l && e.i === i && e.j === j,
                  ) ?? null) as unknown as Json,
              })
        })
        return out as unknown as Contract<TrainState, KanPath>
      },
    },
    hyper,
  )
  return {
    ...base,
    fit: (data, options) => base.fit(data, options) as KanFitted,
    revive: (params) => base.revive(params) as KanFitted,
    session: (data, options) => base.session(data, options) as unknown as KanSession,
    with: (more) => kanTrainer(kind, { ...hyper, ...more }),
  }
}

/**
 * A Kolmogorov-Arnold regression network trained by squared error with the paper's sparsification penalties (zero by
 * default). `fit(data)` trains `KAN([d, ...hidden, outputs])`; `with(...)` changes any hyperparameter, the slots
 * (`basis`, `base`, `knots`, `optimiser`) included.
 *
 * @concept kolmogorov-arnold-networks
 * @example
 * ```ts
 * import { kanRegressor } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 30 }, (_, i) => [-1 + i / 14.5])
 * const y = x.map(([t]) => (t ?? 0) ** 2)
 * const fit = kanRegressor.with({ hidden: [], epochs: 30, learningRate: 0.05 }).fit({ x, y }, { key: key(1) })
 * fit.network.widths // => [1, 1]
 * ```
 */
export const kanRegressor: KanEstimator = kanTrainer('regressor')

/**
 * A Kolmogorov-Arnold classifier: a logit per class, trained by cross-entropy.
 *
 * @concept kolmogorov-arnold-networks
 * @example
 * ```ts
 * import { kanClassifier } from 'mdml-engine/learn'
 * kanClassifier.name // => 'kan-classifier'
 * ```
 */
export const kanClassifier: KanEstimator = kanTrainer('classifier')
