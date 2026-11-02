/**
 * The trainer base: an estimator whose fit is minibatch training of a module from `nn`, run as a session. A particular
 * declares the model, the objective, the optimiser (given the learning rate, which the settings' `schedule` shapes from
 * the `schedules` table), the data source (rows, or a stream drawn into the rows the budget needs), the batch size and
 * budget, and optionally validation data and metrics; the base owns the loop (an `Algorithm` over `fitAlgorithm`, one
 * step an optimiser step or, as an option, one epoch), the standard paths, the parameter checkpoints that let a probe
 * run at any step, the kind's probes, the scikit-learn `fit` surface, and `method` and `session` like the iterative
 * base. One module and one optimiser: a trainer with several (a GAN's two) writes its own step over `trainState`.
 *
 * @index structure Trainer base: minibatch training of an nn module as a session.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { resolveKey } from '../../context/index.ts'
import { rowCount, type TabularData, take } from '../../data/index.ts'
import type { Exposition } from '../../exposition/index.ts'
import { type Callback, fitAlgorithm, type Module, type Params, type TrainState } from '../../nn/index.ts'
import { type GradientTransformation, type LearningRate, schedules } from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../random/index.ts'
import { type DictSpace, dict, type SpaceValue } from '../../space/index.ts'
import { asTensor, type Tensor, toFlat, treePaths, type Value, zeros } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineMethod,
  derived,
  event,
  Log,
  type Method,
  type Run,
  type Session,
  type Subscription,
  scalar,
  session,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import type { Estimator, EstimatorKind, FitOptions, Fitted, FittedFacts, LearnData, Verbs } from '../estimator.ts'
import { withInputRows } from '../explained.ts'
import {
  curvesOf,
  defaultMetrics,
  evaluate,
  heldOut,
  type Layer,
  type LayerPath,
  layersOf,
  type Metrics,
  networkVerbs,
  recordLayers,
  tabular,
} from './_trainer.ts'
import { fittedSession, type IterativeSession, type IterativeSessionOptions } from './iterative.ts'
import { type KindPath, kindProbes } from './kinds.ts'

/** Data that arrives rather than sits: `draw(n, key)` gives `n` fresh rows, and a trainer draws what its budget needs. */
export interface LearnStream {
  draw(n: number, key: Key): LearnData
}

/** The training schedule a particular declares from its hyperparameters. */
export interface TrainerSettings {
  readonly learningRate: number
  /** A configuration of `schedules` (`{ kind: 'cosine', steps: 200 }`) shaping the learning rate; default constant. */
  readonly schedule?: SpaceValue
  /** Passes over the rows (default 1); `steps` instead counts minibatches in all (required for a stream). */
  readonly epochs?: number
  readonly steps?: number
  /** Minibatch size (default the whole data; required for a stream). */
  readonly batchSize?: number
  /** What one session step is: a minibatch (default) or an epoch. */
  readonly unit?: 'step' | 'epoch'
  readonly clipNorm?: number
  readonly accumulateGradients?: number
  readonly callbacks?: readonly Callback[]
  /** Hold out this share of the rows as validation data when the particular gives none. */
  readonly validationFraction?: number
  /** Rows the layer activations are computed on (default 64). */
  readonly probeRows?: number
}

/**
 * The paths of a session of the trainer base: the particular's own (`K`), the base's (`train/...`, `eval/<metric>`
 * for the validation metrics, `layer/<i>/...` for the layers, `model/params`) and the kind's probes. With `K` left at
 * `string` every path is allowed.
 *
 * @example
 * ```ts
 * import type { TrainerPath } from 'mdml-engine/learn'
 * type P = TrainerPath<'gan/critic'>   // 'gan/critic' | 'train/loss' | 'train/gradNorm' | ...
 * ```
 */
export type TrainerPath<K extends string = string> =
  | K
  | 'train/loss'
  | 'train/gradNorm'
  | 'train/lr'
  | 'train/step/end'
  | 'train/epoch/end'
  | 'eval/loss'
  | `eval/${string}`
  | LayerPath
  | 'model/params'
  | KindPath

/**
 * What a particular declares over the trainer base. The last parameter is the type of the paths it adds to the base's
 * (`string` by default: unchecked); type them as for the iterative base (`IterativeParticular`): a `FooPath` union, the
 * `contract` returning `Contract<TrainState, FooPath>`, the estimator typed `TrainerEstimator<H, FooPath>`.
 */
export interface TrainerParticular<H extends object, K extends string = string> {
  readonly name: string
  readonly kind: EstimatorKind
  readonly defaults: H
  readonly hyperSpace?: (hyper: H) => DictSpace
  /** The module (a classifier ends in logits). Its parameters at the given key are the starting point. */
  model(data: LearnData, hyper: H, key: Key): Module
  /** The training objective of predictions and targets. */
  objective(hyper: H, data: LearnData): (prediction: Value, target: Value) => Value
  /**
   * A regulariser of the model's parameters, added to every minibatch loss (weight decay, a smoothness penalty on
   * spline coefficients). `train/loss` then includes it; `eval/loss` is the objective alone.
   */
  penalty?(hyper: H, data: LearnData, model: Module): (params: Params) => Value
  /** The optimiser, given the learning rate (a number, or the schedule the settings asked for). */
  optimiser(learningRate: LearningRate, hyper: H): GradientTransformation
  settings(hyper: H, data: LearnData): TrainerSettings
  /** Where the batches come from (default the rows given to `fit`). */
  source?(data: LearnData, hyper: H): LearnData | LearnStream
  /** Validation data (else `validationFraction` of the rows, else none). */
  validation?(data: LearnData, hyper: H, key: Key): LearnData | undefined
  /** Metrics of predictions and targets logged as `eval/<name>` (default accuracy for a classifier, rmse otherwise). */
  metrics?(hyper: H): Readonly<Record<string, (prediction: Value, target: Value) => number>>
  /**
   * Verbs and facts beyond those of the kind (predicted distributions, per-term effects), over the model and the
   * parameters and buffers of a state; they are merged over the standard ones.
   */
  verbs?(
    model: Module,
    state: { readonly params: Params; readonly buffers: Params },
    data: LearnData,
    hyper: H,
  ): Verbs & FittedFacts & Readonly<Record<string, unknown>>
  /** What the state can tell beyond the standard paths. */
  contract?(data: LearnData, hyper: H): Contract<TrainState, K>
  records?(state: TrainState, data: LearnData, hyper: H): readonly Exposition[]
  readonly projections?: Readonly<Record<string, (run: Run, data: LearnData, hyper: H) => Exposition | undefined>>
  /** What `fit` keeps by default (default the loss, gradient norm and learning rate, and `eval/` when validating). */
  readonly logs?: Subscription<NoInfer<TrainerPath<K>>>
}

/** The parameters of a trained network as data, with what rebuilds its architecture. */
export type TrainerParams = {
  readonly params: Json
  readonly buffers: Json
  /** Input features, class labels (classifiers) and target width (`null` for a vector of targets). */
  readonly d: number
  readonly classes: number[] | null
  readonly outputs: number | null
}

/** A fitted trainer value: the verbs of its kind over the network, its records and the tape. */
export type TrainerFitted = Fitted<TrainerParams> &
  Verbs &
  FittedFacts & { readonly exposition: readonly Exposition[]; readonly run: Run }

/**
 * A session of training: the iterative session, whose `fittedAt(t)` is the network at a reached step. The parameter is
 * the type of the session's paths ({@link TrainerPath} of the particular's).
 */
export type TrainerSession<K extends string = string> = IterativeSession<
  TrainState,
  TrainerParams,
  Verbs & FittedFacts,
  K
>

/** An estimator on the trainer base: `fit`, its method and a session of its training. */
export interface TrainerEstimator<H extends object, K extends string = string> extends Estimator<H, TrainerFitted> {
  method(data: LearnData): Method<TrainState, Config, TrainerPath<K>>
  session(data: LearnData, options?: IterativeSessionOptions<TrainerPath<K>>): TrainerSession<TrainerPath<K>>
  with(changes: Partial<H>): TrainerEstimator<H, K>
}

const isStream = (s: LearnData | LearnStream): s is LearnStream => 'draw' in s
const EMPTY_RUN: Run = { schema: 'mdml.run/1', events: [] }

/** Everything the loop needs for one (data, hyperparameters, key), built once. */
interface Built {
  readonly model: Module
  readonly algorithm: Algorithm<TrainState, TrainerPath>
  readonly validation: TabularData | undefined
  readonly loss: (prediction: Value, target: Value) => Value
  readonly metrics: Metrics
  readonly layers: readonly Layer[]
  readonly probe: Tensor
}

/** The plan of a run, which depends on the settings and the number of rows but not on the key. */
interface Plan {
  readonly settings: TrainerSettings
  readonly batch: number
  readonly perEpoch: number
  readonly total: number
  /** Session steps: `total` minibatches, or epochs. */
  readonly steps: number
}

/**
 * An estimator over the trainer base.
 *
 * @param p - The particular.
 * @param hyper - Its hyperparameters (default the particular's defaults).
 * @returns The estimator, with `method` and `session` beside `fit`.
 * @concept log-contract
 * @example
 * ```ts
 * import { defineTrainer } from 'mdml-engine/learn'
 * import { mlp } from 'mdml-engine/nn'
 * import { adam } from 'mdml-engine/optim'
 * import { crossEntropy } from 'mdml-engine/losses'
 * import { key } from 'mdml-engine/random'
 * import { toFlat } from 'mdml-engine/tensor'
 * const net = defineTrainer({
 *   name: 'tiny-mlp', kind: 'classifier', defaults: { hidden: 8 },
 *   model: (_d, h, k) => mlp([2, h.hidden, 2], { key: k }),
 *   objective: () => (p, t) => crossEntropy(p, t),
 *   optimiser: (lr) => adam({ learningRate: lr }),
 *   settings: () => ({ learningRate: 0.05, epochs: 2, batchSize: 2 }),
 * })
 * const data = { x: [[0, 0], [1, 1], [0, 1], [1, 0]], y: [0, 1, 0, 1] }
 * toFlat(net.fit(data, { key: key(1) }).predict(data.x)).length // => 4
 * ```
 */
export function defineTrainer<H extends object, K extends string = string>(
  p: TrainerParticular<H, K>,
  hyper: H = p.defaults,
): TrainerEstimator<H, K> {
  const space: DictSpace = p.hyperSpace?.(hyper) ?? dict({})
  const knobs = Object.keys(space.spaces)
  const hyperOf = (c: Config): H => ({ ...hyper, ...c }) as H
  const configOf = (h: H): Config =>
    Object.fromEntries(knobs.map((k) => [k, (h as Record<string, unknown>)[k] as SpaceValue])) as Config
  const planOf = (data: LearnData, h: H): Plan => {
    const settings = p.settings(h, data)
    const src = p.source?.(data, h) ?? data
    const stream = isStream(src)
    if (stream && (settings.steps === undefined || settings.batchSize === undefined))
      throw new MdmlError('value', `${p.name}: a stream needs settings.steps and settings.batchSize`)
    const n = stream ? (settings.steps ?? 1) * (settings.batchSize ?? 1) : (asTensor(src.x).shape[0] ?? 0)
    const rows = stream || p.validation !== undefined ? n : n - heldOut(n, settings.validationFraction)
    const batch = Math.min(settings.batchSize ?? rows, rows)
    const perEpoch = Math.ceil(rows / batch)
    const total = settings.steps ?? (settings.epochs ?? 1) * perEpoch
    return { settings, batch, perEpoch, total, steps: settings.unit === 'epoch' ? Math.ceil(total / perEpoch) : total }
  }
  const cache = new WeakMap<LearnData, Map<string, Built>>()
  const builtOf = (data: LearnData, h: H, key: Key): Built => {
    const id = `${JSON.stringify(h)}|${key.words.join(',')}`
    const forData = cache.get(data) ?? new Map<string, Built>()
    cache.set(data, forData)
    const made = forData.get(id) ?? build(data, h, key)
    forData.set(id, made)
    return made
  }
  const build = (data: LearnData, h: H, key: Key): Built => {
    const plan = planOf(data, h)
    const { settings } = plan
    const src = p.source?.(data, h) ?? data
    let rows = tabular(p.kind, isStream(src) ? src.draw(plan.total * plan.batch, foldIn(key, 3)) : src)
    let validation: TabularData | undefined
    const given = p.validation?.(data, h, foldIn(key, 4))
    if (given !== undefined) validation = tabular(p.kind, given)
    else if ((settings.validationFraction ?? 0) > 0 && !isStream(src)) {
      const order = Array.from(permutation(foldIn(key, 4), rowCount(rows)).data)
      const cut = order.length - heldOut(order.length, settings.validationFraction)
      validation = take(rows, order.slice(cut))
      rows = take(rows, order.slice(0, cut))
    }
    const model = p.model(data, h, key)
    const loss = p.objective(h, data)
    const lr: LearningRate =
      settings.schedule === undefined ? settings.learningRate : schedules.make(settings.schedule)(settings.learningRate)
    const inner = fitAlgorithm(model, rows, {
      loss,
      optimizer: p.optimiser(lr, h),
      epochs: Math.ceil(plan.total / plan.perEpoch),
      batchSize: plan.batch,
      key: foldIn(key, 1),
      ...(p.penalty === undefined ? {} : { penalty: p.penalty(h, data, model) }),
      ...(settings.clipNorm === undefined ? {} : { clipNorm: settings.clipNorm }),
      ...(settings.accumulateGradients === undefined ? {} : { accumulateGradients: settings.accumulateGradients }),
      ...(settings.callbacks === undefined ? {} : { callbacks: settings.callbacks }),
    })
    const metrics = p.metrics?.(h) ?? defaultMetrics(p.kind)
    const layers = layersOf(model)
    const probe = take(
      rows,
      Array.from({ length: Math.min(settings.probeRows ?? 64, rowCount(rows)) }, (_, i) => i),
    ).x
    const rate = (updates: number): number => (typeof lr === 'number' ? lr : lr(updates))
    const algorithm: Algorithm<TrainState, TrainerPath> = {
      ...inner,
      name: p.name,
      step: (s, ctx) => {
        // One minibatch, or the rest of the epoch; the losses of an epoch are averaged.
        const todo =
          settings.unit === 'epoch' ? Math.min(plan.perEpoch - (s.step % plan.perEpoch), plan.total - s.step) : 1
        let st = s
        let sum = 0
        let used = rate(s.updates) * s.lrScale
        for (let i = 0; i < todo && (i === 0 || inner.done?.(st, ctx.t) === undefined); i++) {
          used = rate(st.updates) * st.lrScale
          st = inner.step(st, ctx)
          sum += st.loss
        }
        ctx.log('train/loss', todo > 1 ? sum / todo : st.loss)
        ctx.log('train/gradNorm', st.gradNorm)
        ctx.log('train/lr', used)
        ctx.emit('train/step/end', { t: ctx.t, step: st.step })
        if (st.step % plan.perEpoch === 0 || st.step >= plan.total)
          ctx.emit('train/epoch/end', { t: ctx.t, epoch: Math.ceil(st.step / plan.perEpoch) })
        recordLayers(layers, probe, st, ctx)
        if (validation !== undefined && ctx.wants('eval/loss')) {
          const ev = evaluate(model, validation, st, loss, metrics)
          for (const [name, v] of Object.entries(ev)) ctx.log(`eval/${name}`, v)
        }
        return st
      },
      done: (s, t) => inner.done?.(s, t) ?? (s.step >= plan.total ? 'completed' : undefined),
    }
    return { model, algorithm, validation, loss, metrics, layers, probe }
  }
  const classesOf = (data: LearnData): number[] | undefined =>
    p.kind === 'classifier' && data.y !== undefined
      ? [...new Set(toFlat(asTensor(data.y)))].sort((a, b) => a - b)
      : undefined
  const fittedOf = (b: Built, state: TrainState, data: LearnData): Fitted<TrainerParams> & Verbs & FittedFacts => {
    const classes = classesOf(data)
    const y = data.y === undefined ? undefined : asTensor(data.y)
    return {
      ...networkVerbs(p.kind, b.model, state, classes),
      ...(p.verbs?.(b.model, state, data, hyper) ?? {}),
      estimator: p.name,
      kind: p.kind,
      nFeaturesIn: asTensor(data.x).shape[1] ?? 1,
      params: {
        params: stateToJson(state.params),
        buffers: stateToJson(state.buffers),
        d: asTensor(data.x).shape[1] ?? 1,
        classes: classes ?? null,
        outputs: y === undefined || y.shape.length < 2 ? null : (y.shape[1] ?? 1),
      },
    }
  }
  const contractOf = (data: LearnData, h: H, key: Key): Contract<TrainState, TrainerPath<K>> => {
    const b = builtOf(data, h, key)
    const out: Record<string, Contract<TrainState>[string]> = {
      'train/loss': scalar({ description: 'minibatch loss (the epoch’s mean per epoch step)', direction: 'lower' }),
      'train/gradNorm': scalar({ description: 'global gradient norm' }),
      'train/lr': scalar({ description: 'learning rate (times the plateau scale)' }),
      'train/step/end': event({ context: { t: 'the step just finished', step: 'minibatches so far' } }),
      'train/epoch/end': event({ context: { t: 'the step just finished', epoch: 'epochs so far' } }),
    }
    if (b.validation !== undefined) {
      out['eval/loss'] = scalar({ description: 'the objective on the validation rows', direction: 'lower' })
      for (const name of Object.keys(b.metrics))
        out[`eval/${name}`] = scalar({ description: `${name} on the validation rows` })
    }
    for (const l of b.layers) {
      out[`layer/${l.index}/activations`] = tensor({
        role: 'activation',
        description: 'the layer’s output on the probe rows',
      })
      if (Object.keys(treePaths(l.module.params)).length === 0) continue
      out[`layer/${l.index}/weights`] = tensor({
        role: 'param',
        description: 'the layer’s weight (summarised unless asked in full)',
      })
      out[`layer/${l.index}/params`] = tensor({
        role: 'param',
        description: 'all of the layer’s parameters, flattened',
      })
    }
    const params = (s: TrainState): Json => ({ params: stateToJson(s.params), buffers: stateToJson(s.buffers) })
    out['model/params'] = derived<TrainState>({ description: 'the network’s parameters at this step', compute: params })
    Object.assign(out, kindProbes<TrainState>(p.kind, { data, fitted: (s) => fittedOf(b, s, data) }))
    // The layers and the validation metrics depend on the data, so `out` is a plain record of `TrainerPath` keys.
    return { ...out, ...(p.contract?.(data, h) ?? {}) } as Contract<TrainState, TrainerPath<K>>
  }
  const defaultLogs = (data: LearnData): Subscription<TrainerPath<K>> => {
    const known = contractOf(data, hyper, makeKey(0))
    const out: Record<string, ReturnType<typeof Log>> = {}
    for (const path of ['train/loss', 'train/gradNorm', 'train/lr']) out[path] = Log()
    for (const path of Object.keys(known))
      if (path.startsWith('eval/')) out[path] = Log({ every: ['train/epoch/end', 1] })
    return out as Subscription<TrainerPath<K>>
  }
  const method = (data: LearnData): Method<TrainState, Config, TrainerPath<K>> =>
    defineMethod<TrainState, Config, TrainerPath<K>>({
      name: p.name,
      config: space,
      make: (c, key) => builtOf(data, hyperOf(c), key).algorithm,
      contract: (c, key) => contractOf(data, hyperOf(c), key),
      defaults: p.logs ?? defaultLogs(data),
      projections: {
        curves: (run) => curvesOf(run, planOf(data, hyper).settings.unit ?? 'step'),
        ...Object.fromEntries(
          Object.entries(p.projections ?? {}).map(([n, f]) => [n, (run: Run) => f(run, data, hyper)]),
        ),
      },
    })
  const wrap = (
    data: LearnData,
    key: Key,
    s: Session<TrainState, Config, TrainerPath<K>>,
  ): TrainerSession<TrainerPath<K>> =>
    fittedSession(s, (state, config) => fittedOf(builtOf(data, hyperOf(config), foldIn(key, 7)), state, data))
  const sessionOf = (
    data: LearnData,
    options: IterativeSessionOptions<TrainerPath<K>>,
    key: Key,
  ): TrainerSession<TrainerPath<K>> => {
    const steps = options.steps ?? planOf(data, hyper).steps
    return wrap(
      data,
      key,
      session(method(data), {
        key,
        config: configOf(hyper),
        logs: options.logs ?? (options.explain === false ? 'off' : 'defaults'),
        mode: options.mode ?? 'live',
        steps,
        ...(options.past === undefined ? {} : { past: options.past }),
        checkpointEvery: Math.max(1, Math.floor(steps / 100)),
      }),
    )
  }
  const fit = (data: LearnData, options: FitOptions = {}): TrainerFitted => {
    const s = sessionOf(
      data,
      { ...options, mode: 'eager', past: false } as IterativeSessionOptions<TrainerPath<K>>,
      resolveKey(options, p.name),
    )
    const run = s.run()
    const fitted = s.fittedAt(s.t)
    const records: Exposition[] = [
      ...(p.records?.(s.state, data, hyper) ?? []),
      ...[
        curvesOf(run, planOf(data, hyper).settings.unit ?? 'step'),
        ...Object.values(p.projections ?? {}).map((f) => f(run, data, hyper)),
      ].flatMap((r) => r ?? []),
    ]
    const exposition = options.explain === false ? [] : withInputRows(records, data, fitted.classes, fitted.kind)
    return { ...fitted, exposition, run }
  }
  return {
    name: p.name,
    kind: p.kind,
    hyper,
    ...(knobs.length === 0 ? {} : { hyperSpace: space }),
    fit,
    revive: (params) => {
      const j = params as TrainerParams
      const x = zeros([1, j.d])
      const y = j.classes ?? (j.outputs === null ? [0] : [Array.from({ length: j.outputs }, () => 0)])
      const data: LearnData = { x, y }
      const model = p.model(data, hyper, foldIn(makeKey(0), 7))
      const state = { params: stateFromJson(j.params) as Params, buffers: stateFromJson(j.buffers) as Params }
      return {
        ...networkVerbs(p.kind, model, state, j.classes ?? undefined),
        ...(p.verbs?.(model, state, data, hyper) ?? {}),
        estimator: p.name,
        kind: p.kind,
        params: j,
        exposition: [],
        run: EMPTY_RUN,
      }
    },
    with: (changes) => defineTrainer(p, { ...hyper, ...changes }),
    method,
    session: (data, options = {}) => sessionOf(data, options, resolveKey(options, p.name)),
  }
}
