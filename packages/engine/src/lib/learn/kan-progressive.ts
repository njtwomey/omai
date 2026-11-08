/**
 * The Kolmogorov-Arnold paper's loop as one staged `Algorithm` on the iterative base: train a small-grid network with
 * the sparsification penalties, prune it to what mattered, refine the grid by refitting every edge (the function is
 * kept, the capacity grows), train again, and repeat down a list of grid sizes ("progressive KAN"). The grid of the first
 * stage is also moved to the data as it flows through the network (pykan's `update_grid_from_samples`, every few steps at
 * the start), because the hidden layers' inputs drift while the layers below learn. The state is the structure (widths,
 * grid and the families: a `KanSpec`) together with the training state (parameters, masks, ranges, optimiser state), all
 * plain data, so a run can be paused, serialised, resumed or forked mid-stage; a step is one minibatch, or one
 * structural change (a prune, a refinement) that the tape records as an event with the new widths. Beyond the iterative
 * base's `solver/...` channels (loss, parameters, active edges, grid, stage) the tape holds every edge's curve
 * (`kan/curves`), the grids, the sparsity and the stage at any step, so a page can step through the whole procedure.
 */

import type { Json } from '../base/index.ts'
import { mse } from '../losses/index.ts'
import { r2Score } from '../metrics/index.ts'
import {
  edgeSparsity,
  fitAlgorithm,
  KAN,
  type KanSpec,
  type KanState,
  kanBaseFunctions,
  kanBases,
  kanCurves,
  kanGrids,
  kanPenalties,
  kanSpec,
  type Params,
  pruneKan,
  refineGrid,
  type TrainState,
  trainState,
  updateGrid,
} from '../nn/index.ts'
import { optimisers, schedules } from '../optim/index.ts'
import { foldIn, key as makeKey } from '../random/index.ts'
import { dict, int, real, type SpaceValue } from '../space/index.ts'
import { asTensor, reshape, type Tensor, type TensorLike, toFlat, treePaths, type Value } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  derived,
  type EventSource,
  event,
  Log,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { tabular } from './bases/_trainer.ts'
import { type IterativeEstimator, iterative } from './bases/iterative.ts'
import type { FittedFacts, LearnData, Verbs } from './estimator.ts'
import { completeOption, type KanVerbs, kanVerbs, referenceRows } from './kan.ts'

/** One phase of the procedure. */
export type ProgressivePhase =
  | { readonly kind: 'train'; readonly steps: number; readonly regularise: boolean }
  | { readonly kind: 'prune' }
  | { readonly kind: 'refine'; readonly grid: number }

/** The hyperparameters of the progressive KAN. */
export interface ProgressiveKanHyper {
  /** The hidden widths of the starting network (default `[5]`). */
  readonly hidden: readonly number[]
  /** The grid sizes of the successive stages (default `[3, 5, 10]`): the network is refined to each in turn. */
  readonly grids: readonly number[]
  readonly degree: number
  readonly basis: SpaceValue
  readonly base: SpaceValue
  readonly knots: SpaceValue
  /** Minibatches per stage (default 100). */
  readonly stepsPerStage: number
  /** The paper's penalties while a stage is regularised: edge L1, edge entropy, coefficient L1. */
  readonly lambdaL1: number
  readonly lambdaEntropy: number
  readonly lambdaCoef: number
  /** How many of the first stages are regularised (default 1). */
  readonly regularisedStages: number
  /** Prune after this stage (default 0: after the sparsifying one) with these thresholds; `false` for never. */
  readonly prune: false | { readonly nodeThreshold?: number; readonly edgeThreshold?: number; readonly after?: number }
  /** Move the grid to the data every `every` steps while a stage trains, until step `until` of it, in the first `stages` stages. */
  readonly gridUpdate: false | { readonly every: number; readonly until: number; readonly stages: number }
  /** The phases, instead of the ones the fields above make. */
  readonly plan?: readonly ProgressivePhase[]
  readonly optimiser: SpaceValue
  readonly learningRate: number
  readonly schedule?: SpaceValue
  readonly batchSize?: number
  readonly clipNorm?: number
  /** Rows (spread over the data) on which edge magnitudes, curves and sparsity are measured (default 128). */
  readonly regRows: number
  readonly curvePoints: number
  readonly sparsityThreshold: number
}

/** The default hyperparameters. */
export const PROGRESSIVE_KAN_DEFAULTS: ProgressiveKanHyper = {
  hidden: [5],
  grids: [3, 5, 10],
  degree: 3,
  basis: { kind: 'bspline' },
  base: { kind: 'silu' },
  knots: { kind: 'uniform' },
  stepsPerStage: 100,
  lambdaL1: 0.01,
  lambdaEntropy: 0.02,
  lambdaCoef: 0,
  regularisedStages: 1,
  prune: { nodeThreshold: 0.01, edgeThreshold: 0.03, after: 0 },
  gridUpdate: { every: 10, until: 50, stages: 1 },
  optimiser: { kind: 'adam' },
  learningRate: 0.02,
  regRows: 128,
  curvePoints: 41,
  sparsityThreshold: 0.01,
}

/** A structural change on the tape. */
export interface ProgressiveEvent {
  /** The training step it happened at. */
  readonly step: number
  readonly kind: 'prune' | 'refine' | 'grid-update'
  readonly widths: readonly number[]
  readonly grid: number
  /** The parameter count after it. */
  readonly parameters: number
  /** How far the network's function moved on the data (refinement and grid updates). */
  readonly drift?: number
  /** The edges switched off (pruning). */
  readonly edgesOff?: number
}

/** The state of the procedure: the structure, the training state, and where in the plan it is. */
export interface ProgressiveKanState {
  readonly spec: KanSpec
  readonly train: TrainState
  /** The index into the plan, the steps taken in a training phase, and the refinements done so far. */
  readonly phase: number
  readonly inPhase: number
  readonly stage: number
  readonly history: readonly ProgressiveEvent[]
}

/** The paths the progressive KAN adds to the iterative base's. */
export type ProgressiveKanPath =
  | 'kan/curves'
  | 'kan/grids'
  | 'kan/sparsity'
  | 'kan/stage'
  | 'kan/grid-update'
  | 'kan/prune'
  | 'kan/refine'

/** The fitted parameters: the structure and the state of the network, and the reference rows. */
export type ProgressiveKanParams = {
  /** The structure (a `KanSpec`), as data. */
  readonly spec: Json
  readonly params: Json
  readonly buffers: Json
  readonly reference: number[][]
}

/** The plan of a hyperparameter set: train (regularised) and prune, then refine and train for each further grid. */
export function progressivePlan(h: ProgressiveKanHyper): readonly ProgressivePhase[] {
  if (h.plan !== undefined) return h.plan
  const plan: ProgressivePhase[] = []
  h.grids.forEach((grid, s) => {
    if (s > 0) plan.push({ kind: 'refine', grid })
    plan.push({ kind: 'train', steps: h.stepsPerStage, regularise: s < h.regularisedStages })
    if (h.prune !== false && s === (h.prune.after ?? 0)) plan.push({ kind: 'prune' })
  })
  return plan
}

const totalSteps = (plan: readonly ProgressivePhase[]): number => plan.length

const structureOf = (spec: KanSpec): Omit<KanSpec, 'widths'> => {
  const { widths: _w, ...rest } = spec
  return rest
}

const countActive = (buffers: Params): number =>
  Object.values(buffers).reduce<number>((acc, layer) => {
    const mask = (layer as Params)['mask']
    return acc + (mask === undefined ? 0 : toFlat(mask as Value).reduce((a, v) => a + v, 0))
  }, 0)

interface Built {
  readonly model: KAN
  readonly inner: Algorithm<TrainState>
  /** The buffers of the state being stepped: the penalty reads the masks and ranges from here. */
  held: Params
}

function progressiveAlgorithm(
  data: LearnData,
  h: ProgressiveKanHyper,
): Algorithm<ProgressiveKanState, ProgressiveKanPath> {
  const rows = tabular('regressor', data)
  const x = rows.x as Value
  const plan = progressivePlan(h)
  const reference = referenceRows(x, h.regRows)
  const lr = h.schedule === undefined ? h.learningRate : schedules.make(h.schedule)(h.learningRate)
  const optimizer = optimisers.make(h.optimiser)(lr)
  const weights = { l1: h.lambdaL1, entropy: h.lambdaEntropy, coef: h.lambdaCoef }
  const trainPhases = plan.reduce((n, p) => n + (p.kind === 'train' ? p.steps : 0), 0)
  const cache = new Map<string, Built>()
  const built = (spec: KanSpec, regularise: boolean): Built => {
    const id = JSON.stringify([spec, regularise])
    const found = cache.get(id)
    if (found !== undefined) return found
    const model = new KAN(spec.widths, { ...structureOf(spec), key: makeKey(0) })
    const holder: { held: Params } = { held: model.buffers }
    const inner = fitAlgorithm(model, rows, {
      loss: (p, t) => mse(p, reshape(t, p.shape)),
      optimizer,
      epochs: Math.max(1, trainPhases) * 4,
      ...(h.batchSize === undefined ? {} : { batchSize: h.batchSize }),
      ...(h.clipNorm === undefined ? {} : { clipNorm: h.clipNorm }),
      key: makeKey(1),
      ...(regularise
        ? { penalty: (params: Params) => kanPenalties(model, params, reference, weights, holder.held).total }
        : {}),
    })
    const out: Built = {
      model,
      inner,
      get held() {
        return holder.held
      },
      set held(b: Params) {
        holder.held = b
      },
    }
    cache.set(id, out)
    return out
  }
  const rebuilt = (
    s: ProgressiveKanState,
    spec: KanSpec,
    state: KanState,
    event: ProgressiveEvent,
  ): ProgressiveKanState => {
    const t = s.train
    return {
      ...s,
      spec,
      train: { ...t, params: state.params, buffers: state.buffers, optState: optimizer.init(state.params) },
      history: [...s.history, event],
    }
  }
  return {
    name: 'progressive-kan',
    config: { grids: [...h.grids], hidden: [...h.hidden], stepsPerStage: h.stepsPerStage } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'minibatch loss (with the penalties during a regularised stage)', direction: 'lower' },
        gradNorm: { description: 'global gradient norm' },
        parameters: { description: 'parameters of the current network' },
        activeEdges: { description: 'edges switched on' },
        grid: { description: 'the grid size of the current stage' },
        stage: { description: 'refinements done so far' },
      },
    }),
    init: (k) => {
      const spec = kanSpec([asTensor(data.x).shape[1] ?? 1, ...h.hidden, 1], {
        grid: h.grids[0] ?? 3,
        degree: h.degree,
        basis: h.basis,
        base: h.base,
        knots: h.knots,
      })
      const model = new KAN(spec.widths, { ...structureOf(spec), sample: x, key: foldIn(k, 0) })
      return { spec, train: trainState(model, optimizer, foldIn(k, 1)), phase: 0, inPhase: 0, stage: 0, history: [] }
    },
    step: (s, ctx) => {
      const phase = plan[s.phase]
      if (phase === undefined) return s
      const at = (): KanState => ({ params: s.train.params, buffers: s.train.buffers })
      if (phase.kind === 'train') {
        let state = s
        const b = built(s.spec, phase.regularise)
        const gu = h.gridUpdate
        if (gu !== false && s.stage < gu.stages && s.inPhase < gu.until && s.inPhase % gu.every === 0) {
          const u = updateGrid(b.model, at(), x)
          state = rebuilt(s, s.spec, u.state, {
            step: s.train.step,
            kind: 'grid-update',
            widths: s.spec.widths,
            grid: s.spec.grid,
            parameters: b.model.parameterCount(),
            drift: u.drift.rms,
          })
          // Same shapes, so the moments of the optimiser stay.
          state = { ...state, train: { ...state.train, optState: s.train.optState } }
          ctx.emit('kan/grid-update', { t: ctx.t, drift: u.drift.rms })
        }
        b.held = state.train.buffers
        const next = b.inner.step(state.train, ctx)
        const inPhase = state.inPhase + 1
        const over = inPhase >= phase.steps
        return { ...state, train: next, inPhase: over ? 0 : inPhase, phase: over ? state.phase + 1 : state.phase }
      }
      const model = built(s.spec, false).model
      if (phase.kind === 'prune') {
        const p = pruneKan(model, at(), x, h.prune === false ? {} : h.prune)
        const event: ProgressiveEvent = {
          step: s.train.step,
          kind: 'prune',
          widths: p.kan.widths,
          grid: p.kan.spec.grid,
          parameters: p.parameters.after,
          edgesOff: p.edgesOff,
        }
        ctx.emit('kan/prune', { t: ctx.t, widths: [...p.kan.widths], kept: p.kept.map((k) => [...k]) })
        return { ...rebuilt(s, p.kan.spec, p.state, event), phase: s.phase + 1, inPhase: 0 }
      }
      const r = refineGrid(model, at(), phase.grid, x)
      const event: ProgressiveEvent = {
        step: s.train.step,
        kind: 'refine',
        widths: r.kan.widths,
        grid: phase.grid,
        parameters: r.kan.parameterCount(),
        drift: r.drift.rms,
      }
      ctx.emit('kan/refine', { t: ctx.t, grid: phase.grid, drift: r.drift.rms })
      return { ...rebuilt(s, r.kan.spec, r.state, event), phase: s.phase + 1, inPhase: 0, stage: s.stage + 1 }
    },
    done: (s) =>
      s.phase >= plan.length
        ? 'completed'
        : !Number.isNaN(s.train.loss) && !Number.isFinite(s.train.loss)
          ? 'diverged'
          : undefined,
    observe: (s) => ({
      metrics: {
        ...(Number.isNaN(s.train.loss) ? {} : { loss: s.train.loss, gradNorm: s.train.gradNorm }),
        parameters:
          Object.keys(treePaths(s.train.params)).length === 0 ? 0 : built(s.spec, false).model.parameterCount(),
        activeEdges: countActive(s.train.buffers),
        grid: s.spec.grid,
        stage: s.stage,
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ProgressiveKanState,
  }
}

/** What a fitted progressive KAN offers: the verbs of a regressor and the KAN verbs. */
export type ProgressiveKanVerbs = Verbs & FittedFacts & KanVerbs

const rebuiltNetwork = (p: ProgressiveKanParams): { kan: KAN; state: KanState } => ({
  kan: new KAN((p.spec as unknown as KanSpec).widths, {
    ...structureOf(p.spec as unknown as KanSpec),
    key: makeKey(0),
  }),
  state: { params: stateFromJson(p.params) as Params, buffers: stateFromJson(p.buffers) as Params },
})

function progressiveVerbs(p: ProgressiveKanParams, h: ProgressiveKanHyper): ProgressiveKanVerbs {
  const { kan, state } = rebuiltNetwork(p)
  const reference = asTensor(p.reference) as Value
  const predict = (x: TensorLike): Tensor => {
    const out = kan.apply(state.params, asTensor(x), { buffers: state.buffers, train: false })
    return (out.shape[1] === 1 ? reshape(out, [out.shape[0] ?? 0]) : out) as Tensor
  }
  return {
    ...kanVerbs(kan, state, { reference, training: reference }, h),
    predict,
    score: (d) => r2Score(asTensor(d.y ?? []), predict(d.x)).estimate.value,
    nFeaturesIn: (p.spec as unknown as KanSpec).widths[0] as number,
  }
}

/**
 * The paper's progressive procedure as an estimator: sparsify, prune, refine the grid, train again. `fit(data)` runs the
 * plan to the end and returns the final network (its structure is whatever pruning left); `session(data)` steps through
 * it, with `kan/curves`, `kan/stage` and the `solver/...` channels probeable at any step and the structural changes
 * as events (`kan/prune`, `kan/refine`, `kan/grid-update`). `with({ plan })` replaces the plan, `with({ grids })` the
 * sequence of grid sizes.
 *
 * @concept kolmogorov-arnold-networks
 * @example
 * ```ts
 * import { progressiveKan, progressivePlan } from 'omai-engine/learn'
 * progressivePlan(progressiveKan.hyper).map((p) => p.kind) // => ['train', 'prune', 'refine', 'train', 'refine', 'train']
 * ```
 */
export const progressiveKan: ProgressiveKanEstimator = progressiveEstimator()

/** The progressive KAN's estimator type. */
export type ProgressiveKanEstimator = IterativeEstimator<
  ProgressiveKanHyper,
  ProgressiveKanState,
  ProgressiveKanParams,
  ProgressiveKanVerbs,
  ProgressiveKanPath
>

function progressiveEstimator(changes: Partial<ProgressiveKanHyper> = {}): ProgressiveKanEstimator {
  const hyper: ProgressiveKanHyper = {
    ...PROGRESSIVE_KAN_DEFAULTS,
    ...changes,
    basis: completeOption(kanBases, changes.basis ?? PROGRESSIVE_KAN_DEFAULTS.basis),
    base: completeOption(kanBaseFunctions, changes.base ?? PROGRESSIVE_KAN_DEFAULTS.base),
    optimiser: completeOption(optimisers, changes.optimiser ?? PROGRESSIVE_KAN_DEFAULTS.optimiser),
  }
  const base = iterative<
    ProgressiveKanHyper,
    ProgressiveKanState,
    ProgressiveKanParams,
    ProgressiveKanVerbs,
    ProgressiveKanPath
  >(
    {
      name: 'progressive-kan',
      kind: 'regressor',
      defaults: PROGRESSIVE_KAN_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          stepsPerStage: int(1, 10_000, { default: h.stepsPerStage, label: 'steps per stage' }),
          degree: int(1, 7, { default: h.degree, label: 'spline degree' }),
          basis: kanBases.space,
          base: kanBaseFunctions.space,
          optimiser: optimisers.space,
          learningRate: real(1e-4, 1, { log: true, default: h.learningRate, label: 'learning rate' }),
          lambdaL1: real(0, 10, { default: h.lambdaL1, label: 'edge L1 weight' }),
          lambdaEntropy: real(0, 10, { default: h.lambdaEntropy, label: 'edge entropy weight' }),
        }),
      algorithm: (data, h) => progressiveAlgorithm(data, h),
      params: (s, data, h) => {
        const reference = referenceRows(asTensor(data.x) as Value, h.regRows)
        const flat = toFlat(reference)
        const d = reference.shape[1] ?? 1
        return {
          spec: s.spec as unknown as Json,
          params: stateToJson(s.train.params),
          buffers: stateToJson(s.train.buffers),
          reference: Array.from({ length: reference.shape[0] ?? 0 }, (_, r) =>
            Array.from(flat.slice(r * d, (r + 1) * d)),
          ),
        }
      },
      verbs: (p, h) => progressiveVerbs(p, h),
      maxSteps: (h) =>
        totalSteps(progressivePlan(h)) +
        progressivePlan(h).reduce((n, p) => n + (p.kind === 'train' ? p.steps - 1 : 0), 0),
      contract: (data, h): Contract<ProgressiveKanState, ProgressiveKanPath> => {
        const reference = referenceRows(asTensor(data.x) as Value, h.regRows)
        const plan = progressivePlan(h)
        const networkOf = (s: ProgressiveKanState): { kan: KAN; state: KanState } => ({
          kan: new KAN(s.spec.widths, { ...structureOf(s.spec), key: makeKey(0) }),
          state: { params: s.train.params, buffers: s.train.buffers },
        })
        const ev = (what: string, context: Record<string, string>): EventSource =>
          event({ context: { t: `the step ${what}`, ...context } })
        return {
          'kan/curves': derived<ProgressiveKanState>({
            description:
              'every edge’s function on a grid over its input range, with its magnitude and whether it is on',
            cost: 'dear',
            compute: (s) => {
              const { kan, state } = networkOf(s)
              return kanCurves(kan, state, reference, h.curvePoints) as unknown as Json
            },
          }),
          'kan/grids': derived<ProgressiveKanState>({
            description: 'each layer’s input ranges and the interior knots of its spline edges',
            compute: (s) => {
              const { kan, state } = networkOf(s)
              return kanGrids(kan, state) as unknown as Json
            },
          }),
          'kan/sparsity': derived<ProgressiveKanState>({
            description: 'the share of switched-on edges whose mean magnitude is negligible',
            compute: (s) => {
              const { kan, state } = networkOf(s)
              return edgeSparsity(kan, state, reference, h.sparsityThreshold).fraction
            },
          }),
          'kan/stage': derived<ProgressiveKanState>({
            description: 'where the procedure is: the phase, the widths, the grid and the structural changes so far',
            compute: (s) =>
              ({
                phase: plan[s.phase]?.kind ?? 'done',
                inPhase: s.inPhase,
                stage: s.stage,
                widths: [...s.spec.widths],
                grid: s.spec.grid,
                history: s.history,
              }) as unknown as Json,
          }),
          'kan/grid-update': ev('a layer’s grid was moved to the data', { drift: 'rms change of the output' }),
          'kan/prune': ev('the network was pruned', {
            widths: 'the widths after',
            kept: 'the surviving nodes per layer',
          }),
          'kan/refine': ev('the grid was refined', { grid: 'the new grid size', drift: 'rms change of the output' }),
        } as unknown as Contract<ProgressiveKanState, ProgressiveKanPath>
      },
      logs: {
        'solver/loss': Log(),
        'solver/parameters': Log(),
        'solver/activeEdges': Log(),
        'solver/grid': Log(),
        'kan/stage': Log({ every: 10 }),
        'kan/curves': Log({ every: 10 }),
      },
    },
    hyper,
  )
  return { ...base, with: (more) => progressiveEstimator({ ...hyper, ...more }) }
}
