/**
 * Fitting a neural field to samples of a signal as an `Algorithm`: each step draws a minibatch of coordinates, evaluates
 * the field, takes the loss against the signal's values (squared or absolute error) and, for signed distance fields, an
 * eikonal term that keeps the gradient of the field at unit length, differentiates with the library's autodiff and
 * applies the optimiser, which is a slot (any gradient transformation, normally from the `optimisers` table). The
 * contract logs the loss, its parts, the peak signal-to-noise ratio and the gradient norm, and holds two probes: the
 * loss on the whole data and the field decoded on a grid, which a player draws sharpening as the fit goes on.
 */

import { gradAndAux, grad as gradOf } from '../autodiff/index.ts'
import type { Json } from '../base/index.ts'
import { mae, mse } from '../losses/index.ts'
import type { Module, Params } from '../nn/index.ts'
import { applyUpdates, type GradientTransformation, globalNorm } from '../optim/index.ts'
import { foldIn, randint, uniform } from '../random/index.ts'
import {
  add,
  asTensor,
  concat,
  item,
  mean,
  mul,
  primalOf,
  reshape,
  sqrt,
  square,
  sub,
  sum,
  take,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  type RunOptions,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { type DecodeOptions, decodeField, psnrFromMse } from './quality.ts'

/** Samples of a signal: coordinates `[n, d]` and values `[n]` or `[n, k]` (a regression dataset has this shape). */
export interface FieldData {
  readonly x: Value
  readonly y: Value
}

/** The paths of {@link fitFieldAlgorithm}. */
export type FieldPath =
  | 'field/loss'
  | 'field/mse'
  | 'field/psnr'
  | 'field/eikonal'
  | 'field/gradNorm'
  | 'field/fullMse'
  | 'field/render'

/** The state of a field fit: parameters and optimiser state, the step, and what the last minibatch measured. */
export interface FieldState {
  readonly params: Params
  readonly optState: unknown
  readonly step: number
  /** The last minibatch's total loss, its data error (mean squared error) and eikonal term, and the gradient norm. */
  readonly loss: number
  readonly mse: number
  readonly eikonal: number
  readonly gradNorm: number
}

/** Options of {@link fitFieldAlgorithm}. */
export interface FitFieldOptions {
  /** The optimiser, a gradient transformation (the slot; normally `optimisers.make(config)(learningRate)`). */
  readonly optimizer: GradientTransformation
  /** Rows per minibatch, drawn with replacement from the step's key (default all rows, in order). */
  readonly batchSize?: number
  /** The data loss: squared (default) or absolute error. */
  readonly loss?: 'mse' | 'l1'
  /**
   * The eikonal term `E(|grad f| - 1)^2` for a distance field (one output): its weight and the number of extra points per
   * step drawn uniformly in `bounds` (default 0: only the minibatch's own points).
   */
  readonly eikonal?: { readonly weight: number; readonly extraPoints?: number }
  /** The coordinate box of the extra eikonal points and of the decoding probe (default `[-1, 1]` on every axis). */
  readonly bounds?: readonly [number, number]
  /** The peak-to-peak range of the signal for the peak signal-to-noise ratio (default 1). */
  readonly dataRange?: number
  /** The decoding probe's resolution and, for a 3-D field, the plane (default 32 cells per side). */
  readonly decode?: Pick<DecodeOptions, 'resolution' | 'slice'>
}

const rowsOf = (v: Value): number => v.shape[0] ?? 0
const asColumns = (y: Value): Value => (y.shape.length === 1 ? reshape(y, [-1, 1]) : y)

/**
 * The contract of a field fit: the minibatch's total loss, data error, eikonal term, peak signal-to-noise ratio and
 * gradient norm per step, and two derived probes, `field/fullMse` (the error on every row of the data) and
 * `field/render` (the field on a grid, a `FieldGrid`).
 *
 * @param model - The field.
 * @param data - The signal's samples.
 * @param options - The fit's options (the decoding probe's settings).
 * @returns The contract.
 * @concept log-contract
 * @example
 * ```ts
 * import { fieldContract, neuralField } from 'mdml-engine/fields'
 * import { sgd } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * const f = neuralField({ inputDim: 1, outputDim: 1, width: 4, depth: 1, key: key(0) })
 * Object.keys(fieldContract(f, { x: [[0], [1]] as never, y: [0, 1] as never }, { optimizer: sgd({ learningRate: 0.1 }) })).length // => 7
 * ```
 */
export function fieldContract(
  model: Module,
  data: FieldData,
  options: FitFieldOptions,
): Contract<FieldState, FieldPath> {
  const x = asTensor(data.x) as Value
  const y = asColumns(asTensor(data.y) as Value)
  const dim = x.shape[1] ?? 1
  return defineContract({
    'field/loss': scalar({ description: 'the minibatch loss (the data loss plus the weighted eikonal term)' }),
    'field/mse': scalar({ description: 'the minibatch mean squared error against the signal' }),
    'field/psnr': scalar({ description: 'the peak signal-to-noise ratio of the minibatch, in decibels' }),
    'field/eikonal': scalar({ description: 'the eikonal term E(|grad f| - 1)^2 of the minibatch (0 when it is off)' }),
    'field/gradNorm': scalar({ description: 'the global norm of the gradient' }),
    'field/fullMse': derived<FieldState>({
      description: 'the mean squared error on every row of the data',
      compute: (s) => item(mse(model.apply(s.params, x, { train: false }), y)),
    }),
    'field/render': derived<FieldState>({
      description: 'the field decoded on a grid of cell centres: a line, an image or a plane of a volume',
      cost: 'dear',
      compute: (s) =>
        decodeField(model, s.params, {
          dim,
          ...(options.bounds === undefined ? {} : { bounds: options.bounds }),
          ...(options.decode ?? {}),
        }),
    }),
  })
}

/**
 * Fit a neural field to samples of a signal as an {@link Algorithm}: a step draws a minibatch, differentiates the loss
 * (data error, plus the eikonal term when asked) with respect to the parameters and applies the optimiser. A step
 * draws its batch from the step's key, so a run paused and resumed from a serialised state equals one that never
 * stopped. A non-finite loss ends the run as `diverged`.
 *
 * @param model - The field (its parameters are the starting point; it is not modified).
 * @param data - The signal's samples.
 * @param options - The optimiser slot, batch size, loss, eikonal term and probe settings.
 * @returns The algorithm.
 * @concept neural-fields
 * @example
 * ```ts
 * import { fitFieldAlgorithm, neuralField } from 'mdml-engine/fields'
 * import { adam } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { key } from 'mdml-engine/random'
 * // Fit y = x on four points with a small ReLU network.
 * const f = neuralField({ inputDim: 1, outputDim: 1, width: 8, depth: 1, key: key(0) })
 * const data = { x: [[-1], [-0.3], [0.3], [1]] as never, y: [-1, -0.3, 0.3, 1] as never }
 * const alg = fitFieldAlgorithm(f, data, { optimizer: adam({ learningRate: 0.05 }) })
 * runAlgorithm(alg, { maxSteps: 200 }).state.mse < 0.01 // => true
 * ```
 */
export function fitFieldAlgorithm(
  model: Module,
  data: FieldData,
  options: FitFieldOptions,
): Algorithm<FieldState, FieldPath> {
  const x = asTensor(data.x) as Value
  const y = asColumns(asTensor(data.y) as Value)
  const n = rowsOf(x)
  const dim = x.shape[1] ?? 1
  const batch = Math.min(options.batchSize ?? n, n)
  const lossFn = options.loss === 'l1' ? mae : mse
  const eik = options.eikonal
  const [lo, hi] = options.bounds ?? [-1, 1]
  const range = options.dataRange ?? 1
  const optimizer = options.optimizer
  const inputGradient = (params: Params, points: Value): Value =>
    gradOf((v: Value) => sum(model.apply(params, v, { train: true })))(points)

  return {
    name: 'fit-field',
    config: {
      model: model.kind,
      optimizer: optimizer.name,
      batchSize: batch,
      loss: options.loss ?? 'mse',
      eikonal: eik === undefined ? null : { weight: eik.weight, extraPoints: eik.extraPoints ?? 0 },
    } as Json,
    init: () => ({
      params: model.params,
      optState: optimizer.init(model.params),
      step: 0,
      loss: 0,
      mse: 0,
      eikonal: 0,
      gradNorm: 0,
    }),
    step: (s, ctx) => {
      const rows = batch < n ? randint(foldIn(ctx.key, 0), [batch], 0, n) : undefined
      const xb = rows === undefined ? x : (take(x, rows) as Value)
      const yb = rows === undefined ? y : (take(y, rows) as Value)
      const extra = eik?.extraPoints ?? 0
      const points =
        eik === undefined || extra === 0
          ? xb
          : (concat([xb, uniform(foldIn(ctx.key, 1), [extra, dim], { low: lo, high: hi })], 0) as Value)
      const objective = (p: Params): readonly [Value, { readonly data: Value; readonly eikonal: Value }] => {
        const data = lossFn(model.apply(p, xb, { train: true }), yb)
        if (eik === undefined) return [data, { data, eikonal: asTensor(0) as Value }]
        const g = inputGradient(p, points)
        const norm = sqrt(add(sum(square(g), -1), 1e-12))
        const penalty = mean(square(sub(norm, 1)))
        return [add(data, mul(penalty, eik.weight)), { data, eikonal: penalty }]
      }
      const { grad, aux } = gradAndAux(objective)(s.params)
      const parts = aux as { readonly data: Value; readonly eikonal: Value }
      const r = optimizer.update(grad, s.optState, s.params)
      const dataLoss = item(primalOf(parts.data))
      const eikonal = item(primalOf(parts.eikonal))
      const meanSquared = options.loss === 'l1' ? sqErr(model, s.params, xb, yb) : dataLoss
      const total = dataLoss + (eik === undefined ? 0 : eik.weight * eikonal)
      const gradNorm = item(globalNorm(grad))
      ctx.log('field/loss', total)
      ctx.log('field/mse', meanSquared)
      ctx.log('field/psnr', psnrFromMse(meanSquared, range))
      ctx.log('field/eikonal', eikonal)
      ctx.log('field/gradNorm', gradNorm)
      return {
        params: applyUpdates(s.params, r.updates) as Params,
        optState: r.state,
        step: s.step + 1,
        loss: total,
        mse: meanSquared,
        eikonal,
        gradNorm,
      }
    },
    done: (s) => (Number.isFinite(s.loss) ? undefined : 'diverged'),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FieldState,
  }
}

// The minibatch's mean squared error, for logging when the training loss is another.
function sqErr(model: Module, params: Params, x: Value, y: Value): number {
  return item(mse(model.apply(params, x, { train: false }), y))
}

/** The outcome of {@link fitField}. */
export interface FieldFit {
  /** The trained parameters (also loaded into the model). */
  readonly params: Params
  /** The minibatch losses, one per step. */
  readonly losses: readonly number[]
  readonly steps: number
  readonly status: string
}

/**
 * Fit a neural field for a number of steps and load the trained parameters into the model.
 *
 * @param model - The field (updated in place).
 * @param data - The signal's samples.
 * @param options - As {@link fitFieldAlgorithm}, the step count, the key and the run's tracer and policy.
 * @returns The trained parameters, the per-step losses and the run's status.
 * @concept neural-fields
 * @index function Fit a coordinate network to samples of a signal.
 * @example
 * ```ts
 * import { fitField, neuralField } from 'mdml-engine/fields'
 * import { adam } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * const f = neuralField({ inputDim: 1, outputDim: 1, width: 8, depth: 1, key: key(0) })
 * const data = { x: [[-1], [0], [1]] as never, y: [1, 0, 1] as never }
 * const r = fitField(f, data, { optimizer: adam({ learningRate: 0.05 }), steps: 100 })
 * (r.losses.at(-1) ?? 1) < (r.losses[0] ?? 0) // => true
 * ```
 */
export function fitField(
  model: Module,
  data: FieldData,
  options: FitFieldOptions & { readonly steps: number } & Partial<Omit<RunOptions, 'maxSteps'>>,
): FieldFit {
  const { steps, optimizer, batchSize, loss, eikonal, bounds, dataRange, decode, ...run } = options
  const algorithm = fitFieldAlgorithm(model, data, {
    optimizer,
    ...(batchSize === undefined ? {} : { batchSize }),
    ...(loss === undefined ? {} : { loss }),
    ...(eikonal === undefined ? {} : { eikonal }),
    ...(bounds === undefined ? {} : { bounds }),
    ...(dataRange === undefined ? {} : { dataRange }),
    ...(decode === undefined ? {} : { decode }),
  })
  const losses: number[] = []
  const tracked: Algorithm<FieldState> = {
    ...algorithm,
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      losses.push(next.loss)
      return next
    },
  }
  const result = runAlgorithm(tracked, { ...run, maxSteps: steps })
  model.setParams(result.state.params)
  return { params: result.state.params, losses, steps: result.steps, status: result.status }
}
