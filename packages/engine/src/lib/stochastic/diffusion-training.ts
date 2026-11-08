/**
 * Training a diffusion model on low-dimensional data: a small network over the noisy sample and a Fourier encoding of
 * the time (`scoreNetwork`, built on `nn`'s modules), `networkModel` to hand its parameters to a sampler, and
 * `diffusionTrainer`, the training loop as an `Algorithm` (each step draws a minibatch, a time and noise per sample,
 * evaluates the weighted loss and takes an optimiser step), so a toy mixture can be learned and sampled end to end.
 */

import { type Module, mlp, SiLU, type TrainState, trainState, trainStep } from '../nn/index.ts'
import type { GradientTransformation } from '../optim/index.ts'
import { foldIn, type Key, randint } from '../random/index.ts'
import {
  concat,
  item,
  mean,
  promoteFloat,
  reshape,
  type TensorLike,
  take,
  tensor,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  Log,
  type LogSpec,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import {
  type DiffusionModel,
  type DiffusionPrediction,
  type DiffusionWeighting,
  diffusionBatch,
  type NoiseSchedule,
  weightedDiffusionLoss,
} from './diffusion-models.ts'

/** A network for low-dimensional diffusion: an MLP over [x_t, time features]. */
export interface ScoreNetwork {
  readonly module: Module<Value, Value>
  readonly dim: number
  /** The encoding of a time: t and sin and cos of 2 pi k t for k = 1..frequencies. */
  encode(t: number): number[]
  /** The network's input for a batch at one time or at a time per sample. */
  input(xt: TensorLike, t: number | readonly number[]): Value
}

/**
 * A score network for `dim`-dimensional data: an MLP of SiLU layers over the sample concatenated with a Fourier
 * encoding of the time, with outputs the size of the sample (whatever the prediction is).
 *
 * @param options - dim, hidden widths (default [64, 64]), the number of time frequencies (default 6), the encoding of
 *   a time (default Fourier of t) and the initialisation key.
 * @returns The network.
 * @concept diffusion-models
 * @example
 * ```ts
 * import { scoreNetwork } from 'omai-engine/stochastic'
 * scoreNetwork({ dim: 2, hidden: [8] }).module.parameterCount() // => 146
 * ```
 */
export function scoreNetwork(options: {
  readonly dim: number
  readonly hidden?: readonly number[]
  readonly frequencies?: number
  readonly encode?: (t: number) => number[]
  readonly key?: Key
}): ScoreNetwork {
  const frequencies = options.frequencies ?? 6
  const encode =
    options.encode ??
    ((t: number): number[] => [
      t,
      ...Array.from({ length: frequencies }, (_, k) => Math.sin(2 * Math.PI * (k + 1) * t)),
      ...Array.from({ length: frequencies }, (_, k) => Math.cos(2 * Math.PI * (k + 1) * t)),
    ])
  const features = encode(0).length
  const module = mlp([options.dim + features, ...(options.hidden ?? [64, 64]), options.dim], {
    activation: () => new SiLU(),
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  return {
    module: module as unknown as Module<Value, Value>,
    dim: options.dim,
    encode,
    input: (xt, t) => {
      const [x] = promoteFloat(xt) as [Value]
      const n = x.shape[0] ?? 1
      const rows = typeof t === 'number' ? new Array<number[]>(n).fill(encode(t)) : t.map((u) => encode(u))
      return concat([x, tensor(rows)], 1)
    },
  }
}

/**
 * A network with trained parameters as a model for a sampler.
 *
 * @param network - The network.
 * @param params - Its parameters (a `TrainState`'s).
 * @returns The model: the network's output for a batch at a time.
 * @example
 * ```ts
 * import { networkModel, scoreNetwork } from 'omai-engine/stochastic'
 * import { tensor } from 'omai-engine/tensor'
 * const net = scoreNetwork({ dim: 2, hidden: [4] })
 * networkModel(net, net.module.params)(tensor([[0, 0]]), 0.5).shape // => [1, 2]
 * ```
 */
export function networkModel(network: ScoreNetwork, params: Module['params']): DiffusionModel {
  return (xt, t) => network.module.apply(params, network.input(xt, t), { train: false })
}

/** The paths the trainer logs. */
export type DiffusionTrainPath = 'diffusion/loss' | 'diffusion/gradNorm' | 'diffusion/noiseLevel'

/** The contract of the trainer. */
export const diffusionTrainContract: Contract<TrainState, DiffusionTrainPath> = defineContract({
  'diffusion/loss': scalar({ description: 'the weighted denoising loss of the minibatch', direction: 'lower' }),
  'diffusion/gradNorm': scalar({ description: 'the norm of the gradient of the minibatch loss' }),
  'diffusion/noiseLevel': scalar({ description: 'the mean noise scale sigma(t) of the minibatch' }),
})

/** What the trainer's paths keep by default. */
export const diffusionTrainDefaults: Readonly<Record<DiffusionTrainPath, LogSpec>> = {
  'diffusion/loss': Log(),
  'diffusion/gradNorm': Log(),
  'diffusion/noiseLevel': Log(),
}

/** Options of {@link diffusionTrainer}. */
export interface DiffusionTrainerOptions {
  readonly schedule: NoiseSchedule
  /** The training data `[M, D]`. */
  readonly data: TensorLike
  readonly network: ScoreNetwork
  readonly optimizer: GradientTransformation
  /** Minibatch size (default 128). */
  readonly batchSize?: number
  /** The number of steps. */
  readonly steps: number
  /** What the network predicts (default 'epsilon'). */
  readonly prediction?: DiffusionPrediction
  /** The weighting across noise levels (default 'simple'). */
  readonly weighting?: DiffusionWeighting
  /** Draw noise levels log-normally as EDM does (for a schedule whose time is sigma). */
  readonly logNormal?: boolean
}

/**
 * Training a diffusion network as an {@link Algorithm}: each step draws a minibatch of the data, a time per sample and
 * the noise, forms x_t, evaluates the weighted loss ({@link weightedDiffusionLoss}) and takes an optimiser step. The state
 * is nn's `TrainState`, so it serialises, and the parameters at any step feed {@link networkModel}.
 *
 * @param options - Schedule, data, network, optimiser, batch size, steps, prediction and weighting.
 * @returns The algorithm; its paths are {@link DiffusionTrainPath}.
 * @concept diffusion-models
 * @example
 * ```ts
 * import { diffusionTrainer, linearSchedule, scoreNetwork } from 'omai-engine/stochastic'
 * import { adam } from 'omai-engine/optim'
 * const network = scoreNetwork({ dim: 1, hidden: [4] })
 * diffusionTrainer({ schedule: linearSchedule(), data: [[0], [1]], network, optimizer: adam({ learningRate: 0.01 }), steps: 3 }).name // => 'diffusion-train'
 * ```
 */
export function diffusionTrainer(options: DiffusionTrainerOptions): Algorithm<TrainState, DiffusionTrainPath> {
  const [data] = promoteFloat(options.data) as [Value]
  const rows = data.shape[0] ?? 1
  const batchSize = options.batchSize ?? 128
  const prediction = options.prediction ?? 'epsilon'
  const weighting = options.weighting ?? 'simple'
  return {
    name: 'diffusion-train',
    config: { steps: options.steps, batchSize, prediction, weighting, schedule: options.schedule.name },
    init: (key) => trainState(options.network.module, options.optimizer, key),
    step: (state, ctx) => {
      const indices = randint(foldIn(ctx.key, 0), [batchSize], 0, rows)
      const batch = diffusionBatch(options.schedule, take(data, indices, 0), foldIn(ctx.key, 1), {
        ...(options.logNormal === undefined ? {} : { logNormal: options.logNormal }),
      })
      const next = trainStep(
        options.network.module,
        state,
        options.network.input(batch.xt, batch.times),
        (out) =>
          weightedDiffusionLoss(out, batch.x0, batch.noise, batch.times, options.schedule, { prediction, weighting }),
        options.optimizer,
      )
      ctx.log('diffusion/loss', next.loss)
      ctx.log('diffusion/gradNorm', next.gradNorm)
      ctx.log(
        'diffusion/noiseLevel',
        item(mean(reshape(tensor(batch.times.map((t) => options.schedule.sigma(t))), [-1]))),
      )
      return next
    },
    done: (state) => (state.step >= options.steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TrainState,
  }
}
