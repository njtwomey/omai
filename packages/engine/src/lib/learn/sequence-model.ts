/**
 * Sequence models as trainers: a language model or sequence-to-sequence predictor over token sequences, whose layers
 * are the mixers of `nn/blocks` (Mamba, state-space layers, linear attention, RetNet, RWKV, softmax attention, ...)
 * chosen from a table, trained on the trainer base with the optimiser as a slot from `optimisers`. The data is token
 * rows `x` `[n, L]` and targets `y` `[n, L]` with 0 at the positions that carry no target (the padding id), so one
 * trainer serves next-token prediction, copying, recall and induction tasks. Beyond the base's paths the tape holds,
 * for any step, the eigenvalues of the layers' state transitions, the learned step sizes and the norms of the recurrent
 * states on probe rows.
 */

import type { Json } from '../base/index.ts'
import { crossEntropy } from '../losses/index.ts'
import { SequenceModel, sequenceMixers } from '../nn/blocks/index.ts'
import type { TrainState } from '../nn/index.ts'
import type { Params } from '../nn/module/module.ts'
import { optimisers } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import { completeOption, dict, int, real, type SpaceValue } from '../space/index.ts'
import { asTensor, sliceAxis, type TensorLike, toFlat, treePaths, type Value } from '../tensor/index.ts'
import { type Contract, derived, tapContext } from '../trace/index.ts'
import { defineTrainer, type TrainerEstimator } from './bases/trainer.ts'
import type { LearnData } from './estimator.ts'

/** The hyperparameters of the sequence-model trainer. */
export interface SequenceModelHyper {
  /** The model width and the number of mixer blocks. */
  readonly dim: number
  readonly layers: number
  /** The mixer slot: a configuration of `sequenceMixers.space` (`{ kind: 'mamba', stateSize: 16 }`). */
  readonly mixer: SpaceValue
  /** A gated feed-forward sub-block after each mixer (default none, as Mamba). */
  readonly ffn: 'none' | 'gated'
  /** The optimiser slot: a configuration of `optimisers.space` (`{ kind: 'adamw' }`). */
  readonly optimiser: SpaceValue
  readonly learningRate: number
  /** A configuration of `schedules` shaping the learning rate (default constant). */
  readonly schedule?: SpaceValue
  /** Minibatches in all, and their size. */
  readonly steps: number
  readonly batchSize: number
  /** Clip the gradient's global norm (default 1). */
  readonly clipNorm: number
  /** Hold out this share of the rows to validate on (logged as `eval/loss` and `eval/accuracy`). */
  readonly validationFraction?: number
  /** The vocabulary size (default: the largest token id in the data plus one). */
  readonly vocabSize?: number
}

const sequenceModelDefaults: SequenceModelHyper = {
  dim: 16,
  layers: 1,
  mixer: { kind: 'mamba' },
  ffn: 'none',
  optimiser: { kind: 'adamw' },
  learningRate: 0.01,
  steps: 100,
  batchSize: 32,
  clipNorm: 1,
}

/** The paths the sequence-model trainer adds to the trainer base's. */
export type SequenceModelPath = 'mixer/spectrum' | 'mixer/step' | 'mixer/state-norm'

const vocabOf = (data: LearnData, h: SequenceModelHyper): number => {
  if (h.vocabSize !== undefined) return h.vocabSize
  const ids = [...toFlat(asTensor(data.x)), ...(data.y === undefined ? [] : toFlat(asTensor(data.y)))]
  return ids.reduce((m, v) => Math.max(m, v), 0) + 1
}

const build = (data: LearnData, h: SequenceModelHyper, key: Key): SequenceModel =>
  new SequenceModel(h.dim, h.layers, (dim, k) => sequenceMixers.make(h.mixer)(dim, k), {
    inputSize: vocabOf(data, h),
    ffn: h.ffn,
    key,
  })

/**
 * Token accuracy over the positions that carry a target (target id other than 0): the share of those positions where
 * the highest logit is the target.
 *
 * @param prediction - Logits `[n, L, V]`.
 * @param target - Target ids `[n, L]`, 0 where there is none.
 * @returns The accuracy in [0, 1] (0 when no position carries a target).
 * @example
 * ```ts
 * import { tokenAccuracy } from 'omai-engine/learn'
 * tokenAccuracy([[[0, 1], [1, 0]]], [[0, 1]]) // => 0
 * ```
 */
export function tokenAccuracy(prediction: TensorLike, target: TensorLike): number {
  const scores = asTensor(prediction)
  const v = scores.shape[scores.shape.length - 1] ?? 1
  const logits = toFlat(scores)
  const y = toFlat(asTensor(target))
  let right = 0
  let total = 0
  for (const [i, t] of y.entries()) {
    if (t === 0) continue
    total++
    let best = 0
    for (let j = 1; j < v; j++) if ((logits[i * v + j] ?? 0) > (logits[i * v + best] ?? 0)) best = j
    if (best === t) right++
  }
  return total === 0 ? 0 : right / total
}

const blockParams = (params: Params, i: number): Params => ((params['layers'] as Params)[String(i)] ?? {}) as Params
const mixerParams = (params: Params, i: number): Params => (blockParams(params, i)['mixer'] ?? {}) as Params

/** The squared Frobenius norm of every tensor in a state tree. */
const stateNormSquared = (state: object): number =>
  Object.values(treePaths(state)).reduce((s, v) => s + toFlat(v).reduce((a, x) => a + x * x, 0), 0)

/**
 * The sequence-model trainer: next-token (or aligned-target) prediction with a stack of mixer blocks. `fit(data)`
 * trains with the mixer, depth, width and optimiser the hyperparameters name; `with(...)` changes any of them (the
 * mixer slot included) and keeps the rest.
 *
 * @concept sequence-models
 * @example
 * ```ts
 * import { sequenceModelTrainer } from 'omai-engine/learn'
 * sequenceModelTrainer().hyper.layers // => 1
 * ```
 */
export function sequenceModelTrainer(
  changes: Partial<SequenceModelHyper> = {},
): TrainerEstimator<SequenceModelHyper, SequenceModelPath> {
  const hyper: SequenceModelHyper = {
    ...sequenceModelDefaults,
    ...changes,
    mixer: completeOption(sequenceMixers, changes.mixer ?? sequenceModelDefaults.mixer),
    optimiser: completeOption(optimisers, changes.optimiser ?? sequenceModelDefaults.optimiser),
  }
  const estimator = defineTrainer<SequenceModelHyper, SequenceModelPath>(
    {
      name: 'sequence-model',
      kind: 'other',
      defaults: sequenceModelDefaults,
      hyperSpace: (h) =>
        dict({
          dim: int(2, 1024, { log: true, default: h.dim, label: 'model width' }),
          layers: int(1, 48, { default: h.layers, label: 'blocks' }),
          mixer: sequenceMixers.space,
          optimiser: optimisers.space,
          learningRate: real(1e-5, 1, { log: true, default: h.learningRate, label: 'learning rate' }),
        }),
      model: (data, h, key) => build(data, h, key),
      objective: () => (prediction, target) => crossEntropy(prediction, target, { ignoreIndex: 0 }),
      optimiser: (lr, h) => optimisers.make(h.optimiser)(lr),
      settings: (h) => ({
        learningRate: h.learningRate,
        ...(h.schedule === undefined ? {} : { schedule: h.schedule }),
        steps: h.steps,
        batchSize: h.batchSize,
        clipNorm: h.clipNorm,
        ...(h.validationFraction === undefined ? {} : { validationFraction: h.validationFraction }),
      }),
      metrics: () => ({ accuracy: tokenAccuracy }),
      contract: (data, h): Contract<TrainState, SequenceModelPath> => {
        const model = build(data, h, makeKey(0))
        const probe = sliceAxis(asTensor(data.x), 0, 0, Math.min(16, asTensor(data.x).shape[0] ?? 1))
        const rows = (): Value => asTensor(probe, 'int32')
        return {
          'mixer/spectrum': derived<TrainState>({
            description:
              'the eigenvalues of each layer’s state transition (continuous or discrete time), where it has one',
            compute: (s) =>
              Object.fromEntries(
                model.blockList.map((b, i) => [
                  `layers.${i}`,
                  b.mixer.transitionSpectrum(mixerParams(s.params, i)) ?? null,
                ]),
              ) as Json,
          }),
          'mixer/step': derived<TrainState>({
            cost: 'dear',
            description: 'the mean step size (Delta) each selective layer uses on the probe rows',
            compute: (s) => {
              const taps = tapContext()
              model.run(s.params, rows(), undefined, { train: false, taps })
              return Object.fromEntries(
                [...taps.entries].filter(([k]) => k.endsWith('.dt')).map(([k, v]) => [k, mean(toFlat(v.value))]),
              ) as Json
            },
          }),
          'mixer/state-norm': derived<TrainState>({
            cost: 'dear',
            description: 'the norm of each layer’s recurrent state after the probe rows',
            compute: (s) => {
              const r = model.run(s.params, rows(), undefined, { train: false })
              return Object.fromEntries(
                Object.entries(r.state).map(([i, st]) => [`layers.${i}`, Math.sqrt(stateNormSquared(st))]),
              ) as Json
            },
          }),
        }
      },
    },
    hyper,
  )
  return { ...estimator, with: (more) => sequenceModelTrainer({ ...hyper, ...more }) }
}

const mean = (xs: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < xs.length; i++) s += xs[i] ?? 0
  return xs.length === 0 ? 0 : s / xs.length
}
