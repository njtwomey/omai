/**
 * Reading a transformer layer by layer: the logit lens (nostalgebraist 2020) pushes the residual stream after every
 * block through the final normalisation and the unembedding to see what the model would predict if it stopped there,
 * and the tuned lens (Belrose et al. 2023) first passes it through an affine translator per layer, trained to match the
 * final layer's distribution by KL divergence, which repairs the lens where the early stream is not yet in the
 * unembedding's basis. The tuned lens is an `Algorithm` (one Adam step per step, the per-layer divergences logged), and
 * both lenses return the distributions, entropies and, for given targets, the rank and probability of the right token
 * by layer and position.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import type { Json } from '../base/index.ts'
import { OmaiError } from '../base/index.ts'
import { concreteTree } from '../nn/index.ts'
import { adam, applyUpdates, type LearningRate } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import {
  add,
  argmax,
  asTensor,
  div,
  exp,
  fromStorage,
  logSoftmax,
  matmul,
  mul,
  reshape,
  softmax,
  stack,
  sub,
  sum,
  type Tensor,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import type { ResidualModel, TokenRows } from './residual.ts'

/** The tuned lens's translators: an affine map `r + r W_l + b_l` for each block before the last. */
export interface TunedLens {
  readonly w: readonly Value[]
  readonly b: readonly Value[]
}

/** Options of {@link logitLens}. */
export interface LensOptions {
  /** Translators of a trained tuned lens; without them this is the plain logit lens. */
  readonly translators?: TunedLens
  /** The token that should come next at each position (the prompt shifted by one, or the model's own answer), for ranks. */
  readonly targets?: readonly number[]
}

/** What a lens reads off the residual stream, for one prompt: `numLayers + 1` rows (the embeddings, then after each block). */
export interface LensResult {
  /** The logits `[L + 1, T, V]` at every depth. */
  readonly logits: Tensor
  /** The next-token distributions `[L + 1, T, V]`. */
  readonly probabilities: Tensor
  /** The entropy of each distribution in nats `[L + 1, T]`. */
  readonly entropy: Tensor
  /** The most likely token at each depth and position `[L + 1, T]`. */
  readonly prediction: Tensor
  /** The rank of the target (0 is the top choice) `[L + 1, T]`, when targets were given. */
  readonly targetRank?: Tensor
  /** The probability of the target `[L + 1, T]`, when targets were given. */
  readonly targetProbability?: Tensor
}

/** The residual stream a translator acts on, translated: `r + r W + b`. */
function translate(r: Value, lens: TunedLens, layer: number): Value {
  const w = lens.w[layer]
  const b = lens.b[layer]
  if (w === undefined || b === undefined) return r
  return add(add(r, matmul(r, w)), b)
}

/** Logits at every depth `[L + 1, B, T, V]` stacked, the last depth never translated. */
function lensLogits(model: ResidualModel, residual: readonly Value[], lens: TunedLens | undefined): Value[] {
  const last = residual.length - 1
  return residual.map((r, l) => model.unembed(lens !== undefined && l < last ? translate(r, lens, l) : r))
}

/**
 * The logit lens (and, with translators, the tuned lens) of one prompt: the next-token distribution implied by the
 * residual stream after every block. At the last depth it is the model's own output, exactly.
 *
 * @param model - The opened model ({@link residualModel}).
 * @param tokens - The prompt.
 * @param options - Translators of a tuned lens and the targets to rank.
 * @returns Logits, probabilities, entropy and prediction by depth and position; target rank and probability if given.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { logitLens, residualModel } from 'omai-engine/explain'
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const m = residualModel(new blocks.TransformerLM(7, 8, 2, 2, { key: key(0) }))
 * logitLens(m, [1, 2, 3]).logits.shape // => [3, 3, 7]
 * ```
 */
export function logitLens(model: ResidualModel, tokens: readonly number[], options: LensOptions = {}): LensResult {
  const run = model.run([tokens])
  const logits = lensLogits(model, run.residual, options.translators)
  const depth = logits.length
  const t = tokens.length
  const v = model.vocabSize
  const all = stack(
    logits.map((l) => reshape(l, [t, v])),
    0,
  ) as Tensor
  const probabilities = softmax(all, -1) as Tensor
  const entropy = sub(0, sum(mul(probabilities, logSoftmax(all, -1)), -1)) as Tensor
  const prediction = argmax(all, -1) as Tensor
  const out: {
    logits: Tensor
    probabilities: Tensor
    entropy: Tensor
    prediction: Tensor
    targetRank?: Tensor
    targetProbability?: Tensor
  } = { logits: all, probabilities, entropy, prediction }
  if (options.targets !== undefined) {
    if (options.targets.length !== t) throw new OmaiError('shape', 'logitLens: one target per position')
    const rank = new Float64Array(depth * t)
    const prob = new Float64Array(depth * t)
    const flat = toFlat(all)
    const pflat = toFlat(probabilities)
    for (let l = 0; l < depth; l++)
      for (let p = 0; p < t; p++) {
        const target = options.targets[p] as number
        const base = (l * t + p) * v
        const own = flat[base + target] as number
        let better = 0
        for (let k = 0; k < v; k++) if ((flat[base + k] as number) > own) better++
        rank[l * t + p] = better
        prob[l * t + p] = pflat[base + target] as number
      }
    out.targetRank = fromStorage(rank, [depth, t], 'float64')
    out.targetProbability = fromStorage(prob, [depth, t], 'float64')
  }
  return out
}

/**
 * The mean KL divergence from the model's final distribution to each depth's lens distribution over a corpus
 * (`KL(p_final || p_l)`, in nats, averaged over sequences and positions): how well each depth already predicts what the
 * model will say. Zero at the last depth.
 *
 * @param model - The opened model.
 * @param corpus - Equal-length token rows.
 * @param translators - A tuned lens, or none for the logit lens.
 * @returns One divergence per depth, `numLayers + 1` of them.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { lensDivergence, residualModel } from 'omai-engine/explain'
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const m = residualModel(new blocks.TransformerLM(7, 8, 2, 2, { key: key(0) }))
 * lensDivergence(m, [[1, 2, 3], [3, 2, 1]]).at(-1) // => 0
 * ```
 */
export function lensDivergence(model: ResidualModel, corpus: TokenRows, translators?: TunedLens): number[] {
  const run = model.run(corpus)
  const logp = logSoftmax(run.logits, -1)
  const p = exp(logp)
  const n = (corpus.length * (corpus[0]?.length ?? 1)) as number
  return lensLogits(model, run.residual, translators).map((l) => {
    const kl = div(sum(mul(p, sub(logp, logSoftmax(l, -1)))), n)
    return Math.max(0, toFlat(asTensor(kl))[0] as number)
  })
}

/** The state of {@link tunedLensAlgorithm}. */
export interface TunedLensState {
  readonly lens: TunedLens
  readonly optimiser: unknown
  /** The summed divergence over the trained depths. */
  readonly loss: number
  /** The divergence at each depth before this step's update. */
  readonly layerDivergence: readonly number[]
  readonly step: number
}

/** The paths of {@link tunedLensAlgorithm}. */
export type TunedLensPath = 'lens/loss' | 'lens/layerDivergence' | 'lens/translators'

/** What a tuned-lens run logs. */
export const tunedLensContract: Contract<TunedLensState, TunedLensPath> = defineContract({
  'lens/loss': scalar({ description: 'the summed KL from the final distribution over the depths', direction: 'lower' }),
  'lens/layerDivergence': tensorSource({
    role: 'state',
    shape: ['depth'],
    description: 'the KL from the final distribution at each depth',
  }),
  'lens/translators': tensorSource({
    role: 'param',
    shape: ['depth', 'dim', 'dim'],
    description: 'the translators of the affine maps',
  }),
})

/** Options of {@link tunedLensAlgorithm}. */
export interface TunedLensOptions {
  /** Adam's learning rate (default 0.02). */
  readonly learningRate?: LearningRate
  /** The number of steps before the algorithm reports completion (default 200). */
  readonly steps?: number
}

/**
 * Train the tuned lens as an algorithm: the translators start at the identity map (so the first step is the plain
 * logit lens), and each step takes one full-batch Adam step on the summed KL divergence from the model's final
 * distribution to each depth's lens distribution. Logs the loss and the divergence per depth.
 *
 * @param model - The opened model.
 * @param corpus - Equal-length token rows to train on.
 * @param options - Learning rate and the number of steps.
 * @returns The algorithm.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { lensDivergence, residualModel, tunedLensAlgorithm } from 'omai-engine/explain'
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const m = residualModel(new blocks.TransformerLM(7, 8, 2, 2, { key: key(0) }))
 * const corpus = [[1, 2, 3, 4], [4, 3, 2, 1], [2, 2, 5, 1]]
 * const lens = runAlgorithm(tunedLensAlgorithm(m, corpus, { steps: 60 }), { maxSteps: 60 }).state.lens
 * lensDivergence(m, corpus, lens)[0] < lensDivergence(m, corpus)[0] // => true
 * ```
 */
export function tunedLensAlgorithm(
  model: ResidualModel,
  corpus: TokenRows,
  options: TunedLensOptions = {},
): Algorithm<TunedLensState, TunedLensPath> {
  const { dim, numLayers } = model
  const steps = options.steps ?? 200
  const optimiser = adam({ learningRate: options.learningRate ?? 0.02 })
  const run = model.run(corpus)
  const logp = logSoftmax(run.logits, -1)
  const p = exp(logp)
  const count = corpus.length * (corpus[0]?.length ?? 1)
  const residual = run.residual.slice(0, numLayers)
  const loss = (lens: TunedLens): readonly [Value, readonly Value[]] => {
    const kls = residual.map((r, l) => {
      const q = logSoftmax(model.unembed(translate(r, lens, l)), -1)
      return div(sum(mul(p, sub(logp, q))), count)
    })
    const total = kls.reduce((a, b) => add(a, b), 0 as unknown as Value)
    return [total, kls]
  }
  const vg = valueAndGrad<TunedLens>(loss, { hasAux: true })
  return {
    name: 'tunedLens',
    config: { dim, numLayers, steps, corpus: corpus.length } as Json,
    init: (_key: Key) => {
      const lens: TunedLens = {
        w: Array.from({ length: numLayers }, () => zeros([dim, dim]) as Value),
        b: Array.from({ length: numLayers }, () => zeros([dim]) as Value),
      }
      return { lens, optimiser: optimiser.init(lens), loss: Number.NaN, layerDivergence: [], step: 0 }
    },
    step: (s, ctx) => {
      const { value, grad, aux } = vg(s.lens)
      const { updates, state } = optimiser.update(grad, s.optimiser as unknown[], s.lens)
      const lens = applyUpdates(s.lens, updates) as TunedLens
      const total = toFlat(asTensor(value))[0] as number
      const layerDivergence = concreteTree(aux as readonly Value[]).map((k) => toFlat(asTensor(k))[0] as number)
      ctx.log('lens/loss', total)
      ctx.log('lens/layerDivergence', layerDivergence)
      if (ctx.wants('lens/translators')) ctx.log('lens/translators', stack(lens.w, 0) as Tensor)
      return { lens, optimiser: state, loss: total, layerDivergence, step: s.step + 1 }
    },
    done: (s) => (s.step >= steps ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { loss: s.loss },
      tensors: { layerDivergence: { role: 'state', value: s.layerDivergence } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TunedLensState,
  }
}
