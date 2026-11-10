/**
 * The BiLSTM-CRF sequence tagger (Lample et al. 2016, flair's `SequenceTagger`): token ids go through an embedding
 * table, optional dropout, a (bi)directional LSTM and a linear layer to emission scores, and a CRF layer (the
 * `CRF` module of `neural-crf.ts`) scores whole tag sequences, so the loss is the sequence negative log-likelihood and
 * prediction is exact Viterbi, optionally restricted to allowed label transitions. With `crf: false` the same encoder
 * ends in a per-token softmax, the foil that shows what the transition scores add. `taggerAlgorithm` trains it end to
 * end with any optimiser from `optim` as a traced `Algorithm` (minibatches, loss, gradient norm, token accuracy and
 * entity F1 on the tape); `padTagging` pads ragged sequences into a batch and `evaluateTagger` scores predictions.
 */

import { valueAndGrad } from '../../autodiff/index.ts'
import { type Json, OmaiError } from '../../base/index.ts'
import { accuracyScore, entityScores, type TagScheme } from '../../metrics/index.ts'
import {
  type ApplyContext,
  applyGradients,
  assemble,
  dropout,
  Embedding,
  isTraining,
  Linear,
  LSTM,
  Module,
  type Params,
  type TrainState,
  trainState,
} from '../../nn/index.ts'
import { chain, clipByGlobalNorm, type GradientTransformation, globalNorm } from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../random/index.ts'
import {
  asTensor,
  div,
  item,
  logSoftmax,
  mul,
  neg,
  reshape,
  sum,
  take,
  tensor,
  toFlat,
  treeLeaves,
  treePaths,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../../trace/index.ts'
import { CRF, type NeuralCrfAllowed, neuralCrfDecode, neuralCrfNegLogLikelihood } from './neural-crf.ts'

/** A tagged sequence: token ids and one tag id per token. */
export interface TaggedSequence {
  readonly tokens: readonly number[]
  readonly tags: readonly number[]
}

/** A padded batch: `[B][T]` token ids, a mask of real tokens (right-padded sequences) and, for training, the tags. */
export interface TagBatch {
  readonly tokens: readonly (readonly number[])[]
  /** 1 for a token, 0 for padding. */
  readonly mask: readonly (readonly number[])[]
  readonly tags?: readonly (readonly number[])[]
}

/**
 * Pad ragged sequences into a batch: tokens (and tags, when every sequence has them) padded with `padId` (and tag 0) to
 * the longest length, and the mask of real tokens.
 *
 * @param sequences - The sequences; `tags` may be left out for prediction.
 * @param padId - The token id of padding (default 0).
 * @returns The batch.
 * @throws OmaiError with code `value` for an empty batch or an empty sequence, `shape` when tags and tokens differ in
 *   length.
 * @concept sequence-tagging
 * @example
 * ```ts
 * import { padTagging } from 'omai-engine/pgm'
 * padTagging([{ tokens: [3, 4, 5], tags: [0, 1, 1] }, { tokens: [6], tags: [2] }]).mask // => [[1, 1, 1], [1, 0, 0]]
 * ```
 */
export function padTagging(
  sequences: readonly { readonly tokens: readonly number[]; readonly tags?: readonly number[] }[],
  padId = 0,
): TagBatch {
  if (sequences.length === 0) throw new OmaiError('value', 'padTagging: no sequences')
  const T = Math.max(...sequences.map((s) => s.tokens.length))
  if (sequences.some((s) => s.tokens.length === 0)) throw new OmaiError('value', 'padTagging: an empty sequence')
  if (sequences.some((s) => s.tags !== undefined && s.tags.length !== s.tokens.length))
    throw new OmaiError('shape', 'padTagging: tags and tokens must have the same length')
  const pad = <V>(row: readonly V[], fill: V): V[] => [...row, ...new Array<V>(T - row.length).fill(fill)]
  const labelled = sequences.every((s) => s.tags !== undefined)
  return {
    tokens: sequences.map((s) => pad(s.tokens, padId)),
    mask: sequences.map((s) => pad(new Array<number>(s.tokens.length).fill(1), 0)),
    ...(labelled ? { tags: sequences.map((s) => pad(s.tags ?? [], 0)) } : {}),
  }
}

/** Options of {@link SequenceTagger}. */
export interface SequenceTaggerOptions {
  readonly vocabSize: number
  readonly numTags: number
  /** Embedding width (default 16). */
  readonly embeddingDim?: number
  /** Hidden units per direction (default 16). */
  readonly hiddenSize?: number
  readonly numLayers?: number
  /** Read the sentence in both directions (default true). */
  readonly bidirectional?: boolean
  /** Dropout on the embeddings and on the encoder's outputs, in training (default 0). */
  readonly dropout?: number
  /** End in a CRF layer (default true); false ends in a per-token softmax with the same encoder. */
  readonly crf?: boolean
  /** The token id whose embedding stays at zero. */
  readonly paddingIdx?: number
  readonly key?: Key
}

const part = (params: Params, name: string): Params => (params[name] ?? {}) as Params

/**
 * A BiLSTM-CRF tagger as a module: `apply(params, batch, ctx)` gives the emission scores `[B, T, K]` of a padded batch
 * (token ids through embedding, dropout, LSTM, dropout and a linear layer), `loss` the sequence negative log-likelihood
 * under the CRF (per token; the token cross-entropy when built with `crf: false`), and `predict` the tag paths by
 * Viterbi. Parameters are `embedding`, `lstm`, `classifier` and, with the CRF, `crf` (torchcrf's names).
 *
 * @concept sequence-tagging
 * @index module BiLSTM-CRF sequence tagger over token ids, trained end to end (Lample et al., flair SequenceTagger).
 * @example
 * ```ts
 * import { SequenceTagger, padTagging } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const tagger = new SequenceTagger({ vocabSize: 10, numTags: 3, key: key(0) })
 * tagger.predict(padTagging([{ tokens: [1, 2, 3] }, { tokens: [4, 5] }])).map((p) => p.length) // => [3, 2]
 * ```
 */
export class SequenceTagger extends Module<TagBatch, Value> {
  readonly kind = 'SequenceTagger'
  readonly numTags: number
  readonly dropoutRate: number
  readonly embedding: Embedding
  readonly lstm: LSTM
  readonly classifier: Linear
  /** The CRF layer, absent for the softmax foil. */
  readonly crfLayer: CRF | undefined

  constructor(options: SequenceTaggerOptions) {
    const k = options.key ?? makeKey(0)
    const embeddingDim = options.embeddingDim ?? 16
    const hiddenSize = options.hiddenSize ?? 16
    const bidirectional = options.bidirectional ?? true
    const embedding = new Embedding(options.vocabSize, embeddingDim, {
      key: foldIn(k, 0),
      ...(options.paddingIdx === undefined ? {} : { paddingIdx: options.paddingIdx }),
    })
    const lstm = new LSTM(embeddingDim, hiddenSize, {
      batchFirst: true,
      bidirectional,
      numLayers: options.numLayers ?? 1,
      key: foldIn(k, 1),
    })
    const classifier = new Linear(hiddenSize * (bidirectional ? 2 : 1), options.numTags, { key: foldIn(k, 2) })
    const crfLayer =
      options.crf === false ? undefined : new CRF(options.numTags, { batchFirst: true, key: foldIn(k, 3) })
    super(
      assemble([
        ['embedding', embedding.params],
        ['lstm', lstm.params],
        ['classifier', classifier.params],
        ...(crfLayer === undefined ? [] : ([['crf', crfLayer.params]] as const)),
      ]),
    )
    this.numTags = options.numTags
    this.dropoutRate = options.dropout ?? 0
    this.embedding = embedding
    this.lstm = lstm
    this.classifier = classifier
    this.crfLayer = crfLayer
    if (!(this.dropoutRate >= 0 && this.dropoutRate < 1))
      throw new OmaiError('value', 'SequenceTagger: dropout in [0, 1)')
  }

  override children(): readonly (readonly [string, Module<never, unknown>])[] {
    return [
      ['embedding', this.embedding],
      ['lstm', this.lstm],
      ['classifier', this.classifier],
      ...(this.crfLayer === undefined ? [] : ([['crf', this.crfLayer]] as const)),
    ]
  }

  protected override prepare(x: unknown): unknown {
    return x
  }

  /** The emission scores `[B, T, K]` of a padded batch. */
  apply(params: Params, batch: TagBatch, ctx: ApplyContext = {}): Value {
    const training = isTraining(ctx)
    const drop = (x: Value, i: number): Value =>
      this.dropoutRate === 0 || !training
        ? x
        : dropout(x, this.dropoutRate, {
            training,
            ...(ctx.key === undefined ? {} : { key: foldIn(ctx.key, i) }),
          })
    const lengths = batch.mask.map((row) => row.reduce((a, v) => a + (v === 0 ? 0 : 1), 0))
    const embedded = this.embedding.apply(part(params, 'embedding'), asTensor(batch.tokens as number[][], 'int32'))
    const encoded = this.lstm.apply(part(params, 'lstm'), { input: drop(embedded, 0), lengths }).output as Value
    return this.classifier.apply(part(params, 'classifier'), drop(encoded, 1))
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape, this.numTags]
  }

  /**
   * The training loss of a labelled batch, per token: the CRF's negative log-likelihood of the gold sequences, or the
   * softmax cross-entropy without a CRF.
   */
  loss(batch: TagBatch, params: Params = this.params, ctx: ApplyContext = {}): Value {
    if (batch.tags === undefined) throw new OmaiError('value', 'SequenceTagger.loss: the batch has no tags')
    const emissions = this.apply(params, batch, ctx)
    if (this.crfLayer !== undefined)
      return neuralCrfNegLogLikelihood(
        emissions,
        batch.tags as number[][],
        this.crfLayer.layerParameters(part(params, 'crf')),
        {
          batchFirst: true,
          mask: batch.mask as number[][],
          reduction: 'token_mean',
        },
      )
    const [, T = 0, K = 0] = emissions.shape
    const logp = reshape(logSoftmax(emissions), [-1])
    const index = batch.tags.flatMap((row, b) => row.map((k, t) => (b * T + t) * K + k))
    const live = batch.mask.flat()
    const picked = mul(take(logp, index), tensor(live as number[]))
    return neg(
      div(
        sum(picked),
        Math.max(
          1,
          live.reduce((a, v) => a + v, 0),
        ),
      ),
    )
  }

  /**
   * The predicted tag path of each sequence: Viterbi under the CRF (exact; with `allowed`, only valid transitions), or
   * the per-token argmax without one.
   */
  predict(
    batch: TagBatch,
    options: { readonly params?: Params; readonly allowed?: NeuralCrfAllowed } = {},
  ): number[][] {
    const params = options.params ?? this.params
    const emissions = this.apply(params, batch, { train: false })
    const mask = batch.mask as number[][]
    if (this.crfLayer !== undefined)
      return neuralCrfDecode(emissions, this.crfLayer.layerParameters(part(params, 'crf')), {
        batchFirst: true,
        mask,
        ...(options.allowed === undefined ? {} : { allowed: options.allowed }),
      }).paths
    const K = this.numTags
    const e = toFlat(emissions)
    const T = emissions.shape[1] ?? 0
    return mask.map((row, b) =>
      row.flatMap((on, t) => {
        if (on === 0) return []
        let best = 0
        for (let k = 1; k < K; k++) if ((e[(b * T + t) * K + k] ?? 0) > (e[(b * T + t) * K + best] ?? 0)) best = k
        return [best]
      }),
    )
  }

  override extraRepr(): string {
    return `tags=${this.numTags}, crf=${this.crfLayer !== undefined}`
  }
}

/** The scores of a tagger on labelled sequences. */
export interface TaggerScores {
  /** The fraction of tokens tagged right. */
  readonly tokenAccuracy: number
  /** Entity-level F1 (micro) under the BIO scheme, when tag names are given. */
  readonly entityF1?: number
  /** Tags that break the BIO scheme (an I-X after anything but B-X or I-X), when tag names are given. */
  readonly violations?: number
}

const BATCH = 256

/**
 * Score a tagger: token accuracy, and with tag names (`O`, `B-X`, `I-X`) the entity-level F1 and the number of BIO
 * violations in its predictions.
 *
 * @param model - The tagger.
 * @param data - Labelled sequences.
 * @param options - Parameters to score (default the model's), tag names and the allowed transitions of decoding.
 * @returns The scores.
 * @concept sequence-tagging
 * @example
 * ```ts
 * import { SequenceTagger, evaluateTagger } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const tagger = new SequenceTagger({ vocabSize: 4, numTags: 2, key: key(0) })
 * evaluateTagger(tagger, [{ tokens: [1, 2], tags: [0, 1] }]).tokenAccuracy <= 1 // => true
 * ```
 */
export function evaluateTagger(
  model: SequenceTagger,
  data: readonly TaggedSequence[],
  options: {
    readonly params?: Params
    readonly tagNames?: readonly string[]
    readonly allowed?: NeuralCrfAllowed
    readonly scheme?: TagScheme
  } = {},
): TaggerScores {
  const predicted: number[][] = []
  for (let i = 0; i < data.length; i += BATCH)
    predicted.push(
      ...model.predict(padTagging(data.slice(i, i + BATCH)), {
        ...(options.params === undefined ? {} : { params: options.params }),
        ...(options.allowed === undefined ? {} : { allowed: options.allowed }),
      }),
    )
  const truth = data.map((s) => [...s.tags])
  const accuracy = accuracyScore(truth.flat(), predicted.flat()).estimate.value
  const names = options.tagNames
  if (names === undefined) return { tokenAccuracy: accuracy }
  const label = (rows: readonly (readonly number[])[]): string[][] => rows.map((r) => r.map((k) => names[k] ?? 'O'))
  const yTrue = label(truth)
  const yPred = label(predicted)
  return {
    tokenAccuracy: accuracy,
    entityF1: entityScores(yTrue, yPred, options.scheme ?? 'bio').f1,
    violations: yPred.reduce((a, row) => a + bioBreaks(row), 0),
  }
}

/** Tags that break the BIO scheme: an I-X that follows neither B-X nor I-X (or opens a sequence). */
function bioBreaks(tags: readonly string[]): number {
  let n = 0
  for (const [i, t] of tags.entries()) {
    if (!t.startsWith('I-')) continue
    const prev = i === 0 ? 'O' : (tags[i - 1] ?? 'O')
    if (prev === 'O' || prev.slice(2) !== t.slice(2)) n++
  }
  return n
}

/** Options of {@link taggerAlgorithm} and {@link fitTagger}. */
export interface TaggerTrainOptions {
  /** The optimiser (for example `adam({ learningRate: 0.05 })`). */
  readonly optimizer: GradientTransformation
  readonly epochs: number
  /** Sequences per minibatch (default 16). */
  readonly batchSize?: number
  /** Shuffling and dropout keys derive from this (default key(0)). */
  readonly key?: Key
  /** Clip the gradient's global norm before the optimiser. */
  readonly clipNorm?: number
  /** Held-out sequences scored at each epoch end (default: the training sequences). */
  readonly evaluation?: readonly TaggedSequence[]
  /** Tag names (`O`, `B-X`, `I-X`): adds entity F1 and violations to the tape. */
  readonly tagNames?: readonly string[]
  /** Allowed transitions for the decoding of the epoch-end scores (the CRF only). */
  readonly allowed?: NeuralCrfAllowed
}

/**
 * Train a tagger end to end as a traced {@link Algorithm}: one step is one minibatch of shuffled sequences (padded to
 * the batch's longest), the loss is differentiated with `valueAndGrad` through the CRF's forward algorithm, the LSTM and
 * the embeddings, and the optimiser updates every parameter. The tape has `loss`, `gradNorm`, the `epochLoss` and, at
 * each epoch end, `tokenAccuracy` (and `entityF1` and `violations` with tag names) on the evaluation sequences.
 *
 * @param model - The tagger (its parameters are the starting point; it is not modified).
 * @param data - The training sequences.
 * @param options - Optimiser, epochs, batch size, key, clipping and evaluation.
 * @returns The algorithm.
 * @concept sequence-tagging
 * @index algorithm Train a BiLSTM-CRF tagger end to end, scoring each epoch.
 * @example
 * ```ts
 * import { SequenceTagger, taggerAlgorithm } from 'omai-engine/pgm'
 * import { adam } from 'omai-engine/optim'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const tagger = new SequenceTagger({ vocabSize: 4, numTags: 2, embeddingDim: 4, hiddenSize: 4, key: key(0) })
 * const data = [{ tokens: [1, 2, 1], tags: [0, 1, 0] }]
 * const alg = taggerAlgorithm(tagger, data, { optimizer: adam({ learningRate: 0.1 }), epochs: 2 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.step // => 2
 * ```
 */
export function taggerAlgorithm(
  model: SequenceTagger,
  data: readonly TaggedSequence[],
  options: TaggerTrainOptions,
): Algorithm<TrainState> {
  if (data.length === 0) throw new OmaiError('value', 'taggerAlgorithm: no training sequences')
  const batchSize = Math.min(options.batchSize ?? 16, data.length)
  const perEpoch = Math.ceil(data.length / batchSize)
  const total = options.epochs * perEpoch
  const base = options.key ?? makeKey(0)
  const optimizer =
    options.clipNorm === undefined ? options.optimizer : chain(clipByGlobalNorm(options.clipNorm), options.optimizer)
  const mask = model.trainableMask()
  const frozen = treeLeaves(mask).some((m) => item(m) === 0)
  const evaluation = options.evaluation ?? data
  let cached: { epoch: number; order: number[] } | undefined
  const orderOf = (epoch: number): number[] => {
    if (cached?.epoch !== epoch) cached = { epoch, order: Array.from(permutation(foldIn(base, 0), data.length).data) }
    return cached.order
  }
  return {
    name: 'sequence-tagger',
    config: {
      model: model.kind,
      crf: model.crfLayer !== undefined,
      optimizer: optimizer.name,
      epochs: options.epochs,
      batchSize,
      sequences: data.length,
    } as Json,
    schema: channels({
      metrics: {
        loss: { description: 'minibatch loss per token' },
        gradNorm: { description: 'global gradient norm' },
        epochLoss: { description: 'mean minibatch loss of the epoch' },
        tokenAccuracy: { description: 'token accuracy on the evaluation sequences, at the epoch end' },
        entityF1: { description: 'entity-level F1 on the evaluation sequences (with tag names)' },
        violations: { description: 'predicted tags that break the BIO scheme (with tag names)' },
      },
      tensors: { params: { role: 'param', description: 'parameters by layer' } },
    }),
    init: () => ({ ...trainState(model, optimizer, base) }),
    step: (s, ctx) => {
      const epoch = Math.floor(s.step / perEpoch)
      const batchIndex = s.step % perEpoch
      const order = orderOf(epoch)
      const rows = order
        .slice(batchIndex * batchSize, (batchIndex + 1) * batchSize)
        .map((i) => data[i] as TaggedSequence)
      const batch = padTagging(rows)
      const { value, grad } = valueAndGrad((p: Params) =>
        model.loss(batch, p, { train: true, key: foldIn(foldIn(base, 1), s.step) }),
      )(s.params)
      const loss = item(value)
      const applied = applyGradients(s, grad, optimizer, frozen ? { mask } : {})
      const step = s.step + 1
      let epochLossSum = s.epochLossSum + loss
      if (step % perEpoch === 0) {
        ctx.run.metric(step, 'epochLoss', epochLossSum / perEpoch)
        const scores = evaluateTagger(model, evaluation, {
          params: applied.state.params,
          ...(options.tagNames === undefined ? {} : { tagNames: options.tagNames }),
          ...(options.allowed === undefined ? {} : { allowed: options.allowed }),
        })
        ctx.run.metric(step, 'tokenAccuracy', scores.tokenAccuracy)
        if (scores.entityF1 !== undefined) ctx.run.metric(step, 'entityF1', scores.entityF1)
        if (scores.violations !== undefined) ctx.run.metric(step, 'violations', scores.violations)
        epochLossSum = 0
      }
      return {
        ...applied.state,
        step,
        epoch: Math.floor(step / perEpoch),
        loss,
        gradNorm: item(globalNorm(grad)),
        epochLossSum,
      }
    },
    done: (s) =>
      s.step >= total ? 'completed' : !Number.isNaN(s.loss) && !Number.isFinite(s.loss) ? 'diverged' : undefined,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TrainState,
    observe: (s) => ({
      metrics: Number.isNaN(s.loss) ? {} : { loss: s.loss, gradNorm: s.gradNorm },
      tensors: Object.fromEntries(
        Object.entries(treePaths(s.params)).map(([path, v]) => [
          `params.${path}`,
          { role: 'param' as const, value: v },
        ]),
      ),
    }),
  }
}

/** The outcome of {@link fitTagger}. */
export interface TaggerFit {
  /** The minibatch steps taken. */
  readonly steps: number
  /** The loss of the last minibatch. */
  readonly loss: number
  /** The trained parameters (also loaded into the model). */
  readonly params: Params
  /** The scores on the evaluation sequences (default: the training sequences). */
  readonly scores: TaggerScores
}

/**
 * Run {@link taggerAlgorithm} to its end and load the trained parameters into the model.
 *
 * @param model - The tagger (modified: its parameters become the trained ones).
 * @param data - The training sequences.
 * @param options - As for `taggerAlgorithm`.
 * @returns The steps, the last loss, the parameters and the scores.
 * @concept sequence-tagging
 * @example
 * ```ts
 * import { SequenceTagger, fitTagger } from 'omai-engine/pgm'
 * import { adam } from 'omai-engine/optim'
 * import { key } from 'omai-engine/random'
 * const tagger = new SequenceTagger({ vocabSize: 3, numTags: 2, embeddingDim: 4, hiddenSize: 4, key: key(1) })
 * const data = [{ tokens: [1, 2], tags: [0, 1] }]
 * fitTagger(tagger, data, { optimizer: adam({ learningRate: 0.1 }), epochs: 3 }).steps // => 3
 * ```
 */
export function fitTagger(
  model: SequenceTagger,
  data: readonly TaggedSequence[],
  options: TaggerTrainOptions,
): TaggerFit {
  const perEpoch = Math.ceil(data.length / Math.min(options.batchSize ?? 16, data.length))
  const run = runAlgorithm(taggerAlgorithm(model, data, options), {
    maxSteps: options.epochs * perEpoch,
    key: options.key ?? makeKey(0),
  })
  model.setParams(run.state.params)
  return {
    steps: run.steps,
    loss: run.state.loss,
    params: run.state.params,
    scores: evaluateTagger(model, options.evaluation ?? data, {
      ...(options.tagNames === undefined ? {} : { tagNames: options.tagNames }),
      ...(options.allowed === undefined ? {} : { allowed: options.allowed }),
    }),
  }
}
