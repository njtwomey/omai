/**
 * Generation from a sequence model as an algorithm: the model as a `LanguageModel` whose cache is the per-layer mixer
 * state (the SSM state, the convolution window, the key-value cache), so the library's decoding algorithm (greedy or
 * sampled, with temperature, top-k, top-p, penalties and constraints) feeds it one token per step, and the serialisable
 * form of that algorithm, whose state, the tokens and the model's state, can be checkpointed and resumed. The logits it
 * produces are those of the parallel pass.
 */

import type { Json } from '../../base/index.ts'
import { asTensor, isValue, type Value } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { stateFromJson, stateToJson } from '../../trace/index.ts'
import { type DecodeState, decodingAlgorithm, type GenerateOptions } from '../decoding/index.ts'
import type { LanguageModel, TokenBatch, Tokenizer } from '../language-model.ts'
import type { Params } from '../module/module.ts'
import type { SequenceModel, SequenceModelState } from './sequence-model.ts'

/** A tokenizer over bare ids: `decode` joins them with spaces, `encode` reads space-separated integers. */
export function idTokenizer(vocabSize: number): Tokenizer {
  return {
    vocabSize,
    encode: (text) =>
      text
        .split(/\s+/)
        .filter((t) => t !== '')
        .map(Number),
    decode: (ids) => ids.join(' '),
  }
}

/**
 * A sequence model as a language model with an incremental interface: `logits` is the parallel pass, `step` feeds new
 * tokens after those the cache has seen (the cache is the per-layer mixer state) and returns their logits.
 *
 * @param model - The model (a language model: per-position logits).
 * @param params - Its parameters (default the model's own).
 * @param tokenizer - The tokenizer (default bare ids).
 * @returns The language model.
 * @concept language-models
 * @example
 * ```ts
 * import { mambaModel, sequenceLanguageModel } from 'omai-engine/nn/blocks'
 * sequenceLanguageModel(mambaModel(7, 8, 1, { stateSize: 4 })).logits([[1, 2, 3]]).shape // => [1, 3, 7]
 * ```
 */
export function sequenceLanguageModel(
  model: SequenceModel,
  params: Params = model.params,
  tokenizer: Tokenizer = idTokenizer(model.outputSize),
): LanguageModel<SequenceModelState> {
  const ids = (t: TokenBatch): Value => (isValue(t) ? t : asTensor(t as readonly (readonly number[])[], 'int32'))
  return {
    tokenizer,
    vocabSize: model.outputSize,
    logits: (tokens) => model.run(params, ids(tokens), undefined, { train: false }).output,
    initCache: (batch) => model.initialState(batch),
    step: (tokens, cache) => {
      const r = model.run(params, ids(tokens), cache, { train: false })
      return { logits: r.output, cache: r.state }
    },
  }
}

/**
 * Generation as a traced algorithm, one token per step, whose state is the tokens so far and the model's state; it
 * serialises (the state tensors go to JSON), so a run can be checkpointed and resumed. The tokens it chooses and their
 * log-probabilities are those of the parallel pass over the same tokens.
 *
 * @param model - The model.
 * @param params - Its parameters (default the model's own).
 * @param prompt - Prompt token ids.
 * @param options - The decoding options (greedy by default; `sample`, `temperature`, `topK`, ...).
 * @returns The algorithm.
 * @concept decoding
 * @example
 * ```ts
 * import { generationAlgorithm, mambaModel } from 'omai-engine/nn/blocks'
 * generationAlgorithm(mambaModel(7, 8, 1, { stateSize: 4 }), undefined, [1, 2]).name // => 'decode'
 * ```
 */
export function generationAlgorithm(
  model: SequenceModel,
  params: Params | undefined,
  prompt: readonly number[],
  options: GenerateOptions = {},
): Algorithm<DecodeState> {
  const base = decodingAlgorithm(sequenceLanguageModel(model, params ?? model.params), prompt, options)
  return {
    ...base,
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j) as DecodeState,
  }
}
