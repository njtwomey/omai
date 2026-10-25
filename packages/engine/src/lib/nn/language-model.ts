/**
 * The language-model contract used by decoding and the NLP methods: a `Tokenizer` and a `LanguageModel` that returns
 * next-token logits for batches of token sequences. A character tokenizer, a bigram model from a table, an adapter from
 * any module mapping token ids to logits, and helpers for next-token logits, sequence log-probability and perplexity.
 */

import { MdmlError } from '../base/index.ts'
import {
  asTensor,
  isValue,
  logSoftmax,
  reshape,
  sliceAxis,
  type Tensor,
  take,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import type { Module } from './module/module.ts'

/**
 * The tokenizer a language model speaks: text to token ids and back. Structural, so any tokenizer with these members
 * (character, byte-pair, word-piece) satisfies it.
 */
export interface Tokenizer {
  readonly vocabSize: number
  encode(text: string): number[]
  /** Text of ids (special tokens are left out). */
  decode(ids: readonly number[]): string
  readonly bosId?: number
  readonly eosId?: number
  readonly padId?: number
}

/** A batch of token ids: rows of equal length, or an int tensor `[B, T]`. */
export type TokenBatch = readonly (readonly number[])[] | Value

/**
 * The language-model contract: a tokenizer and next-token logits `[B, T, V]` for a batch of token sequences (row t
 * scores the token after position t), optionally embeddings and an incremental interface with a cache, so decoding
 * feeds only new tokens. Local models and adapters to remote models (methods/llm) implement it.
 *
 * @concept language-models
 */
export interface LanguageModel<C = unknown> {
  readonly tokenizer: Tokenizer
  readonly vocabSize: number
  /** The longest sequence the model accepts (default unlimited). */
  readonly maxLength?: number
  /** Next-token logits `[B, T, V]`. */
  logits(tokens: TokenBatch): Value
  /** Token or sequence embeddings `[B, T, D]`, when the model exposes them. */
  embed?(tokens: TokenBatch): Value
  /** An empty cache for a batch (incremental decoding). */
  initCache?(batch: number): C
  /** Logits `[B, t, V]` for t new tokens after those already in the cache, and the extended cache. */
  step?(tokens: TokenBatch, cache: C): { readonly logits: Value; readonly cache: C }
}

/** The ids of a token batch as an int32 `[B, T]` tensor (rows must have equal length). */
function batchTensor(tokens: TokenBatch): Value {
  if (isValue(tokens)) return tokens.dtype === 'int32' ? tokens : asTensor(tokens, 'int32')
  const t = tokens[0]?.length ?? 0
  if (tokens.some((r) => r.length !== t))
    throw new MdmlError('shape', 'language model: rows of a batch must have equal length')
  return tensor(
    tokens.map((r) => [...r]),
    { dtype: 'int32' },
  )
}

/**
 * A character tokenizer over an alphabet: one id per character in order, then optional special tokens (BOS, EOS,
 * PAD) after the alphabet. Unknown characters are an error.
 *
 * @param alphabet - The characters, each once.
 * @param specials - Which special tokens to add.
 * @returns The tokenizer.
 * @example
 * ```ts
 * import { characterTokenizer } from 'mdml-engine/nn'
 * const tok = characterTokenizer('ab', { eos: true })
 * [tok.encode('ba'), tok.eosId, tok.decode([1, 0, 2])] // => [[1, 0], 2, 'ba']
 * ```
 */
export function characterTokenizer(
  alphabet: string,
  specials: { readonly bos?: boolean; readonly eos?: boolean; readonly pad?: boolean } = {},
): Tokenizer {
  const chars = [...alphabet]
  if (new Set(chars).size !== chars.length) throw new MdmlError('value', 'characterTokenizer: repeated characters')
  const index = new Map(chars.map((c, i) => [c, i]))
  let next = chars.length
  const bosId = specials.bos === true ? next++ : undefined
  const eosId = specials.eos === true ? next++ : undefined
  const padId = specials.pad === true ? next++ : undefined
  return {
    vocabSize: next,
    encode: (text) =>
      [...text].map((c) => {
        const i = index.get(c)
        if (i === undefined) throw new MdmlError('value', `characterTokenizer: '${c}' is not in the alphabet`)
        return i
      }),
    decode: (ids) =>
      ids
        .filter((i) => i >= 0 && i < chars.length)
        .map((i) => chars[i] ?? '')
        .join(''),
    ...(bosId === undefined ? {} : { bosId }),
    ...(eosId === undefined ? {} : { eosId }),
    ...(padId === undefined ? {} : { padId }),
  }
}

/**
 * A bigram language model: the logits of the next token are the row of a `[V, V]` table for the current token. The
 * smallest model with the full contract (it is its own cache: only the last token matters), used to test and explain
 * decoding.
 *
 * @param table - Logits `[V, V]`; row i scores the token after i.
 * @param tokenizer - The tokenizer (default: one with V characters 'a', 'b', ... and no special tokens).
 * @returns The model.
 * @concept language-models
 * @example
 * ```ts
 * import { bigramLanguageModel } from 'mdml-engine/nn'
 * bigramLanguageModel([[0, 1], [1, 0]]).logits([[0, 1, 1]]).shape // => [1, 3, 2]
 * ```
 */
export function bigramLanguageModel(table: TensorLikeTable, tokenizer?: Tokenizer): LanguageModel<null> {
  const t = asTensor(table)
  const v = t.shape[0] ?? 0
  if (t.shape.length !== 2 || t.shape[1] !== v)
    throw new MdmlError('shape', 'bigramLanguageModel: table must be [V, V]')
  const tok = tokenizer ?? characterTokenizer(Array.from({ length: v }, (_, i) => String.fromCharCode(97 + i)).join(''))
  if (tok.vocabSize !== v) throw new MdmlError('value', 'bigramLanguageModel: the tokenizer and table sizes differ')
  const logits = (tokens: TokenBatch): Value => {
    const ids = batchTensor(tokens)
    const [b = 0, n = 0] = ids.shape
    return reshape(take(t, reshape(ids, [-1]), 0), [b, n, v])
  }
  return {
    tokenizer: tok,
    vocabSize: v,
    logits,
    initCache: () => null,
    step: (tokens, cache) => ({ logits: logits(tokens), cache }),
  }
}

/** A `[V, V]` table given as nested arrays or a tensor. */
export type TensorLikeTable = readonly (readonly number[])[] | Value

/**
 * A language model from a module mapping token ids `[B, T]` to logits `[B, T, V]` (a decoder-only transformer, an
 * RNN language model), evaluated in evaluation mode with its own parameters.
 *
 * @param module - The module.
 * @param tokenizer - Its tokenizer.
 * @param options - `maxLength`, and an embedding function when the model exposes one.
 * @returns The language model.
 * @example
 * ```ts
 * import { languageModelFromModule, characterTokenizer, Embedding } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const lm = languageModelFromModule(new Embedding(3, 3, { key: key(0) }), characterTokenizer('abc'))
 * lm.logits([[0, 2]]).shape // => [1, 2, 3]
 * ```
 */
export function languageModelFromModule(
  module: Module,
  tokenizer: Tokenizer,
  options: { readonly maxLength?: number; readonly embed?: (tokens: Value) => Value } = {},
): LanguageModel {
  const run = (tokens: TokenBatch): Value => {
    const ids = batchTensor(tokens)
    if (options.maxLength !== undefined && (ids.shape[1] ?? 0) > options.maxLength)
      throw new MdmlError('value', `language model: sequences longer than ${options.maxLength}`)
    const y = module.apply(module.params, ids, { train: false, buffers: module.buffers })
    if (y.shape.length !== 3 || y.shape[2] !== tokenizer.vocabSize)
      throw new MdmlError('shape', 'languageModelFromModule: the module must return [B, T, vocabSize]')
    return y
  }
  const embed = options.embed
  return {
    tokenizer,
    vocabSize: tokenizer.vocabSize,
    ...(options.maxLength === undefined ? {} : { maxLength: options.maxLength }),
    logits: run,
    ...(embed === undefined ? {} : { embed: (tokens: TokenBatch) => embed(batchTensor(tokens)) }),
  }
}

/**
 * The next-token logits after a single sequence (the last row of the model's logits).
 *
 * @param lm - The model.
 * @param prefix - Token ids (at least one).
 * @returns V logits.
 * @example
 * ```ts
 * import { bigramLanguageModel, nextTokenLogits } from 'mdml-engine/nn'
 * nextTokenLogits(bigramLanguageModel([[0, 1], [2, 3]]), [0, 1]) // => [2, 3]
 * ```
 */
export function nextTokenLogits(lm: LanguageModel, prefix: readonly number[]): number[] {
  if (prefix.length === 0) throw new MdmlError('value', 'nextTokenLogits: need at least one token')
  const y = lm.logits([prefix])
  const n = y.shape[1] ?? 0
  return toFlat(sliceAxis(y, 1, n - 1, n))
}

/**
 * The log-probability of a sequence under a model: the sum over t >= 1 of log p(token_t | tokens_<t) (the first
 * token is the context).
 *
 * @param lm - The model.
 * @param tokens - Token ids (at least one).
 * @returns The total log-probability (natural log).
 * @example
 * ```ts
 * import { bigramLanguageModel, sequenceLogProb } from 'mdml-engine/nn'
 * sequenceLogProb(bigramLanguageModel([[0, 0], [0, 0]]), [0, 1, 1]) // => -1.3862943611198906
 * ```
 */
export function sequenceLogProb(lm: LanguageModel, tokens: readonly number[]): number {
  if (tokens.length < 2) return 0
  const lp = logSoftmax(lm.logits([tokens.slice(0, -1)]), -1) as Tensor
  const v = lm.vocabSize
  let total = 0
  for (let t = 1; t < tokens.length; t++) total += lp.data[(t - 1) * v + (tokens[t] ?? 0)] ?? Number.NaN
  return total
}

/**
 * Perplexity, exp(-mean log-probability) of the predicted tokens of a sequence.
 *
 * @param lm - The model.
 * @param tokens - Token ids (at least two).
 * @returns The perplexity (V for a uniform model).
 * @example
 * ```ts
 * import { bigramLanguageModel, perplexity } from 'mdml-engine/nn'
 * perplexity(bigramLanguageModel([[0, 0], [0, 0]]), [0, 1, 0]) // => 2
 * ```
 */
export function perplexity(lm: LanguageModel, tokens: readonly number[]): number {
  if (tokens.length < 2) throw new MdmlError('value', 'perplexity: need at least two tokens')
  return Math.exp(-sequenceLogProb(lm, tokens) / (tokens.length - 1))
}
