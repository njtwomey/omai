/**
 * Autoregressive text generation, one token per step: greedy or sampled decoding with temperature, top-k, top-p and
 * min-p filtering, repetition, frequency and presence penalties, extra logit processors and token constraints. Provided
 * as the traced `decodingAlgorithm` and the `generate` function that runs it and returns the tokens with their
 * log-probabilities.
 */

import { argmaxOf, type Json, maxOf } from '../../base/index.ts'
import { categorical, foldIn, type Key, key as makeKey } from '../../random/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import type { LanguageModel } from '../language-model.ts'
import { chain, eosOf, promptIds, scoreNext } from './_shared.ts'
import type { TokenConstraint } from './constraints.ts'
import { entropyOfLogits, type LogitProcessor, logProbs } from './processors.ts'

const NEG = Number.NEGATIVE_INFINITY

/** Options of {@link generate} and {@link decodingAlgorithm}. */
export interface GenerateOptions {
  /** The most tokens to generate (default 16). */
  readonly maxNewTokens?: number
  /** Sample (default false: greedy). A temperature of 0 is greedy too. */
  readonly sample?: boolean
  readonly temperature?: number
  readonly topK?: number
  readonly topP?: number
  readonly minP?: number
  readonly repetitionPenalty?: number
  readonly frequencyPenalty?: number
  readonly presencePenalty?: number
  /** Further processors, applied after the penalties and the constraint and before temperature and filtering. */
  readonly processors?: readonly LogitProcessor[]
  /** Allowed tokens as a function of the generated tokens. */
  readonly constraint?: TokenConstraint
  /** Stop at this token (default the tokenizer's end-of-sequence id; `null` never stops early). */
  readonly eosId?: number | null
  /** Sampling key (default key(0)). */
  readonly key?: Key
}

/** The state of a decoding run. */
export interface DecodeState {
  /** Prompt and generated tokens. */
  readonly tokens: readonly number[]
  readonly generated: readonly number[]
  /** The model's log-probability of the generated tokens (before processing). */
  readonly logprob: number
  readonly cache: unknown
  /** How many tokens the cache has seen. */
  readonly fed: number
  readonly last?: { readonly token: number; readonly logprob: number; readonly entropy: number }
  /** Why decoding ended: end-of-sequence, length, or a constraint that allows nothing. */
  readonly finished?: 'eos' | 'length' | 'constraint'
}

/**
 * Autoregressive decoding as a traced {@link Algorithm}, one token per step: greedy or sampled with temperature,
 * top-k, top-p and min-p filtering, repetition, frequency and presence penalties, further logit processors and an
 * allowed-token constraint. Records the model's log-probability and the entropy of the processed distribution as
 * metrics and each chosen token as a `token` message; stops at the end-of-sequence token, the length limit, or when
 * a constraint allows nothing.
 *
 * @param lm - The language model.
 * @param prompt - Token ids or text (an empty prompt starts from BOS).
 * @param options - The decoding options.
 * @returns The algorithm (run it for maxNewTokens steps).
 * @concept decoding
 * @example
 * ```ts
 * import { decodingAlgorithm, bigramLanguageModel } from 'whitebox-engine/nn'
 * decodingAlgorithm(bigramLanguageModel([[0, 1], [1, 0]]), [0], {}).name // => 'decode'
 * ```
 */
export function decodingAlgorithm(
  lm: LanguageModel,
  prompt: readonly number[] | string,
  options: GenerateOptions = {},
): Algorithm<DecodeState> {
  const start = promptIds(lm, prompt)
  const maxNew = options.maxNewTokens ?? 16
  const eos = eosOf(lm, options)
  const { pre, post } = chain(options, start.length)
  const sampling = options.sample === true && (options.temperature === undefined || options.temperature > 0)
  return {
    name: 'decode',
    config: {
      strategy: sampling ? 'sample' : 'greedy',
      maxNewTokens: maxNew,
      temperature: options.temperature ?? null,
      topK: options.topK ?? null,
      topP: options.topP ?? null,
      minP: options.minP ?? null,
      repetitionPenalty: options.repetitionPenalty ?? null,
      constrained: options.constraint !== undefined,
    } as Json,
    schema: channels({
      metrics: {
        logprob: { description: "the model's log-probability of the chosen token", unit: 'nats' },
        entropy: { description: 'entropy of the processed next-token distribution', unit: 'nats' },
        totalLogprob: { description: 'log-probability of the generated text so far', unit: 'nats' },
      },
      messages: { token: { description: 'the chosen token (id and text)' } },
    }),
    init: () => ({ tokens: start, generated: [], logprob: 0, cache: undefined, fed: 0 }),
    step: (s, ctx) => {
      const scored = scoreNext(lm, s.tokens, s.cache, s.fed)
      const model = logProbs(scored.logits)
      const processed = post(pre(scored.logits, s.tokens), s.tokens)
      if (maxOf(processed) === NEG) return { ...s, cache: scored.cache, fed: scored.fed, finished: 'constraint' }
      const token = sampling
        ? (categorical(options.key === undefined ? ctx.key : foldIn(options.key, ctx.t), processed, [1], {
            logits: true,
          }).data[0] ?? 0)
        : argmaxOf(processed)
      const lp = model[token] ?? NEG
      const entropy = entropyOfLogits(processed)
      ctx.run.message(ctx.t, 'token', { id: token, text: lm.tokenizer.decode([token]) })
      const generated = [...s.generated, token]
      const finished = token === eos ? 'eos' : generated.length >= maxNew ? 'length' : undefined
      return {
        tokens: [...s.tokens, token],
        generated,
        logprob: s.logprob + lp,
        cache: scored.cache,
        fed: scored.fed,
        last: { token, logprob: lp, entropy },
        ...(finished === undefined ? {} : { finished }),
      }
    },
    done: (s) => (s.finished === undefined ? undefined : 'completed'),
    observe: (s) => ({
      metrics:
        s.last === undefined ? {} : { logprob: s.last.logprob, entropy: s.last.entropy, totalLogprob: s.logprob },
    }),
  }
}

/** Result of {@link generate}. */
export interface GenerateResult {
  /** Prompt and generated tokens. */
  readonly tokens: readonly number[]
  readonly generated: readonly number[]
  /** The generated tokens decoded. */
  readonly text: string
  /** The model's log-probability of the generated tokens. */
  readonly logprob: number
  readonly finished: 'eos' | 'length' | 'constraint'
}

/**
 * Generate a continuation of a prompt (runs {@link decodingAlgorithm}).
 *
 * @param lm - The language model.
 * @param prompt - Token ids or text.
 * @param options - Decoding options and the run's tracer and policy.
 * @returns The tokens, text, log-probability and why decoding stopped.
 * @concept decoding
 * @example
 * ```ts
 * import { generate, bigramLanguageModel } from 'whitebox-engine/nn'
 * generate(bigramLanguageModel([[0, 5, 0], [0, 0, 5], [5, 0, 0]]), 'a', { maxNewTokens: 4 }).text // => 'bcab'
 * ```
 */
export function generate(
  lm: LanguageModel,
  prompt: readonly number[] | string,
  options: GenerateOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): GenerateResult {
  const alg = decodingAlgorithm(lm, prompt, options)
  const { tracer, policy, parent, runId } = options
  const r = runAlgorithm(alg, {
    maxSteps: options.maxNewTokens ?? 16,
    key: options.key ?? makeKey(0),
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
    ...(parent === undefined ? {} : { parent }),
    ...(runId === undefined ? {} : { runId }),
  })
  const s = r.state
  return {
    tokens: s.tokens,
    generated: s.generated,
    text: lm.tokenizer.decode(s.generated),
    logprob: s.logprob,
    finished: s.finished ?? 'length',
  }
}
