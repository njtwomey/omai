import { WhiteboxError } from '../../base/index.ts'
import { sliceAxis, toFlat } from '../../tensor/index.ts'
import type { LanguageModel } from '../language-model.ts'
import { maskProcessor } from './constraints.ts'
import type { GenerateOptions } from './generate.ts'
import {
  composeProcessors,
  frequencyPresenceProcessor,
  type LogitProcessor,
  minPProcessor,
  repetitionPenaltyProcessor,
  temperatureProcessor,
  topKProcessor,
  topPProcessor,
} from './processors.ts'

const _NEG = Number.NEGATIVE_INFINITY

/** Next-token logits for a sequence, feeding only new tokens to a model with a cache. */
export function scoreNext(
  lm: LanguageModel,
  tokens: readonly number[],
  cache: unknown,
  fed: number,
): { logits: number[]; cache: unknown; fed: number } {
  if (lm.step !== undefined && lm.initCache !== undefined) {
    const c = fed === 0 ? lm.initCache(1) : cache
    const r = lm.step([tokens.slice(fed)], c)
    const n = r.logits.shape[1] ?? 0
    return { logits: toFlat(sliceAxis(r.logits, 1, n - 1, n)), cache: r.cache, fed: tokens.length }
  }
  const y = lm.logits([tokens])
  const n = y.shape[1] ?? 0
  return { logits: toFlat(sliceAxis(y, 1, n - 1, n)), cache, fed }
}

/** The processor chain an options record asks for, in Hugging Face's order. */
export function chain(options: GenerateOptions, promptLength: number): { pre: LogitProcessor; post: LogitProcessor } {
  const pre: LogitProcessor[] = []
  if (options.repetitionPenalty !== undefined) pre.push(repetitionPenaltyProcessor(options.repetitionPenalty))
  if (options.frequencyPenalty !== undefined || options.presencePenalty !== undefined)
    pre.push(
      frequencyPresenceProcessor({
        ...(options.frequencyPenalty === undefined ? {} : { frequency: options.frequencyPenalty }),
        ...(options.presencePenalty === undefined ? {} : { presence: options.presencePenalty }),
      }),
    )
  if (options.constraint !== undefined) pre.push(maskProcessor(options.constraint, promptLength))
  pre.push(...(options.processors ?? []))
  const post: LogitProcessor[] = []
  if (options.temperature !== undefined && options.temperature > 0) post.push(temperatureProcessor(options.temperature))
  if (options.topK !== undefined) post.push(topKProcessor(options.topK))
  if (options.topP !== undefined) post.push(topPProcessor(options.topP))
  if (options.minP !== undefined) post.push(minPProcessor(options.minP))
  return { pre: composeProcessors(...pre), post: composeProcessors(...post) }
}

export const promptIds = (lm: LanguageModel, prompt: readonly number[] | string): number[] => {
  const ids = typeof prompt === 'string' ? lm.tokenizer.encode(prompt) : [...prompt]
  if (ids.length === 0) {
    if (lm.tokenizer.bosId === undefined)
      throw new WhiteboxError('value', 'decoding: an empty prompt needs a BOS token')
    return [lm.tokenizer.bosId]
  }
  return ids
}

export const eosOf = (lm: LanguageModel, options: { readonly eosId?: number | null }): number | undefined =>
  options.eosId === null ? undefined : (options.eosId ?? lm.tokenizer.eosId)
