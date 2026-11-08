/**
 * Speculative decoding (Leviathan et al. 2023; Chen et al. 2023): a cheap draft model proposes several tokens, the
 * target model scores them in one pass and a rejection rule keeps a prefix, so the output follows the target model's
 * distribution exactly while calling it less often. A traced `speculativeAlgorithm` and the `speculativeDecode`
 * function, reporting acceptance statistics.
 */

import { argmaxOf, type Json, OmaiError } from '../../base/index.ts'
import { categorical, foldIn, type Key, key as makeKey, uniform } from '../../random/index.ts'
import type { Tensor } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import type { LanguageModel } from '../language-model.ts'
import { eosOf, promptIds, scoreNext } from './_shared.ts'
import { logProbs } from './processors.ts'

const _NEG = Number.NEGATIVE_INFINITY

/** Options of {@link speculativeDecode}. */
export interface SpeculativeOptions {
  /** Draft tokens proposed per round (default 4). */
  readonly gamma?: number
  readonly maxNewTokens?: number
  /** Sampling temperature for both models (default 1; 0 is greedy: accept while the draft agrees with the target). */
  readonly temperature?: number
  readonly eosId?: number | null
  readonly key?: Key
}

/** The state of speculative decoding. */
export interface SpeculativeState {
  readonly tokens: readonly number[]
  readonly generated: readonly number[]
  /** Draft tokens proposed and accepted so far, and target model calls. */
  readonly proposed: number
  readonly accepted: number
  readonly targetCalls: number
  readonly finished?: 'eos' | 'length'
}

/** Probabilities from logits at a temperature (one-hot at the argmax for temperature 0). */
function distribution(logits: readonly number[], temperature: number): number[] {
  if (temperature <= 0) {
    const i = argmaxOf(logits)
    return logits.map((_, j) => (j === i ? 1 : 0))
  }
  return logProbs(logits.map((z) => z / temperature)).map(Math.exp)
}

const draw = (k: Key, p: readonly number[]): number => categorical(k, p, [1]).data[0] ?? 0

/**
 * Speculative decoding as a traced {@link Algorithm} (Leviathan, Kalman and Matias 2023; Chen et al. 2023): each
 * round the draft model proposes gamma tokens from q, the target model scores them all in one call, each draft token
 * is accepted with probability min(1, p / q), the first rejection is replaced by a sample from the normalised residual
 * max(0, p - q), and when all are accepted one more token is sampled from p. The output has exactly the target's
 * distribution; each round emits `token` messages and the acceptance count.
 *
 * @param target - The model whose distribution is sampled.
 * @param draft - A cheaper model with the same vocabulary.
 * @param prompt - Token ids or text.
 * @param options - gamma, length, temperature, end token and key.
 * @returns The algorithm (one round per step).
 * @concept decoding
 * @reference Leviathan, Kalman and Matias (2023), Fast inference from transformers via speculative decoding.
 * @example
 * ```ts
 * import { speculativeAlgorithm, bigramLanguageModel } from 'omai-engine/nn'
 * const lm = bigramLanguageModel([[0, 1], [1, 0]])
 * speculativeAlgorithm(lm, lm, [0]).name // => 'speculativeDecode'
 * ```
 */
export function speculativeAlgorithm(
  target: LanguageModel,
  draft: LanguageModel,
  prompt: readonly number[] | string,
  options: SpeculativeOptions = {},
): Algorithm<SpeculativeState> {
  if (target.vocabSize !== draft.vocabSize) throw new OmaiError('value', 'speculativeDecode: vocabularies differ')
  const start = promptIds(target, prompt)
  const gamma = options.gamma ?? 4
  const maxNew = options.maxNewTokens ?? 16
  const temperature = options.temperature ?? 1
  const eos = eosOf(target, options)
  return {
    name: 'speculativeDecode',
    config: { gamma, maxNewTokens: maxNew, temperature } as Json,
    schema: channels({
      metrics: {
        accepted: { description: 'draft tokens accepted this round' },
        acceptanceRate: { description: 'accepted / proposed draft tokens so far' },
      },
      messages: { token: { description: 'an emitted token (id, text, and whether it was drafted)' } },
    }),
    init: () => ({ tokens: start, generated: [], proposed: 0, accepted: 0, targetCalls: 0 }),
    step: (s, ctx) => {
      const k = options.key === undefined ? ctx.key : foldIn(options.key, ctx.t)
      const room = maxNew - s.generated.length
      const g = Math.min(gamma, room)
      const drafts: number[] = []
      const qs: number[][] = []
      for (let i = 0; i < g; i++) {
        const q = distribution(scoreNext(draft, [...s.tokens, ...drafts], undefined, 0).logits, temperature)
        const x = draw(foldIn(foldIn(k, 0), i), q)
        qs.push(q)
        drafts.push(x)
        if (x === eos) break
      }
      // One target call scores every draft position and the one after.
      const y = target.logits([[...s.tokens, ...drafts]]) as Tensor
      const v = target.vocabSize
      const n = s.tokens.length
      const p = (i: number): number[] =>
        distribution(Array.from(y.data.subarray((n - 1 + i) * v, (n + i) * v)), temperature)
      const u = uniform(foldIn(k, 1), [drafts.length]).data
      const emitted: number[] = []
      let accepted = 0
      let rejected = false
      for (const [i, x] of drafts.entries()) {
        const pi = p(i)
        const qi = qs[i] as number[]
        const ratio = (qi[x] ?? 0) > 0 ? (pi[x] ?? 0) / (qi[x] ?? 1) : 0
        if ((u[i] ?? 1) < Math.min(1, ratio)) {
          emitted.push(x)
          accepted += 1
          if (x === eos) break
          continue
        }
        const residual = pi.map((pv, j) => Math.max(0, pv - (qi[j] ?? 0)))
        const mass = residual.reduce((a, b) => a + b, 0)
        emitted.push(draw(foldIn(foldIn(k, 2), i), mass > 0 ? residual : pi))
        rejected = true
        break
      }
      if (!rejected && emitted.at(-1) !== eos && emitted.length < room)
        emitted.push(draw(foldIn(k, 3), p(drafts.length)))
      const out = emitted.slice(0, room)
      const cut = eos === undefined ? -1 : out.indexOf(eos)
      const kept = cut >= 0 ? out.slice(0, cut + 1) : out
      for (const [i, t] of kept.entries())
        ctx.run.message(ctx.t, 'token', { id: t, text: target.tokenizer.decode([t]), drafted: i < accepted })
      const generated = [...s.generated, ...kept]
      const finished = cut >= 0 ? 'eos' : generated.length >= maxNew ? 'length' : undefined
      const proposed = s.proposed + drafts.length
      const total = s.accepted + accepted
      ctx.run.metric(ctx.t, 'accepted', accepted)
      ctx.run.metric(ctx.t, 'acceptanceRate', proposed === 0 ? 0 : total / proposed)
      return {
        tokens: [...s.tokens, ...kept],
        generated,
        proposed,
        accepted: total,
        targetCalls: s.targetCalls + 1,
        ...(finished === undefined ? {} : { finished }),
      }
    },
    done: (s) => (s.finished === undefined ? undefined : 'completed'),
  }
}

/** Result of {@link speculativeDecode}. */
export interface SpeculativeResult {
  readonly tokens: readonly number[]
  readonly generated: readonly number[]
  readonly text: string
  /** Accepted / proposed draft tokens. */
  readonly acceptanceRate: number
  /** Target model calls (fewer than generated tokens when drafts are accepted). */
  readonly targetCalls: number
}

/**
 * Speculative decoding (runs {@link speculativeAlgorithm}).
 *
 * @param target - The model whose distribution is sampled.
 * @param draft - The draft model.
 * @param prompt - Token ids or text.
 * @param options - gamma, length, temperature, end token, key, and the run's tracer and policy.
 * @returns The tokens, text, acceptance rate and target calls.
 * @concept decoding
 * @example
 * ```ts
 * import { speculativeDecode, bigramLanguageModel } from 'omai-engine/nn'
 * const lm = bigramLanguageModel([[0, 9], [9, 0]])
 * speculativeDecode(lm, lm, [0], { maxNewTokens: 4, temperature: 0 }).generated // => [1, 0, 1, 0]
 * ```
 */
export function speculativeDecode(
  target: LanguageModel,
  draft: LanguageModel,
  prompt: readonly number[] | string,
  options: SpeculativeOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): SpeculativeResult {
  const { tracer, policy } = options
  const maxNew = options.maxNewTokens ?? 16
  const r = runAlgorithm(speculativeAlgorithm(target, draft, prompt, options), {
    maxSteps: maxNew,
    key: options.key ?? makeKey(0),
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
  })
  const s = r.state
  return {
    tokens: s.tokens,
    generated: s.generated,
    text: target.tokenizer.decode(s.generated),
    acceptanceRate: s.proposed === 0 ? 0 : s.accepted / s.proposed,
    targetCalls: s.targetCalls,
  }
}
