import { type Json, maxOf } from '../../base/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import type { LanguageModel } from '../language-model.ts'
import { chain, eosOf, promptIds, scoreNext } from './_shared.ts'
import type { TokenConstraint } from './constraints.ts'
import { type LogitProcessor, logProbs } from './processors.ts'

const NEG = Number.NEGATIVE_INFINITY

/** Options of {@link beamSearch}. */
export interface BeamSearchOptions {
  /** Hypotheses kept per step (default 4). */
  readonly beamWidth?: number
  /** How many finished hypotheses to return (default 1). */
  readonly numReturn?: number
  readonly maxNewTokens?: number
  /** GNMT's length penalty exponent alpha: scores are divided by ((5 + length) / 6)^alpha (default 0: none). */
  readonly lengthPenalty?: number
  readonly eosId?: number | null
  readonly processors?: readonly LogitProcessor[]
  readonly constraint?: TokenConstraint
}

/** A hypothesis of {@link beamSearch}. */
export interface Beam {
  readonly tokens: readonly number[]
  readonly generated: readonly number[]
  /** The sum of the log-probabilities of the generated tokens. */
  readonly logprob: number
  /** The length-normalised score beams are ranked by. */
  readonly score: number
  readonly finished: boolean
}

interface BeamState {
  readonly live: readonly Beam[]
  readonly done: readonly Beam[]
  readonly steps: number
}

const gnmt = (logprob: number, length: number, alpha: number): number =>
  alpha === 0 ? logprob : logprob / ((5 + length) / 6) ** alpha

/**
 * Beam search as a traced {@link Algorithm}: keep the beamWidth most likely partial sequences, extend each by every
 * token, retire hypotheses that emit the end-of-sequence token, and rank by GNMT's length-normalised score. Each step
 * emits a `beams` message with the live hypotheses.
 *
 * @param lm - The language model.
 * @param prompt - Token ids or text.
 * @param options - Width, number returned, length, length penalty, end token, processors and constraint.
 * @returns The algorithm.
 * @concept decoding
 * @reference Wu et al. (2016), Google's neural machine translation system.
 * @example
 * ```ts
 * import { beamSearchAlgorithm, bigramLanguageModel } from 'whitebox-engine/nn'
 * beamSearchAlgorithm(bigramLanguageModel([[0, 1], [1, 0]]), [0]).name // => 'beamSearch'
 * ```
 */
export function beamSearchAlgorithm(
  lm: LanguageModel,
  prompt: readonly number[] | string,
  options: BeamSearchOptions = {},
): Algorithm<BeamState> {
  const start = promptIds(lm, prompt)
  const width = options.beamWidth ?? 4
  const maxNew = options.maxNewTokens ?? 16
  const alpha = options.lengthPenalty ?? 0
  const eos = eosOf(lm, options)
  const { pre } = chain(
    {
      ...(options.processors === undefined ? {} : { processors: options.processors }),
      ...(options.constraint === undefined ? {} : { constraint: options.constraint }),
    },
    start.length,
  )
  return {
    name: 'beamSearch',
    config: { beamWidth: width, maxNewTokens: maxNew, lengthPenalty: alpha } as Json,
    schema: channels({
      metrics: { bestScore: { description: 'score of the best live hypothesis' } },
      messages: { beams: { description: 'the live hypotheses and their scores' } },
    }),
    init: () => ({
      live: [{ tokens: start, generated: [], logprob: 0, score: 0, finished: false }],
      done: [],
      steps: 0,
    }),
    step: (s, ctx) => {
      const candidates: { beam: Beam; token: number; logprob: number }[] = []
      for (const beam of s.live) {
        const logits = pre(scoreNext(lm, beam.tokens, undefined, 0).logits, beam.tokens)
        const lp = logProbs(logits)
        for (const [t, v] of lp.entries()) if (v > NEG) candidates.push({ beam, token: t, logprob: beam.logprob + v })
      }
      candidates.sort((a, b) => b.logprob - a.logprob)
      const live: Beam[] = []
      const done = [...s.done]
      for (const [rank, c] of candidates.entries()) {
        if (live.length >= width) break
        const generated = [...c.beam.generated, c.token]
        const beam: Beam = {
          tokens: [...c.beam.tokens, c.token],
          generated,
          logprob: c.logprob,
          score: gnmt(c.logprob, generated.length, alpha),
          finished: c.token === eos,
        }
        if (beam.finished) {
          if (rank < width) done.push(beam)
        } else live.push(beam)
      }
      ctx.run.message(ctx.t, 'beams', live.map((b) => ({ tokens: [...b.generated], score: b.score })) as Json)
      return { live, done, steps: s.steps + 1 }
    },
    done: (s) => (s.live.length === 0 || s.steps >= maxNew ? 'completed' : undefined),
    observe: (s) => ({ metrics: s.live.length === 0 ? {} : { bestScore: maxOf(s.live.map((b) => b.score)) } }),
  }
}

/**
 * Beam search (runs {@link beamSearchAlgorithm}): the numReturn best hypotheses, finished or not, by length-normalised
 * score.
 *
 * @param lm - The language model.
 * @param prompt - Token ids or text.
 * @param options - Width, number returned, length, length penalty, end token, processors and constraint.
 * @returns The best hypotheses, best first.
 * @concept decoding
 * @example
 * ```ts
 * import { beamSearch, bigramLanguageModel } from 'whitebox-engine/nn'
 * beamSearch(bigramLanguageModel([[0, 5], [5, 0]]), [0], { maxNewTokens: 3, beamWidth: 2 })[0]?.generated // => [1, 0, 1]
 * ```
 */
export function beamSearch(
  lm: LanguageModel,
  prompt: readonly number[] | string,
  options: BeamSearchOptions & Partial<Omit<RunOptions, 'maxSteps'>> = {},
): Beam[] {
  const { tracer, policy } = options
  const r = runAlgorithm(beamSearchAlgorithm(lm, prompt, options), {
    maxSteps: options.maxNewTokens ?? 16,
    ...(tracer === undefined ? {} : { tracer }),
    ...(policy === undefined ? {} : { policy }),
  })
  return [...r.state.done, ...r.state.live].sort((a, b) => b.score - a.score).slice(0, options.numReturn ?? 1)
}
