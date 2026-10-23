import { describe, expect, it } from 'vitest'
import { key, normal } from '../../random/index.ts'
import { toArray } from '../../tensor/index.ts'
import { messages, metricSeries, recorder, tracePolicy } from '../../trace/index.ts'
import { bigramLanguageModel, characterTokenizer, sequenceLogProb } from '../language-model.ts'
import { beamSearch } from './beam.ts'
import { automatonConstraint, maskProcessor, trieConstraint } from './constraints.ts'
import { generate } from './generate.ts'
import {
  entropyOfLogits,
  frequencyPresenceProcessor,
  logProbs,
  minPProcessor,
  repetitionPenaltyProcessor,
  topKProcessor,
  topPProcessor,
} from './processors.ts'
import { speculativeDecode } from './speculative.ts'

const L = Math.log
// a -> b -> c -> a, clear margins; token 3 is EOS.
const cycle = bigramLanguageModel(
  [
    [0, 3, 1, -2],
    [1, 0, 3, -2],
    [3, 1, 0, -2],
    [0, 0, 0, 0],
  ],
  characterTokenizer('abc', { eos: true }),
)
const randomLm = bigramLanguageModel(toArray(normal(key(11), [5, 5])) as number[][])

describe('logit processors', () => {
  it('top-p keeps the smallest set reaching p, min-p a relative threshold, top-k the k largest', () => {
    const z = [L(0.2), L(0.5), L(0.3)]
    const kept = (y: number[]): number[] => y.flatMap((v, i) => (v > Number.NEGATIVE_INFINITY ? [i] : []))
    expect(kept(topPProcessor(0.5)(z, []))).toEqual([1])
    expect(kept(topPProcessor(0.6)(z, []))).toEqual([1, 2])
    expect(kept(topPProcessor(0.81)(z, []))).toEqual([0, 1, 2])
    expect(kept(minPProcessor(0.5)(z, []))).toEqual([1, 2])
    expect(kept(topKProcessor(1)(z, []))).toEqual([1])
    expect(kept(topKProcessor(9)(z, []))).toEqual([0, 1, 2])
  })
  it('penalties and masks', () => {
    expect(repetitionPenaltyProcessor(2)([2, -2, 1], [0, 1, 1])).toEqual([1, -4, 1])
    expect(frequencyPresenceProcessor({ frequency: 1 })([0, 0], [1, 1])).toEqual([0, -2])
    expect(maskProcessor(() => [false, true], 1)([3, 4], [0])).toEqual([Number.NEGATIVE_INFINITY, 4])
    expect(logProbs([Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY])).toEqual([
      Number.NEGATIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ])
    expect(entropyOfLogits([0, Number.NEGATIVE_INFINITY])).toBe(0)
  })
})

describe('generation', () => {
  it('greedy decoding follows the argmax chain and stops at EOS', () => {
    const r = generate(cycle, 'a', { maxNewTokens: 5 })
    expect(r.text).toBe('bcabc')
    expect(r.finished).toBe('length')
    expect(r.logprob).toBeCloseTo(sequenceLogProb(cycle, r.tokens), 12)
    const toEos = bigramLanguageModel(
      [
        [0, 0, 5],
        [0, 0, 0],
        [0, 0, 0],
      ],
      characterTokenizer('ab', { eos: true }),
    )
    const e = generate(toEos, [0], { maxNewTokens: 5 })
    expect(e.generated).toEqual([2])
    expect(e.finished).toBe('eos')
    expect(generate(toEos, [0], { maxNewTokens: 3, eosId: null }).generated.length).toBe(3)
  })

  it('near-zero temperature and top-k 1 sample the greedy path', () => {
    const greedy = generate(randomLm, [0], { maxNewTokens: 6 }).generated
    expect(generate(randomLm, [0], { maxNewTokens: 6, sample: true, temperature: 0 }).generated).toEqual(greedy)
    expect(generate(cycle, [0], { maxNewTokens: 6, sample: true, temperature: 0.01, key: key(1) }).generated).toEqual(
      generate(cycle, [0], { maxNewTokens: 6 }).generated,
    )
    expect(generate(randomLm, [0], { maxNewTokens: 6, sample: true, topK: 1, key: key(2) }).generated).toEqual(greedy)
    // Sampling is reproducible for a key and differs across keys.
    const s = (k: number): readonly number[] =>
      generate(randomLm, [0], { maxNewTokens: 8, sample: true, key: key(k) }).generated
    expect(s(3)).toEqual(s(3))
    expect([4, 5, 6, 7].map((k) => s(k).join(',')).some((x) => x !== s(3).join(','))).toBe(true)
  })

  it('a repetition penalty changes the choice', () => {
    const lm = bigramLanguageModel([
      [0, 2, 1.5],
      [0, 0, 0],
      [0, 0, 0],
    ])
    expect(generate(lm, [1, 0], { maxNewTokens: 1 }).generated).toEqual([1])
    expect(generate(lm, [1, 0], { maxNewTokens: 1, repetitionPenalty: 2 }).generated).toEqual([2])
    expect(generate(lm, [1, 0], { maxNewTokens: 1, presencePenalty: 1 }).generated).toEqual([2])
  })

  it('records one token per step', () => {
    const rec = recorder()
    const r = generate(cycle, 'a', { maxNewTokens: 4, tracer: rec, policy: tracePolicy({ messages: true }) })
    expect(messages(rec.run(), 'token').map((m) => (m.payload as { text: string }).text)).toEqual([...r.text])
    expect(metricSeries(rec.run(), 'entropy').values.length).toBe(4)
  })
})

describe('constrained decoding', () => {
  it('never leaves a trie of allowed sequences', () => {
    const tok = characterTokenizer('abcd', { eos: true })
    const lm = bigramLanguageModel(toArray(normal(key(5), [5, 5])) as number[][], tok)
    const allowed = [[0, 1, 2], [2, 2], [3]]
    const constraint = trieConstraint(allowed, 5, 4)
    for (let k = 0; k < 20; k++) {
      const r = generate(lm, [1], { maxNewTokens: 6, sample: true, key: key(k), constraint })
      expect(r.finished).toBe('eos')
      expect(allowed.map((a) => a.join(','))).toContain(r.generated.slice(0, -1).join(','))
    }
  })
  it('follows an automaton and stops when nothing is allowed', () => {
    const alternate = automatonConstraint(
      { start: 0, next: (s: number, t: number) => (t === s ? 1 - s : undefined) },
      3,
    )
    const r = generate(randomLm, [4], { maxNewTokens: 5, constraint: alternate })
    expect(r.generated).toEqual([0, 1, 0, 1, 0])
    const dead = generate(randomLm, [4], { maxNewTokens: 5, constraint: () => new Set<number>() })
    expect(dead.finished).toBe('constraint')
    expect(dead.generated).toEqual([])
  })
})

describe('beam search', () => {
  it('width 1 is greedy', () => {
    const greedy = generate(randomLm, [2], { maxNewTokens: 5, eosId: null }).generated
    expect(beamSearch(randomLm, [2], { beamWidth: 1, maxNewTokens: 5, eosId: null })[0]?.generated).toEqual(greedy)
  })
  it('finds a sequence more likely than greedy', () => {
    const tiny = 1e-9
    const lm = bigramLanguageModel([
      [L(tiny), L(0.55), L(0.45)],
      [0, 0, 0],
      [L(tiny), L(tiny), 0],
    ])
    const greedy = generate(lm, [0], { maxNewTokens: 2 })
    const best = beamSearch(lm, [0], { beamWidth: 2, maxNewTokens: 2, numReturn: 2 })
    expect(greedy.generated[0]).toBe(1)
    expect(best[0]?.generated).toEqual([2, 2])
    expect(best[0]?.logprob ?? Number.NEGATIVE_INFINITY).toBeGreaterThan(greedy.logprob)
    expect(best.length).toBe(2)
  })
  it('retires hypotheses at EOS and normalises by length', () => {
    const r = beamSearch(cycle, [0], { beamWidth: 3, maxNewTokens: 4, numReturn: 3, lengthPenalty: 0.6 })
    expect(r.length).toBe(3)
    for (const b of r) if (b.finished) expect(b.generated.at(-1)).toBe(3)
    expect(r[0]?.score ?? 0).toBeGreaterThanOrEqual(r[1]?.score ?? 0)
  })
})

describe('speculative decoding', () => {
  it('a draft equal to the target is always accepted; greedy speculative decoding is greedy', () => {
    const r = speculativeDecode(randomLm, randomLm, [0], { maxNewTokens: 8, gamma: 3, key: key(1) })
    expect(r.acceptanceRate).toBe(1)
    expect(r.targetCalls).toBeLessThan(8)
    const other = bigramLanguageModel(toArray(normal(key(12), [5, 5])) as number[][])
    expect(speculativeDecode(randomLm, other, [0], { maxNewTokens: 6, temperature: 0 }).generated).toEqual(
      generate(randomLm, [0], { maxNewTokens: 6 }).generated,
    )
  })

  it("samples the target's distribution whatever the draft", () => {
    const target = bigramLanguageModel([
      [L(0.6), L(0.3), L(0.1)],
      [L(0.2), L(0.2), L(0.6)],
      [L(0.5), L(0.25), L(0.25)],
    ])
    const draft = bigramLanguageModel([
      [L(0.1), L(0.2), L(0.7)],
      [L(0.6), L(0.2), L(0.2)],
      [L(0.3), L(0.3), L(0.4)],
    ])
    const n = 1500
    const first = [0, 0, 0]
    const second = [0, 0, 0]
    for (let i = 0; i < n; i++) {
      const g = speculativeDecode(target, draft, [0], { maxNewTokens: 2, gamma: 2, key: key(1000 + i) }).generated
      first[g[0] ?? 0] = (first[g[0] ?? 0] ?? 0) + 1 / n
      second[g[1] ?? 0] = (second[g[1] ?? 0] ?? 0) + 1 / n
    }
    const p1 = [0.6, 0.3, 0.1]
    const rows = [
      [0.6, 0.3, 0.1],
      [0.2, 0.2, 0.6],
      [0.5, 0.25, 0.25],
    ]
    const p2 = [0, 1, 2].map((j) => p1.reduce((s, p, i) => s + p * (rows[i]?.[j] ?? 0), 0))
    for (let j = 0; j < 3; j++) {
      expect(Math.abs((first[j] ?? 0) - (p1[j] ?? 0))).toBeLessThan(0.04)
      expect(Math.abs((second[j] ?? 0) - (p2[j] ?? 0))).toBeLessThan(0.04)
    }
  })
})
