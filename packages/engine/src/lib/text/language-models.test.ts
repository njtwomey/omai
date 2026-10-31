import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  countNgrams,
  goodTuring,
  type NgramSmoothing,
  ngramFromCounts,
  ngramModel,
  ngramTrainingAlgorithm,
} from './language-models.ts'

const corpus = [
  ['a', 'b'],
  ['a', 'c'],
  ['b', 'c'],
]

const smoothers: NgramSmoothing[] = [
  'add-k',
  'witten-bell',
  'absolute-discounting',
  'kneser-ney',
  'modified-kneser-ney',
  'katz',
]

describe('n-gram models', () => {
  it('counts padded n-grams', () => {
    const c = countNgrams(corpus, { order: 2 })
    expect(c.counts[0]?.get('a')).toBe(2)
    expect(c.counts[0]?.get('</s>')).toBe(3)
    expect(c.counts[1]?.get('<s>\u0000a')).toBe(2)
    expect(c.counts[1]?.get('c\u0000</s>')).toBe(2)
  })

  it('matches the hand-worked interpolated Kneser-Ney bigram (D = 0.75)', () => {
    // Bigrams: <s>a 2, <s>b 1, ab 1, ac 1, bc 1, b</s> 1, c</s> 2. Continuation counts: a 1, b 2, c 2, </s> 2 (total 7).
    const m = ngramModel(corpus, { order: 2, smoothing: 'kneser-ney', discount: 0.75, unigram: 'mle' })
    // P(b | a) = (1 - .75) / 2 + .75 * 2 / 2 * (2 / 7)
    expect(m.prob('b', ['a'])).toBeCloseTo(0.25 / 2 + 0.75 * (2 / 7), 12)
    // P(</s> | c) = (2 - .75) / 2 + .75 * 1 / 2 * (2 / 7)
    expect(m.prob('</s>', ['c'])).toBeCloseTo(1.25 / 2 + 0.75 * 0.5 * (2 / 7), 12)
    // A word never seen after a gets only the discounted mass times its continuation probability.
    expect(m.prob('a', ['a'])).toBeCloseTo(0.75 * (1 / 7), 12)
  })

  it('matches the hand-worked Witten-Bell and absolute-discounting bigrams', () => {
    const wb = ngramModel(corpus, { order: 2, smoothing: 'witten-bell', unigram: 'mle' })
    // Unigram MLE over the 9 target tokens: a 2, b 2, c 2, </s> 3. P(b | a) = (1 + 2 * (2 / 9)) / (2 + 2)
    expect(wb.prob('b', ['a'])).toBeCloseTo((1 + 2 * (2 / 9)) / 4, 12)
    const ad = ngramModel(corpus, { order: 2, smoothing: 'absolute-discounting', discount: 0.5, unigram: 'mle' })
    expect(ad.prob('b', ['a'])).toBeCloseTo(0.5 / 2 + ((0.5 * 2) / 2) * (2 / 9), 12)
  })

  it('add-k is the Lidstone estimate', () => {
    const m = ngramModel(corpus, { order: 2, smoothing: 'add-k', k: 0.5 })
    // The vocabulary is a, b, c, </s> and <unk>: five words.
    expect(m.prob('b', ['a'])).toBeCloseTo((1 + 0.5) / (2 + 0.5 * 5), 12)
    expect(m.prob('a', ['zzz'])).toBeCloseTo(0.5 / (0.5 * 5), 12)
  })

  for (const smoothing of smoothers) {
    for (const order of [1, 2, 3]) {
      it(`${smoothing} of order ${order} is a distribution`, () => {
        const m = ngramModel(
          [
            ['the', 'cat', 'sat'],
            ['the', 'dog', 'sat'],
            ['a', 'cat', 'ran'],
            ['the', 'cat', 'ran', 'away'],
          ],
          { order, smoothing },
        )
        for (const ctx of [[], ['the'], ['the', 'cat'], ['cat'], ['zzz', 'qqq'], ['ran', 'away']]) {
          const p = m.distribution(ctx)
          const total = p.reduce((a, b) => a + b, 0)
          expect(total).toBeCloseTo(1, 9)
          for (const x of p) expect(x).toBeGreaterThanOrEqual(0)
        }
      })
    }
  }

  it('maximum likelihood gives no mass to the unseen, so a smoother wins on held-out text', () => {
    const train = [
      ['a', 'b', 'c'],
      ['a', 'b', 'd'],
      ['a', 'c', 'd'],
    ]
    const test = [['a', 'd', 'c']]
    expect(ngramModel(train, { order: 2, smoothing: 'mle' }).perplexity(test)).toBe(Number.POSITIVE_INFINITY)
    expect(ngramModel(train, { order: 2, smoothing: 'kneser-ney' }).perplexity(test)).toBeLessThan(
      ngramModel(train, { order: 2, smoothing: 'add-k', k: 1 }).perplexity(test),
    )
  })

  it('rare words become the unknown token', () => {
    const m = ngramModel(
      [
        ['a', 'a', 'b'],
        ['a', 'c'],
      ],
      { order: 2, smoothing: 'add-k', minCount: 2 },
    )
    expect(m.vocabulary).toEqual(['</s>', '<unk>', 'a'])
    expect(m.prob('b', ['a'])).toBe(m.prob('<unk>', ['a']))
    expect(m.prob('zzz', ['a'])).toBe(m.prob('<unk>', ['a']))
  })

  it('Katz backoff reserves the discounted mass for unseen continuations', () => {
    const m = ngramModel(
      [
        ['a', 'b'],
        ['a', 'b'],
        ['a', 'c'],
        ['b', 'c'],
        ['b', 'c'],
        ['c', 'a'],
        ['a', 'd'],
      ],
      { order: 2, smoothing: 'katz' },
    )
    expect(m.unseenMass(['a'])).toBeGreaterThan(0)
    expect(m.unseenMass(['a'])).toBeLessThan(1)
  })

  it('Good-Turing adjusts counts and gives the unseen the mass of the singletons', () => {
    const g = goodTuring([1, 1, 1, 2, 2, 3])
    expect(g.adjusted.get(1)).toBeCloseTo((2 * 2) / 3, 12)
    expect(g.adjusted.get(2)).toBeCloseTo(1.5, 12)
    expect(g.adjusted.get(3)).toBe(3)
    expect(g.unseenMass).toBeCloseTo(3 / 10, 12)
  })

  it('samples sentences from the model', () => {
    const m = ngramModel(
      [
        ['x', 'y'],
        ['x', 'y'],
        ['x', 'y'],
      ],
      { order: 2, smoothing: 'mle' },
    )
    expect(m.sample(key(0))).toEqual(['x', 'y'])
  })

  it('counts one sentence per step and the model so far scores held-out text', () => {
    const rec = recorder()
    const alg = ngramTrainingAlgorithm(corpus, { order: 2 }, [['a', 'b']])
    const r = runAlgorithm(alg, { seed: 0, maxSteps: 10, tracer: rec })
    expect(r.steps).toBe(3)
    expect(r.state.counts.sentences).toBe(3)
    const direct = ngramFromCounts(countNgrams(corpus, { order: 2 }), { order: 2 }).perplexity([['a', 'b']])
    expect(ngramFromCounts(r.state.counts, { order: 2 }).perplexity([['a', 'b']])).toBeCloseTo(direct, 12)
    const back = alg.deserialise?.(alg.serialise?.(r.state) as never)
    expect(back?.counts.counts[1]?.get('<s>\u0000a')).toBe(2)
  })
})
