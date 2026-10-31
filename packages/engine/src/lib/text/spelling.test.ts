import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { typoPairs } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { ngramModel } from './language-models.ts'
import { learnChannel, spellingCorrector, uniformChannel } from './spelling.ts'

describe('the channel', () => {
  it('learns which edits are common', () => {
    // Every error in the training pairs swaps e and i.
    const c = learnChannel([
      ['recieve', 'receive'],
      ['beleive', 'believe'],
      ['wierd', 'weird'],
      ['friend', 'friend'],
      ['field', 'field'],
    ])
    expect(c.logProbability('recieve', 'receive')).toBeGreaterThan(c.logProbability('recxive', 'receive'))
    expect(c.rates.transposition).toBeGreaterThan(c.rates.substitution)
    expect(c.rates.match).toBeGreaterThan(0.5)
  })

  it('charges the identity less than any edit, and a uniform channel by the number of edits', () => {
    const u = uniformChannel(0.01)
    expect(u.logProbability('cat', 'cat')).toBeCloseTo(0, 12)
    expect(u.logProbability('cat', 'cut')).toBeCloseTo(Math.log(0.01), 12)
    expect(u.logProbability('ct', 'cut')).toBeCloseTo(Math.log(0.01), 12)
    expect(u.logProbability('cta', 'cat')).toBeCloseTo(Math.log(0.01), 12)
    expect(u.logProbability('xyz', 'cat')).toBeCloseTo(3 * Math.log(0.01), 12)
  })
})

describe('the corrector', () => {
  const counts = new Map([
    ['the', 100],
    ['then', 8],
    ['they', 6],
    ['cat', 20],
    ['cart', 3],
    ['sat', 12],
  ])

  it('picks the likelier intended word among those within range', () => {
    const c = spellingCorrector(counts, { maxDistance: 2 })
    expect(c.correct('teh')).toBe('the')
    expect(c.correct('cta')).toBe('cat')
    expect(c.correct('xylophone')).toBe('xylophone')
    const cands = c.candidates('thn')
    expect(cands[0]?.word).toBe('the')
    expect(cands.every((x) => x.distance <= 2)).toBe(true)
    for (let i = 1; i < cands.length; i++) expect(cands[i]?.score).toBeLessThanOrEqual(cands[i - 1]?.score ?? 0)
  })

  it('lets the language model decide between equally close words by context', () => {
    const lm = ngramModel(
      [
        ['the', 'cat', 'sat'],
        ['the', 'cat', 'sat'],
        ['a', 'cart', 'sat'],
        ['the', 'cat', 'sat'],
        ['the', 'cart', 'rolled'],
      ],
      { order: 2, smoothing: 'kneser-ney' },
    )
    const c = spellingCorrector(
      new Map([
        ['cat', 5],
        ['cart', 5],
        ['sat', 5],
        ['rolled', 1],
        ['the', 5],
        ['a', 1],
      ]),
      {
        languageModel: lm,
        maxDistance: 1,
      },
    )
    // "cxt" is one edit from cat only; "cat"/"cart" both one edit from "car": the context decides.
    expect(c.correctSentence(['the', 'cxt', 'sat'])).toEqual(['the', 'cat', 'sat'])
    expect(c.candidates('car', ['the'])[0]?.word).toBe('cat')
    expect(c.candidates('car', ['a'])[0]?.word).toBe('cart')
  })
})

describe('on typed words with planted intentions', () => {
  const { data } = generate(typoPairs, { key: key(4), knobs: { n: 3000, vocabulary: 150, errorRate: 0.08 } })
  const typed = data.source as string[]
  const intended = data.target as string[]
  const train = 2000
  const trainPairs = typed.slice(0, train).map((t, i) => [t, intended[i] as string] as [string, string])
  const counts = new Map<string, number>()
  for (const w of intended.slice(0, train)) counts.set(w, (counts.get(w) ?? 0) + 1)
  const accuracy = (f: (w: string) => string): number =>
    typed.slice(train).filter((t, i) => f(t) === intended[train + i]).length / (typed.length - train)

  it('corrects far more words than leaving them alone, and a learned channel is no worse than a uniform one', () => {
    const baseline = accuracy((w) => w)
    const uniform = spellingCorrector(counts, { maxDistance: 2 })
    const learned = spellingCorrector(counts, { maxDistance: 2, channel: learnChannel(trainPairs) })
    const accUniform = accuracy((w) => uniform.correct(w))
    const accLearned = accuracy((w) => learned.correct(w))
    expect(baseline).toBeLessThan(0.7)
    expect(accUniform).toBeGreaterThan(baseline + 0.15)
    expect(accLearned).toBeGreaterThan(0.85)
    expect(accLearned).toBeGreaterThanOrEqual(accUniform - 0.01)
  })

  it('finds more candidates as the edit distance grows', () => {
    const c1 = spellingCorrector(counts, { maxDistance: 1 })
    const c2 = spellingCorrector(counts, { maxDistance: 3 })
    const sample = typed.slice(train, train + 50)
    const n = (c: typeof c1): number => sample.reduce((a, w) => a + c.candidates(w).length, 0)
    expect(n(c2)).toBeGreaterThan(n(c1))
  })
})
