import { describe, expect, it } from 'vitest'
import { runAlgorithm } from '../trace/index.ts'
import { bm25, bm25Scores } from './features.ts'
import {
  booleanSearch,
  documentFrequency,
  indexCorpus,
  indexingAlgorithm,
  phraseSearch,
  type RetrievalModel,
  scoreDocuments,
  search,
} from './retrieval.ts'

const docs = [
  'the cat sat on the mat',
  'the dog sat on the log',
  'a cat and a dog played',
  'new york is a big city',
  'the big dog chased a cat',
  'york new york city lights',
].map((d) => d.split(' '))

describe('inverted index', () => {
  it('keeps positions and document frequencies', () => {
    const ix = indexCorpus(docs)
    expect(ix.documents).toBe(6)
    expect(documentFrequency(ix, 'cat')).toBe(3)
    expect(ix.postings.get('the')?.[0]?.positions).toEqual([0, 4])
    expect(ix.tokens).toBe(docs.reduce((a, d) => a + d.length, 0))
  })

  it('grows one document per step and serialises', () => {
    const alg = indexingAlgorithm(docs)
    const r = runAlgorithm(alg, { seed: 0, maxSteps: 100 })
    expect(r.steps).toBe(6)
    expect(r.state.index.postings.size).toBe(indexCorpus(docs).postings.size)
    const back = alg.deserialise?.(alg.serialise?.(r.state) as never)
    expect(back?.index.documents).toBe(6)
  })
})

describe('scoring models', () => {
  it('BM25 over the postings equals rank_bm25 scores when no idf is negative', () => {
    const ix = indexCorpus(docs)
    const reference = bm25Scores(bm25(docs, { k1: 1.2, b: 0.75 }), ['york', 'city'])
    const ours = scoreDocuments(ix, ['york', 'city'], { kind: 'bm25', k1: 1.2, b: 0.75 }).map((s) => s.score)
    for (const [i, v] of reference.entries()) expect(ours[i]).toBeCloseTo(v, 12)
  })

  it('contributions sum to the score and BM25 saturates with term frequency', () => {
    const ix = indexCorpus([
      ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
      ['a', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
      ['y', 'z'],
    ])
    const s = scoreDocuments(ix, ['a', 'a'], { kind: 'bm25', k1: 1.2, b: 0 })
    expect(Object.values(s[0]?.contributions ?? {}).reduce((a, b) => a + b, 0)).toBeCloseTo(s[0]?.score ?? 0, 12)
    const one = scoreDocuments(ix, ['a'], { kind: 'bm25', k1: 1.2, b: 0 })
    // 8 occurrences is at most (k1 + 1) / 1 = 2.2 times one occurrence, not 8 times.
    expect((one[0]?.score ?? 0) / (one[1]?.score ?? 1)).toBeLessThan(2.2)
    expect(s[0]?.score).toBeCloseTo(2 * (one[0]?.score ?? 0), 12)
  })

  it('TF-IDF cosine is 1 for a document scored against itself and bounded by 1', () => {
    const ix = indexCorpus(docs)
    const q = docs[3] as string[]
    const s = scoreDocuments(ix, q, { kind: 'tfidf' })
    expect(s[3]?.score).toBeCloseTo(1, 10)
    for (const x of s) expect(x.score).toBeLessThanOrEqual(1 + 1e-12)
    expect(search(ix, q, { kind: 'tfidf' }, 1)[0]?.doc).toBe(3)
  })

  it('binary independence uses the RSJ weight, and relevance information sharpens it', () => {
    const ix = indexCorpus(docs)
    const base = scoreDocuments(ix, ['cat'], { kind: 'bim' })
    expect(base[0]?.score).toBeCloseTo(Math.log((6 - 3 + 0.5) / (3 + 0.5)), 12)
    const withRel = scoreDocuments(ix, ['cat'], { kind: 'bim', relevant: [0, 2] })
    // r = 2, R = 2, n = 3, N = 6
    expect(withRel[0]?.score).toBeCloseTo(Math.log((2.5 * (6 - 3 - 2 + 2 + 0.5)) / ((1 + 0.5) * 0.5)), 12)
  })

  const smoothers: RetrievalModel[] = [
    { kind: 'query-likelihood', smoothing: 'dirichlet', mu: 5 },
    { kind: 'query-likelihood', smoothing: 'jelinek-mercer', lambda: 0.3 },
    { kind: 'query-likelihood', smoothing: 'absolute', delta: 0.6 },
  ]
  for (const model of smoothers) {
    it(`query likelihood with ${model.kind === 'query-likelihood' ? model.smoothing : ''} smoothing is a distribution over terms`, () => {
      const ix = indexCorpus(docs)
      for (let d = 0; d < docs.length; d++) {
        let total = 0
        for (const t of ix.postings.keys()) total += Math.exp(scoreDocuments(ix, [t], model)[d]?.score ?? 0)
        expect(total).toBeCloseTo(1, 10)
      }
    })
  }

  it('Dirichlet smoothing equals the hand formula and ranks the better match first', () => {
    const ix = indexCorpus(docs)
    const mu = 4
    const pc = 3 / ix.tokens // 'cat' occurs 3 times in the collection
    const s = scoreDocuments(ix, ['cat'], { kind: 'query-likelihood', smoothing: 'dirichlet', mu })
    expect(s[0]?.score).toBeCloseTo(Math.log((1 + mu * pc) / (6 + mu)), 12)
    expect(s[3]?.score).toBeCloseTo(Math.log((0 + mu * pc) / (6 + mu)), 12)
    expect(search(ix, ['york'], { kind: 'query-likelihood', mu })[0]?.doc).toBe(5)
  })
})

describe('structured queries', () => {
  const ix = indexCorpus(docs)
  it('phrase queries need adjacency, with optional slop', () => {
    expect(phraseSearch(ix, ['new', 'york'])).toEqual([3, 5])
    expect(phraseSearch(ix, ['york', 'new', 'york'])).toEqual([5])
    expect(phraseSearch(ix, ['york', 'city'])).toEqual([5])
    expect(phraseSearch(ix, ['new', 'city'], 1)).toEqual([5])
    expect(phraseSearch(ix, ['new', 'city'], 4)).toEqual([3, 5])
    expect(phraseSearch(ix, ['nope'])).toEqual([])
  })
  it('boolean queries combine postings', () => {
    expect(booleanSearch(ix, { all: ['cat'], none: ['dog'] })).toEqual([0])
    expect(booleanSearch(ix, { any: ['mat', 'log'] })).toEqual([0, 1])
    expect(booleanSearch(ix, { all: ['the'], any: ['cat', 'dog'], none: ['mat'] })).toEqual([1, 4])
  })
})
