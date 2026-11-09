import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { topicCorpus } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { topicCoherence } from '../text/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { latentDirichletAllocation, ldaAlgorithm, ldaBound } from './lda.ts'

type Corpus = { x: number[][]; topics: number[][]; vocabulary: string[]; tokens: number[][] }

const planted = (knobs: Record<string, number | string>, seed: number): Corpus => {
  const { data, truth } = generate(topicCorpus, { key: key(seed), knobs })
  const vocabulary = [...(data.vocabulary ?? [])]
  const tokens = (data.tokens ?? []) as number[][]
  const x = tokens.map((doc) => {
    const row = new Array<number>(vocabulary.length).fill(0)
    for (const w of doc) row[w] = (row[w] ?? 0) + 1
    return row
  })
  return { x, topics: truth.latents?.['topics'] as number[][], vocabulary, tokens }
}

const cosine = (a: readonly number[], b: readonly number[]): number => {
  let ab = 0
  let aa = 0
  let bb = 0
  for (const [i, v] of a.entries()) {
    ab += v * (b[i] ?? 0)
    aa += v * v
    bb += (b[i] ?? 0) ** 2
  }
  return ab / Math.sqrt(aa * bb)
}

/** The mean cosine of each planted topic with its best-matching learnt topic (each learnt topic used once, greedily). */
const recovery = (learnt: number[][], truth: number[][]): number => {
  const used = new Set<number>()
  let total = 0
  const pairs = truth.flatMap((t, i) => learnt.map((l, j) => ({ i, j, c: cosine(t, l) }))).sort((a, b) => b.c - a.c)
  const done = new Set<number>()
  for (const { i, j, c } of pairs) {
    if (used.has(j) || done.has(i)) continue
    used.add(j)
    done.add(i)
    total += c
  }
  return total / truth.length
}

const normalise = (rows: number[][]): number[][] => rows.map((r) => r.map((v) => v / r.reduce((a, b) => a + b, 0)))

const topWords = (components: number[][], vocabulary: string[], n: number): string[][] =>
  components.map((row) =>
    row
      .map((v, w) => ({ v, w }))
      .sort((a, b) => b.v - a.v)
      .slice(0, n)
      .map(({ w }) => vocabulary[w] ?? ''),
  )

describe('latent Dirichlet allocation', () => {
  const knobs = { topics: 'dirichlet', nTopics: 4, vocabulary: 40, eta: 0.05, alpha: 0.3, nDocs: 150, docLength: 60 }
  const bars = planted(knobs, 1)
  const hyper = { nComponents: 4, topicWordPrior: 0.05, docTopicPrior: 0.3 }

  it('batch VB recovers the planted bars topics up to permutation, and its bound rises', () => {
    const f = latentDirichletAllocation.with({ ...hyper, maxIter: 30 }).fit({ x: bars.x }, { key: key(2), logs: 'all' })
    expect(recovery(normalise(f.params.components), bars.topics)).toBeGreaterThan(0.97)
    const bound = metricSeries(f.run, 'solver/bound').values
    expect(bound.length).toBe(31)
    for (let i = 1; i < bound.length; i++)
      expect(bound[i] ?? 0).toBeGreaterThan((bound[i - 1] ?? 0) - 1e-3 * Math.abs(bound[i - 1] ?? 0))
    expect(bound.at(-1) ?? 0).toBeGreaterThan(bound[0] ?? 0)
    const perplexity = metricSeries(f.run, 'solver/perplexity').values
    expect(perplexity.at(-1) ?? 0).toBeLessThan((perplexity[0] ?? 0) / 2)
    expect(f.params.nIter).toBe(30)
  })

  it("the learnt topics are more coherent than the first step's", () => {
    const texts = bars.tokens.map((doc) => doc.map((w) => bars.vocabulary[w] ?? ''))
    const coherence = (iters: number): number => {
      const f = latentDirichletAllocation.with({ ...hyper, maxIter: iters }).fit({ x: bars.x }, { key: key(2) })
      return topicCoherence(topWords(f.params.components, bars.vocabulary, 5), texts).mean
    }
    expect(coherence(30)).toBeGreaterThan(coherence(1))
  })

  it('online VB over mini-batches also finds the topics, with the weight decaying', () => {
    const f = latentDirichletAllocation
      .with({ ...hyper, learningMethod: 'online', batchSize: 50, maxIter: 30, learningOffset: 1, learningDecay: 0.6 })
      .fit({ x: bars.x }, { key: key(2), logs: 'all' })
    expect(f.params.nBatchIterations).toBe(90)
    expect(f.params.nIter).toBe(30)
    expect(recovery(normalise(f.params.components), bars.topics)).toBeGreaterThan(0.9)
    const p = metricSeries(f.run, 'solver/perplexity').values
    expect(p.at(-1) ?? 0).toBeLessThan((p[0] ?? 0) / 2)
  })

  it('transform gives proportions; held-out documents from the same topics score like the training ones', () => {
    const all = planted({ ...knobs, nDocs: 190 }, 1).x
    const train = all.slice(0, 150)
    const held = all.slice(150)
    const f = latentDirichletAllocation.with({ ...hyper, maxIter: 20 }).fit({ x: train }, { key: key(2) })
    const theta = toArray(f.transform(train)) as number[][]
    expect(theta.length).toBe(150)
    for (const row of theta) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const gamma = toArray(f.transform(train, { normalize: false })) as number[][]
    expect(gamma[0]?.reduce((a, b) => a + b, 0) ?? 0).toBeGreaterThan(40)
    expect(f.perplexity(held)).toBeLessThan(f.params.perplexity * 1.15)
    // The bound of the training corpus is the score, and is the perplexity's numerator.
    const words = train.flat().reduce((a, b) => a + b, 0)
    expect(Math.exp(-f.score({ x: train }) / words)).toBeCloseTo(f.params.perplexity, 8)
    expect(f.perplexity(held, true)).toBeGreaterThan(0)
    // Fitted topics explain held-out documents far better than flat topics.
    const flat = train[0]?.map(() => 1) ?? []
    const uniform = ldaBound(
      held,
      held.map(() => new Array<number>(4).fill(1)),
      new Array(4).fill(flat),
      { alpha: 0.1, eta: 0.1 },
    )
    expect(f.score({ x: held })).toBeGreaterThan(uniform)
  })

  it('is deterministic for a key, and with a given start and a constant document start', () => {
    const run = (k: number) =>
      latentDirichletAllocation.with({ ...hyper, maxIter: 3 }).fit({ x: bars.x }, { key: key(k) }).params.components
    expect(run(4)).toEqual(run(4))
    expect(run(4)).not.toEqual(run(5))
    const start = Array.from({ length: 4 }, (_, k) =>
      Array.from({ length: 40 }, (_, w) => 1 + ((w * 7 + k * 3) % 11) / 10),
    )
    const given = (k: number) =>
      latentDirichletAllocation
        .with({ ...hyper, maxIter: 3, componentsInit: start, docInit: 'constant' })
        .fit({ x: bars.x }, { key: key(k) }).params.components
    expect(given(0)).toEqual(given(1))
  })

  it('stops early when the perplexity stalls', () => {
    const f = latentDirichletAllocation
      .with({ ...hyper, maxIter: 100, evaluateEvery: 1, perpTol: 0.05 })
      .fit({ x: bars.x }, { key: key(2), logs: 'all' })
    expect(f.params.nIter).toBeLessThan(100)
  })

  it('a session plays the topics forming: the topics record at any step, the model at that step, states round-trip', () => {
    const s = latentDirichletAllocation.with({ ...hyper }).session({ x: bars.x }, { key: key(2), logs: 'all' })
    s.advance({ steps: 4 })
    const early = s.probe(1, 'lda/topics') as { kind: string; space: string; components: number[][] }
    const late = s.probe(4, 'lda/topics') as { components: number[][] }
    expect(early.kind).toBe('components')
    expect(early.space).toBe('vocabulary')
    expect(recovery(late.components, bars.topics)).toBeGreaterThan(recovery(early.components, bars.topics))
    expect((s.probe(2, 'model/embedding') as number[][]).length).toBe(150)
    expect(s.fittedAt(1).params.nIter).toBe(1)
    const state = s.state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })

  it('a fit records the final topics, with the words as features', () => {
    const f = latentDirichletAllocation
      .with({ ...hyper, maxIter: 2, vocabulary: bars.vocabulary })
      .fit({ x: bars.x }, { key: key(2) })
    const record = f.exposition.find((e) => e.kind === 'components')
    expect(record).toBeDefined()
    expect((record as unknown as { features: string[] }).features).toEqual(bars.vocabulary)
    expect(f.featureNamesOut?.().length).toBe(4)
  })

  it('rejects negative counts, ragged documents and an empty corpus', () => {
    const start = [[1, 1]]
    expect(() => ldaAlgorithm([[1, -1]], start)).toThrow(/non-negative/)
    expect(() => ldaAlgorithm([[1, 1], [1]], start)).toThrow(/same number/)
    expect(() => ldaAlgorithm([], start)).toThrow(/at least one/)
    expect(() => ldaAlgorithm([[1, 1]], [[1, 1, 1]])).toThrow(/initial components/)
    expect(() => latentDirichletAllocation.with({ docTopicPrior: -1 }).fit({ x: [[1, 2]] })).toThrow(/positive/)
  })

  it('a document with no words keeps its prior proportions', () => {
    const x = [
      [3, 0, 0],
      [0, 0, 0],
      [0, 4, 4],
    ]
    const f = latentDirichletAllocation.with({ nComponents: 2, maxIter: 3 }).fit({ x }, { key: key(1) })
    const theta = toFlat(f.transform(x))
    expect(theta[2]).toBeCloseTo(0.5, 12)
    expect(theta[3]).toBeCloseTo(0.5, 12)
    const alg = runAlgorithm(
      ldaAlgorithm(
        x,
        [
          [1, 1, 1],
          [1, 1, 1],
        ],
        { docInit: 'constant' },
      ),
      { maxSteps: 2 },
    )
    expect(Number.isFinite(alg.state.bound)).toBe(true)
  })
})
