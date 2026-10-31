import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { analogyCorpus } from '../datasets/index.ts'
import { flatObjective } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { nearestWords } from '../text/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { analogyAccuracy, neighbourPurity } from './evaluation.ts'
import {
  subwordVector,
  trainWord2vec,
  type Word2vecOptions,
  word2vecProblem,
  word2vecSgdAlgorithm,
  word2vecSolvers,
} from './word2vec.ts'

const toy = [
  ['the', 'cat', 'sat', 'on', 'the', 'mat'],
  ['the', 'dog', 'sat', 'on', 'the', 'log'],
  ['a', 'cat', 'saw', 'a', 'dog'],
  ['the', 'dog', 'saw', 'the', 'cat'],
]

describe('the word2vec loss', () => {
  const configs: [string, Word2vecOptions][] = [
    ['skip-gram with negative sampling', { architecture: 'skip-gram', output: 'negative-sampling' }],
    ['CBOW with negative sampling', { architecture: 'cbow', output: 'negative-sampling' }],
    ['skip-gram with the hierarchical softmax', { architecture: 'skip-gram', output: 'hierarchical-softmax' }],
    ['CBOW with the hierarchical softmax', { architecture: 'cbow', output: 'hierarchical-softmax' }],
    ['fastText subwords', { architecture: 'skip-gram', subwords: { minn: 2, maxn: 3, buckets: 20 } }],
  ]
  for (const [name, options] of configs) {
    it(`${name}: one SGD update is a step along the autodiff gradient`, () => {
      const p = word2vecProblem(
        toy,
        { dimension: 3, window: 2, negatives: 3, dynamicWindow: false, ...options },
        key(3),
      )
      // Move the output rows off zero so every part of the gradient is live.
      const x0 = Float64Array.from(p.x0 as Float64Array, (v, i) =>
        i >= p.inputRows * 3 ? Math.sin(i) * 0.3 : v + Math.cos(i) * 0.2,
      )
      const lr = 1e-7
      const probe = { ...p, x0 }
      const alg = word2vecSgdAlgorithm(probe, { learningRate: lr, minLearningRate: 1, epochs: 1 })
      const s0 = { ...alg.init(key(0)), x: Float64Array.from(x0) }
      const s1 = alg.step(s0, { t: 0, key: key(0) } as never)
      const f = flatObjective((q) => p.objective(q, [0]), x0)
      const { grad, value } = f.valueAndGrad(x0)
      for (let i = 0; i < x0.length; i++) {
        const moved = ((x0[i] ?? 0) - (s1.x[i] ?? 0)) / lr
        expect(moved).toBeCloseTo(grad[i] ?? 0, 4)
      }
      expect(s1.value).toBeCloseTo(value, 10)
    })
  }

  it('starts at word2vec initial parameters with the loss of an untrained model', () => {
    const p = word2vecProblem(toy, { dimension: 4, negatives: 5 }, key(0))
    const f = flatObjective(p.objective, p.x0)
    // Output rows are zero, so every logit is 0 and the loss is (1 + negatives) log 2.
    expect(f.value(p.x0 as Float64Array)).toBeCloseTo(6 * Math.LN2, 12)
  })

  it('subsampling and a dynamic window change the examples but not the vocabulary', () => {
    const a = word2vecProblem(toy, { window: 3, dynamicWindow: false }, key(1))
    const b = word2vecProblem(toy, { window: 3, dynamicWindow: true }, key(1))
    expect(b.examples.n).toBeLessThan(a.examples.n)
    expect(b.vocabulary.words).toEqual(a.vocabulary.words)
    const c = word2vecProblem(toy, { window: 3, dynamicWindow: false, subsample: 0.01 }, key(1))
    expect(c.examples.n).toBeLessThan(a.examples.n)
  })
})

describe('training on the planted corpus', () => {
  const { data, truth } = generate(analogyCorpus, {
    key: key(5),
    knobs: { concepts: 4, attributes: 3, contexts: 3, sentences: 2500, noise: 0.15 },
  })
  const sentences = (data.documents ?? []).map((d) => d.split(' '))
  const quads = truth.extra?.['analogies'] as [string, string, string, string][]
  const targets = truth.extra?.['targets'] as string[]
  const concept = truth.extra?.['conceptOf'] as Record<string, number>

  it("word2vec's own SGD recovers the planted analogies and concept clusters", () => {
    const m = trainWord2vec(
      sentences,
      { dimension: 16, window: 4, negatives: 5, solver: { kind: 'word2vecSgd', epochs: 8, learningRate: 0.05 } },
      key(1),
    )
    const chance = 1 / (m.words.length - 3)
    const acc = analogyAccuracy(m.vectors, quads, { candidates: targets })
    expect(acc.total).toBe(quads.length)
    expect(acc.accuracy).toBeGreaterThan(0.9)
    expect(acc.accuracy).toBeGreaterThan(8 * chance)
    // A target's vector is its concept part plus its attribute part, so its nearest targets share one of the two.
    const attribute = truth.extra?.['attributeOf'] as Record<string, number>
    const targetOnly = { words: targets, rows: targets.map((w) => m.vectors.rows[m.words.indexOf(w)] as Float64Array) }
    let related = 0
    let seen = 0
    for (const w of targets)
      for (const n of nearestWords(targetOnly, w, 2)) {
        seen++
        if (concept[n.word] === concept[w] || attribute[n.word] === attribute[w]) related++
      }
    expect(related / seen).toBeGreaterThan(0.9)
    // Context words of one concept occur in the same sentences, so they cluster.
    const contextCluster: Record<string, number> = {}
    for (const w of m.words) if (/^c\d+_\d+$/.test(w)) contextCluster[w] = Number(w.slice(1, w.indexOf('_')))
    expect(neighbourPurity(m.vectors, contextCluster, 2)).toBeGreaterThan(0.9)
  })

  it('CBOW with the hierarchical softmax also clusters the context words', () => {
    const m = trainWord2vec(
      sentences,
      {
        architecture: 'cbow',
        output: 'hierarchical-softmax',
        dimension: 16,
        window: 4,
        solver: { kind: 'word2vecSgd', epochs: 8, learningRate: 0.05 },
      },
      key(2),
    )
    const contextCluster: Record<string, number> = {}
    for (const w of m.words) if (/^c\d+_\d+$/.test(w)) contextCluster[w] = Number(w.slice(1, w.indexOf('_')))
    expect(neighbourPurity(m.vectors, contextCluster, 2)).toBeGreaterThan(0.8)
  })
})

describe('the solver slot', () => {
  it('offers its own SGD beside the generic solvers', () => {
    expect(word2vecSolvers.names[0]).toBe('word2vecSgd')
    for (const n of ['gradientDescent', 'lbfgs', 'bfgs', 'gradient']) expect(word2vecSolvers.names).toContain(n)
  })

  it('L-BFGS, Adam minibatches and word2vec SGD all reduce the same loss', () => {
    const hyper = { dimension: 4, window: 2, negatives: 3, dynamicWindow: false } as const
    const p = word2vecProblem(toy, hyper, key(4))
    const start = flatObjective(p.objective, p.x0).value(p.x0 as Float64Array)
    const finals: number[] = []
    for (const solver of [
      { kind: 'lbfgs' as const },
      { kind: 'gradient' as const, learningRate: 0.05, batch: 8, optimiser: { kind: 'adam' as const } },
      { kind: 'word2vecSgd' as const, epochs: 60, learningRate: 0.1 },
    ]) {
      const m = trainWord2vec(toy, { ...hyper, solver, maxSteps: solver.kind === 'word2vecSgd' ? 5000 : 300 }, key(4))
      finals.push(m.loss)
    }
    for (const f of finals) expect(f).toBeLessThan(start * 0.75)
  })

  it('runs word2vec SGD as a stepped algorithm whose state serialises', () => {
    const p = word2vecProblem(toy, { dimension: 3, window: 1, negatives: 2 }, key(6))
    const alg = word2vecSgdAlgorithm(p, { epochs: 2, batch: 3, shuffle: true })
    const r = runAlgorithm(alg, { seed: 1, maxSteps: 10_000 })
    expect(r.steps).toBe(Math.ceil(p.examples.n / 3) * 2)
    const back = alg.deserialise?.(alg.serialise?.(r.state) as never)
    expect(Array.from(back?.x ?? [])).toEqual(Array.from(r.state.x))
  })
})

describe('fastText subwords', () => {
  it('embeds an unseen word from its character n-grams', () => {
    const corpus = [
      ['walking', 'walked', 'walks', 'talking', 'talked', 'talks'],
      ['walking', 'talking', 'walked', 'talked'],
      ['running', 'runs', 'ran'],
    ]
    const m = trainWord2vec(
      corpus,
      {
        dimension: 8,
        window: 3,
        subwords: { minn: 3, maxn: 5, buckets: 200 },
        solver: { kind: 'word2vecSgd', epochs: 30, learningRate: 0.05 },
      },
      key(7),
    )
    const v = subwordVector(m, 'walker')
    expect(v).toBeDefined()
    expect(v?.some((x) => x !== 0)).toBe(true)
    expect(subwordVector(m, 'walking')).toEqual(m.vectors.rows[m.words.indexOf('walking')])
  })
})
