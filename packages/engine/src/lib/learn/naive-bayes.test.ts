import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { revive, serialiseFitted } from './estimator.ts'
import {
  aode,
  conditionalMutualInformation,
  discreteCounts,
  gaussianNaiveBayes,
  multinomialNaiveBayes,
  naiveBayes,
  naiveBayesInit,
  naiveBayesPartialFit,
  treeAugmentedNaiveBayes,
} from './naive-bayes.ts'

const docs = {
  x: [
    [3, 0, 1],
    [2, 1, 0],
    [4, 0, 0],
    [0, 3, 1],
    [1, 2, 2],
    [0, 4, 1],
  ],
  y: [0, 0, 0, 1, 1, 1],
}

describe('naive Bayes', () => {
  it('classifies and exposes the per-feature log-likelihoods that its joint sums', () => {
    const f = multinomialNaiveBayes().fit(docs)
    expect(
      toFlat(
        f.predict([
          [5, 0, 0],
          [0, 5, 1],
        ]),
      ),
    ).toEqual([0, 1])
    const ll = f.featureLogLikelihood([[2, 1, 0]])
    const joint = f.classLogPrior.map((p, c) => p + (ll[0]?.[c] ?? []).reduce((a, b) => a + b, 0))
    const proba = toFlat(f.predictProba([[2, 1, 0]]))
    const z = Math.log(joint.reduce((a, b) => a + Math.exp(b), 0))
    expect(proba[0]).toBeCloseTo(Math.exp((joint[0] ?? 0) - z), 12)
  })

  it('updates one row at a time to the batch fit, in any order', () => {
    for (const family of ['gaussian', 'multinomial', 'bernoulli', 'complement'] as const) {
      const h = { family, binarize: 0.5 } as const
      const batch = naiveBayes({ family, binarize: 0.5 }).fit(docs).params
      let online = naiveBayesInit([0, 1], 3, h)
      for (const i of [5, 0, 3, 1, 4, 2]) online = naiveBayesPartialFit(online, h, docs.x[i] ?? [], docs.y[i] ?? 0)
      expect(online.classCount).toEqual(batch.classCount)
      if (family !== 'gaussian') expect(online.featureCount).toEqual(batch.featureCount)
      else expect(online.mean.flat()).toEqual(batch.mean.flat().map((v) => expect.closeTo(v, 12)))
    }
  })

  it('logs the prequential accuracy as the rows arrive, and the model at an early step', () => {
    const est = gaussianNaiveBayes()
    const s = est.session(docs, { key: key(0), logs: 'all' })
    s.advance({ steps: 5 })
    const acc = metricSeries(s.run(), 'solver/accuracy')
    expect(acc.values.length).toBeGreaterThan(0)
    expect(s.fittedAt(2).params.classCount).toEqual([3, 0])
    expect(s.probe(3, 'model/params')).toMatchObject({ classCount: [3, 1] })
    expect(toFlat(s.fittedAt(5).predict([[0, 4, 0]]))).toEqual([1])
    const back = revive(est, JSON.parse(JSON.stringify(serialiseFitted(est.fit(docs)))))
    expect(toFlat(back.predict([[0, 4, 0]]))).toEqual([1])
  })

  it('applies a partialFit verb to a fitted value', () => {
    const f = multinomialNaiveBayes().fit(docs)
    const more = f.partialFit([[9, 0, 0]], [1])
    expect(more.classCount).toEqual([3, 4])
    expect(more.featureCount[1]).toEqual([
      (f.params.featureCount[1]?.[0] ?? 0) + 9,
      f.params.featureCount[1]?.[1] ?? 0,
      f.params.featureCount[1]?.[2] ?? 0,
    ])
  })
})

describe('tree-augmented naive Bayes and AODE', () => {
  const x = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ]
  const y = [0, 1, 1, 0, 0, 1, 1, 0]

  it('learns the dependence naive Bayes cannot: the XOR of two features', () => {
    const nb = naiveBayes({ family: 'categorical' }).fit({ x, y })
    expect(nb.score({ x, y })).toBeLessThan(1)
    expect(treeAugmentedNaiveBayes().fit({ x, y }).score({ x, y })).toBe(1)
    expect(aode({ alpha: 0.1 }).fit({ x, y }).score({ x, y })).toBe(1)
  })

  it('grows one edge per step, from naive Bayes at the start to the whole tree', () => {
    const data = {
      x: [
        [0, 0, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 0],
        [0, 0, 0],
        [1, 1, 1],
      ],
      y: [0, 0, 0, 0, 1, 1, 1, 1],
    }
    const est = treeAugmentedNaiveBayes()
    const s = est.session(data, { key: key(0), logs: 'all' })
    s.advance()
    expect(s.fittedAt(0).edges).toEqual([])
    expect(s.fittedAt(1).edges).toHaveLength(1)
    expect(s.fittedAt(1).edges[0]).toMatchObject({ parent: 0, child: 1 })
    expect(s.fittedAt(s.t).edges).toHaveLength(2)
    expect(metricSeries(s.run(), 'solver/edges').values).toEqual([0, 1, 2])
    const cmi = conditionalMutualInformation(discreteCounts(data.x, data.y))
    expect(cmi[0]?.[1]).toBeGreaterThan(cmi[0]?.[2] ?? 0)
    expect(cmi[0]?.[0]).toBe(0)
  })

  it('counts a row at a time for AODE and refuses a category outside the tables', () => {
    const f = aode({ minCategories: 2 }).fit({ x, y })
    const more = f.partialFit([[1, 1]], [0])
    expect(more.classCount).toEqual([5, 4])
    expect(() => f.partialFit([[2, 0]], [0])).toThrow(/outside/)
  })
})
