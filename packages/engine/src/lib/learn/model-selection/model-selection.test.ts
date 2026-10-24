import { describe, expect, it } from 'vitest'
import { gaussianNB, leastSquares, ridge } from '../../../testing/learn.ts'
import { key } from '../../random/index.ts'
import { recorder, runTree } from '../../trace/index.ts'
import { defineEstimator } from '../estimator.ts'
import { bootstrap632, mallowsCp, oneStandardErrorRule } from './criteria.ts'
import { crossValidate, crossValPredict, permutationTestScore } from './cross-validation.ts'
import { gridSearchCV, nestedCrossValidation, parameterGrid, parameterSampler, randomizedSearchCV } from './search.ts'
import { shuffleSplit } from './splitters.ts'

const x = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i), i / 30])
const y = x.map(([a = 0, b = 0, c = 0]) => 2 * a - b + 3 * c)
const labels = x.map(([a = 0]) => (a > 0 ? 1 : 0))

describe('cross-validation', () => {
  it('runs each fold as a child run under the cross-validation run', () => {
    const rec = recorder()
    crossValidate(ridge(), { x, y }, { cv: 3, tracer: rec })
    const tree = runTree(rec.run())
    expect(tree.length).toBe(1)
    expect(tree[0]?.children.length).toBe(3)
  })

  it('substitutes the error score for failing fits when asked', () => {
    const failing = defineEstimator(
      {
        name: 'failing',
        kind: 'regressor',
        fit: () => {
          throw new Error('no')
        },
        methods: () => ({}),
      },
      {},
    )
    expect(crossValidate(failing, { x, y }, { cv: 3, errorScore: -1, scoring: 'r2' }).testScores['r2']).toEqual([
      -1, -1, -1,
    ])
    expect(() => crossValidate(failing, { x, y }, { cv: 3 })).toThrow('no')
  })

  it('needs a partition for out-of-fold predictions', () => {
    expect(() => crossValPredict(ridge(), { x, y }, { cv: shuffleSplit(3) })).toThrow(/partition/)
  })

  it('a permutation test finds a real signal and not a fake one', () => {
    const real = permutationTestScore(
      gaussianNB(),
      { x, y: labels },
      { cv: 3, nPermutations: 19, scoring: 'accuracy', key: key(2) },
    )
    expect(real.pValue).toBeLessThanOrEqual(0.1)
    const fake = permutationTestScore(
      gaussianNB(),
      { x, y: labels.map((_, i) => (i * 7) % 2) },
      { cv: 3, nPermutations: 19, scoring: 'accuracy', key: key(2) },
    )
    expect(fake.pValue).toBeGreaterThan(0.05)
  })
})

describe('searches', () => {
  it('expand grids and sample candidates reproducibly', () => {
    expect(parameterGrid([{ a: [1] }, { b: [2, 3] }])).toEqual([{ a: 1 }, { b: 2 }, { b: 3 }])
    expect(parameterSampler({ a: [1, 2, 3], b: [0, 1] }, 4, key(1))).toEqual(
      parameterSampler({ a: [1, 2, 3], b: [0, 1] }, 4, key(1)),
    )
    const r = randomizedSearchCV(ridge(), undefined, { x, y }, { nIter: 3, cv: 3, scoring: 'r2' })
    expect(r.params.length).toBe(3)
  })

  it('refit the best candidate and nest selection inside cross-validation', () => {
    const r = gridSearchCV(ridge(), { alpha: [1e-6, 10, 1000] }, { x, y }, { cv: 3, scoring: 'r2' })
    expect(r.bestParams).toEqual({ alpha: 1e-6 })
    expect(r.bestEstimator).toBeDefined()
    const nested = nestedCrossValidation(
      ridge(),
      { alpha: [1e-6, 1000] },
      { x, y },
      { outerCv: 3, innerCv: 2, scoring: 'r2' },
    )
    expect(nested.chosen.every((c) => c['alpha'] === 1e-6)).toBe(true)
    expect(nested.mean).toBeGreaterThan(0.99)
  })
})

describe('criteria', () => {
  it('choose the simplest model within one standard error', () => {
    expect(oneStandardErrorRule([0.9, 0.85, 0.7], [0.1, 0.1, 0.1], { complexity: [3, 2, 1] })).toBe(1)
    expect(mallowsCp(10, 1, 10, 5)).toBe(10)
  })

  it('the .632 bootstrap of an exact model is zero and .632+ never falls below the training error', () => {
    const exact = bootstrap632(leastSquares(), { x, y }, { nBootstraps: 30, key: key(4) })
    expect(exact.error).toBeLessThan(1e-12)
    const noisy = y.map((v, i) => v + Math.sin(17 * i))
    const plus = bootstrap632(ridge({ alpha: 0.1 }), { x, y: noisy }, { nBootstraps: 30, plus: true, key: key(4) })
    expect(plus.error).toBeGreaterThanOrEqual(plus.trainingError)
    expect(plus.weight).toBeGreaterThanOrEqual(0.632)
  })
})
