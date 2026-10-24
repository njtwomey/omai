import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { createContext } from '../context/index.ts'
import { batches, generate, trainTestSplit } from '../data/index.ts'
import { key } from '../random/index.ts'
import { item, mean, square, sub, type Tensor, toFlat } from '../tensor/index.ts'
import * as DS from './index.ts'

describe('datasets against sklearn.datasets', () => {
  for (const c of loadFixture('datasets/regression').cases) {
    it(c.name, () => {
      if (c.fn === 'diabetes') {
        const d = DS.diabetes({ scaled: c.params['scaled'] as boolean })
        expectMatches(c.params['part'] === 'x' ? d.x : d.y, c.output, c.rtol, c.atol, c.name)
        return
      }
      const f = (DS as unknown as Record<string, (x: Tensor) => Tensor>)[c.fn]
      if (f === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(f(decode(c.inputs[0]) as Tensor), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('generators', () => {
  it('are reproducible from a seed and respond to knobs', () => {
    const a = generate(DS.friedman1, { ctx: createContext({ seed: 3 }), knobs: { n: 40, noise: 0.5 } })
    const b = generate(DS.friedman1, { ctx: createContext({ seed: 3 }), knobs: { n: 40, noise: 0.5 } })
    expect(toFlat(a.data.y)).toEqual(toFlat(b.data.y))
    expect(a.data.x.shape).toEqual([40, 10])
    expect(a.data.meta.recipe?.generator).toBe('friedman1')
  })

  it('carry a truth whose noise matches the residuals', () => {
    const { data, truth } = generate(DS.friedman1, { key: key(1), knobs: { n: 4000, noise: 2 } })
    const residual = sub(data.y, truth.regressionFunction?.(data.x) as Tensor)
    expect(Math.sqrt(item(mean(square(residual))))).toBeCloseTo(2, 1)
    expect(truth.bayesError).toBe(4)
  })

  it('makeRegression is linear in its informative features', () => {
    const { data, truth } = generate(DS.makeRegression, {
      key: key(2),
      knobs: { n: 20, nFeatures: 4, nInformative: 2, bias: 3 },
    })
    const coef = truth.extra?.['coefficients'] as number[]
    expect(coef.slice(2)).toEqual([0, 0])
    const row = toFlat(data.x).slice(0, 4)
    expect(toFlat(data.y)[0]).toBeCloseTo(3 + row.reduce((s, v, j) => s + v * (coef[j] ?? 0), 0), 10)
  })
})

describe('rows and batches', () => {
  const data = DS.diabetes()
  it('split like sklearn (test size rounded up)', () => {
    const { train, test } = trainTestSplit(data, { testSize: 0.2, key: key(0) })
    expect([train.x.shape[0], test.x.shape[0]]).toEqual([353, 89])
    expect(train.featureNames).toEqual(data.featureNames)
  })

  it('batch every row exactly once per epoch', () => {
    const parts = batches(data, { batchSize: 100, key: key(5) })
    expect(parts.map((p) => p.y.shape[0])).toEqual([100, 100, 100, 100, 42])
    const all = parts.flatMap((p) => toFlat(p.y)).sort((x, y) => x - y)
    expect(all).toEqual(toFlat(data.y).sort((x, y) => x - y))
  })
})

describe('generating programs (the truth as a probprog program)', () => {
  it('friedman1: the program gives the exact log density and regression function', () => {
    const g = generate(DS.friedman1, { ctx: createContext({ seed: 5 }), knobs: { n: 12, noise: 0.7, nFeatures: 6 } })
    const program = g.truth.program
    expect(program?.kind).toBe('program')
    const exact = toFlat(g.truth.logProb?.(g.data.x, g.data.y) ?? g.data.y)
    const viaProgram = toFlat(program?.logProb(g.data.x, g.data.y) ?? g.data.y)
    viaProgram.forEach((v, i) => {
      expect(v).toBeCloseTo(exact[i] ?? 0, 10)
    })
    const f = toFlat(program?.regressionFunction?.(g.data.x) ?? g.data.y)
    toFlat(DS.friedman1Function(g.data.x)).forEach((v, i) => {
      expect(f[i]).toBeCloseTo(v, 12)
    })
    expect(program?.sample(key(0), 4).x.shape).toEqual([4, 6])
  })

  it('blobs: the mixture program reproduces the exact joint density and class posterior', () => {
    const g = generate(DS.blobs, { ctx: createContext({ seed: 2 }), knobs: { n: 9, centers: 3, nFeatures: 2 } })
    const program = g.truth.program
    expect(program?.classes).toBe(3)
    const x = g.data.x
    const y = g.data.y
    const exact = toFlat(g.truth.logProb?.(x, y) ?? y)
    toFlat(program?.logProb(x, y) ?? y).forEach((v, i) => {
      expect(v).toBeCloseTo(exact[i] ?? 0, 10)
    })
    const marginal = toFlat(g.truth.logProb?.(x) ?? y)
    toFlat(program?.logProb(x) ?? y).forEach((v, i) => {
      expect(v).toBeCloseTo(marginal[i] ?? 0, 10)
    })
    const post = toFlat(program?.posterior?.(x) ?? y)
    toFlat(g.truth.posterior?.(x) ?? y).forEach((v, i) => {
      expect(post[i]).toBeCloseTo(v, 10)
    })
  })
})
