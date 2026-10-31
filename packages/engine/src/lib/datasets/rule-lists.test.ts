import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { plantedRuleList } from './index.ts'

type Truth = {
  rules: { literals: { feature: number; value: number }[]; label: number }[]
  predict: (row: ArrayLike<number>) => number
  clean: number[]
  bayesAccuracy: number
  noise: number
}

const draw = (
  knobs: Record<string, number>,
  seed: number,
): { x: Float64Array; y: Float64Array; truth: Truth; d: number } => {
  const { data, truth } = generate(plantedRuleList, { key: key(seed), knobs })
  return {
    x: Float64Array.from(toFlat(data.x)),
    y: Float64Array.from(toFlat(data.y)),
    truth: (truth.extra ?? {}) as unknown as Truth,
    d: data.x.shape[1] ?? 0,
  }
}

describe('plantedRuleList', () => {
  it('labels follow the planted list exactly without noise', () => {
    const { x, y, truth, d } = draw({ n: 300, noise: 0, rules: 3 }, 1)
    for (let i = 0; i < 300; i++) expect(y[i]).toBe(truth.predict(x.subarray(i * d, (i + 1) * d)))
    expect(truth.bayesAccuracy).toBe(1)
    expect(truth.rules.length).toBe(3)
  })

  it('the first matching rule decides and the default class takes the rest', () => {
    const { x, truth, d } = draw({ n: 200, noise: 0, rules: 4, classes: 3 }, 2)
    for (let i = 0; i < 200; i++) {
      const row = x.subarray(i * d, (i + 1) * d)
      const first = truth.rules.find((r) => r.literals.every((l) => row[l.feature] === l.value))
      expect(truth.predict(row)).toBe(first?.label ?? 0)
    }
    expect(truth.rules.every((r) => r.label >= 1 && r.label <= 2)).toBe(true)
  })

  it('flips about the noise share of labels, and the truth says so', () => {
    const { y, truth } = draw({ n: 4000, noise: 0.2 }, 3)
    const flipped = truth.clean.filter((c, i) => c !== y[i]).length / 4000
    expect(flipped).toBeCloseTo(0.2, 1)
    expect(truth.bayesAccuracy).toBeCloseTo(0.8, 12)
  })

  it('draws with different keys come from the same list, a different listSeed from another', () => {
    const a = draw({ n: 50 }, 1).truth.rules
    expect(draw({ n: 50 }, 9).truth.rules).toEqual(a)
    expect(draw({ n: 50, listSeed: 5 }, 1).truth.rules).not.toEqual(a)
  })

  it('is deterministic given the key', () => {
    expect(Array.from(draw({ n: 60 }, 4).y)).toEqual(Array.from(draw({ n: 60 }, 4).y))
  })
})
