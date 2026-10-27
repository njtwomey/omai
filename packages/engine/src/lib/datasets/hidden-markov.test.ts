import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { datasetGenerators } from './catalogue.ts'
import { datasetCollections } from './collections.ts'
import { dishonestCasino, regimeSwitching } from './hidden-markov.ts'

describe('hidden-state sequences', () => {
  it('rolls the dishonest casino: faces 0 to 5, a die per roll, sixes when loaded', () => {
    const { data, truth } = generate(dishonestCasino, { key: key(0), knobs: { length: 4000 } })
    expect(data.x.shape).toEqual([4000, 1])
    expect(data.classes).toEqual(['fair', 'loaded'])
    const faces = toFlat(data.x)
    const dice = toFlat(data.y)
    expect(faces.every((f) => Number.isInteger(f) && f >= 0 && f <= 5)).toBe(true)
    const sixes = (die: number): number => {
      const rolls = faces.filter((_, t) => dice[t] === die)
      return rolls.filter((f) => f === 5).length / rolls.length
    }
    expect(sixes(0)).toBeCloseTo(1 / 6, 1)
    expect(sixes(1)).toBeCloseTo(0.5, 1)
    expect(truth.extra?.['model']).toMatchObject({ kind: 'hmm' })
  })

  it('switches between regimes that differ in level and spread, and is reproducible', () => {
    const a = generate(regimeSwitching, { key: key(2), knobs: { length: 3000, gap: 4 } })
    const b = generate(regimeSwitching, { key: key(2), knobs: { length: 3000, gap: 4 } })
    expect(Array.from(toFlat(a.data.x))).toEqual(Array.from(toFlat(b.data.x)))
    const x = toFlat(a.data.x)
    const regime = toFlat(a.data.y)
    const mean = (r: number): number => {
      const v = x.filter((_, t) => regime[t] === r)
      return v.reduce((s, u) => s + u, 0) / v.length
    }
    expect(mean(0)).toBeCloseTo(0, 0)
    expect(mean(1)).toBeCloseTo(4, 0)
  })

  it('is in the catalogue and in the sequences collection', () => {
    expect(datasetGenerators['dishonestCasino']).toBe(dishonestCasino)
    expect(datasetCollections['sequences']?.entries.map((e) => e.name)).toEqual(['dishonestCasino', 'regimeSwitching'])
  })
})
