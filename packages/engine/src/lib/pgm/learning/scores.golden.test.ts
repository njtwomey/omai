import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import { tensor } from '../../tensor/index.ts'
import * as P from '../index.ts'
import * as L from './index.ts'

describe('structure scores against pgmpy', () => {
  for (const c of loadFixture('pgm/learning/scores').cases) {
    it(c.name, () => {
      const data = c.params['data'] as L.DataRow[]
      const states = c.params['states'] as Record<string, string[]> | undefined
      const score = c.params['score'] as L.DiscreteScore | L.GaussianScore
      const options = states === undefined ? {} : { states }
      if (c.fn === 'structure') {
        const g = P.dag(c.params['edges'] as [string, string][])
        expectMatches(L.structureScore(g, data, score, options), c.output, c.rtol, 0, c.name)
        return
      }
      const families = c.params['families'] as [string, string[]][]
      const values = families.map(([v, ps]) =>
        c.fn === 'gaussian'
          ? L.gaussianLocalScore(data, v, ps, score as L.GaussianScore)
          : L.localScore(data, v, ps, score as L.DiscreteScore, options),
      )
      expectMatches(tensor(values), c.output, c.rtol, 0, c.name)
    })
  }
})

describe('independence tests against pgmpy', () => {
  for (const c of loadFixture('pgm/learning/independence').cases) {
    it(c.name, () => {
      const data = c.params['data'] as L.DataRow[]
      const x = c.params['x'] as string
      const y = c.params['y'] as string
      const z = c.params['z'] as string[]
      const r =
        c.fn === 'chiSquare'
          ? L.chiSquareTest(data, x, y, z)
          : c.fn === 'gSquare'
            ? L.gSquareTest(data, x, y, z)
            : c.fn === 'fisherZ'
              ? L.fisherZTest(data, x, y, z)
              : L.partialCorrelationTest(data, x, y, z)
      const got =
        c.fn === 'chiSquare' || c.fn === 'gSquare' ? [r.statistic, r.pValue, r.dof ?? 0] : [r.statistic, r.pValue]
      expectMatches(tensor(got), c.output, c.rtol, c.atol, c.name)
      expect(r.independent).toBe(r.pValue >= 0.01)
    })
  }
})

describe('independence oracles', () => {
  it('memoise, accept decision functions, and are symmetric', () => {
    const data = Array.from({ length: 60 }, (_, i) => ({ a: i % 2, b: (i % 2) ^ (i % 7 === 0 ? 1 : 0), c: i % 3 }))
    const oracle = L.independenceOracle(data, 'gSquare')
    expect(oracle('a', 'b', [])).toBe(oracle('b', 'a', []))
    expect(L.independenceOracle(data, () => true)('a', 'b', [])).toBe(true)
    expect(() => L.fisherZTest([{ a: 1, b: 2 }], 'a', 'b')).toThrow(/too few/)
  })
})
