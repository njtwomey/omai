import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import * as P from '../index.ts'
import * as L from './index.ts'

describe('Markov equivalence against pgmpy', () => {
  for (const c of loadFixture('pgm/learning/equivalence').cases) {
    it(c.name, () => {
      const nodes = c.params['nodes'] as string[]
      if (c.fn === 'graph') {
        const g = P.dag(c.params['edges'] as [string, string][], nodes)
        expect(L.immoralities(g)).toEqual(c.params['immoralities'])
        const cp = L.cpdag(g)
        expect(cp.directed).toEqual(c.params['directed'])
        expect(cp.undirected).toEqual(c.params['undirected'])
        // Any consistent extension of the CPDAG is Markov equivalent to the DAG.
        const back = L.pdagToDag(cp)
        expect(L.isMarkovEquivalent(back, g)).toBe(true)
        expect(L.structuralHammingDistance(L.cpdag(back), cp)).toBe(0)
        return
      }
      const a = P.dag(c.params['a'] as [string, string][], nodes)
      const b = P.dag(c.params['b'] as [string, string][], nodes)
      expect(L.isMarkovEquivalent(a, b)).toBe(c.params['equivalent'])
      expect(L.structuralHammingDistance(a, b)).toBe(c.params['shd'])
    })
  }
})

describe('PDAGs', () => {
  it('reject conflicting edges and impossible extensions', () => {
    expect(() => L.pdag([], [['a', 'b']], [['b', 'a']])).toThrow(/both/)
    // a - b - c - d - a has no consistent extension (every orientation adds an immorality or a cycle).
    const cycle = L.pdag(
      [],
      [],
      [
        ['a', 'b'],
        ['b', 'c'],
        ['c', 'd'],
        ['d', 'a'],
      ],
    )
    expect(() => L.pdagToDag(cycle)).toThrow(/consistent extension/)
    expect(
      L.meekRules(
        L.pdag(
          [],
          [
            ['a', 'b'],
            ['b', 'c'],
          ],
          [['a', 'c']],
        ),
      ).directed,
    ).toContainEqual(['a', 'c'])
  })
})
