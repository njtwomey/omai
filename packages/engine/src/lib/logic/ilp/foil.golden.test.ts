import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { clauseToString } from '../index.ts'
import { type FoilState, foil, foilAlgorithm, type Tuple } from './index.ts'

describe('FOIL against a reference port on Quinlan tasks', () => {
  for (const c of loadFixture('logic/ilp/foil').cases) {
    it(c.name, () => {
      const p = c.params
      const task = {
        target: p['target'] as string,
        arity: p['arity'] as number,
        positive: p['positive'] as Tuple[],
        background: p['background'] as Record<string, Tuple[]>,
      }
      const options = { recursion: p['recursion'] as boolean }
      const r = foil(task, options)
      expect(r.clauses.map((x) => clauseToString(x))).toEqual(p['clauses'])
      expect(r.status).toBe('complete')
      // The gain of each chosen literal, from the trace of states.
      const gains: number[] = []
      const algorithm = foilAlgorithm(task, options)
      const wrapped = {
        ...algorithm,
        step: (s: FoilState, ctx: Parameters<typeof algorithm.step>[1]) => {
          const n = algorithm.step(s, ctx)
          if (n.event?.kind === 'literal') gains.push(n.event.best[0]?.gain ?? Number.NaN)
          return n
        },
      }
      runAlgorithm(wrapped, { maxSteps: 1000 })
      const expected = p['gains'] as number[]
      expect(gains.length).toBe(expected.length)
      gains.forEach((g, i) => {
        expect(g).toBeCloseTo(expected[i] ?? Number.NaN, 9)
      })
    })
  }
})
